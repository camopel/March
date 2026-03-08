"""Tests for session memory, reset, and guardian.

Covers:
  - Session memory fields and MemoryStore interface
  - Session/orchestrator reset behavior (history, DB, cache, cross-session isolation)
  - Guardian module imports and configuration
  - Log structure expectations (xfail — not yet implemented)

All tests run without external services. Uses pytest + pytest-asyncio.
Each test is independent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.message import Message, Role
from march.core.orchestrator import (
    FinalResponse,
    Orchestrator,
    OrchestratorEvent,
    TextDelta,
)
from march.core.session import Session, SessionStore
from march.llm.base import StreamChunk


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / Mocks  (reuse patterns from test_orchestrator.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MockAgent:
    """Minimal mock Agent whose ``run_stream`` yields predetermined items."""

    def __init__(self, items=None, *, track_history: bool = True):
        from march.core.agent import AgentResponse
        self.items = items or []
        self.memory = None
        self._track_history = track_history

    async def run_stream(self, user_message, session):
        from march.core.agent import AgentResponse
        for item in self.items:
            if isinstance(item, AgentResponse) and self._track_history:
                session.add_exchange(user_message, item.content)
            yield item


class InMemorySessionStore:
    """Minimal in-memory SessionStore matching the real async interface."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._messages: dict[str, list[Message]] = {}

    async def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def create_session(self, source_type, source_id, name="",
                             session_id=None, metadata=None) -> Session:
        s = Session(id=session_id or "", source_type=source_type,
                    source_id=source_id, name=name, metadata=metadata or {})
        self._sessions[s.id] = s
        self._messages[s.id] = []
        return s

    async def get_messages(self, session_id, limit=None, offset=0) -> list[Message]:
        msgs = self._messages.get(session_id, [])
        if limit is not None:
            return msgs[offset:offset + limit]
        return msgs[offset:]

    async def add_message(self, session_id, message, attachments=None) -> str:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)
        return "msg-id"

    async def clear_session(self, session_id) -> None:
        self._messages.pop(session_id, None)
        # Also remove from sessions dict to mirror real DB clear
        self._sessions.pop(session_id, None)


async def _collect_events(orch, session_id, content, **kwargs) -> list[OrchestratorEvent]:
    events = []
    async for ev in orch.handle_message(session_id, content, source="test", **kwargs):
        events.append(ev)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionMemory
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionMemory:
    """Session memory fields and MemoryStore interface."""

    def test_session_has_memory_fields(self):
        """Session has compaction_summary, rolling_summary, and backup_history fields."""
        s = Session(source_type="test", source_id="mem-test")
        # These fields serve as the session's memory/facts/plan/checkpoint equivalents
        assert hasattr(s, "compaction_summary")
        assert hasattr(s, "rolling_summary")
        assert hasattr(s, "backup_history")
        # Verify defaults
        assert s.compaction_summary == ""
        assert s.rolling_summary == ""
        assert s.backup_history == []
        # Also has metadata dict for arbitrary facts
        assert isinstance(s.metadata, dict)

    def test_memory_store_interface(self):
        """MemoryStore has save (append_memory), load (load_long_term), and reset (reset_session) methods."""
        from march.memory.store import MemoryStore

        # Check the class has the key methods
        assert hasattr(MemoryStore, "append_memory")       # save
        assert hasattr(MemoryStore, "load_long_term")       # load
        assert hasattr(MemoryStore, "reset_session")        # reset
        assert hasattr(MemoryStore, "initialize")
        assert hasattr(MemoryStore, "close")

        # Verify they are coroutines (async methods)
        import inspect
        assert inspect.iscoroutinefunction(MemoryStore.append_memory)
        assert inspect.iscoroutinefunction(MemoryStore.load_long_term)
        assert inspect.iscoroutinefunction(MemoryStore.reset_session)


# ═══════════════════════════════════════════════════════════════════════════════
# TestReset
# ═══════════════════════════════════════════════════════════════════════════════


class TestReset:
    """Session and orchestrator reset behavior."""

    def test_reset_clears_history(self):
        """Create session with messages, reset, verify history empty."""
        s = Session(source_type="test", source_id="reset-hist")
        s.add_message(Message.user("hello"))
        s.add_message(Message.assistant("hi"))
        s.add_message(Message.user("how are you"))
        assert len(s.history) == 3

        s.reset()

        assert len(s.history) == 0
        assert s.backup_history == []
        assert s.compaction_summary == ""
        assert s.rolling_summary == ""
        assert s.state == "reset"

    async def test_reset_clears_db_messages(self):
        """Add messages to SessionStore, reset via orchestrator, verify DB empty."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="Hi"),
            AgentResponse(content="Hi"),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        # Drive a conversation to populate DB
        await _collect_events(orch, "sess-db-reset", "Hello")
        msgs_before = await store.get_messages("sess-db-reset")
        assert len(msgs_before) > 0

        # Reset
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("sess-db-reset")

        # DB messages should be gone
        msgs_after = await store.get_messages("sess-db-reset")
        assert len(msgs_after) == 0

    async def test_reset_preserves_other_sessions(self):
        """Reset session A, verify session B still has data."""
        from march.core.agent import AgentResponse

        store = InMemorySessionStore()

        # Session A
        agent_a = MockAgent([
            StreamChunk(delta="A"),
            AgentResponse(content="A"),
        ])
        orch = Orchestrator(agent=agent_a, session_store=store)
        await _collect_events(orch, "sess-A", "Hello A")

        # Session B
        orch.agent = MockAgent([
            StreamChunk(delta="B"),
            AgentResponse(content="B"),
        ])
        await _collect_events(orch, "sess-B", "Hello B")

        # Verify both have messages
        msgs_a = await store.get_messages("sess-A")
        msgs_b = await store.get_messages("sess-B")
        assert len(msgs_a) > 0
        assert len(msgs_b) > 0

        # Reset session A only
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("sess-A")

        # Session A should be cleared
        msgs_a_after = await store.get_messages("sess-A")
        assert len(msgs_a_after) == 0

        # Session B should be untouched
        msgs_b_after = await store.get_messages("sess-B")
        assert len(msgs_b_after) == len(msgs_b)

    async def test_reset_evicts_cache(self):
        """After reset, orchestrator._sessions no longer has the session."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="X"),
            AgentResponse(content="X"),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await _collect_events(orch, "sess-evict-test", "Hello")
        assert "sess-evict-test" in orch._sessions

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("sess-evict-test")

        assert "sess-evict-test" not in orch._sessions

    async def test_reset_returns_confirmation(self):
        """reset_session returns a dict (success indicator)."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="Y"),
            AgentResponse(content="Y"),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await _collect_events(orch, "sess-confirm", "Hello")

        with patch("march.core.compaction.delete_session_memory", return_value=True):
            result = await orch.reset_session("sess-confirm")

        # Should return a dict with cleanup details
        assert isinstance(result, dict)
        # When delete_session_memory returns True, the key should be set
        assert result.get("session_memory_deleted") is True

    async def test_reset_calls_reset_children(self):
        """reset_session calls agent_manager.reset_children when available."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="Z"),
            AgentResponse(content="Z"),
        ])
        # Set up a mock agent_manager on the agent
        mock_manager = AsyncMock()
        mock_manager.reset_children = AsyncMock(return_value=3)
        agent.agent_manager = mock_manager

        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await _collect_events(orch, "sess-children-test", "Hello")

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            result = await orch.reset_session("sess-children-test")

        # Should have called reset_children with the session_id
        mock_manager.reset_children.assert_called_once_with("sess-children-test")
        assert result.get("children_cleaned") == 3

    async def test_reset_without_agent_manager(self):
        """reset_session works fine when agent has no agent_manager."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="W"),
            AgentResponse(content="W"),
        ])
        # Ensure no agent_manager attribute
        assert not hasattr(agent, "agent_manager")

        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await _collect_events(orch, "sess-no-mgr", "Hello")

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            result = await orch.reset_session("sess-no-mgr")

        # Should succeed without children_cleaned key
        assert isinstance(result, dict)
        assert "children_cleaned" not in result

    async def test_reset_children_failure_nonfatal(self):
        """reset_session continues even if reset_children raises."""
        from march.core.agent import AgentResponse

        agent = MockAgent([
            StreamChunk(delta="V"),
            AgentResponse(content="V"),
        ])
        mock_manager = AsyncMock()
        mock_manager.reset_children = AsyncMock(side_effect=RuntimeError("boom"))
        agent.agent_manager = mock_manager

        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await _collect_events(orch, "sess-children-fail", "Hello")

        with patch("march.core.compaction.delete_session_memory", return_value=True):
            result = await orch.reset_session("sess-children-fail")

        # Should still succeed (non-fatal)
        assert isinstance(result, dict)
        assert result.get("session_memory_deleted") is True
        # children_cleaned should not be set since it failed
        assert "children_cleaned" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardian
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardian:
    """Guardian module imports and configuration."""

    def test_guardian_module_imports(self):
        """from march.cli.guardian_cmd import guardian works."""
        from march.cli.guardian_cmd import guardian
        assert guardian is not None
        # Also verify the agent-side guardian imports
        from march.agents.guardian import Guardian, GuardianConfig, run_guardian
        assert Guardian is not None
        assert GuardianConfig is not None
        assert run_guardian is not None

    def test_guardian_config(self):
        """Guardian has configurable check_interval."""
        from march.agents.guardian import GuardianConfig, Guardian

        # Default check_interval
        default_cfg = GuardianConfig()
        assert hasattr(default_cfg, "check_interval")
        assert isinstance(default_cfg.check_interval, int)
        assert default_cfg.check_interval > 0

        # Custom check_interval
        custom_cfg = GuardianConfig(check_interval=60)
        assert custom_cfg.check_interval == 60

        # Guardian uses the config
        g = Guardian(config=custom_cfg)
        assert g.config.check_interval == 60


# ═══════════════════════════════════════════════════════════════════════════════
# TestLogStructure (xfail — not yet implemented)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogStructure:
    """Log directory structure and TTL cleanup verification."""

    def test_log_subdirectories(self):
        """~/.march/logs/{agent,guardian,turns,metrics}/ should exist."""
        from march.core.log_maintenance import ensure_log_subdirectories
        log_base = Path.home() / ".march" / "logs"
        ensure_log_subdirectories(log_base)
        expected_dirs = ["agent", "guardian", "turns", "metrics"]
        for subdir in expected_dirs:
            d = log_base / subdir
            assert d.is_dir(), f"Expected log subdirectory {d} to exist"

    def test_log_ttl_30_days(self):
        """A log TTL cleanup function should exist and be called on startup."""
        # The framework should expose a cleanup_old_logs() or similar function
        # that is invoked on restart to purge logs older than 30 days.
        from march.core import log_maintenance  # noqa: F401 — module should exist
        assert hasattr(log_maintenance, "cleanup_old_logs")
        assert hasattr(log_maintenance, "LOG_TTL_DAYS")
        assert log_maintenance.LOG_TTL_DAYS == 30
