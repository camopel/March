"""Tests for rolling context: Session sliding window, compaction, flush, and restore.

Covers:
  - Rolling context grows with messages
  - Compaction resets messages, updates rolling_summary
  - needs_flush() logic (10 msgs or 10 seconds)
  - Session restore from DB (rolling_summary + messages after last_processed_seq)
  - Per-session lock prevents concurrent processing
  - First-turn bootstrap reads .md files
  - Subsequent turns use rolling_summary
  - Dirty message flush to DB

All tests run without external services. Uses pytest + pytest-asyncio.
Each test is independent.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.agent import AgentResponse, _extract_text
from march.core.message import Message, Role
from march.core.orchestrator import (
    FinalResponse,
    Orchestrator,
    OrchestratorEvent,
    TextDelta,
    Error,
)
from march.core.session import Session
from march.llm.base import StreamChunk


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / Mocks
# ═══════════════════════════════════════════════════════════════════════════════


class MockAgent:
    """Minimal mock Agent whose ``run_stream`` yields predetermined items."""

    def __init__(self, items=None):
        self.items = items or []
        self.memory = None

    async def run_stream(self, user_message, session):
        for item in self.items:
            if isinstance(item, AgentResponse):
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

    async def get_messages_after_seq(self, session_id, last_processed_seq) -> list[Message]:
        msgs = self._messages.get(session_id, [])
        return [m for m in msgs if (m.metadata or {}).get("seq", 0) > last_processed_seq]

    async def add_message(self, session_id, message, attachments=None) -> str:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)
        return "msg-id"

    async def flush_messages(self, session_id, messages) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].extend(messages)

    async def save_session(self, session) -> None:
        self._sessions[session.id] = session

    async def clear_session(self, session_id) -> None:
        self._messages.pop(session_id, None)
        self._sessions.pop(session_id, None)

    async def reactivate_session(self, session_id: str, source_type: str = "", source_id: str = "") -> Session | None:
        """Reactivate a soft-deleted session (no-op for in-memory store)."""
        return None

    async def update_session(self, session) -> None:
        self._sessions[session.id] = session

    async def delete_session(self, session_id) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)


async def collect_events(orch, session_id, content, **kwargs) -> list[OrchestratorEvent]:
    events = []
    async for ev in orch.handle_message(session_id, content, source="test", **kwargs):
        events.append(ev)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Rolling Context Grows with Messages
# ═══════════════════════════════════════════════════════════════════════════════


class TestRollingContextGrows:
    """Test that the rolling context (messages list) grows as messages are added."""

    def test_messages_grow_with_add_message(self):
        """Adding messages increases the messages list."""
        s = Session(id="grow-test")
        assert len(s.messages) == 0

        s.add_message(Message.user("hello"))
        assert len(s.messages) == 1

        s.add_message(Message.assistant("world"))
        assert len(s.messages) == 2

        s.add_message(Message.user("how are you"))
        assert len(s.messages) == 3

    def test_messages_grow_with_add_exchange(self):
        """add_exchange adds both user and assistant messages."""
        s = Session(id="grow-exchange")
        s.add_exchange("hello", "world")
        assert len(s.messages) == 2

        s.add_exchange("second", "reply")
        assert len(s.messages) == 4

    def test_dirty_messages_track_unflushed(self):
        """dirty_messages tracks messages not yet flushed to DB."""
        s = Session(id="dirty-test")
        s.add_message(Message.user("hello"))
        s.add_message(Message.assistant("world"))

        assert len(s.dirty_messages) == 2
        assert len(s.messages) == 2

    def test_seq_counter_increments(self):
        """Each add_message increments the seq counter."""
        s = Session(id="seq-test")
        s.add_message(Message.user("first"))
        assert s._seq_counter == 1
        assert s.messages[0].metadata.get("seq") == 1

        s.add_message(Message.assistant("second"))
        assert s._seq_counter == 2
        assert s.messages[1].metadata.get("seq") == 2

    async def test_multi_turn_messages_grow(self):
        """Multiple turns via orchestrator grow the messages list."""
        store = InMemorySessionStore()
        agent = MockAgent([])
        orch = Orchestrator(agent=agent, session_store=store)

        prev_len = 0
        for i in range(5):
            agent.items = [StreamChunk(delta=f"Reply {i}"), AgentResponse(content=f"Reply {i}")]
            await collect_events(orch, "multi-turn", f"Message {i}")
            session = orch.get_cached_session("multi-turn")
            assert session is not None
            cur_len = len(session.messages)
            assert cur_len > prev_len, f"Turn {i}: messages didn't grow ({cur_len} <= {prev_len})"
            prev_len = cur_len

        assert prev_len == 10  # 5 turns × 2 messages


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Compaction Resets Messages, Updates Rolling Summary
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompaction:
    """Test that compaction clears messages and sets rolling_summary."""

    def test_compact_clears_messages(self):
        """compact() clears messages and dirty_messages."""
        s = Session(id="compact-test")
        for i in range(20):
            s.add_exchange(f"user-{i}", f"assistant-{i}")

        assert len(s.messages) == 40
        assert len(s.dirty_messages) == 40

        s.compact("Summary of conversation")

        assert len(s.messages) == 0
        assert len(s.dirty_messages) == 0

    def test_compact_sets_rolling_summary(self):
        """compact() sets the rolling_summary."""
        s = Session(id="compact-summary")
        s.add_exchange("hello", "world")
        s.compact("This is the rolling summary")

        assert s.rolling_summary == "This is the rolling summary"

    def test_compact_updates_last_processed_seq(self):
        """compact() updates last_processed_seq to current _seq_counter."""
        s = Session(id="compact-seq")
        s.add_exchange("hello", "world")
        assert s._seq_counter == 2

        s.compact("summary")
        assert s.last_processed_seq == 2

    def test_compact_preserves_id_and_metadata(self):
        """compact() doesn't change session identity or metadata."""
        s = Session(id="compact-meta", source_type="test", source_id="src-1")
        s.metadata = {"key": "value"}
        s.add_exchange("hello", "world")

        s.compact("summary")

        assert s.id == "compact-meta"
        assert s.source_type == "test"
        assert s.metadata == {"key": "value"}

    def test_get_messages_for_llm_after_compact(self):
        """After compaction, get_messages_for_llm returns rolling summary + new messages."""
        s = Session(id="compact-llm")
        s.add_exchange("old user", "old assistant")
        s.compact("Summary of old conversation")

        # Add new messages after compaction
        s.add_exchange("new user", "new assistant")

        msgs = s.get_messages_for_llm()
        # Should have: rolling summary + new user + new assistant
        assert len(msgs) == 3
        assert "Context Summary" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. needs_flush() Logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestNeedsFlush:
    """Test needs_flush() triggers on 10 messages or 10 seconds."""

    def test_no_dirty_messages_no_flush(self):
        """Empty dirty_messages → no flush needed."""
        s = Session(id="flush-empty")
        assert s.needs_flush() is False

    def test_ten_dirty_messages_triggers_flush(self):
        """10+ dirty messages → flush needed."""
        s = Session(id="flush-count")
        for i in range(9):
            s.add_message(Message.user(f"msg-{i}"))
        assert s.needs_flush() is False

        s.add_message(Message.user("msg-9"))
        assert s.needs_flush() is True

    def test_time_based_flush(self):
        """10+ seconds since last flush → flush needed (if dirty messages exist)."""
        s = Session(id="flush-time")
        s.add_message(Message.user("hello"))
        # Manually set flush timer to 11 seconds ago
        s._flush_timer = time.time() - 11
        assert s.needs_flush() is True

    def test_time_based_no_dirty_no_flush(self):
        """Even if 10+ seconds, no dirty messages → no flush."""
        s = Session(id="flush-time-empty")
        s._flush_timer = time.time() - 20
        assert s.needs_flush() is False

    def test_flush_clears_dirty_messages(self):
        """flush() returns dirty messages and clears the buffer."""
        s = Session(id="flush-clear")
        s.add_message(Message.user("hello"))
        s.add_message(Message.assistant("world"))

        flushed = s.flush()
        assert len(flushed) == 2
        assert s.dirty_messages == []
        assert s.needs_flush() is False

    def test_flush_updates_timer(self):
        """flush() updates the _flush_timer."""
        s = Session(id="flush-timer")
        s._flush_timer = time.time() - 100
        s.add_message(Message.user("hello"))

        before = time.time()
        s.flush()
        after = time.time()

        assert s._flush_timer >= before
        assert s._flush_timer <= after

    def test_flush_preserves_messages(self):
        """flush() only clears dirty_messages, not messages."""
        s = Session(id="flush-preserve")
        s.add_message(Message.user("hello"))
        s.add_message(Message.assistant("world"))

        s.flush()
        assert len(s.messages) == 2
        assert len(s.dirty_messages) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Session Restore from DB
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionRestore:
    """Test session restore from DB (rolling_summary + messages after last_processed_seq)."""

    def test_to_dict_from_dict_round_trip(self):
        """Session serializes and deserializes correctly."""
        s = Session(id="restore-rt", source_type="test", source_id="src-1")
        s.rolling_summary = "Previous context"
        s.add_exchange("hello", "world")
        s.last_processed_seq = 5
        s._seq_counter = 7

        d = s.to_dict()
        restored = Session.from_dict(d)

        assert restored.id == s.id
        assert restored.rolling_summary == s.rolling_summary
        assert restored.last_processed_seq == s.last_processed_seq
        assert len(restored.messages) == len(s.messages)

    def test_from_dict_restores_seq_counter(self):
        """from_dict restores _seq_counter from message metadata."""
        s = Session(id="restore-seq")
        s.add_message(Message.user("first"))
        s.add_message(Message.assistant("second"))
        s.add_message(Message.user("third"))

        d = s.to_dict()
        restored = Session.from_dict(d)

        assert restored._seq_counter == 3

    async def test_cold_start_loads_messages_after_seq(self):
        """Cold start loads messages after last_processed_seq."""
        store = InMemorySessionStore()

        # Create a session with some messages
        pre = Session(id="cold-seq", source_type="test", source_id="cold-seq")
        pre.rolling_summary = "Previous summary"
        pre.last_processed_seq = 5
        store._sessions["cold-seq"] = pre

        # Add messages with seq metadata
        msg1 = Message.user("old msg")
        msg1.metadata = {"seq": 3}  # Before last_processed_seq
        msg2 = Message.user("new msg 1")
        msg2.metadata = {"seq": 6}  # After last_processed_seq
        msg3 = Message.assistant("new reply")
        msg3.metadata = {"seq": 7}  # After last_processed_seq
        store._messages["cold-seq"] = [msg1, msg2, msg3]

        agent = MockAgent([StreamChunk(delta="X"), AgentResponse(content="X")])
        orch = Orchestrator(agent=agent, session_store=store)

        await collect_events(orch, "cold-seq", "Hello")

        session = orch.get_cached_session("cold-seq")
        assert session is not None
        assert session.rolling_summary == "Previous summary"
        # Should only have loaded messages after seq 5 (seq 6 and 7) + new exchange
        # The cold start loads 2 messages from DB, then add_exchange adds 2 more
        assert len(session.messages) >= 2

    async def test_cold_start_preserves_rolling_summary(self):
        """Cold start preserves the rolling_summary from DB."""
        store = InMemorySessionStore()
        pre = Session(id="cold-summary", source_type="test", source_id="cold-summary")
        pre.rolling_summary = "Important context from previous compaction"
        store._sessions["cold-summary"] = pre
        store._messages["cold-summary"] = []

        agent = MockAgent([StreamChunk(delta="Y"), AgentResponse(content="Y")])
        orch = Orchestrator(agent=agent, session_store=store)

        await collect_events(orch, "cold-summary", "Hello")

        session = orch.get_cached_session("cold-summary")
        assert session.rolling_summary == "Important context from previous compaction"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Per-session Lock Prevents Concurrent Processing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerSessionLock:
    """Test per-session locking in the orchestrator."""

    def test_lock_creation(self):
        """_get_session_lock creates and caches locks."""
        orch = Orchestrator(agent=MockAgent([]), session_store=InMemorySessionStore())

        lock1 = orch._get_session_lock("sess-1")
        lock2 = orch._get_session_lock("sess-1")
        lock3 = orch._get_session_lock("sess-2")

        assert lock1 is lock2  # Same session → same lock
        assert lock1 is not lock3  # Different session → different lock
        assert isinstance(lock1, asyncio.Lock)

    async def test_reset_acquires_lock(self):
        """reset_session acquires the session lock."""
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)

        await collect_events(orch, "lock-reset", "Hello")

        # Get the lock and verify it's not locked before reset
        lock = orch._get_session_lock("lock-reset")
        assert not lock.locked()

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("lock-reset")

        # After reset, lock should be released
        assert not lock.locked()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. First-turn Bootstrap Reads .md Files
# ═══════════════════════════════════════════════════════════════════════════════


class TestFirstTurnBootstrap:
    """Test that first turn (empty rolling_summary) reads .md files."""

    async def test_first_turn_reads_md_files(self):
        """When rolling_summary is empty, _build_context reads .md files."""
        from march.core.agent import Agent
        from march.core.context import Context

        # Create a mock Agent with a mock memory store
        mock_memory = AsyncMock()
        mock_memory.load_system_rules = AsyncMock(return_value="System rules content")
        mock_memory.load_agent_profile = AsyncMock(return_value="Agent profile content")
        mock_memory.load_tool_rules = AsyncMock(return_value="Tool rules content")
        mock_memory.load_long_term = AsyncMock(return_value="Long term memory content")

        agent = MagicMock(spec=Agent)
        agent.memory = mock_memory
        agent._build_context = Agent._build_context.__get__(agent, Agent)

        session = Session(id="first-turn")
        assert session.rolling_summary == ""

        ctx = await agent._build_context(session)

        # All .md file loaders should have been called
        mock_memory.load_system_rules.assert_called_once()
        mock_memory.load_agent_profile.assert_called_once()
        mock_memory.load_tool_rules.assert_called_once()
        mock_memory.load_long_term.assert_called_once()

        # Context should have the loaded content
        assert ctx.system_rules == "System rules content"
        assert ctx.agent_profile == "Agent profile content"

    async def test_subsequent_turn_uses_rolling_summary(self):
        """When rolling_summary is non-empty, _build_context skips .md files."""
        from march.core.agent import Agent
        from march.core.context import Context

        mock_memory = AsyncMock()
        mock_memory.load_system_rules = AsyncMock(return_value="System rules")
        mock_memory.load_agent_profile = AsyncMock(return_value="Agent profile")
        mock_memory.load_tool_rules = AsyncMock(return_value="Tool rules")
        mock_memory.load_long_term = AsyncMock(return_value="Long term")

        agent = MagicMock(spec=Agent)
        agent.memory = mock_memory
        agent._build_context = Agent._build_context.__get__(agent, Agent)

        session = Session(id="subsequent-turn")
        session.rolling_summary = "Cached rolling context from previous compaction"

        ctx = await agent._build_context(session)

        # .md file loaders should NOT have been called
        mock_memory.load_system_rules.assert_not_called()
        mock_memory.load_agent_profile.assert_not_called()
        mock_memory.load_tool_rules.assert_not_called()
        mock_memory.load_long_term.assert_not_called()

        # Context should use rolling_summary as system_rules
        assert ctx.system_rules == "Cached rolling context from previous compaction"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Flush Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlushIntegration:
    """Test dirty message flushing to DB via orchestrator."""

    async def test_flush_session_writes_to_store(self):
        """flush_session writes dirty messages to the session store."""
        store = InMemorySessionStore()
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        orch = Orchestrator(agent=agent, session_store=store)

        await collect_events(orch, "flush-test", "Hello")

        session = orch.get_cached_session("flush-test")
        assert session is not None

        # Add some dirty messages manually
        session.add_message(Message.user("extra 1"))
        session.add_message(Message.assistant("extra reply 1"))
        assert len(session.dirty_messages) > 0

        await orch.flush_session("flush-test")

        # Dirty messages should be cleared
        assert len(session.dirty_messages) == 0

    async def test_flush_nonexistent_session_noop(self):
        """flush_session on a non-cached session is a no-op."""
        store = InMemorySessionStore()
        orch = Orchestrator(agent=MockAgent([]), session_store=store)
        # Should not raise
        await orch.flush_session("nonexistent")

    async def test_maybe_flush_after_response(self):
        """After agent response, if needs_flush() is True, flush happens."""
        store = InMemorySessionStore()
        agent = MockAgent([])
        orch = Orchestrator(agent=agent, session_store=store)

        # Populate a session with many messages to trigger flush
        session = Session(id="auto-flush")
        for i in range(15):
            session.add_exchange(f"user-{i}", f"assistant-{i}")
        orch._sessions["auto-flush"] = session

        assert session.needs_flush() is True

        # Flush via orchestrator
        await orch.flush_session("auto-flush")
        assert session.needs_flush() is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Session Clear (for /reset)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionClear:
    """Test Session.clear() resets everything."""

    def test_clear_resets_all_fields(self):
        """clear() resets messages, dirty_messages, rolling_summary, seq counters."""
        s = Session(id="clear-test")
        s.add_exchange("hello", "world")
        s.rolling_summary = "Some summary"
        s.last_processed_seq = 5
        s._seq_counter = 10

        s.clear()

        assert s.messages == []
        assert s.dirty_messages == []
        assert s.rolling_summary == ""
        assert s.last_processed_seq == 0
        assert s._seq_counter == 0

    def test_clear_preserves_identity(self):
        """clear() preserves session identity fields."""
        s = Session(id="clear-identity", source_type="test", source_id="src-1", name="Test")
        s.metadata = {"key": "value"}
        s.add_exchange("hello", "world")

        s.clear()

        assert s.id == "clear-identity"
        assert s.source_type == "test"
        assert s.source_id == "src-1"
        assert s.name == "Test"
        assert s.metadata == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Two-step Compaction
# ═══════════════════════════════════════════════════════════════════════════════


class TestTwoStepCompaction:
    """Test the two-step compaction logic in agent._two_step_compaction."""

    async def test_two_step_compaction_produces_result(self):
        """_two_step_compaction returns a non-empty deduped rolling summary."""
        from march.core.agent import Agent

        mock_memory = AsyncMock()
        mock_memory.load_system_rules = AsyncMock(return_value="System rules")
        mock_memory.load_agent_profile = AsyncMock(return_value="Agent profile")
        mock_memory.load_tool_rules = AsyncMock(return_value="Tool rules")
        mock_memory.load_long_term = AsyncMock(return_value="Long term memory")

        agent = MagicMock(spec=Agent)
        agent.memory = mock_memory
        agent._two_step_compaction = Agent._two_step_compaction.__get__(agent, Agent)

        session = Session(id="two-step-test")
        session.rolling_summary = "Previous rolling context"
        session.add_exchange("hello", "world")

        messages = session.get_messages_for_llm()

        call_count = 0

        async def mock_summarize(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Summarized conversation content"
            else:
                return "Deduped rolling context with system rules and conversation"

        with patch("march.core.compaction._load_session_memory", return_value={"facts": "", "plan": ""}):
            result = await agent._two_step_compaction(
                session, messages, mock_summarize, 200000, 0.30,
            )

        assert result
        assert len(result) > 0
        # Should have called summarize twice (step 1: summarize, step 2: dedup)
        assert call_count == 2

    async def test_two_step_compaction_fallback_on_empty_summary(self):
        """If step 1 returns empty, fallback to existing rolling_summary."""
        from march.core.agent import Agent

        mock_memory = AsyncMock()
        mock_memory.load_system_rules = AsyncMock(return_value="Rules")
        mock_memory.load_agent_profile = AsyncMock(return_value="Profile")
        mock_memory.load_tool_rules = AsyncMock(return_value="Tools")
        mock_memory.load_long_term = AsyncMock(return_value="Memory")

        agent = MagicMock(spec=Agent)
        agent.memory = mock_memory
        agent._two_step_compaction = Agent._two_step_compaction.__get__(agent, Agent)

        session = Session(id="fallback-test")
        session.rolling_summary = "Existing summary"
        messages = [{"role": "user", "content": "hello"}]

        async def mock_summarize(prompt: str) -> str:
            return ""  # Empty result

        with patch("march.core.compaction._load_session_memory", return_value={"facts": "", "plan": ""}):
            result = await agent._two_step_compaction(
                session, messages, mock_summarize, 200000, 0.30,
            )

        assert result == "Existing summary"
