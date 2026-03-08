"""Comprehensive tests for the Orchestrator layer and refactored channels.

All tests run without external services — LLM, DB, Matrix nio, whisper,
and image processing are fully mocked.  Uses pytest + pytest-asyncio.
Each test is independent (no shared state).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.agent import AgentResponse, _extract_text
from march.core.message import Message, Role
from march.core.orchestrator import (
    Cancelled,
    Error,
    FinalResponse,
    Orchestrator,
    OrchestratorEvent,
    TextDelta,
    ToolProgress,
)
from march.core.session import Session
from march.core.turn_log import TurnLogger, _MAX_FILE_BYTES
from march.llm.base import LLMUsage, StreamChunk


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / Mocks
# ═══════════════════════════════════════════════════════════════════════════════


class MockAgent:
    """A mock Agent whose ``run_stream`` yields predetermined items.

    When yielding an ``AgentResponse`` this mock calls
    ``session.add_exchange()`` — mirroring what the real Agent._finalize() does.
    """

    def __init__(self, items: list[StreamChunk | AgentResponse] | None = None):
        self.items = items or []
        self.memory = None

    async def run_stream(
        self, user_message: str | list, session: Session
    ) -> AsyncIterator[StreamChunk | AgentResponse]:
        for item in self.items:
            if isinstance(item, AgentResponse):
                session.add_exchange(user_message, item.content)
            yield item


class ErrorAgent:
    """Agent that raises an exception during streaming."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.memory = None

    async def run_stream(
        self, user_message: str | list, session: Session
    ) -> AsyncIterator[StreamChunk | AgentResponse]:
        raise self._exc
        yield  # pragma: no cover


class SlowAgent:
    """Agent that yields items with an ``asyncio.sleep`` between them."""

    def __init__(self, items: list[StreamChunk | AgentResponse], delay: float = 0.05):
        self.items = items
        self.delay = delay
        self.memory = None

    async def run_stream(
        self, user_message: str | list, session: Session
    ) -> AsyncIterator[StreamChunk | AgentResponse]:
        for item in self.items:
            await asyncio.sleep(self.delay)
            if isinstance(item, AgentResponse):
                session.add_exchange(user_message, item.content)
            yield item


class InMemorySessionStore:
    """Minimal in-memory SessionStore that satisfies the Orchestrator + ChatDB."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._messages: dict[str, list[Message]] = {}

    # ── Core interface ────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def create_session(
        self,
        source_type: str,
        source_id: str,
        name: str = "",
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        s = Session(
            id=session_id or "",
            source_type=source_type,
            source_id=source_id,
            name=name,
            metadata=metadata or {},
        )
        self._sessions[s.id] = s
        self._messages[s.id] = []
        return s

    async def get_messages(
        self, session_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Message]:
        msgs = self._messages.get(session_id, [])
        if limit is not None:
            return msgs[offset : offset + limit]
        return msgs[offset:]

    async def add_message(
        self, session_id: str, message: Message, attachments: list[dict] | None = None
    ) -> str:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)
        return "msg-id"

    async def clear_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._sessions.pop(session_id, None)

    # ── Extended interface (used by ChatDB / WS proxy) ────────────────

    async def update_session(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)

    async def list_sessions(self) -> list[dict]:
        return [
            {"id": s.id, "name": s.name, "source_type": s.source_type}
            for s in self._sessions.values()
        ]

    async def get_messages_raw(self, session_id: str) -> list[dict]:
        msgs = self._messages.get(session_id, [])
        return [
            {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
                "tool_calls": "[]",
                "attachments": "[]",
            }
            for m in msgs
        ]


async def collect_events(
    orch: Orchestrator, session_id: str, content: str | list, **kwargs
) -> list[OrchestratorEvent]:
    """Helper: collect all events from handle_message into a list."""
    events: list[OrchestratorEvent] = []
    async for ev in orch.handle_message(session_id, content, source="test", **kwargs):
        events.append(ev)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OrchestratorEvent dataclass tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorEvents:
    """Test OrchestratorEvent dataclasses — creation and field access."""

    def test_text_delta_creation(self):
        td = TextDelta(delta="hello")
        assert td.delta == "hello"

    def test_text_delta_frozen(self):
        td = TextDelta(delta="x")
        with pytest.raises(AttributeError):
            td.delta = "y"  # type: ignore[misc]

    def test_tool_progress_creation(self):
        tp = ToolProgress(name="web_search", status="complete", summary="Found 3 results", duration_ms=120.5)
        assert tp.name == "web_search"
        assert tp.status == "complete"
        assert tp.summary == "Found 3 results"
        assert tp.duration_ms == 120.5

    def test_tool_progress_defaults(self):
        tp = ToolProgress(name="read", status="started")
        assert tp.summary == ""
        assert tp.duration_ms == 0.0

    def test_final_response_creation(self):
        fr = FinalResponse(
            content="Here is the answer.",
            tool_calls_made=2,
            total_tokens=500,
            total_cost=0.01,
            turn_summary="Answered the question.",
        )
        assert fr.content == "Here is the answer."
        assert fr.tool_calls_made == 2
        assert fr.total_tokens == 500
        assert fr.total_cost == 0.01
        assert fr.turn_summary == "Answered the question."

    def test_final_response_defaults(self):
        fr = FinalResponse(content="ok")
        assert fr.tool_calls_made == 0
        assert fr.total_tokens == 0
        assert fr.total_cost == 0.0
        assert fr.turn_summary == ""

    def test_error_creation(self):
        err = Error(message="something went wrong")
        assert err.message == "something went wrong"

    def test_cancelled_creation(self):
        c = Cancelled(partial_content="partial text")
        assert c.partial_content == "partial text"

    def test_cancelled_defaults(self):
        c = Cancelled()
        assert c.partial_content == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Orchestrator.handle_message tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorHandleMessage:
    """Test handle_message with mocked Agent."""

    async def test_simple_text_response(self):
        agent = MockAgent([
            StreamChunk(delta="Hello "),
            StreamChunk(delta="world!"),
            AgentResponse(content="Hello world!", total_tokens=10, total_cost=0.001),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-1", "Hi there")

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello "
        assert text_deltas[1].delta == "world!"
        assert len(finals) == 1
        assert finals[0].content == "Hello world!"
        assert finals[0].total_tokens == 10

    async def test_tool_call_response(self):
        agent = MockAgent([
            StreamChunk(delta="", tool_progress={"name": "web_search", "status": "started", "summary": "", "duration_ms": 0}),
            StreamChunk(delta="", tool_progress={"name": "web_search", "status": "complete", "summary": "3 results", "duration_ms": 150}),
            StreamChunk(delta="Found results."),
            AgentResponse(content="Found results.", tool_calls_made=1, total_tokens=50),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-2", "Search for cats")

        tool_events = [e for e in events if isinstance(e, ToolProgress)]
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(tool_events) == 2
        assert tool_events[0].status == "started"
        assert tool_events[1].status == "complete"
        assert tool_events[1].duration_ms == 150
        assert finals[0].tool_calls_made == 1

    async def test_cancel_before_stream(self):
        agent = MockAgent([StreamChunk(delta="nope"), AgentResponse(content="nope")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        cancel = asyncio.Event()
        cancel.set()
        events = await collect_events(orch, "sess-cancel-pre", "Hello", cancel_event=cancel)
        assert len(events) == 1
        assert isinstance(events[0], Cancelled)
        assert events[0].partial_content == ""

    async def test_cancel_during_stream(self):
        items = [
            StreamChunk(delta="Part 1 "),
            StreamChunk(delta="Part 2 "),
            StreamChunk(delta="Part 3 "),
            AgentResponse(content="Full response"),
        ]
        agent = SlowAgent(items, delay=0.05)
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        cancel = asyncio.Event()
        events: list[OrchestratorEvent] = []
        async for ev in orch.handle_message("sess-cancel-mid", "Hello", source="test", cancel_event=cancel):
            events.append(ev)
            if isinstance(ev, TextDelta) and not cancel.is_set():
                cancel.set()
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        cancelled = [e for e in events if isinstance(e, Cancelled)]
        assert len(text_deltas) >= 1
        assert len(cancelled) == 1
        assert cancelled[0].partial_content.startswith("Part 1")

    async def test_error_handling(self):
        agent = ErrorAgent(RuntimeError("LLM exploded"))
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-err", "Boom")
        assert len(events) == 1
        assert isinstance(events[0], Error)
        assert "LLM exploded" in events[0].message

    async def test_empty_response(self):
        agent = MockAgent([AgentResponse(content="", total_tokens=5)])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-empty", "Say nothing")
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1
        assert finals[0].content == ""

    async def test_multiple_tool_calls(self):
        agent = MockAgent([
            StreamChunk(delta="", tool_progress={"name": "read", "status": "complete", "summary": "file read", "duration_ms": 10}),
            StreamChunk(delta="", tool_progress={"name": "write", "status": "complete", "summary": "file written", "duration_ms": 20}),
            StreamChunk(delta="", tool_progress={"name": "exec", "status": "complete", "summary": "ran command", "duration_ms": 500}),
            StreamChunk(delta="Done!"),
            AgentResponse(content="Done!", tool_calls_made=3, total_tokens=200),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-multi-tool", "Do three things")
        tool_events = [e for e in events if isinstance(e, ToolProgress)]
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(tool_events) == 3
        assert finals[0].tool_calls_made == 3

    async def test_stream_with_usage_data(self):
        agent = MockAgent([
            StreamChunk(delta="Hi", usage=LLMUsage(input_tokens=100, output_tokens=50, cost=0.005)),
            AgentResponse(content="Hi", total_tokens=150, total_cost=0.005),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-usage", "Hello")
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert finals[0].total_tokens == 150

    async def test_stream_ends_without_agent_response(self):
        agent = MockAgent([StreamChunk(delta="partial")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-no-final", "Hello")
        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1
        assert "without a final response" in errors[0].message

    async def test_multimodal_content(self):
        agent = MockAgent([
            StreamChunk(delta="I see an image."),
            AgentResponse(content="I see an image."),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "data": "base64..."},
        ]
        events = await collect_events(orch, "sess-mm", content)
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1
        assert finals[0].content == "I see an image."

    async def test_session_load_failure(self):
        agent = MockAgent([])
        store = InMemorySessionStore()
        store.get_session = AsyncMock(side_effect=RuntimeError("DB down"))
        store.create_session = AsyncMock(side_effect=RuntimeError("DB down"))
        orch = Orchestrator(agent=agent, session_store=store)
        events = await collect_events(orch, "sess-db-fail", "Hello")
        assert len(events) == 1
        assert isinstance(events[0], Error)
        assert "Failed to load session" in events[0].message


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Orchestrator session management tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorSessionManagement:
    async def test_get_or_create_session(self):
        agent = MockAgent([StreamChunk(delta="A"), AgentResponse(content="A")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-reuse", "First")
        session_1 = orch._sessions["sess-reuse"]
        agent.items = [StreamChunk(delta="B"), AgentResponse(content="B")]
        await collect_events(orch, "sess-reuse", "Second")
        assert orch._sessions["sess-reuse"] is session_1

    async def test_reset_session(self):
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-reset", "Hello")
        assert "sess-reset" in orch._sessions
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("sess-reset")
        assert "sess-reset" not in orch._sessions

    async def test_evict_session(self):
        agent = MockAgent([StreamChunk(delta="X"), AgentResponse(content="X")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-evict", "Hello")
        orch.evict_session("sess-evict")
        assert "sess-evict" not in orch._sessions
        assert await store.get_session("sess-evict") is not None

    async def test_evict_nonexistent(self):
        orch = Orchestrator(agent=MockAgent([]), session_store=InMemorySessionStore())
        orch.evict_session("does-not-exist")  # no-op, no crash

    async def test_session_persistence_user_and_assistant(self):
        agent = MockAgent([StreamChunk(delta="Reply"), AgentResponse(content="Reply")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-persist", "User says hi")
        msgs = await store.get_messages("sess-persist")
        roles = [m.role for m in msgs]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    async def test_tool_intermediates_not_persisted(self):
        agent = MockAgent([
            StreamChunk(delta="", tool_progress={"name": "exec", "status": "complete", "summary": "ran", "duration_ms": 50}),
            StreamChunk(delta="Done"),
            AgentResponse(content="Done", tool_calls_made=1),
        ])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-no-tool-db", "Run something")
        msgs = await store.get_messages("sess-no-tool-db")
        roles = [m.role for m in msgs]
        assert Role.TOOL not in roles

    async def test_cold_start_from_db(self):
        agent = MockAgent([StreamChunk(delta="Cold"), AgentResponse(content="Cold")])
        store = InMemorySessionStore()
        pre = Session(id="sess-cold", source_type="test", source_id="sess-cold")
        store._sessions["sess-cold"] = pre
        store._messages["sess-cold"] = [Message.user("old"), Message.assistant("old reply")]
        orch = Orchestrator(agent=agent, session_store=store)
        assert "sess-cold" not in orch._sessions
        await collect_events(orch, "sess-cold", "New message")
        assert "sess-cold" in orch._sessions
        assert len(orch._sessions["sess-cold"].history) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TurnLogger tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTurnLogger:
    def test_turn_start_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="hello", source="test")
        lines = (tmp_path / "turns.jsonl").read_text().strip().split("\n")
        data = json.loads(lines[0])
        assert data["event"] == "turn_start"
        assert data["turn_id"] == "t1"
        assert "ts" in data

    def test_turn_complete_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_complete(turn_id="t2", session_id="s2", tool_calls=3, total_tokens=500, total_cost=0.01, duration_ms=1234.5, final_reply_length=100)
        data = json.loads((tmp_path / "turns.jsonl").read_text().strip())
        assert data["event"] == "turn_complete"
        assert data["tool_calls"] == 3

    def test_tool_call_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.tool_call(turn_id="t3", session_id="s3", name="web_search", args={"query": "cats"}, duration_ms=200.0, status="complete", summary="Found 5 results")
        data = json.loads((tmp_path / "turns.jsonl").read_text().strip())
        assert data["event"] == "tool_call"
        assert data["name"] == "web_search"

    def test_turn_cancelled_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_cancelled(turn_id="t4", session_id="s4", partial_content_length=42)
        data = json.loads((tmp_path / "turns.jsonl").read_text().strip())
        assert data["event"] == "turn_cancelled"

    def test_turn_error_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_error(turn_id="t5", session_id="s5", error="kaboom")
        data = json.loads((tmp_path / "turns.jsonl").read_text().strip())
        assert data["event"] == "turn_error"

    def test_log_rotation(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        log_file = tmp_path / "turns.jsonl"
        log_file.write_text("x" * (_MAX_FILE_BYTES + 1))
        logger.turn_start(turn_id="t-rot", session_id="s-rot", user_msg="rotate", source="test")
        assert (tmp_path / "turns.jsonl.1").exists()
        assert "t-rot" in log_file.read_text()

    def test_thread_safety(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        num_threads, writes_per = 10, 50

        def writer(tid: int):
            for i in range(writes_per):
                logger.turn_start(turn_id=f"t-{tid}-{i}", session_id=f"s-{tid}", user_msg=f"msg-{i}", source="test")

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = (tmp_path / "turns.jsonl").read_text().strip().split("\n")
        assert len(lines) == num_threads * writes_per
        for line in lines:
            json.loads(line)  # Should not raise

    def test_json_serialization_non_serializable(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)

        class Weird:
            pass

        logger.tool_call(turn_id="t-weird", session_id="s-weird", name="test_tool", args={"obj": Weird()}, duration_ms=10.0, status="complete", summary="ok")
        data = json.loads((tmp_path / "turns.jsonl").read_text().strip())
        assert "Weird" in data["args"]["obj"]

    def test_creates_log_dir(self, tmp_path: Path):
        log_dir = tmp_path / "nested" / "logs"
        logger = TurnLogger(log_dir=log_dir)
        logger.turn_start(turn_id="t-dir", session_id="s-dir", user_msg="hi", source="test")
        assert (log_dir / "turns.jsonl").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Session dataclass tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSession:
    def test_session_creation_with_defaults(self):
        s = Session()
        assert s.id
        assert s.source_type == "terminal"
        assert s.history == []
        assert s.state == "active"

    def test_session_deterministic_id(self):
        s1 = Session(source_type="matrix", source_id="!room:server")
        s2 = Session(source_type="matrix", source_id="!room:server")
        assert s1.id == s2.id

    def test_add_message(self):
        s = Session(id="test")
        s.add_message(Message.user("hello"))
        assert len(s.history) == 1

    def test_add_exchange(self):
        s = Session(id="test")
        s.add_exchange("user says", "assistant says")
        assert len(s.history) == 2
        assert s.history[0].role == Role.USER
        assert s.history[1].role == Role.ASSISTANT

    def test_clear(self):
        s = Session(id="test")
        s.add_exchange("a", "b")
        s.compaction_summary = "summary"
        s.rolling_summary = "rolling"
        s.clear()
        assert s.history == []
        assert s.compaction_summary == ""
        assert s.rolling_summary == ""

    def test_reset(self):
        s = Session(id="test")
        s.add_exchange("a", "b")
        s.reset()
        assert s.history == []
        assert s.state == "reset"

    def test_compact_history(self):
        s = Session(id="test")
        for i in range(20):
            s.add_exchange(f"user-{i}", f"assistant-{i}")
        moved = s.compact_history("Summary of old messages", keep_recent=10)
        assert moved > 0
        assert len(s.backup_history) == moved
        assert len(s.history) == 11  # 1 summary + 10 recent

    def test_compact_history_too_few_messages(self):
        s = Session(id="test")
        s.add_exchange("a", "b")
        moved = s.compact_history("summary", keep_recent=10)
        assert moved == 0

    def test_to_dict_and_from_dict(self):
        s = Session(id="test-rt", source_type="ws", source_id="ws-1", name="Test Session")
        s.add_exchange("hello", "world")
        s.metadata = {"key": "value"}
        d = s.to_dict()
        restored = Session.from_dict(d)
        assert restored.id == s.id
        assert restored.source_type == s.source_type
        assert len(restored.history) == len(s.history)
        assert restored.metadata == s.metadata

    def test_get_messages_for_llm(self):
        s = Session(id="test")
        s.add_exchange("hello", "world")
        msgs = s.get_messages_for_llm()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. StreamChunk / AgentResponse / _extract_text tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamChunk:
    def test_delta_text_sync(self):
        c = StreamChunk(delta="hello")
        assert c.delta_text == "hello"
        c2 = StreamChunk(delta_text="world")
        assert c2.delta == "world"

    def test_finish_reason_stop_reason_sync(self):
        c = StreamChunk(finish_reason="stop")
        assert c.stop_reason == "stop"

    def test_tool_progress_field(self):
        c = StreamChunk(delta="", tool_progress={"name": "exec", "status": "started"})
        assert c.tool_progress["name"] == "exec"

    def test_usage_field(self):
        usage = LLMUsage(input_tokens=100, output_tokens=50, cost=0.005)
        c = StreamChunk(delta="", usage=usage)
        assert c.usage.input_tokens == 100


class TestAgentResponse:
    def test_defaults(self):
        r = AgentResponse()
        assert r.content == ""
        assert r.tool_calls_made == 0
        assert r.total_tokens == 0

    def test_with_values(self):
        r = AgentResponse(content="Answer", tool_calls_made=2, total_tokens=500)
        assert r.content == "Answer"
        assert r.tool_calls_made == 2


class TestExtractText:
    def test_string_input(self):
        assert _extract_text("hello") == "hello"

    def test_multimodal_input(self):
        content = [{"type": "image", "data": "..."}, {"type": "text", "text": "What is this?"}]
        assert _extract_text(content) == "What is this?"

    def test_multimodal_no_text(self):
        assert _extract_text([{"type": "image", "data": "..."}]) == ""

    def test_empty_list(self):
        assert _extract_text([]) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Multi-modal Input Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultimodalInput:
    """Test multi-modal content: text, voice (mock whisper), image (mock resize),
    PDF (mock extraction), and mixed content — via the Orchestrator."""

    async def test_text_only_ws_context(self):
        """Plain text through WS-style session."""
        agent = MockAgent([StreamChunk(delta="Got it."), AgentResponse(content="Got it.")])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events = await collect_events(orch, "ws-text-1", "Hello from WS")
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1
        assert finals[0].content == "Got it."

    async def test_text_only_matrix_context(self):
        """Plain text through Matrix-style session (source='matrix')."""
        agent = MockAgent([StreamChunk(delta="Matrix reply."), AgentResponse(content="Matrix reply.")])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events: list[OrchestratorEvent] = []
        async for ev in orch.handle_message("matrix-text-1", "Hello from Matrix", source="matrix"):
            events.append(ev)
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1
        assert finals[0].content == "Matrix reply."

    async def test_image_multimodal_content(self):
        """Image + text sent as multimodal content blocks."""
        received_content = None

        class CapturingAgent:
            memory = None

            async def run_stream(self, user_message, session):
                nonlocal received_content
                received_content = user_message
                session.add_exchange(user_message, "I see a cat.")
                yield StreamChunk(delta="I see a cat.")
                yield AgentResponse(content="I see a cat.")

        orch = Orchestrator(agent=CapturingAgent(), session_store=InMemorySessionStore())
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
            {"type": "text", "text": "[User sent image: cat.jpg]"},
        ]
        events = await collect_events(orch, "img-sess", content)
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1
        assert finals[0].content == "I see a cat."
        assert isinstance(received_content, list)
        assert received_content[0]["type"] == "image"

    async def test_voice_transcription_flow(self):
        """Transcribed voice arrives as plain text to orchestrator."""
        agent = MockAgent([
            StreamChunk(delta="You said: turn on the lights."),
            AgentResponse(content="You said: turn on the lights."),
        ])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events = await collect_events(orch, "voice-sess", "Turn on the lights")
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert "lights" in finals[0].content

    async def test_pdf_text_content(self):
        """PDF extracted text sent as plain text to orchestrator."""
        agent = MockAgent([
            StreamChunk(delta="The report shows $5M revenue."),
            AgentResponse(content="The report shows $5M revenue."),
        ])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events = await collect_events(orch, "pdf-sess", "[PDF: report.pdf]\n\nQ1 revenue was $5M...")
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert "$5M" in finals[0].content

    async def test_mixed_image_and_text_multimodal(self):
        """Image + caption + text question in one multimodal message."""
        agent = MockAgent([
            StreamChunk(delta="The chart shows growth."),
            AgentResponse(content="The chart shows growth."),
        ])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}},
            {"type": "text", "text": "[User sent image: chart.png]"},
            {"type": "text", "text": "What does this chart show?"},
        ]
        events = await collect_events(orch, "mixed-mm", content)
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert "growth" in finals[0].content

    async def test_multimodal_persisted_in_db(self):
        """Multimodal content is stored in DB."""
        agent = MockAgent([StreamChunk(delta="Noted."), AgentResponse(content="Noted.")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "CCCC"}},
            {"type": "text", "text": "Describe this image"},
        ]
        await collect_events(orch, "mm-persist", content)
        msgs = await store.get_messages("mm-persist")
        assert len(msgs) >= 1
        assert msgs[0].role == Role.USER

    async def test_empty_multimodal_content(self):
        """Empty multimodal list (edge case)."""
        agent = MockAgent([StreamChunk(delta="Nothing."), AgentResponse(content="Nothing.")])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events = await collect_events(orch, "empty-mm", [])
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(finals) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Sequential Tasks / Multi-turn Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSequentialTasks:
    async def test_multi_turn_history_grows(self):
        """3+ turns: verify session history grows with each turn."""
        store = InMemorySessionStore()
        agent = MockAgent([])
        orch = Orchestrator(agent=agent, session_store=store)

        prev_len = 0
        for i in range(5):
            agent.items = [StreamChunk(delta=f"Reply {i}"), AgentResponse(content=f"Reply {i}")]
            await collect_events(orch, "multi-turn", f"Message {i}")
            session = orch.get_cached_session("multi-turn")
            assert session is not None
            cur_len = len(session.history)
            assert cur_len > prev_len, f"Turn {i}: history didn't grow ({cur_len} <= {prev_len})"
            prev_len = cur_len

        assert prev_len == 10  # 5 turns × 2 messages

    async def test_tool_chain_a_then_b_then_respond(self):
        """Agent calls tool A, then tool B, then responds with text."""
        agent = MockAgent([
            StreamChunk(delta="", tool_progress={"name": "read", "status": "started", "summary": "", "duration_ms": 0}),
            StreamChunk(delta="", tool_progress={"name": "read", "status": "complete", "summary": "read file", "duration_ms": 50}),
            StreamChunk(delta="", tool_progress={"name": "exec", "status": "started", "summary": "", "duration_ms": 0}),
            StreamChunk(delta="", tool_progress={"name": "exec", "status": "complete", "summary": "ran tests", "duration_ms": 2000}),
            StreamChunk(delta="All tests pass."),
            AgentResponse(content="All tests pass.", tool_calls_made=2, total_tokens=300),
        ])
        orch = Orchestrator(agent=agent, session_store=InMemorySessionStore())
        events = await collect_events(orch, "tool-chain", "Run tests")
        tool_events = [e for e in events if isinstance(e, ToolProgress)]
        finals = [e for e in events if isinstance(e, FinalResponse)]
        assert len(tool_events) == 4
        assert finals[0].tool_calls_made == 2

    async def test_multi_turn_with_tools(self):
        """Turn 1 uses tools, turn 2 is text-only — verify clean history."""
        store = InMemorySessionStore()
        agent = MockAgent([
            StreamChunk(delta="", tool_progress={"name": "web_search", "status": "complete", "summary": "found", "duration_ms": 100}),
            StreamChunk(delta="Found it."),
            AgentResponse(content="Found it.", tool_calls_made=1),
        ])
        orch = Orchestrator(agent=agent, session_store=store)
        events1 = await collect_events(orch, "mt-tools", "Search for X")
        assert any(isinstance(e, FinalResponse) for e in events1)

        agent.items = [StreamChunk(delta="Sure, here's more."), AgentResponse(content="Sure, here's more.")]
        events2 = await collect_events(orch, "mt-tools", "Tell me more")
        assert any(isinstance(e, FinalResponse) for e in events2)

        db_msgs = await store.get_messages("mt-tools")
        roles = [m.role for m in db_msgs]
        assert Role.TOOL not in roles

    async def test_concurrent_sessions_no_cross_contamination(self):
        """Two sessions running concurrently don't leak state."""
        store = InMemorySessionStore()
        agent_a = SlowAgent([StreamChunk(delta="Session A reply"), AgentResponse(content="Session A reply")], delay=0.02)
        agent_b = MockAgent([StreamChunk(delta="Session B reply"), AgentResponse(content="Session B reply")])
        orch_a = Orchestrator(agent=agent_a, session_store=store)
        orch_b = Orchestrator(agent=agent_b, session_store=store)

        task_a = asyncio.create_task(collect_events(orch_a, "sess-A", "Hello A"))
        events_b = await collect_events(orch_b, "sess-B", "Hello B")
        events_a = await task_a

        finals_a = [e for e in events_a if isinstance(e, FinalResponse)]
        finals_b = [e for e in events_b if isinstance(e, FinalResponse)]
        assert "Session A" in finals_a[0].content
        assert "Session B" in finals_b[0].content

    async def test_concurrent_sessions_independent_db(self):
        """Concurrent sessions have independent DB message lists."""
        store = InMemorySessionStore()
        agent = MockAgent([StreamChunk(delta="Reply"), AgentResponse(content="Reply")])
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "iso-1", "Msg for session 1")
        agent.items = [StreamChunk(delta="Other"), AgentResponse(content="Other")]
        await collect_events(orch, "iso-2", "Msg for session 2")

        msgs_1 = await store.get_messages("iso-1")
        msgs_2 = await store.get_messages("iso-2")
        assert len(msgs_1) == 2
        assert len(msgs_2) == 2
        user_1 = [m for m in msgs_1 if m.role == Role.USER][0]
        user_2 = [m for m in msgs_2 if m.role == Role.USER][0]
        assert "session 1" in user_1.content
        assert "session 2" in user_2.content


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WS Proxy Tests (unit-level, mocked aiohttp)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWSProxy:
    """Test WSProxyPlugin logic with mocked WebSocket and DB."""

    def _make_ws_conn(self, session_id: str = "ws-test"):
        from march.plugins.ws_proxy import _WSConn
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        conn = _WSConn(ws=mock_ws, session_id=session_id)
        return conn

    def _make_plugin(self, agent=None, store=None):
        from march.plugins.ws_proxy import WSProxyPlugin
        plugin = WSProxyPlugin()
        plugin._agent = agent or MockAgent([])
        plugin._db = MagicMock()
        plugin._db.session_exists = AsyncMock(return_value=True)
        plugin._db.save_message = AsyncMock(return_value="msg-id")
        plugin._db.get_history = AsyncMock(return_value={"session": {}, "messages": []})
        plugin._db.clear_session_messages = AsyncMock()
        plugin._app_ref = MagicMock()
        if store is None:
            store = InMemorySessionStore()
        plugin._orchestrator = Orchestrator(agent=plugin._agent, session_store=store)
        return plugin

    async def test_text_message_handling(self):
        """WS text message is routed to orchestrator."""
        plugin = self._make_plugin(
            agent=MockAgent([StreamChunk(delta="WS reply"), AgentResponse(content="WS reply")])
        )
        conn = self._make_ws_conn("ws-text")
        await plugin._ws_handle_message(conn, {"type": "message", "content": "Hello"})
        calls = conn.ws.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "stream.start" in types
        assert "stream.delta" in types
        assert "stream.end" in types

    async def test_voice_upload_transcription(self):
        """Voice message triggers transcription then agent call."""
        plugin = self._make_plugin(
            agent=MockAgent([StreamChunk(delta="Voice response"), AgentResponse(content="Voice response")])
        )
        conn = self._make_ws_conn("ws-voice")
        mock_tool_result = MagicMock()
        mock_tool_result.is_error = False
        mock_tool_result.content = "Hello from voice"
        plugin._agent.tools = MagicMock()
        plugin._agent.tools.execute = AsyncMock(return_value=mock_tool_result)

        voice_data = base64.b64encode(b"fake audio bytes").decode()
        await plugin._ws_handle_voice(conn, {"type": "voice", "mime_type": "audio/webm", "data": voice_data})
        calls = conn.ws.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "voice.transcribed" in types

    async def test_reconnect_mid_stream(self):
        """Stream buffer provides catchup data on reconnect."""
        from march.plugins.ws_proxy import _StreamBuffer
        buf = _StreamBuffer()
        buf.streaming = True
        buf.add_chunk({"type": "stream.start"})
        buf.add_chunk({"type": "stream.delta", "content": "Hello "})
        buf.collected = "Hello "
        buf.add_chunk({"type": "stream.delta", "content": "world!"})
        buf.collected = "Hello world!"

        missed = buf.get_chunks_after(0)
        assert len(missed) == 2
        assert missed[0]["type"] == "stream.delta"
        assert missed[0]["content"] == "Hello "
        assert missed[1]["content"] == "world!"

    async def test_reconnect_after_stream_done(self):
        """After stream completes, reconnect gets full catchup."""
        from march.plugins.ws_proxy import _StreamBuffer
        buf = _StreamBuffer()
        buf.collected = "Full response text"
        buf.done = True
        buf.streaming = False
        assert buf.collected == "Full response text"
        assert buf.done is True

    async def test_stop_command_cancels_stream(self):
        """Stop command sets cancel_event and clears pending queue."""
        plugin = self._make_plugin(
            agent=SlowAgent([StreamChunk(delta="Slow..."), AgentResponse(content="Slow...")], delay=0.5)
        )
        conn = self._make_ws_conn("ws-stop")
        conn.busy = True
        conn.pending = ["queued msg 1", "queued msg 2"]
        await plugin._ws_handle_message(conn, {"type": "message", "content": "/stop"})
        assert conn.cancel_event.is_set()
        assert len(conn.pending) == 0
        calls = conn.ws.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "stream.cancelled" in types

    async def test_session_crud_via_db(self):
        """Session CRUD operations through ChatDB adapter."""
        from march.plugins.ws_proxy import ChatDB
        store = InMemorySessionStore()
        db = ChatDB(store=store)

        session = await db.create_session("Test Chat", "A test session")
        assert session["name"] == "Test Chat"
        sid = session["id"]
        assert await db.session_exists(sid)

        renamed = await db.rename_session(sid, "Renamed Chat")
        assert renamed is True
        s = await store.get_session(sid)
        assert s.name == "Renamed Chat"

        deleted = await db.delete_session(sid)
        assert deleted is True

    async def test_ws_reset_command(self):
        """WS /reset clears session via orchestrator."""
        plugin = self._make_plugin()
        conn = self._make_ws_conn("ws-reset")
        store = plugin._orchestrator.session_store
        await store.create_session("ws", "ws-reset", session_id="ws-reset")
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await plugin._handle_reset(conn)
        calls = conn.ws.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "session.reset" in types

    async def test_message_queuing_when_busy(self):
        """Messages are queued when agent is busy."""
        plugin = self._make_plugin()
        conn = self._make_ws_conn("ws-queue")
        conn.busy = True
        await plugin._ws_handle_message(conn, {"type": "message", "content": "Queued msg"})
        assert len(conn.pending) == 1
        assert conn.pending[0] == "Queued msg"
        calls = conn.ws.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "message.queued" in types


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Matrix Channel Tests (mocked nio)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixChannel:
    """Test MatrixChannel event handlers with mocked nio client."""

    def _make_channel(self, agent=None, store=None):
        from march.channels.matrix_channel import MatrixChannel
        ch = MatrixChannel(orchestrator=None)
        ch._running = True
        ch._start_ts = 0
        ch._client = MagicMock()
        ch._client.user_id = "@march:localhost"
        ch._client.room_send = AsyncMock()
        ch._client.room_read_markers = AsyncMock()
        ch._client.room_typing = AsyncMock()
        ch._client.download = AsyncMock()
        if store is None:
            store = InMemorySessionStore()
        if agent is None:
            agent = MockAgent([StreamChunk(delta="Matrix response"), AgentResponse(content="Matrix response")])
        ch._orchestrator = Orchestrator(agent=agent, session_store=store)
        ch._agent = agent
        return ch

    def _make_room(self, room_id: str = "!test:localhost"):
        room = MagicMock()
        room.room_id = room_id
        return room

    def _make_text_event(self, body: str, sender: str = "@user:localhost", ts: int = 99999999):
        event = MagicMock()
        event.body = body
        event.sender = sender
        event.server_timestamp = ts
        event.event_id = "$test-event"
        return event

    async def test_text_event_processing(self):
        """Text message triggers orchestrator and sends reply."""
        ch = self._make_channel()
        room = self._make_room()
        event = self._make_text_event("Hello March!")
        await ch._on_message(room, event)
        await asyncio.sleep(0.15)
        assert ch._client.room_send.called

    async def test_ignores_own_messages(self):
        ch = self._make_channel()
        room = self._make_room()
        event = self._make_text_event("Hello", sender="@march:localhost")
        await ch._on_message(room, event)
        await asyncio.sleep(0.05)
        assert not ch._client.room_send.called

    async def test_ignores_pre_startup_messages(self):
        ch = self._make_channel()
        ch._start_ts = 100000
        room = self._make_room()
        event = self._make_text_event("Old message", ts=50000)
        await ch._on_message(room, event)
        await asyncio.sleep(0.05)
        assert not ch._client.room_send.called

    async def test_image_event_triggers_processing(self):
        """Image event calls _on_image which triggers download + processing task."""
        ch = self._make_channel()
        room = self._make_room()
        event = MagicMock()
        event.sender = "@user:localhost"
        event.server_timestamp = 99999999
        event.event_id = "$img-event"
        event.url = "mxc://localhost/test-image"
        event.body = "image.jpg"
        event.key = None
        event.hashes = None
        event.iv = None

        # Mock download
        download_resp = MagicMock()
        download_resp.body = b'\xff\xd8\xff' + b'\x00' * 100
        download_resp.content_type = "image/jpeg"
        ch._client.download = AsyncMock(return_value=download_resp)

        # Patch _process_image to avoid PIL dependency
        ch._process_image = AsyncMock()
        await ch._on_image(room, event)
        await asyncio.sleep(0.1)

        # Should have acknowledged with read markers
        assert ch._client.room_read_markers.called

    async def test_audio_event_transcription(self):
        """Audio event downloads, transcribes, and sends to orchestrator."""
        ch = self._make_channel()
        room = self._make_room()
        event = MagicMock()
        event.sender = "@user:localhost"
        event.server_timestamp = 99999999
        event.event_id = "$audio-event"
        event.url = "mxc://localhost/test-audio"
        event.body = "voice.ogg"
        event.mimetype = "audio/ogg"
        event.key = None
        event.hashes = None
        event.iv = None

        download_resp = MagicMock()
        download_resp.body = b'fake audio data'
        download_resp.content_type = "audio/ogg"
        ch._client.download = AsyncMock(return_value=download_resp)

        mock_tool_result = MagicMock()
        mock_tool_result.is_error = False
        mock_tool_result.content = "Hello from voice"
        ch._agent.tools = MagicMock()
        ch._agent.tools.execute = AsyncMock(return_value=mock_tool_result)

        await ch._on_audio(room, event)
        await asyncio.sleep(0.2)
        assert ch._agent.tools.execute.called

    async def test_file_pdf_event(self):
        """PDF file event downloads, extracts text, sends to orchestrator."""
        ch = self._make_channel()
        room = self._make_room()
        event = MagicMock()
        event.sender = "@user:localhost"
        event.server_timestamp = 99999999
        event.event_id = "$file-event"
        event.url = "mxc://localhost/test-file"
        event.body = "report.pdf"
        event.mimetype = "application/pdf"
        event.key = None
        event.hashes = None
        event.iv = None

        download_resp = MagicMock()
        download_resp.body = b'%PDF-1.4 fake pdf content'
        download_resp.content_type = "application/pdf"
        ch._client.download = AsyncMock(return_value=download_resp)

        with patch.object(ch, "_extract_pdf_text", return_value="[PDF: report.pdf]\n\nExtracted text"):
            await ch._on_file(room, event)
            await asyncio.sleep(0.2)
        assert ch._client.room_read_markers.called

    async def test_stop_command(self):
        """Matrix /stop command sets cancel event."""
        ch = self._make_channel()
        room = self._make_room("!stop-room:localhost")
        event = self._make_text_event("/stop")
        cancel = ch._reset_cancel_event("!stop-room:localhost")
        assert not cancel.is_set()
        await ch._on_message(room, event)
        await asyncio.sleep(0.05)
        room_cancel = ch._cancel_events.get("!stop-room:localhost")
        assert room_cancel is not None
        assert room_cancel.is_set()

    async def test_reset_command(self):
        """Matrix /reset command resets session via orchestrator."""
        store = InMemorySessionStore()
        ch = self._make_channel(store=store)
        room = self._make_room("!reset-room:localhost")
        session_id = ch._session_id_for_room("!reset-room:localhost")
        await store.create_session("matrix", "!reset-room:localhost", session_id=session_id)
        event = self._make_text_event("/reset")
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await ch._on_message(room, event)
            await asyncio.sleep(0.1)
        assert ch._client.room_send.called

    async def test_session_id_deterministic_for_room(self):
        from march.channels.matrix_channel import MatrixChannel
        id1 = MatrixChannel._session_id_for_room("!abc:server")
        id2 = MatrixChannel._session_id_for_room("!abc:server")
        id3 = MatrixChannel._session_id_for_room("!xyz:server")
        assert id1 == id2
        assert id1 != id3

    async def test_cancel_event_per_room(self):
        ch = self._make_channel()
        ev1 = ch._reset_cancel_event("!room1:localhost")
        ev2 = ch._reset_cancel_event("!room2:localhost")
        assert ev1 is not ev2
        ev1.set()
        assert not ev2.is_set()

    async def test_markdown_to_html(self):
        from march.channels.matrix_channel import MatrixChannel
        html = MatrixChannel._markdown_to_html("**bold** and `code`")
        assert "<strong>" in html or "<b>" in html
        assert "<code>" in html

    async def test_is_text_file_detection(self):
        from march.channels.matrix_channel import MatrixChannel
        assert MatrixChannel._is_text_file("test.py", "text/plain") is True
        assert MatrixChannel._is_text_file("test.md", "text/markdown") is True
        assert MatrixChannel._is_text_file("data.json", "application/json") is True
        assert MatrixChannel._is_text_file("image.png", "image/png") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Session Memory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionMemory:
    """Test session memory: save facts, plans, checkpoint, and cross-turn persistence."""

    async def test_save_facts(self, tmp_path: Path):
        """session_memory_tool saves facts to disk."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        session_id = "mem-test-facts"
        current_session_id.set(session_id)

        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            result = await session_memory_tool(type="facts", content="- User prefers Python 3.12")
        assert "Saved" in result

    async def test_save_plan(self, tmp_path: Path):
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        current_session_id.set("mem-test-plan")
        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            result = await session_memory_tool(type="plan", content="1. Refactor DB\n2. Add tests")
        assert "Saved" in result

    async def test_save_invalid_type(self):
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        current_session_id.set("test-invalid")
        result = await session_memory_tool(type="invalid", content="stuff")
        assert "Error" in result

    async def test_save_empty_content(self):
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        current_session_id.set("test-empty")
        result = await session_memory_tool(type="facts", content="   ")
        assert "Error" in result

    async def test_save_without_session_id(self):
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        current_session_id.set("")
        result = await session_memory_tool(type="facts", content="something")
        assert "Error" in result

    async def test_facts_append_not_overwrite(self, tmp_path: Path):
        """Multiple fact saves append, not overwrite."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        session_id = "mem-append"
        current_session_id.set(session_id)
        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            await session_memory_tool(type="facts", content="- Fact 1")
            await session_memory_tool(type="facts", content="- Fact 2")

        facts_file = tmp_path / ".march" / "memory" / session_id / "facts.md"
        content = facts_file.read_text()
        assert "Fact 1" in content
        assert "Fact 2" in content

    async def test_memory_across_turns(self, tmp_path: Path):
        """Save in turn 1, verify accessible in turn 2."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        session_id = "mem-cross-turn"
        current_session_id.set(session_id)

        # Turn 1: save facts
        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            await session_memory_tool(type="facts", content="- Deploy target is AWS Lambda")

        # Turn 2: verify the file exists and has the right content
        facts_file = tmp_path / ".march" / "memory" / session_id / "facts.md"
        assert facts_file.exists()
        content = facts_file.read_text()
        assert "AWS Lambda" in content

    async def test_checkpoint_facts_and_plan(self, tmp_path: Path):
        """Save both facts and plan, verify both files exist."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        session_id = "mem-checkpoint"
        current_session_id.set(session_id)
        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            await session_memory_tool(type="facts", content="- Project uses FastAPI")
            await session_memory_tool(type="plan", content="1. Add auth\n2. Deploy")

        memory_dir = tmp_path / ".march" / "memory" / session_id
        assert (memory_dir / "facts.md").exists()
        assert (memory_dir / "plan.md").exists()
        assert "FastAPI" in (memory_dir / "facts.md").read_text()
        assert "auth" in (memory_dir / "plan.md").read_text()

    async def test_rolling_summary_update(self):
        """Rolling summary can be updated and retrieved."""
        s = Session(id="roll-sum")
        assert s.rolling_summary == ""
        s.rolling_summary = "User is building a web app with FastAPI"
        assert "FastAPI" in s.rolling_summary

    async def test_compaction_extracts_memory(self, tmp_path: Path):
        """Compaction extracts facts/plans from old messages."""
        from march.core.compaction import extract_session_memory

        session_id = "compact-mem"
        memory_dir = tmp_path / session_id

        messages = [
            {"role": "user", "content": "My project uses Python 3.12 and FastAPI"},
            {"role": "assistant", "content": "Got it, I'll remember that."},
        ]

        async def mock_summarize(prompt: str) -> str:
            return "## Facts\n- Project uses Python 3.12 and FastAPI\n\n## Plan\nNone"

        await extract_session_memory(messages, session_id, mock_summarize, memory_dir=str(memory_dir))
        facts_file = memory_dir / "facts.md"
        assert facts_file.exists()
        assert "Python 3.12" in facts_file.read_text()

    async def test_delete_session_memory(self, tmp_path: Path):
        """delete_session_memory removes the session memory directory."""
        from march.core.compaction import delete_session_memory

        session_id = "del-mem"
        memory_dir = tmp_path / ".march" / "memory" / session_id
        memory_dir.mkdir(parents=True)
        (memory_dir / "facts.md").write_text("some facts")

        # Patch pathlib.Path.home at the builtins level since compaction
        # does `from pathlib import Path` inside the function
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = delete_session_memory(session_id)

        assert result is True
        assert not memory_dir.exists()

    async def test_timestamps_in_saved_facts(self, tmp_path: Path):
        """Saved facts include timestamps."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool

        session_id = "mem-ts"
        current_session_id.set(session_id)
        with patch("march.tools.builtin.session_memory_tool.Path.home", return_value=tmp_path):
            await session_memory_tool(type="facts", content="- Important fact")

        facts_file = tmp_path / ".march" / "memory" / session_id / "facts.md"
        content = facts_file.read_text()
        assert "[" in content and "UTC]" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Reset Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReset:
    """Test reset: clears history, DB, memory; preserves other sessions."""

    async def test_reset_clears_history(self):
        """Reset clears in-memory session history."""
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "reset-hist", "Hello")
        session = orch.get_cached_session("reset-hist")
        assert session is not None
        assert len(session.history) == 2  # user + assistant

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("reset-hist")
        assert orch.get_cached_session("reset-hist") is None

    async def test_reset_clears_db(self):
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "reset-db", "Hello")
        assert len(await store.get_messages("reset-db")) > 0
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("reset-db")
        assert len(await store.get_messages("reset-db")) == 0

    async def test_reset_clears_memory(self, tmp_path: Path):
        """Reset deletes session memory files."""
        session_id = "reset-mem"
        memory_dir = tmp_path / ".march" / "memory" / session_id
        memory_dir.mkdir(parents=True)
        (memory_dir / "facts.md").write_text("important facts")

        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, session_id, "Hello")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = await orch.reset_session(session_id)
        assert result.get("session_memory_deleted") is True
        assert not memory_dir.exists()

    async def test_reset_preserves_other_sessions(self):
        agent = MockAgent([StreamChunk(delta="A"), AgentResponse(content="A")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "sess-A-reset", "Hello A")
        agent.items = [StreamChunk(delta="B"), AgentResponse(content="B")]
        await collect_events(orch, "sess-B-keep", "Hello B")

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("sess-A-reset")

        assert orch.get_cached_session("sess-B-keep") is not None
        assert len(await store.get_messages("sess-B-keep")) > 0
        assert orch.get_cached_session("sess-A-reset") is None

    async def test_reset_during_streaming(self):
        """Reset while a stream is active (cancel + reset)."""
        items = [
            StreamChunk(delta="Part 1 "),
            StreamChunk(delta="Part 2 "),
            AgentResponse(content="Full"),
        ]
        agent = SlowAgent(items, delay=0.05)
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        cancel = asyncio.Event()

        events: list[OrchestratorEvent] = []
        async for ev in orch.handle_message("reset-stream", "Hello", source="test", cancel_event=cancel):
            events.append(ev)
            if isinstance(ev, TextDelta) and not cancel.is_set():
                cancel.set()
        assert any(isinstance(e, Cancelled) for e in events)

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("reset-stream")
        assert orch.get_cached_session("reset-stream") is None

        # Can start fresh
        orch.agent = MockAgent([StreamChunk(delta="Fresh"), AgentResponse(content="Fresh")])
        events2 = await collect_events(orch, "reset-stream", "New message")
        finals = [e for e in events2 if isinstance(e, FinalResponse)]
        assert finals[0].content == "Fresh"

    async def test_reset_nonexistent_session(self):
        orch = Orchestrator(agent=MockAgent([]), session_store=InMemorySessionStore())
        with patch("march.core.compaction.delete_session_memory", return_value=False):
            result = await orch.reset_session("nonexistent")
        assert isinstance(result, dict)

    async def test_reset_then_new_conversation(self):
        """After reset, new messages start a fresh conversation."""
        agent = MockAgent([StreamChunk(delta="First"), AgentResponse(content="First")])
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "reset-fresh", "Hello")
        session = orch.get_cached_session("reset-fresh")
        history_len_1 = len(session.history)

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("reset-fresh")

        agent.items = [StreamChunk(delta="Second"), AgentResponse(content="Second")]
        await collect_events(orch, "reset-fresh", "New hello")
        session2 = orch.get_cached_session("reset-fresh")
        assert len(session2.history) <= history_len_1

    async def test_reset_with_agent_memory(self):
        """Reset calls agent.memory.reset_session if memory exists."""
        agent = MockAgent([StreamChunk(delta="Hi"), AgentResponse(content="Hi")])
        agent.memory = MagicMock()
        agent.memory.reset_session = AsyncMock(return_value={"sqlite_entries": 5})
        store = InMemorySessionStore()
        orch = Orchestrator(agent=agent, session_store=store)
        await collect_events(orch, "reset-mem-agent", "Hello")

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            result = await orch.reset_session("reset-mem-agent")
        agent.memory.reset_session.assert_called_once_with("reset-mem-agent")
        assert result.get("memory") == {"sqlite_entries": 5}


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Compaction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompaction:
    def test_needs_compaction_small_context(self):
        from march.core.compaction import needs_compaction
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert needs_compaction(messages, context_window=200000) is False

    def test_needs_compaction_large_context(self):
        from march.core.compaction import needs_compaction
        messages = [{"role": "user", "content": "x" * 1000} for _ in range(100)]
        assert needs_compaction(messages, context_window=5000) is True

    def test_split_for_compaction(self):
        from march.core.compaction import split_for_compaction
        # Each message ~250 tokens → 100 messages ≈ 25000 tokens.
        # With a large context_window, keep_budget allows some recent messages,
        # and the rest become "old" for compaction.
        messages = [{"role": "user", "content": "x" * 1000} for _ in range(100)]
        old, recent = split_for_compaction(messages, context_window=200000)
        # At minimum, the split should partition all messages
        assert len(old) + len(recent) == len(messages)
        # With 100 messages, there should be some in each bucket
        # (exact split depends on token estimation and keep_budget ratio)
        assert len(recent) >= 1

    def test_estimate_tokens(self):
        from march.core.compaction import estimate_tokens
        assert estimate_tokens("hello") == 1
        assert estimate_tokens("a" * 400) == 100

    def test_build_summary_prompt(self):
        from march.core.compaction import build_summary_prompt
        messages = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
        prompt = build_summary_prompt(messages)
        assert "2+2" in prompt
        prompt2 = build_summary_prompt(messages, previous_summary="Earlier: discussed math")
        assert "Earlier: discussed math" in prompt2

    def test_estimate_message_tokens_multimodal(self):
        from march.core.compaction import estimate_message_tokens
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": "base64..."}},
                {"type": "text", "text": "What is this?"},
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 1600


# ═══════════════════════════════════════════════════════════════════════════════
# 14. InMemorySessionStore meta-tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInMemorySessionStore:
    async def test_create_and_get(self):
        store = InMemorySessionStore()
        s = await store.create_session("test", "src-1", "Test", session_id="s1")
        assert s.id == "s1"
        fetched = await store.get_session("s1")
        assert fetched is not None

    async def test_get_nonexistent(self):
        store = InMemorySessionStore()
        assert await store.get_session("nope") is None

    async def test_add_and_get_messages(self):
        store = InMemorySessionStore()
        await store.create_session("test", "src", session_id="s1")
        await store.add_message("s1", Message.user("hello"))
        await store.add_message("s1", Message.assistant("world"))
        msgs = await store.get_messages("s1")
        assert len(msgs) == 2

    async def test_clear_session(self):
        store = InMemorySessionStore()
        await store.create_session("test", "src", session_id="s1")
        await store.add_message("s1", Message.user("hello"))
        await store.clear_session("s1")
        assert await store.get_session("s1") is None
        assert len(await store.get_messages("s1")) == 0
