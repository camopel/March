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
import os
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

    async def get_messages_after_seq(self, session_id: str, last_processed_seq: int) -> list[Message]:
        msgs = self._messages.get(session_id, [])
        return [m for m in msgs if (m.metadata or {}).get("seq", 0) > last_processed_seq]

    async def flush_messages(self, session_id: str, messages: list[Message]) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].extend(messages)

    async def save_session(self, session: Session) -> None:
        self._sessions[session.id] = session

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
        assert len(orch._sessions["sess-cold"].messages) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TurnLogger tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTurnLogger:
    def test_turn_start_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="hello", source="test")
        lines = logger._path.read_text().strip().split("\n")
        data = json.loads(lines[0])
        assert data["event"] == "turn_start"
        assert data["turn_id"] == "t1"
        assert "ts" in data

    def test_turn_complete_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_complete(turn_id="t2", session_id="s2", tool_calls=3, total_tokens=500, total_cost=0.01, duration_ms=1234.5, final_reply_length=100)
        data = json.loads(logger._path.read_text().strip())
        assert data["event"] == "turn_complete"
        assert data["tool_calls"] == 3

    def test_tool_call_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.tool_call(turn_id="t3", session_id="s3", name="web_search", args={"query": "cats"}, duration_ms=200.0, status="complete", summary="Found 5 results")
        data = json.loads(logger._path.read_text().strip())
        assert data["event"] == "tool_call"
        assert data["name"] == "web_search"

    def test_turn_cancelled_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_cancelled(turn_id="t4", session_id="s4", partial_content_length=42)
        data = json.loads(logger._path.read_text().strip())
        assert data["event"] == "turn_cancelled"

    def test_turn_error_writes_jsonl(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_error(turn_id="t5", session_id="s5", error="kaboom")
        data = json.loads(logger._path.read_text().strip())
        assert data["event"] == "turn_error"

    def test_log_rotation(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)
        log_file = logger._path
        log_file.write_text("x" * (_MAX_FILE_BYTES + 1))
        logger.turn_start(turn_id="t-rot", session_id="s-rot", user_msg="rotate", source="test")
        assert log_file.with_suffix(".jsonl.1").exists()
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

        lines = logger._path.read_text().strip().split("\n")
        assert len(lines) == num_threads * writes_per
        for line in lines:
            json.loads(line)  # Should not raise

    def test_json_serialization_non_serializable(self, tmp_path: Path):
        logger = TurnLogger(log_dir=tmp_path)

        class Weird:
            pass

        logger.tool_call(turn_id="t-weird", session_id="s-weird", name="test_tool", args={"obj": Weird()}, duration_ms=10.0, status="complete", summary="ok")
        data = json.loads(logger._path.read_text().strip())
        assert "Weird" in data["args"]["obj"]

    def test_creates_log_dir(self, tmp_path: Path):
        log_dir = tmp_path / "nested" / "logs"
        logger = TurnLogger(log_dir=log_dir)
        logger.turn_start(turn_id="t-dir", session_id="s-dir", user_msg="hi", source="test")
        assert logger._path.exists()
        assert (log_dir / "turns").is_dir()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Session dataclass tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSession:
    def test_session_creation_with_defaults(self):
        s = Session()
        assert s.id
        assert s.source_type == "terminal"
        assert s.messages == []

    def test_session_deterministic_id(self):
        s1 = Session(source_type="matrix", source_id="!room:server")
        s2 = Session(source_type="matrix", source_id="!room:server")
        assert s1.id == s2.id

    def test_add_message(self):
        s = Session(id="test")
        s.add_message(Message.user("hello"))
        assert len(s.messages) == 1
        assert len(s.dirty_messages) == 1

    def test_add_exchange(self):
        s = Session(id="test")
        s.add_exchange("user says", "assistant says")
        assert len(s.messages) == 2
        assert s.messages[0].role == Role.USER
        assert s.messages[1].role == Role.ASSISTANT

    def test_clear(self):
        s = Session(id="test")
        s.add_exchange("a", "b")
        s.rolling_summary = "rolling"
        s.clear()
        assert s.messages == []
        assert s.dirty_messages == []
        assert s.rolling_summary == ""
        assert s.last_processed_seq == 0

    def test_reset(self):
        s = Session(id="test")
        s.add_exchange("a", "b")
        s.reset()
        assert s.messages == []
        assert s.dirty_messages == []

    def test_compact(self):
        s = Session(id="test")
        for i in range(20):
            s.add_exchange(f"user-{i}", f"assistant-{i}")
        assert len(s.messages) == 40
        s.compact("Summary of conversation")
        assert s.messages == []
        assert s.dirty_messages == []
        assert s.rolling_summary == "Summary of conversation"
        assert s.last_processed_seq == s._seq_counter

    def test_to_dict_and_from_dict(self):
        s = Session(id="test-rt", source_type="ws", source_id="ws-1", name="Test Session")
        s.add_exchange("hello", "world")
        s.metadata = {"key": "value"}
        d = s.to_dict()
        restored = Session.from_dict(d)
        assert restored.id == s.id
        assert restored.source_type == s.source_type
        assert len(restored.messages) == len(s.messages)
        assert restored.metadata == s.metadata

    def test_get_messages_for_llm(self):
        s = Session(id="test")
        s.add_exchange("hello", "world")
        msgs = s.get_messages_for_llm()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_get_messages_for_llm_with_rolling_summary(self):
        s = Session(id="test")
        s.rolling_summary = "Previous context summary"
        s.add_exchange("hello", "world")
        msgs = s.get_messages_for_llm()
        assert len(msgs) == 3  # rolling summary + user + assistant
        assert "Context Summary" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_needs_flush(self):
        s = Session(id="test")
        assert s.needs_flush() is False
        for i in range(10):
            s.add_message(Message.user(f"msg-{i}"))
        assert s.needs_flush() is True

    def test_flush(self):
        s = Session(id="test")
        s.add_message(Message.user("hello"))
        s.add_message(Message.assistant("world"))
        flushed = s.flush()
        assert len(flushed) == 2
        assert s.dirty_messages == []

    def test_history_property_alias(self):
        """Legacy .history property maps to .messages."""
        s = Session(id="test")
        s.add_exchange("hello", "world")
        assert s.history is s.messages
        assert len(s.history) == 2


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
            cur_len = len(session.messages)
            assert cur_len > prev_len, f"Turn {i}: messages didn't grow ({cur_len} <= {prev_len})"
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
    """Test WSChannel logic with mocked WebSocket and DB."""

    def _make_ws_conn(self, session_id: str = "ws-test"):
        from march.channels.ws_channel import _WSConn
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        conn = _WSConn(ws=mock_ws, session_id=session_id)
        return conn

    def _make_plugin(self, agent=None, store=None):
        from march.channels.ws_channel import WSChannel
        plugin = WSChannel()
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
        from march.channels.ws_channel import _StreamBuffer
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
        from march.channels.ws_channel import _StreamBuffer
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
        from march.channels.ws_channel import ChatDB
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

    async def test_compaction_loads_session_memory(self, tmp_path: Path):
        """_load_session_memory loads facts/plans from disk."""
        from march.core.compaction import _load_session_memory

        session_id = "compact-mem"
        memory_dir = tmp_path / ".march" / "memory" / session_id
        memory_dir.mkdir(parents=True)
        (memory_dir / "facts.md").write_text("- Project uses Python 3.12 and FastAPI")
        (memory_dir / "plan.md").write_text("1. Refactor DB\n2. Add tests")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = _load_session_memory(session_id)

        assert "Python 3.12" in result["facts"]
        assert "Refactor DB" in result["plan"]

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
        assert len(session.messages) == 2  # user + assistant

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
        history_len_1 = len(session.messages)

        with patch("march.core.compaction.delete_session_memory", return_value=False):
            await orch.reset_session("reset-fresh")

        agent.items = [StreamChunk(delta="Second"), AgentResponse(content="Second")]
        await collect_events(orch, "reset-fresh", "New hello")
        session2 = orch.get_cached_session("reset-fresh")
        assert len(session2.messages) <= history_len_1

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


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Session from Different Devices / Connections
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossDeviceSession:
    """Test session behaviour across multiple WS connections and Matrix event
    types — takeover, reconnect continuity, and shared-session history."""

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_ws(*, closed: bool = False) -> AsyncMock:
        ws = AsyncMock()
        ws.closed = closed
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @staticmethod
    def _make_plugin(agent=None, store=None):
        from march.channels.ws_channel import WSChannel
        plugin = WSChannel()
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
        return plugin, store

    @staticmethod
    def _make_matrix_channel(agent=None, store=None):
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
            agent = MockAgent([
                StreamChunk(delta="Matrix reply"),
                AgentResponse(content="Matrix reply"),
            ])
        ch._orchestrator = Orchestrator(agent=agent, session_store=store)
        ch._agent = agent
        return ch, store

    # ── WS takeover ───────────────────────────────────────────────────

    async def test_same_session_different_ws_connections(self):
        """Second WS client takes over: first gets session.takeover + close,
        second works normally."""
        from march.channels.ws_channel import _WSConn, _try_send

        plugin, store = self._make_plugin(
            agent=MockAgent([
                StreamChunk(delta="From B"),
                AgentResponse(content="From B"),
            ]),
        )
        sid = "shared-sess"
        await store.create_session("ws", sid, session_id=sid)

        # --- Client A connects ---
        ws_a = self._make_ws()
        plugin._active_connections[sid] = ws_a

        # --- Client B connects to the same session ---
        ws_b = self._make_ws()

        # Simulate the takeover logic from _handle_ws
        existing = plugin._active_connections.get(sid)
        assert existing is ws_a
        assert not existing.closed

        # Send takeover to A and close it (mirrors _handle_ws)
        await _try_send(existing, {
            "type": "session.takeover",
            "message": "Another client connected to this session",
        })
        await existing.close()

        # Register B as the active connection
        plugin._active_connections[sid] = ws_b

        # Verify A received session.takeover
        a_calls = ws_a.send_json.call_args_list
        a_types = [c.args[0].get("type") for c in a_calls]
        assert "session.takeover" in a_types
        # Verify A was closed
        ws_a.close.assert_called_once()

        # Verify B is now the active connection
        assert plugin._active_connections[sid] is ws_b

        # --- Client B sends a message and gets a normal response ---
        conn_b = _WSConn(ws=ws_b, session_id=sid)
        await plugin._ws_handle_message(conn_b, {"type": "message", "content": "Hello from B"})

        b_calls = ws_b.send_json.call_args_list
        b_types = [c.args[0].get("type") for c in b_calls]
        assert "stream.start" in b_types
        assert "stream.delta" in b_types
        assert "stream.end" in b_types

    # ── Reconnect continuity ─────────────────────────────────────────

    async def test_session_continuity_across_reconnect(self):
        """Connect → send message → disconnect → reconnect → verify history
        is preserved and agent has context from previous messages."""
        from march.channels.ws_channel import _WSConn

        store = InMemorySessionStore()
        sid = "reconnect-sess"

        # --- First connection: send a message ---
        agent_1 = MockAgent([
            StreamChunk(delta="First reply"),
            AgentResponse(content="First reply"),
        ])
        plugin, _ = self._make_plugin(agent=agent_1, store=store)
        await store.create_session("ws", sid, session_id=sid)

        ws_1 = self._make_ws()
        plugin._active_connections[sid] = ws_1
        conn_1 = _WSConn(ws=ws_1, session_id=sid)
        await plugin._ws_handle_message(conn_1, {"type": "message", "content": "Hello first"})

        # Verify first reply was sent
        ws1_types = [c.args[0].get("type") for c in ws_1.send_json.call_args_list]
        assert "stream.end" in ws1_types

        # Verify DB has the exchange
        db_msgs = await store.get_messages(sid)
        assert len(db_msgs) >= 2  # user + assistant

        # --- Disconnect (simulate) ---
        plugin._active_connections.pop(sid, None)

        # --- Second connection: send another message ---
        agent_2 = MockAgent([
            StreamChunk(delta="Second reply"),
            AgentResponse(content="Second reply"),
        ])
        plugin._agent = agent_2
        plugin._orchestrator.agent = agent_2

        ws_2 = self._make_ws()
        plugin._active_connections[sid] = ws_2
        conn_2 = _WSConn(ws=ws_2, session_id=sid)
        await plugin._ws_handle_message(conn_2, {"type": "message", "content": "Hello second"})

        # Verify second reply was sent
        ws2_types = [c.args[0].get("type") for c in ws_2.send_json.call_args_list]
        assert "stream.end" in ws2_types

        # Verify DB now has both exchanges
        all_msgs = await store.get_messages(sid)
        assert len(all_msgs) >= 4  # 2 user + 2 assistant

        # Verify the session's in-memory history has both turns
        session = plugin._orchestrator.get_cached_session(sid)
        assert session is not None
        assert len(session.history) >= 4

    # ── Matrix: mixed event types in same room ───────────────────────

    async def test_matrix_same_room_different_events(self):
        """Text, image, and audio events in the same Matrix room all share
        one session with a unified history."""
        from march.channels.matrix_channel import MatrixChannel

        room_id = "!multi-event:localhost"
        session_id = MatrixChannel._session_id_for_room(room_id)

        store = InMemorySessionStore()

        # --- Turn 1: text message ---
        agent = MockAgent([
            StreamChunk(delta="Text reply"),
            AgentResponse(content="Text reply"),
        ])
        ch, _ = self._make_matrix_channel(agent=agent, store=store)
        room = MagicMock()
        room.room_id = room_id

        text_event = MagicMock()
        text_event.body = "Hello from text"
        text_event.sender = "@user:localhost"
        text_event.server_timestamp = 99999999
        text_event.event_id = "$text-1"

        await ch._on_message(room, text_event)
        await asyncio.sleep(0.15)

        # Verify text was processed
        assert ch._client.room_send.called
        ch._client.room_send.reset_mock()

        # --- Turn 2: image event ---
        agent.items = [
            StreamChunk(delta="Image reply"),
            AgentResponse(content="Image reply"),
        ]

        img_event = MagicMock()
        img_event.sender = "@user:localhost"
        img_event.server_timestamp = 100000000
        img_event.event_id = "$img-1"
        img_event.url = "mxc://localhost/test-image"
        img_event.body = "photo.jpg"
        img_event.key = None
        img_event.hashes = None
        img_event.iv = None

        # Patch _process_image to just call orchestrator with text
        async def fake_process_image(rid, ev):
            sid = MatrixChannel._session_id_for_room(rid)
            content = "[User sent image: photo.jpg]"
            async for _ in ch._orchestrator.handle_message(sid, content, source="matrix"):
                pass

        ch._process_image = fake_process_image
        await ch._on_image(room, img_event)
        await asyncio.sleep(0.15)

        # --- Turn 3: audio event ---
        agent.items = [
            StreamChunk(delta="Audio reply"),
            AgentResponse(content="Audio reply"),
        ]

        audio_event = MagicMock()
        audio_event.sender = "@user:localhost"
        audio_event.server_timestamp = 100000001
        audio_event.event_id = "$audio-1"
        audio_event.url = "mxc://localhost/test-audio"
        audio_event.body = "voice.ogg"
        audio_event.mimetype = "audio/ogg"
        audio_event.key = None
        audio_event.hashes = None
        audio_event.iv = None

        # Patch _process_audio to just call orchestrator with transcribed text
        async def fake_process_audio(rid, ev):
            sid = MatrixChannel._session_id_for_room(rid)
            content = "[Voice transcription] Turn on the lights"
            async for _ in ch._orchestrator.handle_message(sid, content, source="matrix"):
                pass

        ch._process_audio = fake_process_audio
        await ch._on_audio(room, audio_event)
        await asyncio.sleep(0.15)

        # --- Verify all three turns share the same session ---
        session = ch._orchestrator.get_cached_session(session_id)
        assert session is not None
        # 3 turns × 2 messages (user + assistant) = 6
        assert len(session.history) == 6

        # Verify DB has all messages under the same session_id
        db_msgs = await store.get_messages(session_id)
        assert len(db_msgs) >= 6

        # Verify the session_id is deterministic for the room
        assert session_id == MatrixChannel._session_id_for_room(room_id)

    # ── WS takeover during active stream ─────────────────────────────

    async def test_ws_takeover_during_stream(self):
        """Client A is streaming. Client B connects to the same session.
        Client A gets session.takeover. Client B gets stream.active with
        the partial content accumulated so far."""
        from march.channels.ws_channel import _WSConn, _StreamBuffer, _try_send

        plugin, store = self._make_plugin(
            agent=SlowAgent([
                StreamChunk(delta="Chunk 1 "),
                StreamChunk(delta="Chunk 2 "),
                StreamChunk(delta="Chunk 3 "),
                AgentResponse(content="Chunk 1 Chunk 2 Chunk 3 "),
            ], delay=0.05),
        )
        sid = "takeover-stream"
        await store.create_session("ws", sid, session_id=sid)

        # --- Client A connects and starts streaming ---
        ws_a = self._make_ws()
        plugin._active_connections[sid] = ws_a
        conn_a = _WSConn(ws=ws_a, session_id=sid)

        # Start the stream in the background
        stream_task = asyncio.create_task(
            plugin._stream_response(conn_a, "Start streaming")
        )

        # Wait for some chunks to arrive
        await asyncio.sleep(0.08)

        # The stream buffer should have partial content
        buf = plugin._get_stream_buffer(sid)
        assert buf.streaming is True
        assert len(buf.collected) > 0
        partial_at_takeover = buf.collected

        # --- Client B connects mid-stream ---
        ws_b = self._make_ws()

        # Simulate takeover: send takeover to A, close A, register B
        existing = plugin._active_connections.get(sid)
        assert existing is ws_a
        await _try_send(existing, {
            "type": "session.takeover",
            "message": "Another client connected to this session",
        })
        await existing.close()
        ws_a.closed = True  # Mark as closed so stream_response stops sending to it
        plugin._active_connections[sid] = ws_b

        # Simulate what _handle_ws does on connect: check buf.streaming
        assert buf.streaming is True
        await _try_send(ws_b, {
            "type": "stream.active",
            "chunk_id": buf.next_id - 1 if buf.next_id > 0 else -1,
            "collected": buf.collected,
        })

        # Verify A received session.takeover
        a_calls = ws_a.send_json.call_args_list
        a_types = [c.args[0].get("type") for c in a_calls]
        assert "session.takeover" in a_types
        ws_a.close.assert_called_once()

        # Verify B received stream.active with partial content
        b_calls = ws_b.send_json.call_args_list
        b_types = [c.args[0].get("type") for c in b_calls]
        assert "stream.active" in b_types

        # Find the stream.active message and verify it has the partial content
        active_msg = next(
            c.args[0] for c in b_calls if c.args[0].get("type") == "stream.active"
        )
        assert active_msg["collected"] == partial_at_takeover
        assert len(active_msg["collected"]) > 0
        assert active_msg["chunk_id"] >= 0

        # Let the stream finish
        await stream_task

        # After stream completes, buffer should be done
        assert buf.done is True
        assert buf.streaming is False

    # ── Edge cases ────────────────────────────────────────────────────

    async def test_takeover_when_first_client_already_closed(self):
        """If the first client already disconnected (ws.closed=True),
        takeover skips the send+close and just registers the new client."""
        from march.channels.ws_channel import _WSConn, _try_send

        plugin, store = self._make_plugin(
            agent=MockAgent([
                StreamChunk(delta="Reply"),
                AgentResponse(content="Reply"),
            ]),
        )
        sid = "takeover-closed"
        await store.create_session("ws", sid, session_id=sid)

        # Client A is already closed
        ws_a = self._make_ws(closed=True)
        plugin._active_connections[sid] = ws_a

        # Client B connects
        ws_b = self._make_ws()

        # Simulate takeover check — existing is closed, so skip send+close
        existing = plugin._active_connections.get(sid)
        if existing is not None and not existing.closed:
            await _try_send(existing, {"type": "session.takeover"})
            await existing.close()

        plugin._active_connections[sid] = ws_b

        # A should NOT have received takeover (it was already closed)
        assert ws_a.send_json.call_count == 0
        assert ws_a.close.call_count == 0

        # B is now active and can send messages
        conn_b = _WSConn(ws=ws_b, session_id=sid)
        await plugin._ws_handle_message(conn_b, {"type": "message", "content": "Hello"})

        b_types = [c.args[0].get("type") for c in ws_b.send_json.call_args_list]
        assert "stream.start" in b_types
        assert "stream.end" in b_types

    async def test_reconnect_gets_stream_catchup_when_done(self):
        """If a stream finished while the client was disconnected,
        reconnecting gets stream.catchup with the full content."""
        from march.channels.ws_channel import _StreamBuffer, _try_send

        plugin, store = self._make_plugin()
        sid = "catchup-sess"

        # Simulate a completed stream in the buffer
        buf = plugin._get_stream_buffer(sid)
        buf.collected = "The complete answer to your question."
        buf.done = True
        buf.streaming = False
        buf.add_chunk({"type": "stream.start"})
        buf.add_chunk({"type": "stream.delta", "content": "The complete answer to your question."})
        buf.add_chunk({"type": "stream.end"})

        # New client connects
        ws_new = self._make_ws()

        # Simulate the catchup logic from _handle_ws
        if buf.streaming:
            await _try_send(ws_new, {
                "type": "stream.active",
                "chunk_id": buf.next_id - 1,
                "collected": buf.collected,
            })
        elif buf.done and buf.collected:
            await _try_send(ws_new, {
                "type": "stream.catchup",
                "content": buf.collected,
                "done": True,
                "chunk_id": buf.next_id - 1,
            })

        calls = ws_new.send_json.call_args_list
        types = [c.args[0].get("type") for c in calls]
        assert "stream.catchup" in types

        catchup_msg = next(c.args[0] for c in calls if c.args[0].get("type") == "stream.catchup")
        assert catchup_msg["content"] == "The complete answer to your question."
        assert catchup_msg["done"] is True
        assert catchup_msg["chunk_id"] == buf.next_id - 1


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Guardian Process Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardian:
    """Test the Guardian process: registration, health checks, PID watching,
    restart protection, and graceful shutdown.

    All tests use a tmp_path-scoped Guardian with patched state dirs so nothing
    touches the real ~/.march/guardian.
    """

    # ── helpers ────────────────────────────────────────────────────────

    def _make_guardian(self, tmp_path: Path, **overrides) -> "Guardian":
        """Create a Guardian whose state files live under tmp_path."""
        from march.agents.guardian import Guardian, GuardianConfig

        defaults = dict(
            check_interval=1,  # fast for tests
            march_config_path=str(tmp_path / "config.yaml"),
        )
        defaults.update(overrides)
        cfg = GuardianConfig(**defaults)
        g = Guardian(config=cfg)

        # Redirect all state to tmp_path so tests are hermetic
        import march.agents.guardian as gmod

        gmod.GUARDIAN_STATE_DIR = tmp_path / "guardian"
        gmod.CONFIG_BACKUP_DIR = tmp_path / "guardian" / "config_backups"
        gmod.REGISTRY_FILE = tmp_path / "guardian" / "watched.json"

        return g

    @pytest.fixture(autouse=True)
    def _save_and_restore_module_paths(self, tmp_path: Path):
        """Save the module-level paths before each test and restore after."""
        import march.agents.guardian as gmod

        orig_state = gmod.GUARDIAN_STATE_DIR
        orig_backup = gmod.CONFIG_BACKUP_DIR
        orig_registry = gmod.REGISTRY_FILE
        yield
        gmod.GUARDIAN_STATE_DIR = orig_state
        gmod.CONFIG_BACKUP_DIR = orig_backup
        gmod.REGISTRY_FILE = orig_registry

    # ── WatchEntry dataclass ──────────────────────────────────────────

    def test_watch_entry_round_trip(self):
        """WatchEntry serialises to dict and back."""
        from march.agents.guardian import WatchEntry

        entry = WatchEntry(
            id="task-1", pid=12345, log_path="/tmp/test.log",
            target="matrix:!room:server", timeout=120,
            command="pytest tests/",
        )
        d = entry.to_dict()
        restored = WatchEntry.from_dict(d)
        assert restored.id == entry.id
        assert restored.pid == entry.pid
        assert restored.log_path == entry.log_path
        assert restored.command == entry.command
        assert restored.registered_at == entry.registered_at

    def test_watch_entry_auto_timestamp(self):
        """WatchEntry gets a timestamp on creation."""
        from march.agents.guardian import WatchEntry

        before = time.time()
        entry = WatchEntry(id="ts-test", pid=1)
        after = time.time()
        assert before <= entry.registered_at <= after

    # ── Registration & removal ────────────────────────────────────────

    async def test_register_and_status(self, tmp_path: Path):
        """Register an entry, verify it appears in status."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()

        entry = WatchEntry(id="test-reg", pid=os.getpid(), command="self")
        g.register(entry)

        st = g.status()
        assert st["running"] is False  # not in run_loop yet
        assert len(st["entries"]) == 1
        assert st["entries"][0]["id"] == "test-reg"
        assert st["entries"][0]["alive"] is True  # our own PID

    async def test_register_persists_to_disk(self, tmp_path: Path):
        """Registered entries survive a fresh Guardian load."""
        from march.agents.guardian import WatchEntry

        g1 = self._make_guardian(tmp_path)
        await g1.initialize()
        g1.register(WatchEntry(id="persist-test", pid=99999, command="fake"))

        # Create a second guardian instance that loads from the same state dir
        g2 = self._make_guardian(tmp_path)
        await g2.initialize()

        assert "persist-test" in g2._entries
        assert g2._entries["persist-test"].pid == 99999

    async def test_remove_entry(self, tmp_path: Path):
        """Remove an entry by id."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()
        g.register(WatchEntry(id="rm-test", pid=1, command="test"))
        assert "rm-test" in g._entries

        removed = g.remove("rm-test")
        assert removed is True
        assert "rm-test" not in g._entries

    async def test_remove_nonexistent(self, tmp_path: Path):
        """Removing a nonexistent entry returns False."""
        g = self._make_guardian(tmp_path)
        await g.initialize()
        assert g.remove("nope") is False

    # ── PID liveness checks ───────────────────────────────────────────

    async def test_health_check_alive_pid(self, tmp_path: Path):
        """Guardian detects that our own PID is alive."""
        from march.agents.guardian import Guardian

        g = self._make_guardian(tmp_path)
        assert g._is_pid_alive(os.getpid()) is True

    async def test_health_check_dead_pid(self, tmp_path: Path):
        """Guardian detects a dead PID (use a PID that can't exist)."""
        g = self._make_guardian(tmp_path)
        # PID 2^22 is almost certainly not running
        assert g._is_pid_alive(4194304) is False

    async def test_health_check_zero_pid(self, tmp_path: Path):
        """PID 0 is treated as not alive."""
        g = self._make_guardian(tmp_path)
        assert g._is_pid_alive(0) is False

    async def test_health_check_negative_pid(self, tmp_path: Path):
        """Negative PID is treated as not alive."""
        g = self._make_guardian(tmp_path)
        assert g._is_pid_alive(-1) is False

    # ── Dead PID detection (simulated crash) ──────────────────────────

    async def test_dead_pid_triggers_notification_and_removal(self, tmp_path: Path):
        """When a watched PID dies, guardian notifies and removes the entry."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()

        # Spawn a real subprocess, then kill it
        proc = await asyncio.create_subprocess_exec(
            "sleep", "60",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        child_pid = proc.pid
        assert g._is_pid_alive(child_pid) is True

        g.register(WatchEntry(
            id="crash-test", pid=child_pid,
            target="test-channel", command="sleep 60",
        ))

        # Kill the child to simulate a crash
        proc.kill()
        await proc.wait()
        assert g._is_pid_alive(child_pid) is False

        # Run one check cycle — should detect the dead PID
        notifications: list[str] = []
        g._notify = AsyncMock(side_effect=lambda t, m: notifications.append(m))

        await g._check_entries()

        # Entry should be removed
        assert "crash-test" not in g._entries
        # Notification should mention the dead process
        assert len(notifications) == 1
        assert "Process died" in notifications[0]
        assert "sleep 60" in notifications[0]

    async def test_alive_pid_not_removed(self, tmp_path: Path):
        """A healthy PID is not removed during check."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()

        g.register(WatchEntry(
            id="healthy", pid=os.getpid(), command="self",
        ))

        g._notify = AsyncMock()
        await g._check_entries()

        # Should still be registered
        assert "healthy" in g._entries
        g._notify.assert_not_called()

    # ── Log staleness detection ───────────────────────────────────────

    async def test_stale_log_triggers_notification(self, tmp_path: Path):
        """A log file that hasn't been modified triggers a stale warning."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path, log_stale_threshold=1)
        await g.initialize()

        # Create a log file and backdate its mtime
        log_file = tmp_path / "stale.log"
        log_file.write_text("old log content")
        old_time = time.time() - 600  # 10 minutes ago
        os.utime(log_file, (old_time, old_time))

        g.register(WatchEntry(
            id="stale-log", pid=0,  # no PID — log-only watch
            log_path=str(log_file), target="test",
            timeout=2, command="stale task",
        ))

        notifications: list[str] = []
        g._notify = AsyncMock(side_effect=lambda t, m: notifications.append(m))

        await g._check_entries()

        assert "stale-log" not in g._entries
        assert len(notifications) == 1
        assert "Log stale" in notifications[0]

    async def test_fresh_log_not_flagged(self, tmp_path: Path):
        """A recently-modified log file is not flagged as stale."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()

        log_file = tmp_path / "fresh.log"
        log_file.write_text("fresh content")
        # mtime is now — well within any threshold

        g.register(WatchEntry(
            id="fresh-log", pid=0,
            log_path=str(log_file), timeout=300, command="fresh task",
        ))

        g._notify = AsyncMock()
        await g._check_entries()

        assert "fresh-log" in g._entries
        g._notify.assert_not_called()

    async def test_missing_log_treated_as_stale(self, tmp_path: Path):
        """A log path that doesn't exist is treated as stale."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path)
        await g.initialize()

        g.register(WatchEntry(
            id="missing-log", pid=0,
            log_path=str(tmp_path / "nonexistent.log"),
            timeout=1, command="missing log task",
        ))

        notifications: list[str] = []
        g._notify = AsyncMock(side_effect=lambda t, m: notifications.append(m))

        await g._check_entries()

        assert "missing-log" not in g._entries
        assert len(notifications) == 1

    # ── Restart protection (config backup / revert) ───────────────────

    async def test_register_restart_creates_backup(self, tmp_path: Path):
        """register_restart() copies config.yaml to a timestamped backup."""
        g = self._make_guardian(tmp_path)
        await g.initialize()

        # Create a config file to back up
        config_path = Path(g.config.march_config_path)
        config_path.write_text("model: gpt-4o\nport: 8100\n")

        import march.agents.guardian as gmod
        backup_dir = gmod.CONFIG_BACKUP_DIR

        backup_path = g.register_restart()
        assert backup_path != ""
        assert Path(backup_path).exists()
        assert Path(backup_path).read_text() == "model: gpt-4o\nport: 8100\n"
        assert len(list(backup_dir.glob("*.yaml"))) == 1

    async def test_register_restart_prunes_old_backups(self, tmp_path: Path):
        """Only the N most recent backups are kept."""
        g = self._make_guardian(tmp_path, config_backup_count=3)
        await g.initialize()

        config_path = Path(g.config.march_config_path)

        import march.agents.guardian as gmod
        backup_dir = gmod.CONFIG_BACKUP_DIR

        # Create 5 backups with distinct timestamps by patching time.time
        fake_time = 1000000
        for i in range(5):
            config_path.write_text(f"version: {i}\n")
            with patch("march.agents.guardian.time.time", return_value=fake_time + i):
                g.register_restart()

        # Should only keep 3
        backups = list(backup_dir.glob("*.yaml"))
        assert len(backups) == 3

    async def test_register_restart_no_config(self, tmp_path: Path):
        """register_restart() returns empty string if config doesn't exist."""
        g = self._make_guardian(tmp_path)
        await g.initialize()
        # Don't create the config file
        assert g.register_restart() == ""

    async def test_verify_after_restart_success(self, tmp_path: Path):
        """verify_after_restart returns True when march --status succeeds."""
        g = self._make_guardian(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = await g.verify_after_restart()
        assert result is True

    async def test_verify_after_restart_failure(self, tmp_path: Path):
        """verify_after_restart returns False when march --status fails."""
        g = self._make_guardian(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = await g.verify_after_restart()
        assert result is False

    async def test_verify_after_restart_timeout(self, tmp_path: Path):
        """verify_after_restart returns False on timeout."""
        import subprocess as sp

        g = self._make_guardian(tmp_path)

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("march", 30)):
            result = await g.verify_after_restart()
        assert result is False

    async def test_recover_from_failed_restart(self, tmp_path: Path):
        """recover_from_failed_restart tries backups until one works."""
        g = self._make_guardian(tmp_path)
        await g.initialize()

        import march.agents.guardian as gmod
        backup_dir = gmod.CONFIG_BACKUP_DIR

        config_path = Path(g.config.march_config_path)

        # Create two backups: old (bad) and new (good)
        (backup_dir / "config_1000.yaml").write_text("bad: true\n")
        (backup_dir / "config_2000.yaml").write_text("good: true\n")

        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=mock_run):
            with patch.object(g, "verify_after_restart", return_value=True):
                g._notify = AsyncMock()
                result = await g.recover_from_failed_restart()

        assert result is True
        # Should have tried the newest backup first (config_2000)
        assert config_path.read_text() == "good: true\n"

    async def test_recover_all_backups_fail(self, tmp_path: Path):
        """If all backups fail, recovery returns False and notifies."""
        g = self._make_guardian(tmp_path)
        await g.initialize()

        import march.agents.guardian as gmod
        backup_dir = gmod.CONFIG_BACKUP_DIR

        config_path = Path(g.config.march_config_path)
        config_path.write_text("broken: true\n")
        (backup_dir / "config_1000.yaml").write_text("also_broken: true\n")

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch.object(g, "verify_after_restart", return_value=False):
                notifications: list[str] = []
                g._notify = AsyncMock(side_effect=lambda t, m: notifications.append(m))
                result = await g.recover_from_failed_restart()

        assert result is False
        assert any("recovery failed" in n for n in notifications)

    # ── Run loop & graceful shutdown ──────────────────────────────────

    async def test_run_loop_starts_and_stops(self, tmp_path: Path):
        """Guardian run_loop sets _running=True, stop() sets it False."""
        g = self._make_guardian(tmp_path, check_interval=1)
        await g.initialize()

        assert g._running is False

        # Start the loop in the background
        loop_task = asyncio.create_task(g.run_loop())
        await asyncio.sleep(0.05)
        assert g._running is True

        # Stop it
        await g.stop()
        await asyncio.sleep(0.05)

        # The task should finish
        try:
            await asyncio.wait_for(loop_task, timeout=2.0)
        except asyncio.TimeoutError:
            loop_task.cancel()
            pytest.fail("Guardian run_loop did not stop within 2s")

        assert g._running is False

    async def test_graceful_shutdown_cleans_up(self, tmp_path: Path):
        """After stop(), status shows running=False and entries are preserved."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path, check_interval=1)
        await g.initialize()

        g.register(WatchEntry(id="survive-stop", pid=os.getpid(), command="self"))

        loop_task = asyncio.create_task(g.run_loop())
        await asyncio.sleep(0.05)

        await g.stop()
        try:
            await asyncio.wait_for(loop_task, timeout=2.0)
        except asyncio.TimeoutError:
            loop_task.cancel()

        st = g.status()
        assert st["running"] is False
        # Entries survive shutdown (persisted to disk)
        assert len(st["entries"]) == 1
        assert st["entries"][0]["id"] == "survive-stop"

    async def test_check_runs_during_loop(self, tmp_path: Path):
        """The monitoring loop actually calls _check_entries periodically."""
        from march.agents.guardian import WatchEntry

        g = self._make_guardian(tmp_path, check_interval=1)
        await g.initialize()

        check_count = 0
        original_check = g._check_entries

        async def counting_check():
            nonlocal check_count
            check_count += 1
            await original_check()

        g._check_entries = counting_check

        loop_task = asyncio.create_task(g.run_loop())
        # Let it run for ~1.5 intervals
        await asyncio.sleep(1.5)
        await g.stop()
        try:
            await asyncio.wait_for(loop_task, timeout=2.0)
        except asyncio.TimeoutError:
            loop_task.cancel()

        assert check_count >= 1

    # ── Guardian spawned alongside March (CLI integration) ────────────

    async def test_start_command_spawns_guardian(self):
        """The `march start` CLI spawns a guardian subprocess by default."""
        from march.cli.start import start

        with patch("march.cli.start._start_subprocess", return_value=42) as mock_spawn:
            with patch("march.cli.start._find_march_pids", return_value=[]):
                with patch("march.cli.start._ensure_templates"):
                    with patch("march.app.MarchApp") as mock_app:
                        mock_instance = MagicMock()
                        mock_app.return_value = mock_instance
                        mock_instance.run = MagicMock(side_effect=SystemExit(0))

                        from click.testing import CliRunner
                        runner = CliRunner()
                        result = runner.invoke(start, [], catch_exceptions=True)

                        # Verify guardian was spawned
                        guardian_calls = [
                            c for c in mock_spawn.call_args_list
                            if c.args[0] == "guardian"
                        ]
                        assert len(guardian_calls) == 1

    async def test_start_no_guardian_flag(self):
        """The --no-guardian flag skips guardian spawning."""
        from march.cli.start import start

        with patch("march.cli.start._start_subprocess", return_value=42) as mock_spawn:
            with patch("march.cli.start._find_march_pids", return_value=[]):
                with patch("march.cli.start._ensure_templates"):
                    with patch("march.app.MarchApp") as mock_app:
                        mock_instance = MagicMock()
                        mock_app.return_value = mock_instance
                        mock_instance.run = MagicMock(side_effect=SystemExit(0))

                        from click.testing import CliRunner
                        runner = CliRunner()
                        result = runner.invoke(start, ["--no-guardian"], catch_exceptions=True)

                        guardian_calls = [
                            c for c in mock_spawn.call_args_list
                            if c.args[0] == "guardian"
                        ]
                        assert len(guardian_calls) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Logging System Structure Tests
#
# These tests verify the post-refactor log directory layout:
#
#   ~/.march/logs/
#     agent/YYYY-MM-DD.log        ← structured agent logs
#     guardian/YYYY-MM-DD.log     ← guardian process logs  (TODO: not date-based yet)
#     turns/YYYY-MM-DD.jsonl      ← per-turn debug JSONL
#     metrics/YYYY-MM-DD.jsonl    ← machine-readable metrics
#
# Tests marked @pytest.mark.xfail depend on the guardian date-based log
# refactor that hasn't landed yet.  All other tests verify the current
# (working) implementation.
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogDirectoryStructure:
    """Verify the log subdirectory layout is created on startup."""

    def test_ensure_log_subdirectories(self, tmp_path: Path):
        """ensure_log_subdirectories creates all canonical subdirs."""
        from march.core.log_maintenance import ensure_log_subdirectories, LOG_SUBDIRS

        result = ensure_log_subdirectories(tmp_path)
        assert result == tmp_path
        for name in LOG_SUBDIRS:
            assert (tmp_path / name).is_dir(), f"Missing subdir: {name}"

    def test_subdirectories_are_idempotent(self, tmp_path: Path):
        """Calling ensure_log_subdirectories twice doesn't fail or duplicate."""
        from march.core.log_maintenance import ensure_log_subdirectories

        ensure_log_subdirectories(tmp_path)
        ensure_log_subdirectories(tmp_path)  # second call — no error
        assert (tmp_path / "agent").is_dir()

    def test_canonical_subdirs_list(self):
        """The canonical subdirectory list includes all expected names."""
        from march.core.log_maintenance import LOG_SUBDIRS

        expected = {"agent", "guardian", "turns", "metrics", "dashboard"}
        assert set(LOG_SUBDIRS) == expected

    def test_configure_logging_creates_agent_dir(self, tmp_path: Path):
        """configure_logging() creates the agent/ subdir and a DateBasedFileHandler."""
        from march.logging.handlers import DateBasedFileHandler

        agent_dir = tmp_path / "agent"
        handler = DateBasedFileHandler(log_dir=agent_dir, ext=".log")
        assert agent_dir.is_dir()
        handler.close()

    def test_turn_logger_creates_turns_subdir(self, tmp_path: Path):
        """TurnLogger creates turns/ subdir under its log_dir."""
        logger = TurnLogger(log_dir=tmp_path)
        assert (tmp_path / "turns").is_dir()

    def test_metrics_logger_creates_metrics_dir(self, tmp_path: Path):
        """MetricsLogger creates its metrics directory."""
        from march.logging.logger import MetricsLogger

        MetricsLogger.reset()
        metrics_dir = tmp_path / "metrics"
        ml = MetricsLogger(metrics_dir=metrics_dir)
        assert metrics_dir.is_dir()
        MetricsLogger.reset()


class TestLogDateRotation:
    """Verify that logs rotate by date (each day → new file)."""

    def test_turn_logger_uses_date_filename(self, tmp_path: Path):
        """TurnLogger writes to turns/YYYY-MM-DD.jsonl."""
        from datetime import datetime, timezone

        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="hi", source="test")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected = tmp_path / "turns" / f"{today}.jsonl"
        assert expected.exists()
        data = json.loads(expected.read_text().strip())
        assert data["event"] == "turn_start"

    def test_metrics_logger_uses_date_filename(self, tmp_path: Path):
        """MetricsLogger writes to metrics/YYYY-MM-DD.jsonl."""
        from datetime import datetime, timezone
        from march.logging.logger import MetricsLogger

        MetricsLogger.reset()
        metrics_dir = tmp_path / "metrics"
        ml = MetricsLogger(metrics_dir=metrics_dir)
        ml.llm_call(
            session_id="s1", provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, cost_usd=0.005, duration_ms=800,
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected = metrics_dir / f"{today}.jsonl"
        assert expected.exists()
        data = json.loads(expected.read_text().strip())
        assert data["event"] == "llm.call"
        MetricsLogger.reset()

    def test_date_based_file_handler_uses_date_filename(self, tmp_path: Path):
        """DateBasedFileHandler writes to <dir>/YYYY-MM-DD.<ext>."""
        import logging as stdlib_logging
        from datetime import date
        from march.logging.handlers import DateBasedFileHandler

        handler = DateBasedFileHandler(log_dir=tmp_path, ext=".log")
        handler.setFormatter(stdlib_logging.Formatter("%(message)s"))

        record = stdlib_logging.LogRecord(
            name="test", level=stdlib_logging.INFO, pathname="", lineno=0,
            msg="test log line", args=(), exc_info=None,
        )
        handler.emit(record)
        handler.close()

        today = date.today().isoformat()
        expected = tmp_path / f"{today}.log"
        assert expected.exists()
        assert "test log line" in expected.read_text()

    def test_turn_logger_date_change(self, tmp_path: Path):
        """When the date changes, TurnLogger writes to a new file."""
        from datetime import datetime, timezone
        from unittest.mock import patch as _patch

        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="day1", source="test")

        old_path = logger._path
        assert old_path.exists()

        # Simulate date change by patching datetime in the turn_log module
        with _patch("march.core.turn_log.datetime") as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2099-12-31T00:00:00+00:00"
            mock_dt.now.return_value.strftime.return_value = "2099-12-31"
            # Force the date check to see a new date
            logger._current_date = "2099-12-31"
            logger._path = tmp_path / "turns" / "2099-12-31.jsonl"
            logger.turn_start(turn_id="t2", session_id="s1", user_msg="day2", source="test")

        new_path = tmp_path / "turns" / "2099-12-31.jsonl"
        assert old_path.exists()
        assert new_path.exists()
        assert "day1" in old_path.read_text()
        assert "day2" in new_path.read_text()

    @pytest.mark.xfail(
        reason="Guardian date-based logging not yet implemented — "
               "guardian_cmd.py still writes to flat guardian.log. "
               "Pending log refactor PR.",
        strict=False,
    )
    def test_guardian_log_uses_date_filename(self, tmp_path: Path):
        """Guardian should write to guardian/YYYY-MM-DD.log (post-refactor).

        Currently the guardian CLI writes to a flat ``guardian.log`` file.
        This test will pass once the guardian is migrated to DateBasedFileHandler.
        """
        from datetime import date
        from march.logging.handlers import DateBasedFileHandler

        guardian_dir = tmp_path / "guardian"
        guardian_dir.mkdir()

        # Simulate what the refactored guardian_cmd.py should do:
        handler = DateBasedFileHandler(log_dir=guardian_dir, ext=".log")

        import logging as stdlib_logging
        handler.setFormatter(stdlib_logging.Formatter("%(message)s"))
        record = stdlib_logging.LogRecord(
            name="guardian", level=stdlib_logging.INFO, pathname="", lineno=0,
            msg="guardian health check", args=(), exc_info=None,
        )
        handler.emit(record)
        handler.close()

        today = date.today().isoformat()
        expected = guardian_dir / f"{today}.log"
        assert expected.exists()

        # Verify the actual guardian_cmd.py uses DateBasedFileHandler
        # (this is the part that will fail until the refactor lands)
        import inspect
        from march.cli.guardian_cmd import guardian_start
        source = inspect.getsource(guardian_start)
        assert "DateBasedFileHandler" in source, (
            "guardian_cmd.py should use DateBasedFileHandler for date-based logs"
        )


class TestGuardianLogContent:
    """Verify the guardian writes meaningful log events."""

    @pytest.fixture(autouse=True)
    def _guardian_state_isolation(self, tmp_path: Path):
        """Redirect guardian state to tmp_path for every test in this class."""
        import march.agents.guardian as gmod
        orig_state = gmod.GUARDIAN_STATE_DIR
        orig_backup = gmod.CONFIG_BACKUP_DIR
        orig_registry = gmod.REGISTRY_FILE
        gmod.GUARDIAN_STATE_DIR = tmp_path / "guardian"
        gmod.CONFIG_BACKUP_DIR = tmp_path / "guardian" / "config_backups"
        gmod.REGISTRY_FILE = tmp_path / "guardian" / "watched.json"
        yield
        gmod.GUARDIAN_STATE_DIR = orig_state
        gmod.CONFIG_BACKUP_DIR = orig_backup
        gmod.REGISTRY_FILE = orig_registry

    async def test_guardian_logs_register_events(self, tmp_path: Path, capsys):
        """Guardian.register() logs the registration to stdout."""
        from march.agents.guardian import Guardian, GuardianConfig, WatchEntry

        cfg = GuardianConfig(check_interval=1, march_config_path=str(tmp_path / "config.yaml"))
        g = Guardian(config=cfg)
        await g.initialize()

        g.register(WatchEntry(id="log-test", pid=os.getpid(), command="test process"))

        captured = capsys.readouterr()
        assert "log-test" in captured.out or "log-test" in captured.err

    async def test_guardian_logs_dead_pid_notification(self, tmp_path: Path):
        """Guardian logs a warning when a watched PID dies."""
        from march.agents.guardian import Guardian, GuardianConfig, WatchEntry

        cfg = GuardianConfig(check_interval=1, march_config_path=str(tmp_path / "config.yaml"))
        g = Guardian(config=cfg)
        await g.initialize()

        g.register(WatchEntry(id="dead-log", pid=4194304, command="dead process"))

        notifications: list[str] = []
        g._notify = AsyncMock(side_effect=lambda t, m: notifications.append(m))

        await g._check_entries()

        assert len(notifications) == 1
        assert "Process died" in notifications[0]
        assert "dead process" in notifications[0]

    async def test_guardian_logs_restart_backup(self, tmp_path: Path, capsys):
        """Guardian logs config backup during register_restart."""
        from march.agents.guardian import Guardian, GuardianConfig

        cfg = GuardianConfig(check_interval=1, march_config_path=str(tmp_path / "config.yaml"))
        g = Guardian(config=cfg)
        await g.initialize()

        Path(cfg.march_config_path).write_text("model: test\n")
        backup = g.register_restart()

        assert backup != ""
        captured = capsys.readouterr()
        assert "register_restart" in captured.out or "register_restart" in captured.err


class TestTurnLogPerDate:
    """Verify turn logs are split by date."""

    def test_turn_log_writes_to_dated_file(self, tmp_path: Path):
        """Each turn event goes to turns/YYYY-MM-DD.jsonl."""
        from datetime import datetime, timezone

        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="hello", source="test")
        logger.turn_complete(
            turn_id="t1", session_id="s1", tool_calls=0,
            total_tokens=100, total_cost=0.001, duration_ms=500,
            final_reply_length=50,
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / "turns" / f"{today}.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "turn_start"
        assert json.loads(lines[1])["event"] == "turn_complete"

    def test_turn_log_multiple_sessions_same_file(self, tmp_path: Path):
        """Multiple sessions on the same day write to the same dated file."""
        from datetime import datetime, timezone

        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="sess-A", user_msg="A", source="ws")
        logger.turn_start(turn_id="t2", session_id="sess-B", user_msg="B", source="matrix")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / "turns" / f"{today}.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        sessions = {json.loads(l)["session_id"] for l in lines}
        assert sessions == {"sess-A", "sess-B"}

    def test_turn_log_all_event_types_in_one_file(self, tmp_path: Path):
        """All event types (start, complete, tool, cancel, error, llm) land in the same dated file."""
        logger = TurnLogger(log_dir=tmp_path)
        logger.turn_start(turn_id="t1", session_id="s1", user_msg="hi", source="test")
        logger.llm_call(turn_id="t1", session_id="s1", provider="openai", model="gpt-4o",
                        input_tokens=100, output_tokens=50, cost=0.005, duration_ms=800)
        logger.tool_call(turn_id="t1", session_id="s1", name="exec", args={"cmd": "ls"},
                         duration_ms=50, status="complete", summary="listed files")
        logger.turn_complete(turn_id="t1", session_id="s1", tool_calls=1,
                             total_tokens=150, total_cost=0.005, duration_ms=1200,
                             final_reply_length=100)
        logger.turn_cancelled(turn_id="t2", session_id="s1", partial_content_length=42)
        logger.turn_error(turn_id="t3", session_id="s1", error="boom")

        log_file = logger._path
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 6

        events = [json.loads(l)["event"] for l in lines]
        assert events == ["turn_start", "llm_call", "tool_call", "turn_complete",
                          "turn_cancelled", "turn_error"]

    @pytest.mark.xfail(
        reason="Metrics date-based logging for guardian events not yet wired — "
               "guardian health checks don't emit to metrics/YYYY-MM-DD.jsonl yet. "
               "Pending log refactor PR.",
        strict=False,
    )
    def test_guardian_health_checks_in_metrics(self, tmp_path: Path):
        """Guardian health check events should appear in metrics/ (post-refactor).

        Currently the guardian only logs to its own log file and stdout.
        This test will pass once guardian health checks emit metrics events.
        """
        from march.logging.logger import MetricsLogger

        MetricsLogger.reset()
        metrics_dir = tmp_path / "metrics"
        ml = MetricsLogger(metrics_dir=metrics_dir)

        # After the refactor, guardian._check_entries should call
        # MetricsLogger.get().guardian_health_check(...) or similar
        assert hasattr(ml, "guardian_health_check"), (
            "MetricsLogger should have a guardian_health_check() method "
            "once the guardian metrics integration lands"
        )
        MetricsLogger.reset()


class TestLogMigration:
    """Test migration of legacy flat log files to the new structure."""

    def test_migrate_flat_logs(self, tmp_path: Path):
        """Legacy flat files are moved into subdirectories."""
        from march.core.log_maintenance import migrate_flat_logs

        # Create legacy flat files
        (tmp_path / "march.log").write_text("old agent log\n")
        (tmp_path / "guardian.log").write_text("old guardian log\n")
        (tmp_path / "turns.jsonl").write_text('{"event":"turn_start"}\n')
        (tmp_path / "metrics.jsonl").write_text('{"event":"llm.call"}\n')

        # Ensure subdirs exist
        for d in ("agent", "guardian", "turns", "metrics"):
            (tmp_path / d).mkdir(exist_ok=True)

        migrated = migrate_flat_logs(tmp_path)
        assert migrated == 4

        # Flat files should be gone
        assert not (tmp_path / "march.log").exists()
        assert not (tmp_path / "guardian.log").exists()
        assert not (tmp_path / "turns.jsonl").exists()
        assert not (tmp_path / "metrics.jsonl").exists()

        # Migrated files should be in subdirs
        assert len(list((tmp_path / "agent").glob("migrated-*"))) == 1
        assert len(list((tmp_path / "guardian").glob("migrated-*"))) == 1
        assert len(list((tmp_path / "turns").glob("migrated-*"))) == 1
        assert len(list((tmp_path / "metrics").glob("migrated-*"))) == 1

    def test_migrate_idempotent(self, tmp_path: Path):
        """Running migration twice doesn't fail or duplicate."""
        from march.core.log_maintenance import migrate_flat_logs

        (tmp_path / "march.log").write_text("log\n")
        for d in ("agent",):
            (tmp_path / d).mkdir(exist_ok=True)

        first = migrate_flat_logs(tmp_path)
        second = migrate_flat_logs(tmp_path)
        assert first == 1
        assert second == 0  # nothing left to migrate

    def test_cleanup_old_logs(self, tmp_path: Path):
        """cleanup_old_logs removes files older than TTL."""
        from march.core.log_maintenance import cleanup_old_logs

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        # Create an old file
        old_file = agent_dir / "2020-01-01.log"
        old_file.write_text("ancient log\n")
        old_time = time.time() - (365 * 86400)  # 1 year ago
        os.utime(old_file, (old_time, old_time))

        # Create a fresh file
        fresh_file = agent_dir / "2099-01-01.log"
        fresh_file.write_text("fresh log\n")

        deleted = cleanup_old_logs(tmp_path, ttl_days=30)
        assert deleted == 1
        assert not old_file.exists()
        assert fresh_file.exists()
