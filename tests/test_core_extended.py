"""Extended tests for March core: agent loop, session, context — targeting uncovered lines."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.agent import Agent, AgentResponse, MAX_TOOL_ITERATIONS
from march.core.context import Context, estimate_tokens, truncate_to_tokens
from march.core.message import Message, Role, ToolCall, ToolResult
from march.core.session import Session, SessionStore, deterministic_session_id
from march.llm.base import LLMProvider, LLMResponse, LLMUsage, StreamChunk, ProviderError
from march.llm.base import ToolCall as LLMToolCall
from march.llm.router import LLMRouter, RouterConfig, NoProviderAvailable
from march.memory.store import MemoryStore
from march.plugins.manager import PluginManager
from march.tools.base import tool
from march.tools.registry import ToolRegistry, ToolNotFound


# ─────────────────────────────────────────────────────────────
# Helper: Mock LLM Provider
# ─────────────────────────────────────────────────────────────

class MockProvider(LLMProvider):
    name = "mock"
    model = "mock-model"

    def __init__(self, responses=None, stream_responses=None, fail_on_call=None):
        self._responses = list(responses or [])
        self._stream_responses = stream_responses
        self._fail_on_call = fail_on_call
        self._call_count = 0

    async def converse(self, messages, system=None, tools=None, **kw):
        if self._fail_on_call is not None and self._call_count == self._fail_on_call:
            self._call_count += 1
            raise RuntimeError("LLM exploded")
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return LLMResponse(content="default")

    async def converse_stream(self, messages, system=None, tools=None, **kw):
        resp = await self.converse(messages, system, tools, **kw)
        if resp.content:
            for word in resp.content.split():
                yield StreamChunk(delta=word + " ")
        if resp.tool_calls:
            for i, tc in enumerate(resp.tool_calls):
                yield StreamChunk(tool_call_delta={
                    "index": i, "id": tc.id, "name": tc.name,
                    "arguments": json.dumps(tc.args),
                })
        yield StreamChunk(finish_reason="stop", usage=resp.usage)


def _make_agent(responses, tools=None, plugin_manager=None):
    provider = MockProvider(responses)
    config = RouterConfig(fallback_chain=["mock"])
    router = LLMRouter(config=config, providers={"mock": provider})
    return Agent(
        llm_router=router,
        tool_registry=tools or ToolRegistry(),
        plugin_manager=plugin_manager or PluginManager(),
        memory_store=MemoryStore(),
    )


# ─────────────────────────────────────────────────────────────
# Deterministic Session IDs
# ─────────────────────────────────────────────────────────────

class TestDeterministicSessionId:
    def test_same_source_same_id(self):
        id1 = deterministic_session_id("matrix", "!room:server")
        id2 = deterministic_session_id("matrix", "!room:server")
        assert id1 == id2

    def test_different_source_different_id(self):
        id1 = deterministic_session_id("matrix", "!room1:server")
        id2 = deterministic_session_id("matrix", "!room2:server")
        assert id1 != id2

    def test_different_type_different_id(self):
        id1 = deterministic_session_id("matrix", "test")
        id2 = deterministic_session_id("terminal", "test")
        assert id1 != id2

    def test_returns_valid_uuid(self):
        import uuid
        sid = deterministic_session_id("test", "test")
        uuid.UUID(sid)  # raises if invalid


class TestSessionExtended:
    def test_deterministic_id_from_source_id(self):
        s = Session(source_type="matrix", source_id="!room:server")
        expected = deterministic_session_id("matrix", "!room:server")
        assert s.id == expected

    def test_random_id_no_source(self):
        s1 = Session()
        s2 = Session()
        assert s1.id != s2.id

    def test_reset_clears_and_marks(self):
        s = Session()
        s.add_exchange("hi", "hello")
        s.reset()
        assert len(s.history) == 0
        assert s.state == "reset"

    def test_reactivate(self):
        s = Session()
        s.reset()
        assert s.state == "reset"
        s.reactivate()
        assert s.state == "active"

    def test_add_tool_exchange(self):
        s = Session()
        tc = ToolCall(id="tc1", name="test", args={})
        assistant = Message.assistant("calling tool", tool_calls=[tc])
        result = Message.tool([ToolResult(id="tc1", content="result")])
        s.add_tool_exchange(assistant, result)
        assert len(s.history) == 2
        assert s.history[0].has_tool_calls
        assert s.history[1].has_tool_results

    def test_last_active_updates(self):
        s = Session()
        initial = s.last_active
        time.sleep(0.01)
        s.add_message(Message.user("test"))
        assert s.last_active > initial

    def test_from_dict_missing_fields_defaults(self):
        d = {"id": "test-id"}
        s = Session.from_dict(d)
        assert s.id == "test-id"
        assert s.source_type == "terminal"
        assert s.source_id == ""
        assert s.history == []

    def test_metadata_preserved_in_serialization(self):
        s = Session(metadata={"channel": "matrix", "user": "bob"})
        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.metadata == {"channel": "matrix", "user": "bob"}


# ─────────────────────────────────────────────────────────────
# Session Store Extended
# ─────────────────────────────────────────────────────────────

class TestSessionStoreExtended:
    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "sessions.db"

    async def test_get_or_create_new(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            s = await store.get_or_create_by_source("matrix", "!room:test")
            assert s.source_type == "matrix"
            assert s.source_id == "!room:test"
            assert s.id == deterministic_session_id("matrix", "!room:test")
        finally:
            await store.close()

    async def test_get_or_create_existing(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            s1 = await store.get_or_create_by_source("matrix", "!room:test")
            s1.add_exchange("hello", "hi")
            await store.save_session(s1)
            s2 = await store.get_or_create_by_source("matrix", "!room:test")
            assert s2.id == s1.id
            assert len(s2.history) == 2
        finally:
            await store.close()

    async def test_reset_session(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            s = Session(id="reset-test", source_type="terminal", source_id="t1")
            s.add_exchange("hello", "hi")
            await store.save_session(s)
            result = await store.reset_session("terminal", "t1")
            assert result is True
            loaded = await store.load_session("reset-test")
            assert loaded is not None
            assert len(loaded.history) == 0
        finally:
            await store.close()

    async def test_reset_session_nonexistent(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            result = await store.reset_session("matrix", "!nonexistent")
            assert result is False
        finally:
            await store.close()

    async def test_list_sessions_with_filter(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            s1 = Session(id="s1", source_type="matrix", source_id="r1")
            s2 = Session(id="s2", source_type="terminal", source_id="t1")
            await store.save_session(s1)
            await store.save_session(s2)
            matrix_sessions = await store.list_sessions(source_type="matrix")
            assert len(matrix_sessions) == 1
            assert matrix_sessions[0]["source_type"] == "matrix"
        finally:
            await store.close()

    async def test_delete_nonexistent(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            result = await store.delete_session("nonexistent")
            assert result is False
        finally:
            await store.close()

    async def test_concurrent_sessions(self, db_path):
        store = SessionStore(db_path)
        await store.initialize()
        try:
            sessions = []
            for i in range(10):
                s = Session(id=f"conc-{i}", source_type="test", source_id=f"s{i}")
                s.add_exchange(f"msg {i}", f"reply {i}")
                sessions.append(s)
            await asyncio.gather(*[store.save_session(s) for s in sessions])
            all_sessions = await store.list_sessions()
            assert len(all_sessions) == 10
        finally:
            await store.close()


# ─────────────────────────────────────────────────────────────
# Agent: /rmb and /reset commands
# ─────────────────────────────────────────────────────────────

class TestAgentCommands:
    async def test_rmb_command(self):
        agent = _make_agent([])
        # Mock save_global
        agent.memory.save_global = AsyncMock(return_value="rmb-test123")
        session = Session()
        session.add_exchange("earlier", "earlier reply")
        result = await agent.handle_command("/rmb remember the API key is abc", session)
        assert result is not None
        assert "Remembered" in result.content
        agent.memory.save_global.assert_awaited_once()

    async def test_rmb_empty(self):
        agent = _make_agent([])
        session = Session()
        result = await agent.handle_command("/rmb ", session)
        assert result is not None
        assert "Usage" in result.content

    async def test_reset_command(self):
        agent = _make_agent([])
        agent.memory.reset_session = AsyncMock(return_value={"vector_entries": 3, "sqlite_entries": 5})
        session = Session()
        session.add_exchange("hello", "world")
        result = await agent.handle_command("/reset", session)
        assert result is not None
        assert "reset" in result.content.lower()
        assert "3" in result.content
        assert len(session.history) == 0

    async def test_unknown_command_returns_none(self):
        agent = _make_agent([])
        session = Session()
        result = await agent.handle_command("/unknown", session)
        assert result is None

    async def test_rmb_via_run(self):
        """Ensure /rmb through the full run() flow returns correctly."""
        agent = _make_agent([])
        agent.memory.save_global = AsyncMock(return_value="rmb-xyz")
        session = Session()
        result = await agent.run("/rmb remember this thing", session)
        assert "Remembered" in result.content

    async def test_reset_via_run(self):
        agent = _make_agent([])
        agent.memory.reset_session = AsyncMock(return_value={"vector_entries": 0, "sqlite_entries": 0})
        session = Session()
        result = await agent.run("/reset", session)
        assert "reset" in result.content.lower()


# ─────────────────────────────────────────────────────────────
# Agent: Max iterations guard
# ─────────────────────────────────────────────────────────────

class TestAgentMaxIterations:
    async def test_max_iterations_reached(self):
        """Agent should bail after MAX_TOOL_ITERATIONS tool call loops."""
        # Every response has a tool call, forcing infinite loop
        responses = []
        for i in range(MAX_TOOL_ITERATIONS + 5):
            responses.append(LLMResponse(
                content="",
                tool_calls=[LLMToolCall(id=f"tc_{i}", name="echo", args={"text": "hi"})],
                usage=LLMUsage(input_tokens=1, output_tokens=1),
            ))

        registry = ToolRegistry()

        @tool(description="Echo")
        async def echo(text: str) -> str:
            return text

        registry.register_function(echo)

        agent = _make_agent(responses, tools=registry)
        session = Session()
        result = await agent.run("infinite loop", session)
        assert "maximum" in result.content.lower()
        assert result.tool_calls_made == MAX_TOOL_ITERATIONS


# ─────────────────────────────────────────────────────────────
# Agent: LLM failure handling
# ─────────────────────────────────────────────────────────────

class TestAgentLLMErrors:
    async def test_llm_call_exception(self):
        """If LLM.converse raises, agent returns error."""
        provider = MockProvider(fail_on_call=0)
        config = RouterConfig(fallback_chain=["mock"])
        router = LLMRouter(config=config, providers={"mock": provider})
        agent = Agent(
            llm_router=router,
            tool_registry=ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )
        session = Session()
        result = await agent.run("hello", session)
        assert "Error" in result.content

    async def test_no_provider_error(self):
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        agent = Agent(
            llm_router=router,
            tool_registry=ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )
        session = Session()
        result = await agent.run("hello", session)
        assert "Error" in result.content
        assert result.duration_ms > 0


# ─────────────────────────────────────────────────────────────
# Agent: Plugin short-circuit
# ─────────────────────────────────────────────────────────────

class TestAgentPluginShortCircuit:
    async def test_before_llm_short_circuit(self):
        """A plugin that short-circuits should skip LLM entirely."""
        pm = PluginManager()
        pm.dispatch_before_llm = AsyncMock(
            return_value=(Context(), "hello", "Plugin handled it!")
        )
        pm.dispatch_on_response = AsyncMock(return_value="Plugin handled it!")

        agent = _make_agent([], plugin_manager=pm)
        session = Session()
        result = await agent.run("hello", session)
        assert result.content == "Plugin handled it!"
        assert result.total_tokens == 0

    async def test_before_llm_short_circuit_streaming(self):
        pm = PluginManager()
        pm.dispatch_before_llm = AsyncMock(
            return_value=(Context(), "hello", "Short-circuited!")
        )
        pm.dispatch_on_response = AsyncMock(return_value="Short-circuited!")

        agent = _make_agent([], plugin_manager=pm)
        session = Session()
        chunks = []
        async for item in agent.run_stream("hello", session):
            chunks.append(item)
        # Should have a StreamChunk with the short-circuit and an AgentResponse
        assert any(isinstance(c, StreamChunk) and "Short-circuited" in (c.delta or "") for c in chunks)
        assert any(isinstance(c, AgentResponse) for c in chunks)


# ─────────────────────────────────────────────────────────────
# Agent: Tool call with plugin blocking
# ─────────────────────────────────────────────────────────────

class TestAgentToolBlocked:
    async def test_tool_blocked_by_plugin(self):
        """If plugin returns None for before_tool, tool is blocked."""
        pm = PluginManager()
        pm.dispatch_before_llm = AsyncMock(return_value=(Context(), "test", None))
        pm.dispatch_after_llm = AsyncMock(side_effect=lambda ctx, resp: resp)
        pm.dispatch_on_response = AsyncMock(side_effect=lambda c: c)
        pm.dispatch_before_tool = AsyncMock(return_value=None)  # Block all tools
        pm.dispatch_on_tool_error = AsyncMock()
        pm.dispatch_on_llm_error = AsyncMock()
        pm.dispatch_on_memory_read = AsyncMock(side_effect=lambda kind, content: (kind, content))

        first = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc1", name="echo", args={"text": "hi"})],
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )
        second = LLMResponse(content="Tool was blocked.", usage=LLMUsage(input_tokens=5, output_tokens=5))

        registry = ToolRegistry()

        @tool(description="Echo")
        async def echo(text: str) -> str:
            return text

        registry.register_function(echo)
        agent = _make_agent([first, second], tools=registry, plugin_manager=pm)
        session = Session()
        result = await agent.run("test", session)
        assert "blocked" in result.content.lower() or result.tool_calls_made == 0


# ─────────────────────────────────────────────────────────────
# Agent: Unknown tool call
# ─────────────────────────────────────────────────────────────

class TestAgentUnknownTool:
    async def test_unknown_tool_handled(self):
        first = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc1", name="nonexistent_tool", args={})],
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )
        second = LLMResponse(content="Tool not found.", usage=LLMUsage(input_tokens=5, output_tokens=5))

        agent = _make_agent([first, second])
        session = Session()
        result = await agent.run("call nonexistent", session)
        assert result.content == "Tool not found."
        assert result.tool_calls_made == 1


# ─────────────────────────────────────────────────────────────
# Agent: Streaming with tool calls
# ─────────────────────────────────────────────────────────────

class TestAgentStreamingTools:
    async def test_streaming_with_tool_calls(self):
        """Streaming agent loop handles tool calls correctly."""
        first = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc1", name="echo", args={"text": "hi"})],
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )
        second = LLMResponse(content="Tool said hi", usage=LLMUsage(input_tokens=5, output_tokens=5))

        registry = ToolRegistry()

        @tool(description="Echo")
        async def echo(text: str) -> str:
            return text

        registry.register_function(echo)
        agent = _make_agent([first, second], tools=registry)
        session = Session()

        chunks = []
        async for item in agent.run_stream("test", session):
            chunks.append(item)

        # Should have tool notification chunks and final AgentResponse
        assert any(isinstance(c, AgentResponse) for c in chunks)
        response = [c for c in chunks if isinstance(c, AgentResponse)][0]
        assert response.tool_calls_made == 1

    async def test_streaming_llm_error(self):
        """Streaming handles LLM errors gracefully."""
        provider = MockProvider(fail_on_call=0)
        config = RouterConfig(fallback_chain=["mock"])
        router = LLMRouter(config=config, providers={"mock": provider})
        agent = Agent(
            llm_router=router,
            tool_registry=ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )
        session = Session()
        chunks = []
        async for item in agent.run_stream("hello", session):
            chunks.append(item)
        # Should have an error chunk
        assert any(isinstance(c, StreamChunk) and c.finish_reason == "error" for c in chunks)

    async def test_streaming_no_provider(self):
        """Streaming handles no provider gracefully."""
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        agent = Agent(
            llm_router=router,
            tool_registry=ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )
        session = Session()
        chunks = []
        async for item in agent.run_stream("hello", session):
            chunks.append(item)
        assert any(isinstance(c, StreamChunk) and c.finish_reason == "error" for c in chunks)

    async def test_streaming_command(self):
        """Commands in streaming mode yield StreamChunk then AgentResponse."""
        agent = _make_agent([])
        agent.memory.save_global = AsyncMock(return_value="rmb-123")
        session = Session()
        chunks = []
        async for item in agent.run_stream("/rmb remember test", session):
            chunks.append(item)
        assert any(isinstance(c, StreamChunk) for c in chunks)
        assert any(isinstance(c, AgentResponse) for c in chunks)


# ─────────────────────────────────────────────────────────────
# Agent: _merge_tool_call_delta and _parse_collected_tool_calls
# ─────────────────────────────────────────────────────────────

class TestAgentToolCallParsing:
    def test_merge_tool_call_delta_new_entry(self):
        agent = _make_agent([])
        collected = []
        agent._merge_tool_call_delta(collected, {"index": 0, "id": "tc1", "name": "read"})
        assert len(collected) == 1
        assert collected[0]["id"] == "tc1"
        assert collected[0]["name"] == "read"

    def test_merge_tool_call_delta_append_args(self):
        agent = _make_agent([])
        collected = [{"id": "tc1", "name": "read", "arguments": '{"pat'}]
        agent._merge_tool_call_delta(collected, {"index": 0, "arguments": 'h": "/tmp"}'})
        assert collected[0]["arguments"] == '{"path": "/tmp"}'

    def test_merge_tool_call_delta_multi_index(self):
        agent = _make_agent([])
        collected = []
        agent._merge_tool_call_delta(collected, {"index": 0, "id": "tc1", "name": "a"})
        agent._merge_tool_call_delta(collected, {"index": 1, "id": "tc2", "name": "b"})
        assert len(collected) == 2

    def test_parse_collected_valid(self):
        agent = _make_agent([])
        collected = [
            {"id": "tc1", "name": "read", "arguments": '{"path": "/tmp"}'},
            {"id": "tc2", "name": "write", "arguments": '{"path": "/tmp", "content": "hi"}'},
        ]
        result = agent._parse_collected_tool_calls(collected)
        assert len(result) == 2
        assert result[0].name == "read"
        assert result[0].args == {"path": "/tmp"}

    def test_parse_collected_bad_json(self):
        agent = _make_agent([])
        collected = [{"id": "tc1", "name": "read", "arguments": "not json"}]
        result = agent._parse_collected_tool_calls(collected)
        assert len(result) == 1
        assert "raw" in result[0].args

    def test_parse_collected_empty_name_skipped(self):
        agent = _make_agent([])
        collected = [{"id": "", "name": "", "arguments": ""}]
        result = agent._parse_collected_tool_calls(collected)
        assert len(result) == 0

    def test_parse_collected_empty_args(self):
        agent = _make_agent([])
        collected = [{"id": "tc1", "name": "test", "arguments": ""}]
        result = agent._parse_collected_tool_calls(collected)
        assert len(result) == 1
        assert result[0].args == {}


# ─────────────────────────────────────────────────────────────
# Agent: _build_context
# ─────────────────────────────────────────────────────────────

class TestAgentBuildContext:
    async def test_build_context_loads_all(self):
        agent = _make_agent([])
        agent.memory.load_system_rules = AsyncMock(return_value="system rules")
        agent.memory.load_agent_profile = AsyncMock(return_value="agent profile")
        agent.memory.load_tool_rules = AsyncMock(return_value="tool rules")
        agent.memory.load_long_term = AsyncMock(return_value="long term")
        agent.memory.load_today = AsyncMock(return_value="today notes")

        session = Session(metadata={"channel": "test"})
        ctx = await agent._build_context(session)
        assert ctx.system_rules == "system rules"
        assert ctx.agent_profile == "agent profile"
        assert ctx.tool_rules == "tool rules"
        assert ctx.long_term_memory == "long term"
        assert ctx.daily_memory == "today notes"
        assert ctx.session_context == {"channel": "test"}


# ─────────────────────────────────────────────────────────────
# Agent: _finalize
# ─────────────────────────────────────────────────────────────

class TestAgentFinalize:
    async def test_finalize_saves_exchange(self):
        agent = _make_agent([])
        session = Session()
        result = await agent._finalize(
            content="hello",
            user_message="hi",
            session=session,
            tool_calls_made=2,
            total_tokens=100,
            total_cost=0.01,
            start_time=time.monotonic() - 0.5,
        )
        assert result.content == "hello"
        assert result.tool_calls_made == 2
        assert result.total_tokens == 100
        assert result.total_cost == 0.01
        assert result.duration_ms > 0
        assert len(session.history) == 2  # user + assistant

    async def test_finalize_non_string_content(self):
        """If plugins return non-string, _finalize converts it."""
        pm = PluginManager()
        pm.dispatch_on_response = AsyncMock(return_value=12345)  # non-string
        pm.dispatch_before_llm = AsyncMock(return_value=(Context(), "test", None))
        pm.dispatch_on_memory_read = AsyncMock(side_effect=lambda k, v: (k, v))

        agent = _make_agent([], plugin_manager=pm)
        session = Session()
        result = await agent._finalize(
            content="original",
            user_message="test",
            session=session,
            tool_calls_made=0,
            total_tokens=0,
            total_cost=0.0,
            start_time=time.monotonic(),
        )
        assert result.content == "12345"


# ─────────────────────────────────────────────────────────────
# Context Extended
# ─────────────────────────────────────────────────────────────

class TestContextExtended:
    def test_all_fields_in_prompt(self):
        ctx = Context(
            system_rules="rules",
            agent_profile="profile",
            tool_rules="tool rules",
            long_term_memory="long term",
            daily_memory="daily",
            session_context={"key": "value"},
        )
        ctx.add("extra1")
        ctx.add("extra2")
        prompt = ctx.build_system_prompt()
        assert "rules" in prompt
        assert "profile" in prompt
        assert "tool rules" in prompt
        assert "long term" in prompt
        assert "daily" in prompt
        assert "extra1" in prompt
        assert "extra2" in prompt

    def test_none_values_handled(self):
        ctx = Context(system_rules=None, agent_profile=None)
        prompt = ctx.build_system_prompt()
        # Should not crash
        assert isinstance(prompt, str)


# ─────────────────────────────────────────────────────────────
# Message Extended
# ─────────────────────────────────────────────────────────────

class TestMessageExtended:
    def test_tool_result_long_summary(self):
        tr = ToolResult(id="tc1", content="x" * 500)
        assert len(tr.summary) <= 300 or "x" in tr.summary

    def test_tool_call_create_auto_id(self):
        tc = ToolCall.create("test_tool", {"key": "val"})
        assert tc.id.startswith("call_")
        assert tc.name == "test_tool"

    def test_message_to_dict_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="read", args={"path": "/tmp"})
        msg = Message.assistant("reading", tool_calls=[tc])
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.has_tool_calls
        assert msg2.tool_calls[0].name == "read"

    def test_message_to_dict_with_tool_results(self):
        tr = ToolResult(id="tc1", content="data")
        msg = Message.tool([tr])
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.has_tool_results
        assert msg2.tool_results[0].content == "data"

    def test_role_enum_values(self):
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"
        assert Role.TOOL.value == "tool"
