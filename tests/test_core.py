"""Tests for the March core: agent loop, messages, context, sessions, and tools."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from march.core.agent import Agent, AgentResponse
from march.core.context import Context, estimate_tokens, truncate_to_tokens
from march.core.message import Message, Role, ToolCall, ToolResult
from march.core.session import Session, SessionStore
from march.llm.base import LLMProvider, LLMResponse, LLMUsage, StreamChunk
from march.llm.base import ToolCall as LLMToolCall
from march.llm.router import LLMRouter, RouterConfig
from march.memory.store import MemoryStore
from march.plugins.manager import PluginManager
from march.tools.base import tool, Tool, ToolMeta, _extract_schema
from march.tools.registry import ToolRegistry, ToolNotFound


# ─────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────


class TestMessage:
    def test_user_message(self) -> None:
        msg = Message.user("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert not msg.has_tool_calls
        assert not msg.has_tool_results

    def test_assistant_message(self) -> None:
        msg = Message.assistant("Hi there")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there"

    def test_system_message(self) -> None:
        msg = Message.system("You are helpful")
        assert msg.role == Role.SYSTEM

    def test_tool_call_create(self) -> None:
        tc = ToolCall.create("read_file", {"path": "/tmp/test"})
        assert tc.name == "read_file"
        assert tc.args == {"path": "/tmp/test"}
        assert tc.id.startswith("call_")

    def test_tool_call_serialization(self) -> None:
        tc = ToolCall(id="tc_1", name="read_file", args={"path": "/tmp"})
        d = tc.to_dict()
        assert d["id"] == "tc_1"
        assert d["function"]["name"] == "read_file"
        assert d["function"]["arguments"] == {"path": "/tmp"}

        # Roundtrip
        tc2 = ToolCall.from_dict(d)
        assert tc2.id == "tc_1"
        assert tc2.name == "read_file"
        assert tc2.args == {"path": "/tmp"}

    def test_tool_call_from_flat_dict(self) -> None:
        tc = ToolCall.from_dict({"id": "tc_1", "name": "test", "args": {"x": 1}})
        assert tc.name == "test"
        assert tc.args == {"x": 1}

    def test_tool_result(self) -> None:
        tr = ToolResult(id="tc_1", content="file contents", duration_ms=42.0)
        assert not tr.is_error
        assert tr.summary == "file contents"

    def test_tool_result_error(self) -> None:
        tr = ToolResult(id="tc_1", error="File not found")
        assert tr.is_error
        assert "File not found" in tr.summary

    def test_tool_result_serialization(self) -> None:
        tr = ToolResult(id="tc_1", content="result", duration_ms=10.0)
        d = tr.to_dict()
        tr2 = ToolResult.from_dict(d)
        assert tr2.id == "tc_1"
        assert tr2.content == "result"
        assert tr2.duration_ms == 10.0

    def test_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", name="read", args={"path": "/tmp"})
        msg = Message.assistant("Let me read that", tool_calls=[tc])
        assert msg.has_tool_calls
        assert len(msg.tool_calls) == 1

    def test_message_serialization(self) -> None:
        msg = Message.user("Hello")
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.role == Role.USER
        assert msg2.content == "Hello"

    def test_message_to_llm_messages_user(self) -> None:
        msg = Message.user("Hello")
        llm_msgs = msg.to_llm_messages()
        assert len(llm_msgs) == 1
        assert llm_msgs[0]["role"] == "user"
        assert llm_msgs[0]["content"] == "Hello"

    def test_message_to_llm_messages_tool_results(self) -> None:
        results = [
            ToolResult(id="tc_1", content="result1"),
            ToolResult(id="tc_2", content="result2"),
        ]
        msg = Message.tool(results)
        llm_msgs = msg.to_llm_messages()
        assert len(llm_msgs) == 2
        assert llm_msgs[0]["role"] == "tool"
        assert llm_msgs[0]["tool_call_id"] == "tc_1"

    def test_message_to_llm_messages_assistant_with_tools(self) -> None:
        tc = ToolCall(id="tc_1", name="read", args={})
        msg = Message.assistant("Reading...", tool_calls=[tc])
        llm_msgs = msg.to_llm_messages()
        assert len(llm_msgs) == 1
        assert llm_msgs[0]["role"] == "assistant"
        assert len(llm_msgs[0]["tool_calls"]) == 1

    def test_role_from_string(self) -> None:
        msg = Message(role="user", content="test")
        assert msg.role == Role.USER


# ─────────────────────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────────────────────


class TestContext:
    def test_empty_context(self) -> None:
        ctx = Context()
        assert ctx.build_system_prompt() == ""

    def test_basic_prompt(self) -> None:
        ctx = Context(system_rules="Be helpful", agent_profile="Coding agent")
        prompt = ctx.build_system_prompt()
        assert "Be helpful" in prompt
        assert "Coding agent" in prompt

    def test_add_extra_context(self) -> None:
        ctx = Context()
        ctx.add("Extra info")
        prompt = ctx.build_system_prompt()
        assert "Extra info" in prompt

    def test_add_empty_string_ignored(self) -> None:
        ctx = Context()
        ctx.add("")
        ctx.add("   ")
        assert len(ctx.extra_context) == 0

    def test_session_context(self) -> None:
        ctx = Context(session_context={"channel": "terminal"})
        prompt = ctx.build_system_prompt()
        assert "terminal" in prompt

    def test_token_budget_truncation(self) -> None:
        ctx = Context(
            system_rules="A" * 10000,
            daily_memory="B" * 10000,
        )
        prompt = ctx.build_system_prompt(max_tokens=100)
        # Should be truncated
        assert len(prompt) < 10000

    def test_estimated_tokens(self) -> None:
        ctx = Context(system_rules="Hello world")
        assert ctx.estimated_tokens > 0

    def test_estimate_tokens_function(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("Hello world!") == 3  # 12 chars / 4

    def test_truncate_to_tokens(self) -> None:
        text = "A" * 1000
        truncated = truncate_to_tokens(text, 10)
        assert len(truncated) < 1000
        assert "truncated" in truncated


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────


class TestSession:
    def test_session_creation(self) -> None:
        session = Session()
        assert session.id
        assert session.created_at > 0

    def test_add_message(self) -> None:
        session = Session()
        session.add_message(Message.user("Hello"))
        assert len(session.history) == 1
        assert session.history[0].role == Role.USER

    def test_add_exchange(self) -> None:
        session = Session()
        session.add_exchange("Hello", "Hi there")
        assert len(session.history) == 2
        assert session.history[0].role == Role.USER
        assert session.history[1].role == Role.ASSISTANT

    def test_clear(self) -> None:
        session = Session()
        session.add_exchange("Hello", "Hi")
        session.clear()
        assert len(session.history) == 0

    def test_serialization(self) -> None:
        session = Session(source_type="terminal", source_id="test")
        session.add_exchange("Hello", "Hi")
        d = session.to_dict()
        session2 = Session.from_dict(d)
        assert session2.id == session.id
        assert len(session2.history) == 2

    def test_get_messages_for_llm(self) -> None:
        session = Session()
        session.add_exchange("Hello", "Hi")
        msgs = session.get_messages_for_llm()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


# ─────────────────────────────────────────────────────────────
# Session Persistence (SQLite)
# ─────────────────────────────────────────────────────────────


class TestSessionStore:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_sessions.db"

    async def test_save_and_load(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()

        try:
            session = Session(id="test-1", source_type="terminal", source_id="t1")
            session.add_exchange("Hello", "Hi there")
            await store.save_session(session)

            loaded = await store.load_session("test-1")
            assert loaded is not None
            assert loaded.id == "test-1"
            assert len(loaded.history) == 2
            assert loaded.history[0].content == "Hello"
            assert loaded.history[1].content == "Hi there"
        finally:
            await store.close()

    async def test_save_with_tool_calls(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()

        try:
            session = Session(id="test-tools", source_type="terminal", source_id="t1")
            tc = ToolCall(id="tc_1", name="read_file", args={"path": "/tmp"})
            session.add_message(Message.assistant("Reading...", tool_calls=[tc]))
            session.add_message(
                Message.tool([ToolResult(id="tc_1", content="file contents")])
            )
            await store.save_session(session)

            loaded = await store.load_session("test-tools")
            assert loaded is not None
            assert len(loaded.history) == 2
            assert loaded.history[0].has_tool_calls
            assert loaded.history[0].tool_calls[0].name == "read_file"
            assert loaded.history[1].has_tool_results
            assert loaded.history[1].tool_results[0].content == "file contents"
        finally:
            await store.close()

    async def test_find_by_source(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()

        try:
            session = Session(id="test-find", source_type="matrix", source_id="!room:server")
            session.add_exchange("Hello", "Hi")
            await store.save_session(session)

            found = await store.find_by_source("matrix", "!room:server")
            assert found is not None
            assert found.id == "test-find"
        finally:
            await store.close()

    async def test_load_nonexistent(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()
        try:
            loaded = await store.load_session("does-not-exist")
            assert loaded is None
        finally:
            await store.close()

    async def test_list_sessions(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()

        try:
            for i in range(3):
                session = Session(
                    id=f"test-{i}", source_type="terminal", source_id=f"t{i}"
                )
                await store.save_session(session)

            sessions = await store.list_sessions()
            assert len(sessions) == 3
        finally:
            await store.close()

    async def test_delete_session(self, db_path: Path) -> None:
        store = SessionStore(db_path)
        await store.initialize()

        try:
            session = Session(id="to-delete", source_type="terminal", source_id="t1")
            await store.save_session(session)
            assert await store.delete_session("to-delete")
            assert await store.load_session("to-delete") is None
        finally:
            await store.close()


# ─────────────────────────────────────────────────────────────
# Tool System
# ─────────────────────────────────────────────────────────────


class TestToolDecorator:
    def test_basic_decorator(self) -> None:
        @tool(description="Test tool")
        async def my_tool(name: str, count: int = 5) -> str:
            return f"{name}: {count}"

        meta: ToolMeta = my_tool._tool_meta
        assert meta.name == "my_tool"
        assert meta.description == "Test tool"
        assert "name" in meta.parameters["properties"]
        assert "count" in meta.parameters["properties"]
        assert "name" in meta.parameters["required"]
        assert "count" not in meta.parameters["required"]

    def test_decorator_with_custom_name(self) -> None:
        @tool(name="custom_name", description="Custom")
        async def my_func() -> str:
            return "ok"

        assert my_func._tool_meta.name == "custom_name"

    def test_type_extraction(self) -> None:
        @tool(description="Types test")
        async def typed_tool(
            path: str, count: int, flag: bool, ratio: float
        ) -> str:
            return ""

        params = typed_tool._tool_meta.parameters
        assert params["properties"]["path"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["flag"]["type"] == "boolean"
        assert params["properties"]["ratio"]["type"] == "number"

    def test_list_type(self) -> None:
        @tool(description="List test")
        async def list_tool(items: list[str]) -> str:
            return ""

        params = list_tool._tool_meta.parameters
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"

    def test_extract_schema_direct(self) -> None:
        async def my_fn(x: str, y: int = 10) -> str:
            return ""

        schema = _extract_schema(my_fn)
        assert schema["type"] == "object"
        assert "x" in schema["required"]
        assert "y" not in schema["required"]
        assert schema["properties"]["y"]["default"] == 10


class TestToolRegistry:
    async def test_register_and_execute(self) -> None:
        registry = ToolRegistry()

        @tool(description="Echo tool")
        async def echo(text: str) -> str:
            return f"Echo: {text}"

        registry.register_function(echo)
        assert registry.has("echo")
        assert registry.tool_count == 1

        tc = ToolCall(id="tc_1", name="echo", args={"text": "hello"})
        result = await registry.execute(tc)
        assert result.content == "Echo: hello"
        assert result.duration_ms > 0
        assert not result.is_error

    async def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        tc = ToolCall(id="tc_1", name="unknown", args={})
        with pytest.raises(ToolNotFound):
            await registry.execute(tc)

    async def test_execute_tool_error(self) -> None:
        registry = ToolRegistry()

        @tool(description="Failing tool")
        async def fail_tool() -> str:
            raise ValueError("Something went wrong")

        registry.register_function(fail_tool)
        tc = ToolCall(id="tc_1", name="fail_tool", args={})
        result = await registry.execute(tc)
        assert result.is_error
        assert "Something went wrong" in result.error

    def test_definitions_format(self) -> None:
        registry = ToolRegistry()

        @tool(description="Test tool")
        async def test_tool(param: str) -> str:
            return ""

        registry.register_function(test_tool)
        defs = registry.definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "test_tool"
        assert "parameters" in defs[0]["function"]
        assert defs[0]["function"]["parameters"]["type"] == "object"

    def test_unregister(self) -> None:
        registry = ToolRegistry()

        @tool(description="Temp")
        async def temp() -> str:
            return ""

        registry.register_function(temp)
        assert registry.has("temp")
        assert registry.unregister("temp")
        assert not registry.has("temp")

    async def test_execute_batch(self) -> None:
        registry = ToolRegistry()

        @tool(description="Add tool")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        registry.register_function(add)

        calls = [
            ToolCall(id="tc_1", name="add", args={"a": 1, "b": 2}),
            ToolCall(id="tc_2", name="add", args={"a": 10, "b": 20}),
        ]
        results = await registry.execute_batch(calls)
        assert len(results) == 2
        assert results[0].content == "3"
        assert results[1].content == "30"


# ─────────────────────────────────────────────────────────────
# Mock LLM Provider
# ─────────────────────────────────────────────────────────────


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing the agent loop."""

    name = "mock"
    model = "mock-model"

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response
        return LLMResponse(content="Default mock response")

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        response = await self.converse(messages, system, tools)
        # Stream the content word by word
        words = response.content.split()
        for i, word in enumerate(words):
            delta = word if i == 0 else " " + word
            yield StreamChunk(delta=delta)

        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamChunk(
                    tool_call_delta={
                        "index": 0,
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    }
                )

        yield StreamChunk(
            finish_reason="stop",
            usage=response.usage,
        )


# ─────────────────────────────────────────────────────────────
# Agent Loop Tests
# ─────────────────────────────────────────────────────────────


class TestAgentLoop:
    def _make_agent(
        self,
        responses: list[LLMResponse],
        tools: ToolRegistry | None = None,
    ) -> Agent:
        provider = MockLLMProvider(responses)
        config = RouterConfig(fallback_chain=["mock"])
        router = LLMRouter(config=config, providers={"mock": provider})
        return Agent(
            llm_router=router,
            tool_registry=tools or ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )

    async def test_simple_response(self) -> None:
        """Agent loop with no tool calls — single LLM response."""
        agent = self._make_agent([
            LLMResponse(
                content="Hello! How can I help?",
                usage=LLMUsage(input_tokens=10, output_tokens=5),
            )
        ])
        session = Session()
        result = await agent.run("Hi", session)

        assert result.content == "Hello! How can I help?"
        assert result.tool_calls_made == 0
        assert result.total_tokens == 15
        assert len(session.history) == 2  # user + assistant

    async def test_tool_call_execution(self) -> None:
        """Agent loop with tool calls → execution → final response."""
        # First LLM call returns a tool call
        first_response = LLMResponse(
            content="",
            tool_calls=[
                LLMToolCall(id="tc_1", name="echo", args={"text": "hello"}),
            ],
            usage=LLMUsage(input_tokens=20, output_tokens=10),
        )
        # Second LLM call (after tool result) returns the final response
        second_response = LLMResponse(
            content="The tool said: Echo: hello",
            usage=LLMUsage(input_tokens=30, output_tokens=15),
        )

        # Register the tool
        registry = ToolRegistry()

        @tool(description="Echo tool")
        async def echo(text: str) -> str:
            return f"Echo: {text}"

        registry.register_function(echo)

        agent = self._make_agent([first_response, second_response], tools=registry)
        session = Session()
        result = await agent.run("Say hello", session)

        assert result.content == "The tool said: Echo: hello"
        assert result.tool_calls_made == 1
        assert result.total_tokens == 75  # 20+10 + 30+15

    async def test_tool_error_handling(self) -> None:
        """Tool that raises an error gets caught and reported to LLM."""
        first_response = LLMResponse(
            content="",
            tool_calls=[
                LLMToolCall(id="tc_1", name="fail", args={}),
            ],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        second_response = LLMResponse(
            content="Sorry, the tool failed.",
            usage=LLMUsage(input_tokens=15, output_tokens=8),
        )

        registry = ToolRegistry()

        @tool(description="Failing tool")
        async def fail() -> str:
            raise RuntimeError("Boom!")

        registry.register_function(fail)

        agent = self._make_agent([first_response, second_response], tools=registry)
        session = Session()
        result = await agent.run("Do something", session)

        assert result.content == "Sorry, the tool failed."
        assert result.tool_calls_made == 1

    async def test_no_provider_available(self) -> None:
        """Agent handles no LLM provider gracefully."""
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        agent = Agent(
            llm_router=router,
            tool_registry=ToolRegistry(),
            plugin_manager=PluginManager(),
            memory_store=MemoryStore(),
        )
        session = Session()
        result = await agent.run("Hi", session)
        assert "Error" in result.content

    async def test_streaming_simple(self) -> None:
        """Streaming agent loop yields chunks then AgentResponse."""
        agent = self._make_agent([
            LLMResponse(
                content="Hello streaming!",
                usage=LLMUsage(input_tokens=5, output_tokens=3),
            )
        ])
        session = Session()
        chunks: list[Any] = []
        async for item in agent.run_stream("Hi", session):
            chunks.append(item)

        # Should have stream chunks and a final AgentResponse
        assert any(isinstance(c, StreamChunk) for c in chunks)
        assert any(isinstance(c, AgentResponse) for c in chunks)
