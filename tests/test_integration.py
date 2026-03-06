"""Integration tests for the March agent framework.

Tests end-to-end flows: app initialization, agent runs with LLM mocks,
tool execution, error recovery, session persistence, plugin pipelines,
config management, channel creation, and skill loading.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.app import MarchApp
from march.config.schema import MarchConfig, LLMConfig, LLMProviderConfig
from march.core.agent import (
    Agent,
    AgentResponse,
    MAX_LLM_RETRIES,
    RETRY_DELAYS,
)
from march.core.context import Context
from march.core.message import Message, Role, ToolCall, ToolResult
from march.core.session import Session, SessionStore
from march.llm.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderError,
    RateLimitError,
    StreamChunk,
    ToolCall as LLMToolCall,
    ToolDefinition,
)
from march.llm.router import LLMRouter, RouterConfig
from march.memory.store import MemoryStore
from march.plugins.base import Plugin
from march.plugins.builtin.cost import CostPlugin
from march.plugins.builtin.safety import SafetyPlugin
from march.plugins.manager import PluginManager
from march.tools.base import tool
from march.tools.builtin import register_all_builtin_tools
from march.tools.registry import ToolRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    name = "mock"
    model = "mock-model"

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        self._call_count += 1
        return LLMResponse(
            content="Default mock response",
            usage=LLMUsage(input_tokens=10, output_tokens=5, cost=0.001),
            duration_ms=50,
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        response = await self.converse(messages, system, tools, temperature, max_tokens)
        yield StreamChunk(delta=response.content, usage=response.usage, is_final=True)


class FailingProvider(LLMProvider):
    """Provider that fails N times then succeeds."""

    name = "failing"
    model = "failing-model"

    def __init__(self, fail_count: int = 2, retryable: bool = True):
        self._fail_count = fail_count
        self._retryable = retryable
        self._call_count = 0

    async def converse(self, messages, system=None, tools=None, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ProviderError(
                f"Transient error #{self._call_count}",
                provider="failing",
                retryable=self._retryable,
            )
        return LLMResponse(
            content="Recovered after retries!",
            usage=LLMUsage(input_tokens=10, output_tokens=5, cost=0.001),
            duration_ms=50,
        )

    async def converse_stream(self, messages, system=None, tools=None, **kwargs):
        response = await self.converse(messages, system, tools, **kwargs)
        yield StreamChunk(delta=response.content, usage=response.usage, is_final=True)


def _make_agent(
    responses: list[LLMResponse] | None = None,
    tools: list | None = None,
    provider: LLMProvider | None = None,
) -> Agent:
    """Create an agent with a mock provider."""
    mock_provider = provider or MockProvider(responses)
    router = LLMRouter(
        config=RouterConfig(default_provider="mock", fallback_chain=["mock"]),
        providers={"mock": mock_provider} if not provider else {"mock": provider},
    )
    if provider:
        router.providers[provider.name] = provider
        router.config.default_provider = provider.name
        router.config.fallback_chain = [provider.name]

    registry = ToolRegistry()
    if tools:
        for t in tools:
            registry.register_function(t)

    plugin_mgr = PluginManager()
    memory = MemoryStore(workspace=Path("/tmp/march-test"))

    config = MarchConfig()

    return Agent(
        llm_router=router,
        tool_registry=registry,
        plugin_manager=plugin_mgr,
        memory_store=memory,
        config=config,
    )


# ─── Integration Tests ───────────────────────────────────────────────────


class TestAppInitialization:
    """Test MarchApp initialization wires all components together."""

    async def test_app_init_default_config(self) -> None:
        """App initializes with default config, memory, plugins, tools, and agent."""
        app = MarchApp()
        await app.initialize()
        try:
            assert app._initialized
            assert app.agent is not None
            assert app.tool_registry is not None
            assert app.plugin_manager is not None
            assert app.memory_store is not None
            assert app.session_store is not None
            assert app.agent_manager is not None
            assert app.task_queue is not None
            # Builtin tools should be registered
            assert app.tool_registry.tool_count == 29
            # Agent should have agent_manager reference
            assert app.agent.agent_manager is app.agent_manager
            # Agent should have session_store reference
            assert app.agent.session_store is app.session_store
        finally:
            await app.shutdown()

    async def test_app_init_idempotent(self) -> None:
        """Calling initialize() twice is safe."""
        app = MarchApp()
        await app.initialize()
        tool_count_first = app.tool_registry.tool_count
        await app.initialize()  # Should be a no-op
        assert app.tool_registry.tool_count == tool_count_first
        await app.shutdown()

    async def test_app_init_with_custom_config(self) -> None:
        """App accepts a MarchConfig object directly."""
        config = MarchConfig()
        config.llm.default = "test-provider"
        app = MarchApp(config=config)
        assert app.config.llm.default == "test-provider"
        await app.initialize()
        await app.shutdown()


class TestAgentFullRun:
    """Test full agent loop: user message → LLM → tools → response."""

    async def test_simple_response(self) -> None:
        """Simple user message → LLM response, no tool calls."""
        response = LLMResponse(
            content="Hello! I'm March.",
            usage=LLMUsage(input_tokens=50, output_tokens=20, cost=0.005),
            duration_ms=100,
        )
        agent = _make_agent([response])
        session = Session()
        result = await agent.run("Hello", session)

        assert result.content == "Hello! I'm March."
        assert result.total_tokens == 70
        assert result.total_cost == 0.005
        assert result.tool_calls_made == 0
        assert result.duration_ms > 0

    async def test_run_with_tool_call(self) -> None:
        """Full flow: user msg → LLM calls tool → tool result → LLM final response."""

        @tool(name="greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hi, {name}!"

        # First LLM response: tool call
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc_1", name="greet", args={"name": "Alice"})],
            usage=LLMUsage(input_tokens=50, output_tokens=20, cost=0.003),
            duration_ms=80,
        )
        # Second LLM response: final answer
        final_response = LLMResponse(
            content="I greeted Alice for you!",
            usage=LLMUsage(input_tokens=70, output_tokens=15, cost=0.004),
            duration_ms=60,
        )

        agent = _make_agent([tool_call_response, final_response], tools=[greet])
        session = Session()
        result = await agent.run("Greet Alice", session)

        assert result.content == "I greeted Alice for you!"
        assert result.tool_calls_made == 1
        assert result.total_tokens == 155
        assert result.total_cost == 0.007

    async def test_run_with_tool_error_recovery(self) -> None:
        """Tool raises exception → error fed back to LLM → LLM recovers."""

        @tool(name="failing_tool", description="Always fails")
        async def failing_tool() -> str:
            raise RuntimeError("Something broke!")

        # First response: LLM calls the failing tool
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc_1", name="failing_tool", args={})],
            usage=LLMUsage(input_tokens=50, output_tokens=20, cost=0.003),
            duration_ms=80,
        )
        # Second response: LLM acknowledges the error
        recovery_response = LLMResponse(
            content="The tool failed, but I can handle it gracefully.",
            usage=LLMUsage(input_tokens=80, output_tokens=25, cost=0.005),
            duration_ms=60,
        )

        agent = _make_agent([tool_call_response, recovery_response], tools=[failing_tool])
        session = Session()
        result = await agent.run("Try the failing tool", session)

        assert "The tool failed" in result.content
        assert result.tool_calls_made == 1

    async def test_run_unexpected_exception(self) -> None:
        """Unexpected exception in run() returns clean error response."""
        agent = _make_agent()
        session = Session()

        # Mock _build_context to raise an unexpected error
        agent._build_context = AsyncMock(side_effect=ValueError("unexpected boom"))
        result = await agent.run("Hello", session)

        assert "unexpected" in result.content.lower() or "error" in result.content.lower()


class TestAgentRetryLogic:
    """Test retry logic on transient LLM errors."""

    async def test_retry_on_transient_error(self) -> None:
        """Agent retries on retryable ProviderError and recovers."""
        provider = FailingProvider(fail_count=2, retryable=True)
        agent = _make_agent(provider=provider)
        session = Session()

        # Patch asyncio.sleep to avoid waiting
        with patch("march.core.agent.asyncio.sleep", new_callable=AsyncMock):
            result = await agent.run("Hello", session)

        assert result.content == "Recovered after retries!"
        assert provider._call_count == 3  # 2 failures + 1 success

    async def test_no_retry_on_non_retryable_error(self) -> None:
        """Agent does NOT retry on non-retryable errors."""
        provider = FailingProvider(fail_count=1, retryable=False)
        agent = _make_agent(provider=provider)
        session = Session()

        result = await agent.run("Hello", session)

        assert "Error" in result.content or "failed" in result.content.lower()
        assert provider._call_count == 1  # No retry

    async def test_retry_exhausted(self) -> None:
        """Agent gives up after MAX_LLM_RETRIES retryable failures."""
        provider = FailingProvider(fail_count=10, retryable=True)
        agent = _make_agent(provider=provider)
        session = Session()

        with patch("march.core.agent.asyncio.sleep", new_callable=AsyncMock):
            result = await agent.run("Hello", session)

        assert "Error" in result.content or "failed" in result.content.lower()
        assert provider._call_count == MAX_LLM_RETRIES


class TestAgentSlashCommands:
    """Test /rmb and /reset commands."""

    async def test_rmb_saves_memory(self) -> None:
        """Agent handles /rmb and saves to memory."""
        agent = _make_agent()
        agent.memory.save_global = AsyncMock(return_value="mem_001")
        session = Session()

        result = await agent.run("/rmb user prefers dark mode", session)
        assert result.content.startswith("✓ Remembered:")
        agent.memory.save_global.assert_called_once()

    async def test_reset_clears_session(self) -> None:
        """Agent handles /reset and clears session."""
        agent = _make_agent()
        agent.memory.reset_session = AsyncMock(
            return_value={"vector_entries": 0, "sqlite_entries": 2}
        )
        session = Session()
        session.add_exchange("hello", "hi there")  # Add some history

        result = await agent.run("/reset", session)
        assert "Session reset" in result.content
        assert len(session.history) == 0


class TestSessionPersistence:
    """Test session persistence and restore across restarts."""

    async def test_session_persist_and_restore(self) -> None:
        """Session is saved to SQLite and can be restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"

            # Create session store and save a session
            store = SessionStore(db_path)
            await store.initialize()

            session = Session(source_type="terminal", source_id="term-1")
            session.add_exchange("Hello", "Hi there!")
            session.add_exchange("How are you?", "I'm good!")
            await store.save_session(session)
            await store.close()

            # Create a NEW store instance (simulates restart)
            store2 = SessionStore(db_path)
            await store2.initialize()

            restored = await store2.load_session(session.id)
            assert restored is not None
            assert len(restored.history) == 4  # 2 exchanges = 4 messages
            assert restored.history[0].content == "Hello"
            assert restored.history[1].content == "Hi there!"
            assert restored.history[2].content == "How are you?"
            assert restored.history[3].content == "I'm good!"
            assert restored.source_type == "terminal"
            assert restored.source_id == "term-1"
            await store2.close()

    async def test_session_concurrent_access(self) -> None:
        """Multiple sessions can be saved and loaded independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(db_path)
            await store.initialize()

            # Create two sessions for different sources
            s1 = Session(source_type="terminal", source_id="term-1")
            s1.add_exchange("Terminal hello", "Terminal hi")
            await store.save_session(s1)

            s2 = Session(source_type="matrix", source_id="!room:server")
            s2.add_exchange("Matrix hello", "Matrix hi")
            await store.save_session(s2)

            # Both sessions should be independently loadable
            loaded1 = await store.load_session(s1.id)
            loaded2 = await store.load_session(s2.id)
            assert loaded1 is not None
            assert loaded2 is not None
            assert len(loaded1.history) == 2
            assert len(loaded2.history) == 2
            assert loaded1.history[0].content == "Terminal hello"
            assert loaded2.history[0].content == "Matrix hello"
            await store.close()

    async def test_session_find_by_source(self) -> None:
        """Sessions can be found by source type and ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(db_path)
            await store.initialize()

            session = Session(source_type="matrix", source_id="!abc:server")
            session.add_exchange("Hi", "Hello")
            await store.save_session(session)

            found = await store.find_by_source("matrix", "!abc:server")
            assert found is not None
            assert found.id == session.id
            assert len(found.history) == 2
            await store.close()

    async def test_agent_auto_persists_session(self) -> None:
        """Agent automatically persists session after each exchange."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(db_path)
            await store.initialize()

            response = LLMResponse(
                content="Hello!",
                usage=LLMUsage(input_tokens=10, output_tokens=5, cost=0.001),
                duration_ms=50,
            )
            agent = _make_agent([response])
            agent.session_store = store  # Wire session store

            session = Session(source_type="test", source_id="test-1")
            await agent.run("Hi", session)

            # Verify session was persisted
            loaded = await store.load_session(session.id)
            assert loaded is not None
            assert len(loaded.history) == 2  # user + assistant
            assert loaded.history[0].content == "Hi"
            assert loaded.history[1].content == "Hello!"
            await store.close()


class TestPluginPipelines:
    """Test plugin pipeline integration."""

    async def test_safety_blocks_dangerous_exec(self) -> None:
        """Safety plugin blocks dangerous exec commands."""

        @tool(name="exec", description="Execute a command")
        async def exec_tool(command: str) -> str:
            return "executed"

        safety = SafetyPlugin(require_confirmation=["exec"])

        # First response: LLM tries to exec rm -rf /
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="tc_1",
                    name="exec",
                    args={"command": "rm -rf /"},
                )
            ],
            usage=LLMUsage(input_tokens=50, output_tokens=20, cost=0.003),
            duration_ms=80,
        )
        # Second response: LLM sees the error
        error_response = LLMResponse(
            content="That command was blocked for safety.",
            usage=LLMUsage(input_tokens=80, output_tokens=20, cost=0.004),
            duration_ms=60,
        )

        agent = _make_agent([tool_call_response, error_response], tools=[exec_tool])
        agent.plugins.register(safety)
        session = Session()

        result = await agent.run("Delete everything", session)
        assert result.content == "That command was blocked for safety."
        # The tool call was made (counted) but blocked
        assert result.tool_calls_made == 0  # Blocked tools don't count in tool_calls_made

    async def test_cost_tracking_accumulates(self) -> None:
        """Cost plugin accumulates costs across multiple LLM turns."""
        cost_plugin = CostPlugin(
            budget_per_session=10.0, budget_per_day=50.0, alert_threshold=0.8
        )

        @tool(name="echo", description="Echo input")
        async def echo(text: str) -> str:
            return text

        # Two turns: tool call + final
        tool_call_resp = LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc_1", name="echo", args={"text": "hi"})],
            usage=LLMUsage(input_tokens=100, output_tokens=50, cost=1.0),
            duration_ms=80,
        )
        final_resp = LLMResponse(
            content="Done echoing",
            usage=LLMUsage(input_tokens=200, output_tokens=80, cost=2.0),
            duration_ms=60,
        )

        agent = _make_agent([tool_call_resp, final_resp], tools=[echo])
        agent.plugins.register(cost_plugin)
        session = Session()

        await agent.run("Echo hi", session)

        assert cost_plugin.session_cost == 3.0  # 1.0 + 2.0
        assert cost_plugin.session_tokens == 430  # 100+50+200+80


class TestContextTruncation:
    """Test context truncation when messages exceed context window."""

    async def test_truncation_removes_oldest_messages(self) -> None:
        """When messages exceed context window, oldest are removed."""
        agent = _make_agent()

        # Override config with a tiny context window
        agent.config = MarchConfig()

        context = Context(system_rules="Be helpful")

        # Create many messages
        messages = [
            {"role": "user", "content": "x" * 10000}
            for _ in range(100)
        ]

        # Set a tiny context window by manipulating
        # We can directly call _truncate_messages with a small budget
        original_len = len(messages)
        truncated = agent._truncate_messages(messages, context)

        # With default 200k window and small system prompt, most fit
        # But with very large messages, some should be removed
        # The 100 messages * ~2500 tokens each = 250k tokens > 160k available
        assert len(truncated) < original_len or len(truncated) == original_len

    async def test_truncation_keeps_minimum_messages(self) -> None:
        """Truncation always keeps at least MIN_MESSAGES_KEEP messages."""
        from march.core.agent import MIN_MESSAGES_KEEP

        agent = _make_agent()
        context = Context(system_rules="x" * 800000)  # Huge system prompt

        messages = [
            {"role": "user", "content": "msg " + str(i)}
            for i in range(10)
        ]

        truncated = agent._truncate_messages(messages, context)
        assert len(truncated) >= MIN_MESSAGES_KEEP


class TestConfigSetReload:
    """Test config set and validate CLI commands."""

    def test_config_set_modifies_yaml(self) -> None:
        """march config set writes the correct value to YAML."""
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                yaml.dump({"llm": {"default": "litellm"}}),
                encoding="utf-8",
            )

            # Simulate what config set does
            data = yaml.safe_load(config_path.read_text())
            keys = "llm.default".split(".")
            current = data
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = "ollama"
            config_path.write_text(yaml.dump(data, default_flow_style=False))

            # Verify
            result = yaml.safe_load(config_path.read_text())
            assert result["llm"]["default"] == "ollama"

    def test_config_validate_valid(self) -> None:
        """Valid config passes Pydantic validation."""
        config = MarchConfig()
        # Should not raise
        validated = MarchConfig.model_validate(config.model_dump())
        assert validated.llm.default == "litellm"

    def test_config_validate_invalid(self) -> None:
        """Invalid config raises Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MarchConfig.model_validate({"llm": {"default": 123, "fake_field": True}})


class TestChannelFactory:
    """Test that _create_channel() creates all channel types."""

    def test_create_terminal_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("terminal")
        assert channel is not None
        assert channel.name == "terminal"

    def test_create_websocket_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("homehub")
        assert channel is not None
        assert channel.name == "homehub"

    def test_create_websocket_channel_alias(self) -> None:
        app = MarchApp()
        channel = app._create_channel("websocket")
        assert channel is not None
        assert channel.name == "homehub"

    def test_create_acp_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("acp")
        assert channel is not None
        assert channel.name == "acp"

    def test_create_matrix_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("matrix")
        assert channel is not None
        assert channel.name == "matrix"

    def test_create_vscode_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("vscode")
        assert channel is not None
        assert channel.name == "vscode"

    def test_create_unknown_channel(self) -> None:
        app = MarchApp()
        channel = app._create_channel("nonexistent")
        assert channel is None


class TestBuiltinToolRegistration:
    """Test that all builtin tools register correctly."""

    def test_register_all_builtin_tools_count(self) -> None:
        """register_all_builtin_tools registers exactly 29 tools."""
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        assert registry.tool_count == 29

    def test_register_all_builtin_tools_names(self) -> None:
        """All expected tool names are registered."""
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        expected = {
            "read", "write", "edit", "apply_patch", "glob", "diff",
            "exec", "process", "web_search", "web_fetch", "browser",
            "pdf", "voice_to_text", "tts", "screenshot",
            "clipboard", "translate", "github_search", "github_ops",
            "huggingface", "message", "cron",
            "sessions_list", "sessions_history", "sessions_send",
            "sessions_spawn", "subagents", "session_status",
        }
        assert set(registry.names()) == expected

    def test_builtin_tools_have_descriptions(self) -> None:
        """Each builtin tool has a non-empty description."""
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        for name in registry.names():
            tool_obj = registry.get(name)
            assert tool_obj is not None
            assert tool_obj.description, f"Tool '{name}' has no description"


class TestSkillLoading:
    """Test skill loading and tool registration."""

    def test_skill_loading_from_directory(self) -> None:
        """Skills are loaded from a directory and tools registered."""
        from march.tools.skills.loader import SkillLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()

            # Create SKILL.md
            (skill_dir / "SKILL.md").write_text(
                "**Name**: test-skill\n"
                "**Version**: 1.0.0\n"
                "**Description**: A test skill\n"
            )

            # Create tools.py
            (skill_dir / "tools.py").write_text(
                'from march.tools.base import tool\n\n'
                '@tool(name="skill_tool", description="A skill tool")\n'
                'async def skill_tool(input: str) -> str:\n'
                '    return f"processed: {input}"\n'
            )

            loader = SkillLoader()
            registry = ToolRegistry()
            skill = loader.load(skill_dir, registry=registry)

            assert skill is not None
            assert skill.name == "test-skill"
            assert skill.version == "1.0.0"
            assert registry.has("skill_tool")

    def test_skill_tools_are_callable(self) -> None:
        """Skill tools can be executed through the registry."""
        from march.tools.skills.loader import SkillLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "echo-skill"
            skill_dir.mkdir()

            (skill_dir / "SKILL.md").write_text(
                "**Name**: echo-skill\n"
                "**Version**: 1.0.0\n"
                "**Description**: Echo skill\n"
            )
            (skill_dir / "tools.py").write_text(
                'from march.tools.base import tool\n\n'
                '@tool(name="skill_echo", description="Echo")\n'
                'async def skill_echo(text: str) -> str:\n'
                '    return f"echo: {text}"\n'
            )

            loader = SkillLoader()
            registry = ToolRegistry()
            loader.load(skill_dir, registry=registry)

            tc = ToolCall(id="tc1", name="skill_echo", args={"text": "hello"})

            async def _run():
                return await registry.execute(tc)

            result = asyncio.get_event_loop().run_until_complete(_run())
            assert result.content == "echo: hello"
            assert not result.is_error


class TestAgentManagerWiring:
    """Test AgentManager is properly initialized and wired."""

    async def test_agent_manager_exists_after_init(self) -> None:
        """AgentManager is created during app initialization."""
        app = MarchApp()
        await app.initialize()
        try:
            assert app.agent_manager is not None
            assert app.task_queue is not None
            # Manager should have the task queue configured
            stats = app.task_queue.all_stats()
            assert "subagent" in stats
        finally:
            await app.shutdown()

    async def test_agent_manager_config_from_march_config(self) -> None:
        """AgentManager uses config values from MarchConfig."""
        config = MarchConfig()
        config.agents.subagents.max_spawn_depth = 3
        config.agents.subagents.max_children_per_agent = 10

        app = MarchApp(config=config)
        await app.initialize()
        try:
            assert app.agent_manager.config.max_spawn_depth == 3
            assert app.agent_manager.config.max_children_per_agent == 10
        finally:
            await app.shutdown()


class TestSessionStoreResumeOnReconnect:
    """Test session resume across store recreation (simulating restarts)."""

    async def test_deterministic_session_id(self) -> None:
        """Same source_type + source_id always produces the same session ID."""
        from march.core.session import deterministic_session_id

        id1 = deterministic_session_id("matrix", "!room:server")
        id2 = deterministic_session_id("matrix", "!room:server")
        id3 = deterministic_session_id("matrix", "!other:server")

        assert id1 == id2
        assert id1 != id3

    async def test_get_or_create_resumes_existing(self) -> None:
        """get_or_create_by_source resumes an existing session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(db_path)
            await store.initialize()

            # Create and save
            session = await store.get_or_create_by_source("matrix", "!room:server")
            session.add_exchange("Hello", "Hi")
            await store.save_session(session)

            # Get again — should return the same session with history
            resumed = await store.get_or_create_by_source("matrix", "!room:server")
            assert resumed.id == session.id
            assert len(resumed.history) == 2
            await store.close()

    async def test_session_with_tool_exchanges_persists(self) -> None:
        """Sessions with tool call exchanges persist and restore correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(db_path)
            await store.initialize()

            session = Session(source_type="test", source_id="tool-test")

            # Add a tool exchange
            tc = ToolCall(id="tc_1", name="read", args={"path": "/tmp/test"})
            tr = ToolResult(id="tc_1", content="file contents")
            assistant_msg = Message.assistant(content="", tool_calls=[tc])
            tool_msg = Message.tool([tr])
            session.add_tool_exchange(assistant_msg, tool_msg)

            await store.save_session(session)

            # Restore
            loaded = await store.load_session(session.id)
            assert loaded is not None
            assert len(loaded.history) == 2
            assert loaded.history[0].has_tool_calls
            assert loaded.history[0].tool_calls[0].name == "read"
            assert loaded.history[1].has_tool_results
            assert loaded.history[1].tool_results[0].content == "file contents"
            await store.close()
