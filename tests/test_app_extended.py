"""Extended tests for MarchApp lifecycle, decorators, channel creation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.app import MarchApp
from march.config.schema import MarchConfig
from march.core.session import Session
from march.llm.base import LLMProvider, LLMResponse, LLMUsage
from march.plugins.base import Plugin
from march.tools.base import tool


# ─────────────────────────────────────────────────────────────
# App Construction
# ─────────────────────────────────────────────────────────────

class TestMarchAppInit:
    def test_init_default_config(self):
        app = MarchApp()
        assert app.config is not None
        assert isinstance(app.config, MarchConfig)
        assert not app._initialized

    def test_init_with_config_object(self):
        config = MarchConfig()
        app = MarchApp(config=config)
        assert app.config is config

    def test_init_components_created(self):
        app = MarchApp()
        assert app.llm_router is not None
        assert app.tool_registry is not None
        assert app.plugin_manager is not None
        assert app.memory_store is not None
        assert app.agent is not None
        assert app.session_store is None  # Not initialized until .initialize()

    def test_init_with_config_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
llm:
  default: mock
memory:
  system_rules: SYSTEM.md
""")
        app = MarchApp(config=str(config_file))
        assert app.config is not None


# ─────────────────────────────────────────────────────────────
# App Initialize / Shutdown
# ─────────────────────────────────────────────────────────────

class TestMarchAppLifecycle:
    async def test_initialize(self, tmp_path):
        """App initializes all components."""
        config = MarchConfig()
        app = MarchApp(config=config)
        app.memory_store = MagicMock()
        app.memory_store.initialize = AsyncMock()
        app.memory_store.close = AsyncMock()
        app.plugin_manager = MagicMock()
        app.plugin_manager.dispatch_simple = AsyncMock()
        app.plugin_manager.load_directory = MagicMock()

        # Run in tmp dir to avoid filesystem side effects
        with patch("march.app.Path.cwd", return_value=tmp_path):
            await app.initialize()

        assert app._initialized
        app.memory_store.initialize.assert_awaited_once()

    async def test_initialize_idempotent(self, tmp_path):
        config = MarchConfig()
        app = MarchApp(config=config)
        app.memory_store = MagicMock()
        app.memory_store.initialize = AsyncMock()
        app.memory_store.close = AsyncMock()
        app.plugin_manager = MagicMock()
        app.plugin_manager.dispatch_simple = AsyncMock()

        with patch("march.app.Path.cwd", return_value=tmp_path):
            await app.initialize()
            await app.initialize()  # Second call should be no-op

        app.memory_store.initialize.assert_awaited_once()

    async def test_shutdown(self, tmp_path):
        config = MarchConfig()
        app = MarchApp(config=config)
        app.memory_store = MagicMock()
        app.memory_store.initialize = AsyncMock()
        app.memory_store.close = AsyncMock()
        app.plugin_manager = MagicMock()
        app.plugin_manager.dispatch_simple = AsyncMock()

        with patch("march.app.Path.cwd", return_value=tmp_path):
            await app.initialize()
            await app.shutdown()

        assert not app._initialized
        app.memory_store.close.assert_awaited_once()
        assert app.plugin_manager.dispatch_simple.await_count >= 2  # on_start + on_shutdown


# ─────────────────────────────────────────────────────────────
# App Decorators
# ─────────────────────────────────────────────────────────────

class TestMarchAppDecorators:
    def test_tool_decorator(self):
        app = MarchApp()

        @app.tool(name="my_echo", description="Echo back")
        async def my_echo(text: str) -> str:
            return f"Echo: {text}"

        assert app.tool_registry.has("my_echo")
        defs = app.tool_registry.definitions()
        names = [d["function"]["name"] for d in defs]
        assert "my_echo" in names

    def test_tool_decorator_auto_name(self):
        app = MarchApp()

        @app.tool(description="Auto-named tool")
        async def auto_named(x: int) -> str:
            return str(x)

        assert app.tool_registry.has("auto_named")

    def test_plugin_decorator(self):
        app = MarchApp()

        @app.plugin(name="my-plugin", priority=5)
        class MyPlugin(Plugin):
            async def before_llm(self, context, message):
                return context, message

        assert len(app.plugin_manager._plugins) >= 1
        # Check it was registered with correct name
        registered = [p for p in app.plugin_manager._plugins if p.name == "my-plugin"]
        assert len(registered) == 1
        assert registered[0].priority == 5


# ─────────────────────────────────────────────────────────────
# App Registration Methods
# ─────────────────────────────────────────────────────────────

class TestMarchAppRegistration:
    def test_register_tool(self):
        app = MarchApp()

        @tool(description="Direct register")
        async def direct_tool(x: str) -> str:
            return x

        app.register_tool(direct_tool)
        assert app.tool_registry.has("direct_tool")

    def test_register_plugin(self):
        app = MarchApp()

        class TestPlugin(Plugin):
            name = "test-direct"
            priority = 50

        plugin = TestPlugin()
        app.register_plugin(plugin)
        registered = [p for p in app.plugin_manager._plugins if p.name == "test-direct"]
        assert len(registered) == 1

    def test_register_provider(self):
        app = MarchApp()

        class MockProvider(LLMProvider):
            name = "mock_provider"
            model = "mock"
            async def converse(self, messages, **kw):
                return LLMResponse(content="mock")
            async def converse_stream(self, messages, **kw):
                yield

        provider = MockProvider()
        app.register_provider("mock_provider", provider)
        assert "mock_provider" in app.llm_router.providers
        assert "mock_provider" in app.llm_router.config.fallback_chain

    def test_register_provider_no_duplicate_chain(self):
        app = MarchApp()

        class MockProvider(LLMProvider):
            name = "mock"
            model = "mock"
            async def converse(self, messages, **kw):
                return LLMResponse(content="mock")
            async def converse_stream(self, messages, **kw):
                yield

        # Pre-populate the chain
        app.llm_router.config.fallback_chain.append("mock")
        app.register_provider("mock", MockProvider())
        # Should not duplicate
        assert app.llm_router.config.fallback_chain.count("mock") == 1


# ─────────────────────────────────────────────────────────────
# App Skill Loading
# ─────────────────────────────────────────────────────────────

class TestMarchAppSkills:
    def test_load_skill_nonexistent(self, tmp_path):
        app = MarchApp()
        result = app.load_skill(tmp_path / "nonexistent")
        assert result is None

    def test_load_skill_invalid(self, tmp_path):
        skill_dir = tmp_path / "bad_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.yaml").write_text("invalid: yaml: content")
        app = MarchApp()
        # Should not crash, returns None
        result = app.load_skill(skill_dir)
        # Result depends on skill loader behavior, but should not crash


# ─────────────────────────────────────────────────────────────
# App Channel Creation
# ─────────────────────────────────────────────────────────────

class TestMarchAppChannels:
    def test_create_terminal_channel(self):
        app = MarchApp()
        channel = app._create_channel("terminal")
        assert channel is not None
        assert channel.name == "terminal"

    def test_create_unknown_channel(self):
        app = MarchApp()
        channel = app._create_channel("nonexistent")
        assert channel is None


# ─────────────────────────────────────────────────────────────
# App Built-in Plugin Loading
# ─────────────────────────────────────────────────────────────

class TestMarchAppBuiltinPlugins:
    def test_load_safety_plugin(self):
        config = MarchConfig()
        config.plugins.enabled = ["safety"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "safety" in names

    def test_load_cost_plugin(self):
        config = MarchConfig()
        config.plugins.enabled = ["cost"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "cost" in names

    def test_load_rate_limiter_plugin(self):
        config = MarchConfig()
        config.plugins.enabled = ["rate_limiter"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "rate_limiter" in names

    def test_load_logger_plugin(self):
        config = MarchConfig()
        config.plugins.enabled = ["logger"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "logger" in names

    def test_load_git_context_plugin(self):
        config = MarchConfig()
        config.plugins.enabled = ["git_context"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "git_context" in names

    def test_load_multiple_plugins(self):
        config = MarchConfig()
        config.plugins.enabled = ["safety", "cost", "logger"]
        app = MarchApp(config=config)
        app._load_builtin_plugins()
        names = [p.name for p in app.plugin_manager._plugins]
        assert "safety" in names
        assert "cost" in names
        assert "logger" in names


# ─────────────────────────────────────────────────────────────
# App Run (basic path testing)
# ─────────────────────────────────────────────────────────────

class TestMarchAppRun:
    async def test_run_async_no_channels(self, tmp_path):
        app = MarchApp()
        app.memory_store = MagicMock()
        app.memory_store.initialize = AsyncMock()
        app.memory_store.close = AsyncMock()
        app.plugin_manager = MagicMock()
        app.plugin_manager.dispatch_simple = AsyncMock()

        with patch("march.app.Path.cwd", return_value=tmp_path):
            # Empty channel list
            await app._run_async([], None)
            # Should have handled gracefully

    async def test_run_async_unknown_channel(self, tmp_path):
        app = MarchApp()
        app.memory_store = MagicMock()
        app.memory_store.initialize = AsyncMock()
        app.memory_store.close = AsyncMock()
        app.plugin_manager = MagicMock()
        app.plugin_manager.dispatch_simple = AsyncMock()

        with patch("march.app.Path.cwd", return_value=tmp_path):
            await app._run_async(["nonexistent_channel"], None)
            # Should handle gracefully without crash
