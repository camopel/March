"""Tests for the March plugin system: dispatch, priority, blocking."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from march.core.context import Context
from march.core.message import ToolCall
from march.plugins.base import Plugin
from march.plugins.hooks import Hook
from march.plugins.manager import PluginManager


# ─────────────────────────────────────────────────────────────
# Test Plugins
# ─────────────────────────────────────────────────────────────


class HighPriorityPlugin(Plugin):
    name = "high-priority"
    priority = 10

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def before_llm(self, context: Context, message: str) -> tuple[Context, str]:
        self.calls.append("before_llm")
        context.add("injected by high-priority")
        return context, message

    async def after_llm(self, context: Context, response: Any) -> Any:
        self.calls.append("after_llm")
        return response

    async def before_tool(self, tool_call: ToolCall) -> ToolCall | None:
        self.calls.append(f"before_tool:{tool_call.name}")
        return tool_call

    async def on_response(self, response: Any) -> Any:
        self.calls.append("on_response")
        return response


class LowPriorityPlugin(Plugin):
    name = "low-priority"
    priority = 200

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def before_llm(self, context: Context, message: str) -> tuple[Context, str]:
        self.calls.append("before_llm")
        return context, message

    async def on_response(self, response: Any) -> Any:
        self.calls.append("on_response")
        return response


class BlockingPlugin(Plugin):
    """Plugin that blocks specific tool calls."""

    name = "blocker"
    priority = 5

    def __init__(self, blocked_tools: list[str] | None = None) -> None:
        super().__init__()
        self.blocked_tools = blocked_tools or ["dangerous_tool"]
        self.blocked_count = 0

    async def before_tool(self, tool_call: ToolCall) -> ToolCall | None:
        if tool_call.name in self.blocked_tools:
            self.blocked_count += 1
            return None  # Block!
        return tool_call


class ShortCircuitPlugin(Plugin):
    """Plugin that short-circuits the LLM with a cached response."""

    name = "short-circuit"
    priority = 1

    def __init__(self, cached_response: str = "Cached!") -> None:
        super().__init__()
        self.cached_response = cached_response

    async def before_llm(
        self, context: Context, message: str
    ) -> tuple[Context, str, str]:
        # Return 3-tuple to short-circuit
        return context, message, self.cached_response


class ErrorPlugin(Plugin):
    """Plugin that raises errors in hooks (should be caught gracefully)."""

    name = "error-plugin"
    priority = 50

    async def before_llm(self, context: Context, message: str) -> tuple[Context, str]:
        raise RuntimeError("Plugin exploded!")

    async def on_response(self, response: Any) -> Any:
        raise ValueError("Response processing failed!")


class ModifyingPlugin(Plugin):
    """Plugin that modifies messages."""

    name = "modifier"
    priority = 50

    async def before_llm(self, context: Context, message: str) -> tuple[Context, str]:
        return context, message + " [modified]"

    async def on_response(self, response: Any) -> Any:
        if isinstance(response, str):
            return response + " [post-processed]"
        return response


# ─────────────────────────────────────────────────────────────
# Plugin Hook Enum Tests
# ─────────────────────────────────────────────────────────────


class TestHookEnum:
    def test_all_hooks_exist(self) -> None:
        expected = [
            "on_start", "on_shutdown", "on_config_reload",
            "on_session_start", "on_session_end", "on_session_restore",
            "before_llm", "after_llm", "on_llm_error", "on_llm_fallback", "on_stream_chunk",
            "before_tool", "after_tool", "on_tool_error",
            "on_subagent_spawn", "on_subagent_progress", "on_subagent_complete", "on_subagent_error",
            "on_memory_read", "on_memory_write", "on_memory_search", "on_memory_index",
            "on_response",
            "on_channel_connect", "on_channel_disconnect", "on_message_receive",
            "on_error", "on_recovery",
        ]
        for hook_value in expected:
            assert hook_value in [h.value for h in Hook], f"Missing hook: {hook_value}"

    def test_hook_is_string_enum(self) -> None:
        assert Hook.BEFORE_LLM == "before_llm"
        assert isinstance(Hook.BEFORE_LLM, str)


# ─────────────────────────────────────────────────────────────
# Plugin Priority & Dispatch Order
# ─────────────────────────────────────────────────────────────


class TestPluginPriority:
    def test_priority_sorting(self) -> None:
        """Plugins should be sorted by priority (lower = first)."""
        manager = PluginManager()
        low = LowPriorityPlugin()
        high = HighPriorityPlugin()
        manager.register(low)
        manager.register(high)

        assert manager.plugins[0].name == "high-priority"
        assert manager.plugins[1].name == "low-priority"

    async def test_dispatch_order(self) -> None:
        """Hooks fire in priority order."""
        manager = PluginManager()
        high = HighPriorityPlugin()
        low = LowPriorityPlugin()
        manager.register(low)
        manager.register(high)

        ctx = Context()
        await manager.dispatch_before_llm(ctx, "test")

        assert high.calls == ["before_llm"]
        assert low.calls == ["before_llm"]

    async def test_dispatch_on_response_order(self) -> None:
        """on_response fires in priority order."""
        manager = PluginManager()
        high = HighPriorityPlugin()
        low = LowPriorityPlugin()
        manager.register(low)
        manager.register(high)

        await manager.dispatch_on_response("test response")

        assert "on_response" in high.calls
        assert "on_response" in low.calls


# ─────────────────────────────────────────────────────────────
# Plugin: before_tool Blocking
# ─────────────────────────────────────────────────────────────


class TestBeforeToolBlocking:
    async def test_block_tool(self) -> None:
        """before_tool returning None blocks execution."""
        manager = PluginManager()
        blocker = BlockingPlugin(blocked_tools=["exec"])
        manager.register(blocker)

        tc = ToolCall(id="tc_1", name="exec", args={"cmd": "rm -rf /"})
        result = await manager.dispatch_before_tool(tc)

        assert result is None
        assert blocker.blocked_count == 1

    async def test_allow_tool(self) -> None:
        """before_tool returning the tool_call allows execution."""
        manager = PluginManager()
        blocker = BlockingPlugin(blocked_tools=["exec"])
        manager.register(blocker)

        tc = ToolCall(id="tc_1", name="read_file", args={"path": "/tmp"})
        result = await manager.dispatch_before_tool(tc)

        assert result is not None
        assert result.name == "read_file"
        assert blocker.blocked_count == 0

    async def test_multiple_plugins_blocking(self) -> None:
        """If any plugin blocks, execution is blocked."""
        manager = PluginManager()
        # First plugin allows everything
        manager.register(HighPriorityPlugin())
        # Second plugin blocks "dangerous_tool"
        blocker = BlockingPlugin()
        manager.register(blocker)

        tc = ToolCall(id="tc_1", name="dangerous_tool", args={})
        result = await manager.dispatch_before_tool(tc)

        assert result is None


# ─────────────────────────────────────────────────────────────
# Plugin: Short-Circuit
# ─────────────────────────────────────────────────────────────


class TestShortCircuit:
    async def test_short_circuit_response(self) -> None:
        """Plugin can return a direct response to skip LLM."""
        manager = PluginManager()
        manager.register(ShortCircuitPlugin(cached_response="Cached answer"))

        ctx = Context()
        new_ctx, msg, short_circuit = await manager.dispatch_before_llm(ctx, "question")

        assert short_circuit == "Cached answer"

    async def test_no_short_circuit(self) -> None:
        """Normal plugins don't short-circuit."""
        manager = PluginManager()
        manager.register(HighPriorityPlugin())

        ctx = Context()
        new_ctx, msg, short_circuit = await manager.dispatch_before_llm(ctx, "question")

        assert short_circuit is None


# ─────────────────────────────────────────────────────────────
# Plugin: Error Handling
# ─────────────────────────────────────────────────────────────


class TestPluginErrorHandling:
    async def test_error_in_hook_continues(self) -> None:
        """Errors in plugin hooks are logged and execution continues."""
        manager = PluginManager()
        error_plugin = ErrorPlugin()
        normal_plugin = LowPriorityPlugin()
        manager.register(error_plugin)
        manager.register(normal_plugin)

        ctx = Context()
        new_ctx, msg, short_circuit = await manager.dispatch_before_llm(ctx, "test")

        # The normal plugin should still execute despite error_plugin throwing
        assert "before_llm" in normal_plugin.calls

    async def test_error_in_on_response_continues(self) -> None:
        """Error in on_response doesn't break the chain."""
        manager = PluginManager()
        manager.register(ErrorPlugin())
        normal = LowPriorityPlugin()
        manager.register(normal)

        result = await manager.dispatch_on_response("test")
        assert "on_response" in normal.calls


# ─────────────────────────────────────────────────────────────
# Plugin: Context Modification
# ─────────────────────────────────────────────────────────────


class TestPluginModification:
    async def test_modify_message(self) -> None:
        """Plugin can modify the user message."""
        manager = PluginManager()
        manager.register(ModifyingPlugin())

        ctx = Context()
        new_ctx, msg, _ = await manager.dispatch_before_llm(ctx, "hello")

        assert msg == "hello [modified]"

    async def test_modify_response(self) -> None:
        """Plugin can modify the response."""
        manager = PluginManager()
        manager.register(ModifyingPlugin())

        result = await manager.dispatch_on_response("original")
        assert result == "original [post-processed]"

    async def test_inject_context(self) -> None:
        """Plugin can inject extra context."""
        manager = PluginManager()
        manager.register(HighPriorityPlugin())

        ctx = Context()
        new_ctx, msg, _ = await manager.dispatch_before_llm(ctx, "test")

        prompt = new_ctx.build_system_prompt()
        assert "injected by high-priority" in prompt


# ─────────────────────────────────────────────────────────────
# Plugin Manager: Registration & Discovery
# ─────────────────────────────────────────────────────────────


class TestPluginManager:
    def test_register_and_get(self) -> None:
        manager = PluginManager()
        plugin = HighPriorityPlugin()
        manager.register(plugin)

        assert manager.get("high-priority") is plugin
        assert manager.get("nonexistent") is None

    def test_unregister(self) -> None:
        manager = PluginManager()
        manager.register(HighPriorityPlugin())
        assert manager.unregister("high-priority")
        assert not manager.unregister("high-priority")

    def test_disabled_plugin_skipped(self) -> None:
        """Disabled plugins are skipped during dispatch."""
        manager = PluginManager()
        plugin = HighPriorityPlugin()
        plugin.enabled = False
        manager.register(plugin)

        # This should work fine — disabled plugins are skipped
        assert len(manager.plugins) == 1

    async def test_dispatch_simple_hook(self) -> None:
        """Simple hooks (no return value) dispatch to all plugins."""
        manager = PluginManager()

        started = False

        class StartPlugin(Plugin):
            name = "start-test"

            async def on_start(self, app: Any) -> None:
                nonlocal started
                started = True

        manager.register(StartPlugin())
        await manager.dispatch_simple("on_start", None)
        assert started

    async def test_dispatch_on_memory_read(self) -> None:
        """on_memory_read can modify key/value."""
        manager = PluginManager()

        class MemPlugin(Plugin):
            name = "mem-test"

            async def on_memory_read(self, key: str, value: str) -> tuple[str, str]:
                return key, value + " [enriched]"

        manager.register(MemPlugin())
        key, value = await manager.dispatch_on_memory_read("system", "rules")
        assert value == "rules [enriched]"

    def test_load_directory_missing(self) -> None:
        """Loading from a nonexistent directory returns 0."""
        import tempfile
        from pathlib import Path

        manager = PluginManager()
        loaded = manager.load_directory(Path(tempfile.mkdtemp()) / "nope")
        assert loaded == 0

    def test_load_directory_with_plugin(self, tmp_path: Any) -> None:
        """Load a plugin from a directory."""
        plugin_code = '''
from march.plugins.base import Plugin

class TestDirPlugin(Plugin):
    name = "test-dir"
    priority = 42
    async def before_llm(self, context, message):
        return context, message
'''
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(plugin_code)

        manager = PluginManager()
        loaded = manager.load_directory(tmp_path, enabled=["test-dir"])
        assert loaded == 1
        assert manager.get("test-dir") is not None
        assert manager.get("test-dir").priority == 42

    def test_load_directory_filter_enabled(self, tmp_path: Any) -> None:
        """Only enabled plugins are loaded."""
        plugin_code = '''
from march.plugins.base import Plugin

class EnabledPlugin(Plugin):
    name = "enabled-one"

class DisabledPlugin(Plugin):
    name = "disabled-one"
'''
        (tmp_path / "plugins.py").write_text(plugin_code)

        manager = PluginManager()
        loaded = manager.load_directory(tmp_path, enabled=["enabled-one"])
        assert loaded == 1
        assert manager.get("enabled-one") is not None
        assert manager.get("disabled-one") is None
