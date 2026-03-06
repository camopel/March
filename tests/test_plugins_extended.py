"""Extended tests for plugin base, plugin manager, and plugin dispatch."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from march.plugins.base import Plugin
from march.plugins.manager import PluginManager


# ── Plugin Base Tests ──

class TestPluginBase:
    """Test all default no-op hook methods on the Plugin base class."""

    async def test_on_start(self):
        p = Plugin()
        await p.on_start(MagicMock())  # Should not raise

    async def test_on_shutdown(self):
        p = Plugin()
        await p.on_shutdown(MagicMock())

    async def test_on_config_reload(self):
        p = Plugin()
        await p.on_config_reload(MagicMock())

    async def test_on_session_start(self):
        p = Plugin()
        await p.on_session_start(MagicMock())

    async def test_on_session_end(self):
        p = Plugin()
        await p.on_session_end(MagicMock())

    async def test_on_session_restore(self):
        p = Plugin()
        await p.on_session_restore(MagicMock())

    async def test_before_llm_passthrough(self):
        p = Plugin()
        ctx = MagicMock()
        result = await p.before_llm(ctx, "hello")
        assert result == (ctx, "hello")

    async def test_after_llm_passthrough(self):
        p = Plugin()
        resp = MagicMock()
        result = await p.after_llm(MagicMock(), resp)
        assert result is resp

    async def test_on_llm_error(self):
        p = Plugin()
        await p.on_llm_error(RuntimeError("test"))

    async def test_on_llm_fallback(self):
        p = Plugin()
        await p.on_llm_fallback("provider_a", "provider_b")

    async def test_on_stream_chunk(self):
        p = Plugin()
        chunk = MagicMock()
        result = await p.on_stream_chunk(chunk)
        assert result is chunk

    async def test_before_tool_passthrough(self):
        p = Plugin()
        tc = MagicMock()
        result = await p.before_tool(tc)
        assert result is tc

    async def test_after_tool_passthrough(self):
        p = Plugin()
        tc = MagicMock()
        tr = MagicMock()
        result = await p.after_tool(tc, tr)
        assert result is tr

    async def test_on_tool_error(self):
        p = Plugin()
        await p.on_tool_error(MagicMock(), RuntimeError("test"))

    async def test_on_subagent_spawn(self):
        p = Plugin()
        config = {"task": "test"}
        result = await p.on_subagent_spawn(config)
        assert result == config

    async def test_on_subagent_progress(self):
        p = Plugin()
        await p.on_subagent_progress("agent-1", {"status": "running"})

    async def test_on_subagent_complete(self):
        p = Plugin()
        await p.on_subagent_complete("agent-1", "result")

    async def test_on_subagent_error(self):
        p = Plugin()
        await p.on_subagent_error("agent-1", RuntimeError("test"))

    async def test_on_memory_read(self):
        p = Plugin()
        result = await p.on_memory_read("key", "value")
        assert result == ("key", "value")

    async def test_on_memory_write(self):
        p = Plugin()
        result = await p.on_memory_write("key", "value")
        assert result == ("key", "value")

    async def test_on_memory_search(self):
        p = Plugin()
        results_in = [1, 2, 3]
        result = await p.on_memory_search("query", results_in)
        assert result == results_in

    async def test_on_memory_index(self):
        p = Plugin()
        result = await p.on_memory_index("key", "content")
        assert result == ("key", "content")

    async def test_on_response(self):
        p = Plugin()
        result = await p.on_response("text response")
        assert result == "text response"

    async def test_on_channel_connect(self):
        p = Plugin()
        await p.on_channel_connect("matrix", {"room": "test"})

    async def test_on_channel_disconnect(self):
        p = Plugin()
        await p.on_channel_disconnect("matrix")

    async def test_on_message_receive(self):
        p = Plugin()
        msg = MagicMock()
        result = await p.on_message_receive(msg)
        assert result is msg

    async def test_on_error(self):
        p = Plugin()
        await p.on_error(RuntimeError("test"))

    async def test_on_recovery(self):
        p = Plugin()
        await p.on_recovery(RuntimeError("test"), "retry")

    def test_default_attributes(self):
        p = Plugin()
        assert p.name == "unnamed"
        assert p.version == "0.1.0"
        assert p.priority == 100
        assert p.enabled is True


# ── Plugin Manager Tests ──

class TestPluginManagerExtended:
    def test_register_sorts_by_priority(self):
        pm = PluginManager()

        class HighPriority(Plugin):
            name = "high"
            priority = 1

        class LowPriority(Plugin):
            name = "low"
            priority = 200

        pm.register(LowPriority())
        pm.register(HighPriority())
        assert pm.plugins[0].name == "high"
        assert pm.plugins[1].name == "low"

    def test_unregister(self):
        pm = PluginManager()
        p = Plugin()
        p.name = "test"
        pm.register(p)
        assert pm.unregister("test")
        assert not pm.unregister("test")  # Already removed

    def test_get(self):
        pm = PluginManager()
        p = Plugin()
        p.name = "findme"
        pm.register(p)
        assert pm.get("findme") is p
        assert pm.get("nonexistent") is None

    async def test_dispatch_before_llm_disabled_plugin(self):
        pm = PluginManager()

        class DisabledPlugin(Plugin):
            name = "disabled"
            enabled = False

            async def before_llm(self, ctx, msg):
                return ctx, msg, "should not appear"

        pm.register(DisabledPlugin())
        ctx = MagicMock()
        result = await pm.dispatch_before_llm(ctx, "hello")
        assert result[2] is None  # No short-circuit

    async def test_dispatch_before_llm_short_circuit(self):
        pm = PluginManager()

        class ShortCircuit(Plugin):
            name = "sc"

            async def before_llm(self, ctx, msg):
                return ctx, msg, "I handled it!"

        pm.register(ShortCircuit())
        ctx = MagicMock()
        result = await pm.dispatch_before_llm(ctx, "hello")
        assert result[2] == "I handled it!"

    async def test_dispatch_before_llm_modify(self):
        pm = PluginManager()

        class Modifier(Plugin):
            name = "mod"

            async def before_llm(self, ctx, msg):
                return ctx, msg + " modified"

        pm.register(Modifier())
        ctx = MagicMock()
        result = await pm.dispatch_before_llm(ctx, "hello")
        assert result[1] == "hello modified"
        assert result[2] is None

    async def test_dispatch_before_llm_plugin_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def before_llm(self, ctx, msg):
                raise RuntimeError("plugin crashed")

        pm.register(Broken())
        ctx = MagicMock()
        result = await pm.dispatch_before_llm(ctx, "hello")
        # Should continue gracefully
        assert result[1] == "hello"

    async def test_dispatch_after_llm(self):
        pm = PluginManager()

        class AfterMod(Plugin):
            name = "after"

            async def after_llm(self, ctx, resp):
                resp.modified = True
                return resp

        pm.register(AfterMod())
        resp = MagicMock()
        result = await pm.dispatch_after_llm(MagicMock(), resp)
        assert result.modified

    async def test_dispatch_after_llm_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def after_llm(self, ctx, resp):
                raise RuntimeError("crash")

        pm.register(Broken())
        resp = MagicMock()
        result = await pm.dispatch_after_llm(MagicMock(), resp)
        assert result is resp  # Original response preserved

    async def test_dispatch_before_tool(self):
        pm = PluginManager()
        tc = MagicMock()
        tc.name = "test_tool"
        result = await pm.dispatch_before_tool(tc)
        assert result is tc

    async def test_dispatch_before_tool_block(self):
        pm = PluginManager()

        class Blocker(Plugin):
            name = "blocker"

            async def before_tool(self, tc):
                return None

        pm.register(Blocker())
        tc = MagicMock()
        tc.name = "dangerous_tool"
        result = await pm.dispatch_before_tool(tc)
        assert result is None

    async def test_dispatch_before_tool_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def before_tool(self, tc):
                raise RuntimeError("crash")

        pm.register(Broken())
        tc = MagicMock()
        tc.name = "tool"
        result = await pm.dispatch_before_tool(tc)
        assert result is tc  # Preserved despite error

    async def test_dispatch_after_tool(self):
        pm = PluginManager()
        tc = MagicMock()
        tr = MagicMock()
        result = await pm.dispatch_after_tool(tc, tr)
        assert result is tr

    async def test_dispatch_after_tool_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def after_tool(self, tc, tr):
                raise RuntimeError("crash")

        pm.register(Broken())
        tc = MagicMock()
        tr = MagicMock()
        result = await pm.dispatch_after_tool(tc, tr)
        assert result is tr

    async def test_dispatch_on_tool_error(self):
        pm = PluginManager()
        called = False

        class ErrorHandler(Plugin):
            name = "handler"

            async def on_tool_error(self, tc, error):
                nonlocal called
                called = True

        pm.register(ErrorHandler())
        await pm.dispatch_on_tool_error(MagicMock(), RuntimeError("test"))
        assert called

    async def test_dispatch_on_tool_error_handler_crashes(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_tool_error(self, tc, error):
                raise RuntimeError("handler crash")

        pm.register(Broken())
        # Should not raise
        await pm.dispatch_on_tool_error(MagicMock(), RuntimeError("original"))

    async def test_dispatch_on_response(self):
        pm = PluginManager()

        class Modifier(Plugin):
            name = "mod"

            async def on_response(self, resp):
                return resp + " modified"

        pm.register(Modifier())
        result = await pm.dispatch_on_response("original")
        assert result == "original modified"

    async def test_dispatch_on_response_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_response(self, resp):
                raise RuntimeError("crash")

        pm.register(Broken())
        result = await pm.dispatch_on_response("original")
        assert result == "original"

    async def test_dispatch_on_stream_chunk(self):
        pm = PluginManager()
        chunk = MagicMock()
        result = await pm.dispatch_on_stream_chunk(chunk)
        assert result is chunk

    async def test_dispatch_on_stream_chunk_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_stream_chunk(self, chunk):
                raise RuntimeError("crash")

        pm.register(Broken())
        chunk = MagicMock()
        result = await pm.dispatch_on_stream_chunk(chunk)
        assert result is chunk

    async def test_dispatch_simple(self):
        pm = PluginManager()
        called = False

        class Handler(Plugin):
            name = "handler"

            async def on_start(self, app):
                nonlocal called
                called = True

        pm.register(Handler())
        await pm.dispatch_simple("on_start", MagicMock())
        assert called

    async def test_dispatch_simple_nonexistent_hook(self):
        pm = PluginManager()
        pm.register(Plugin())
        await pm.dispatch_simple("nonexistent_hook", MagicMock())

    async def test_dispatch_simple_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_start(self, app):
                raise RuntimeError("crash")

        pm.register(Broken())
        await pm.dispatch_simple("on_start", MagicMock())

    async def test_dispatch_on_memory_read(self):
        pm = PluginManager()

        class Modifier(Plugin):
            name = "mod"

            async def on_memory_read(self, key, value):
                return key + "_modified", value + "_modified"

        pm.register(Modifier())
        key, value = await pm.dispatch_on_memory_read("k", "v")
        assert key == "k_modified"
        assert value == "v_modified"

    async def test_dispatch_on_memory_read_error(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_memory_read(self, key, value):
                raise RuntimeError("crash")

        pm.register(Broken())
        key, value = await pm.dispatch_on_memory_read("k", "v")
        assert key == "k"
        assert value == "v"

    async def test_dispatch_on_llm_error(self):
        pm = PluginManager()
        called = False

        class Handler(Plugin):
            name = "handler"

            async def on_llm_error(self, error):
                nonlocal called
                called = True

        pm.register(Handler())
        await pm.dispatch_on_llm_error(RuntimeError("test"))
        assert called

    async def test_dispatch_on_llm_error_handler_crash(self):
        pm = PluginManager()

        class Broken(Plugin):
            name = "broken"

            async def on_llm_error(self, error):
                raise RuntimeError("handler crash")

        pm.register(Broken())
        await pm.dispatch_on_llm_error(RuntimeError("original"))

    def test_load_directory_nonexistent(self):
        pm = PluginManager()
        loaded = pm.load_directory(Path("/nonexistent"))
        assert loaded == 0

    def test_load_directory_with_plugins(self, tmp_path):
        plugin_code = '''
from march.plugins.base import Plugin

class TestPlugin(Plugin):
    name = "test_loaded"
    priority = 50
'''
        (tmp_path / "test_plugin.py").write_text(plugin_code)
        pm = PluginManager()
        loaded = pm.load_directory(tmp_path)
        assert loaded == 1
        assert pm.get("test_loaded") is not None

    def test_load_directory_with_filter(self, tmp_path):
        plugin_code = '''
from march.plugins.base import Plugin

class PluginA(Plugin):
    name = "plugin_a"

class PluginB(Plugin):
    name = "plugin_b"
'''
        (tmp_path / "plugins.py").write_text(plugin_code)
        pm = PluginManager()
        loaded = pm.load_directory(tmp_path, enabled=["plugin_a"])
        assert loaded == 1
        assert pm.get("plugin_a") is not None
        assert pm.get("plugin_b") is None

    def test_load_directory_skip_underscore(self, tmp_path):
        (tmp_path / "_internal.py").write_text("# internal")
        pm = PluginManager()
        loaded = pm.load_directory(tmp_path)
        assert loaded == 0

    def test_load_directory_broken_plugin(self, tmp_path):
        (tmp_path / "broken.py").write_text("raise ImportError('oops')")
        pm = PluginManager()
        loaded = pm.load_directory(tmp_path)
        assert loaded == 0

    async def test_multiple_plugins_in_order(self):
        pm = PluginManager()
        order = []

        class First(Plugin):
            name = "first"
            priority = 1

            async def before_llm(self, ctx, msg):
                order.append("first")
                return ctx, msg

        class Second(Plugin):
            name = "second"
            priority = 2

            async def before_llm(self, ctx, msg):
                order.append("second")
                return ctx, msg

        pm.register(Second())
        pm.register(First())
        await pm.dispatch_before_llm(MagicMock(), "hello")
        assert order == ["first", "second"]
