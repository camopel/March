"""Plugin manager for the March agent framework.

Handles plugin loading, priority sorting, and hook dispatch.
Supports auto-discovery from a plugins directory and graceful error handling.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any

from march.logging import get_logger
from march.plugins._base import Plugin
from march.plugins._hooks import Hook

logger = get_logger("march.plugins")


class PluginManager:
    """Load plugins, sort by priority, and dispatch hooks.

    Plugins are sorted by priority (lower = runs first). Hook dispatch is
    await-safe and handles plugin errors gracefully by logging and continuing.
    """

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []

    @property
    def plugins(self) -> list[Plugin]:
        """Get all loaded plugins in priority order."""
        return list(self._plugins)

    def register(self, plugin: Plugin) -> None:
        """Register a plugin and re-sort by priority."""
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda p: p.priority)

    def unregister(self, name: str) -> bool:
        """Remove a plugin by name. Returns True if found and removed."""
        before = len(self._plugins)
        self._plugins = [p for p in self._plugins if p.name != name]
        return len(self._plugins) < before

    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        for p in self._plugins:
            if p.name == name:
                return p
        return None

    def load_directory(self, plugin_dir: Path, enabled: list[str] | None = None) -> int:
        """Auto-discover and load plugins from a directory.

        Scans the directory for Python files, imports them, finds Plugin subclasses,
        and registers those whose names are in the enabled list (or all if enabled is None).

        Returns:
            Number of plugins loaded.
        """
        if not plugin_dir.is_dir():
            return 0

        loaded = 0
        for path in sorted(plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                classes = self._import_plugins_from_file(path)
                for cls in classes:
                    if enabled is not None and cls.name not in enabled:
                        continue
                    plugin = cls()
                    self.register(plugin)
                    loaded += 1
                    logger.debug(
                        "Loaded plugin: %s (priority=%d)", plugin.name, plugin.priority
                    )
            except Exception as e:
                logger.error("Failed to load plugin from %s: %s", path, e)

        return loaded

    def _import_plugins_from_file(self, path: Path) -> list[type[Plugin]]:
        """Import a Python file and find all Plugin subclasses."""
        module_name = f"march_plugins_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        classes: list[type[Plugin]] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj is not Plugin and obj.name != "unnamed":
                classes.append(obj)

        return classes

    async def dispatch_before_llm(
        self, context: Any, message: str
    ) -> tuple[Any, str, str | None]:
        """Dispatch the before_llm hook through all plugins.

        Returns:
            (context, message, short_circuit_response) where short_circuit_response
            is None unless a plugin short-circuits with a direct response.
        """
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                start = time.monotonic()
                result = await plugin.before_llm(context, message)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=before_llm duration=%.0fms",
                    plugin.name, duration_ms,
                )

                if result is None:
                    # Plugin signals to block/skip
                    return context, message, None

                if len(result) == 3:
                    # Short-circuit: plugin returned a response directly
                    return result[0], result[1], result[2]
                elif len(result) == 2:
                    context, message = result[0], result[1]
                else:
                    logger.warning(
                        "Plugin %s before_llm returned unexpected tuple length: %d",
                        plugin.name, len(result),
                    )
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=before_llm error=%s",
                    plugin.name, e,
                )

        return context, message, None

    async def dispatch_after_llm(self, context: Any, response: Any) -> Any:
        """Dispatch the after_llm hook. Plugins can modify the response."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                start = time.monotonic()
                result = await plugin.after_llm(context, response)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=after_llm duration=%.0fms",
                    plugin.name, duration_ms,
                )
                if result is not None:
                    response = result
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=after_llm error=%s",
                    plugin.name, e,
                )
        return response

    async def dispatch_before_tool(self, tool_call: Any) -> Any | None:
        """Dispatch the before_tool hook. Returns None if any plugin blocks."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                start = time.monotonic()
                result = await plugin.before_tool(tool_call)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=before_tool duration=%.0fms",
                    plugin.name, duration_ms,
                )
                if result is None:
                    logger.info(
                        "Plugin %s blocked tool call: %s",
                        plugin.name, tool_call.name,
                    )
                    return None
                tool_call = result
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=before_tool error=%s",
                    plugin.name, e,
                )
        return tool_call

    async def dispatch_after_tool(self, tool_call: Any, result: Any) -> Any:
        """Dispatch the after_tool hook. Plugins can modify the result."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                start = time.monotonic()
                modified = await plugin.after_tool(tool_call, result)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=after_tool duration=%.0fms",
                    plugin.name, duration_ms,
                )
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=after_tool error=%s",
                    plugin.name, e,
                )
        return result

    async def dispatch_on_tool_error(self, tool_call: Any, error: Exception) -> None:
        """Dispatch the on_tool_error hook to all plugins."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                await plugin.on_tool_error(tool_call, error)
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=on_tool_error error=%s",
                    plugin.name, e,
                )

    async def dispatch_on_response(self, response: Any) -> Any:
        """Dispatch the on_response hook. Plugins can modify the response."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                start = time.monotonic()
                modified = await plugin.on_response(response)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=on_response duration=%.0fms",
                    plugin.name, duration_ms,
                )
                if modified is not None:
                    response = modified
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=on_response error=%s",
                    plugin.name, e,
                )
        return response

    async def dispatch_on_stream_chunk(self, chunk: Any) -> Any:
        """Dispatch the on_stream_chunk hook."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                modified = await plugin.on_stream_chunk(chunk)
                if modified is not None:
                    chunk = modified
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=on_stream_chunk error=%s",
                    plugin.name, e,
                )
        return chunk

    async def dispatch_simple(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Dispatch a simple notification hook (no return value processing).

        Used for hooks like on_start, on_shutdown, on_error, on_session_start, etc.
        """
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            handler = getattr(plugin, hook, None)
            if handler is None:
                continue
            try:
                start = time.monotonic()
                await handler(*args, **kwargs)
                duration_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "plugin.hook plugin=%s hook=%s duration=%.0fms",
                    plugin.name, hook, duration_ms,
                )
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=%s error=%s",
                    plugin.name, hook, e,
                )

    async def dispatch_on_llm_error(self, error: Exception) -> None:
        """Dispatch on_llm_error to all plugins."""
        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                await plugin.on_llm_error(error)
            except Exception as e:
                logger.error(
                    "plugin.error plugin=%s hook=on_llm_error error=%s",
                    plugin.name, e,
                )
