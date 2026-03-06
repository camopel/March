"""Tool registry for the March agent framework.

Unified registry for builtin, MCP, and skill tools. Handles registration,
schema generation for LLM APIs, and tool call execution.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Coroutine

from march.core.message import ToolCall, ToolResult
from march.logging import get_logger
from march.tools.base import Tool, ToolMeta

logger = get_logger("march.tools")


class ToolNotFound(Exception):
    """Raised when a tool call references an unknown tool."""

    pass


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""

    pass


class ToolRegistry:
    """Unified registry for all agent tools.

    Manages tool registration, provides LLM-compatible definitions,
    and executes tool calls.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} (source={tool.source})")

    def register_function(
        self,
        fn: Callable[..., Coroutine[Any, Any, str]],
        name: str | None = None,
        description: str | None = None,
        source: str = "builtin",
    ) -> None:
        """Register an async function as a tool.

        If the function has _tool_meta from the @tool decorator, that metadata
        is used. Otherwise, metadata is extracted from the function signature.
        """
        meta: ToolMeta | None = getattr(fn, "_tool_meta", None)
        if meta is None:
            from march.tools.base import _extract_schema

            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]
            params = _extract_schema(fn)
        else:
            tool_name = name or meta.name
            tool_desc = description or meta.description
            params = meta.parameters

        tool = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            fn=fn,
            source=source,
        )
        self.register(tool)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name. Returns True if found and removed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in LLM-compatible JSON Schema format.

        Suitable for passing to OpenAI/Anthropic tool_use parameters.
        """
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def definitions_anthropic(self) -> list[dict[str, Any]]:
        """Return tool definitions in Anthropic's format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        Args:
            tool_call: The tool call to execute.

        Returns:
            ToolResult with the execution output or error.

        Raises:
            ToolNotFound: If the tool name is not registered.
        """
        tool = self._tools.get(tool_call.name)
        if not tool:
            raise ToolNotFound(f"Unknown tool: {tool_call.name}")

        start = time.monotonic()
        try:
            result = await tool.fn(**tool_call.args)
            duration_ms = (time.monotonic() - start) * 1000

            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)

            return ToolResult(
                id=tool_call.id,
                content=result,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(f"Tool execution failed: {tool_call.name} — {e}")
            return ToolResult(
                id=tool_call.id,
                content="",
                error=str(e),
                duration_ms=duration_ms,
            )

    async def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls sequentially.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of results in the same order as the input calls.
        """
        results: list[ToolResult] = []
        for tc in tool_calls:
            result = await self.execute(tc)
            results.append(result)
        return results
