"""Plugin base class for the March agent framework.

All March plugins inherit from this class. Override the hook methods you need;
the defaults are no-ops that pass through their arguments unchanged.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from march.core.context import Context
    from march.core.message import Message, ToolCall, ToolResult
    from march.llm.base import LLMResponse, StreamChunk


class Plugin:
    """Base class for all March plugins.

    Override only the hooks you need. All hooks have default no-op implementations
    that pass through their arguments unchanged.

    Attributes:
        name: Unique plugin name.
        version: Plugin version string.
        priority: Execution priority (lower = runs first). Default 100.
        enabled: Whether the plugin is active.
    """

    name: str = "unnamed"
    version: str = "0.1.0"
    priority: int = 100
    enabled: bool = True

    # ── Application lifecycle ──

    async def on_start(self, app: Any) -> None:
        """Called when the March app starts. Use for initialization."""
        pass

    async def on_shutdown(self, app: Any) -> None:
        """Called when the March app shuts down. Use for cleanup."""
        pass

    # ── Session lifecycle ──

    async def on_session_connect(self, session: Any) -> None:
        """Called when a session connects (terminal start, WebSocket connect)."""
        pass

    async def on_session_reset(self, session: Any) -> None:
        """Called when a session is reset (history cleared)."""
        pass

    # ── LLM pipeline ──

    async def before_llm(
        self, context: "Context", message: str
    ) -> tuple["Context", str] | tuple["Context", str, str]:
        """Called before the LLM is invoked.

        Can do one of three things:
        1. Pass through: return (context, message) unchanged
        2. Modify: return (modified_context, modified_message)
        3. Short-circuit: return (context, message, response_text) to skip the LLM

        Returns:
            Tuple of (context, message) or (context, message, response) to short-circuit.
        """
        return context, message

    async def after_llm(self, context: "Context", response: "LLMResponse") -> "LLMResponse":
        """Called after the LLM responds. Can modify the response."""
        return response

    async def on_llm_error(self, error: Exception) -> None:
        """Called when an LLM call fails."""
        pass

    async def on_stream_chunk(self, chunk: "StreamChunk") -> "StreamChunk":
        """Called for each streaming chunk. Can modify the chunk."""
        return chunk

    # ── Tool pipeline ──

    async def before_tool(self, tool_call: "ToolCall") -> "ToolCall | None":
        """Called before a tool is executed.

        Return the tool_call to proceed, or None to block execution.
        """
        return tool_call

    async def after_tool(
        self, tool_call: "ToolCall", result: "ToolResult"
    ) -> "ToolResult":
        """Called after a tool executes. Can modify the result."""
        return result

    async def on_tool_error(self, tool_call: "ToolCall", error: Exception) -> None:
        """Called when a tool execution fails."""
        pass

    # ── Response ──

    async def on_response(self, response: Any) -> Any:
        """Called before the final response is sent to the user. Can modify."""
        return response

    # ── Error handling ──

    async def on_error(self, error: Exception) -> None:
        """Called on any unhandled error in the agent loop."""
        pass
