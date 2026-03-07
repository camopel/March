"""LoggerPlugin — Log all LLM calls, tool executions, and plugin actions.

Provides structured logging of all agent activity for debugging and auditing.
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from march.logging import get_logger
from march.plugins._base import Plugin

if TYPE_CHECKING:
    from march.core.context import Context
    from march.core.message import ToolCall, ToolResult
    from march.llm.base import LLMResponse

logger = get_logger("march.plugins.logger")


class LoggerPlugin(Plugin):
    """Log all LLM calls, tool executions, and sub-agent events.

    Attributes:
        log_tool_results: Whether to log tool result summaries.
        log_llm_calls: Whether to log LLM call details.
    """

    name = "logger"
    version = "0.1.0"
    priority = 95  # Runs late — after most plugins

    def __init__(
        self,
        log_tool_results: bool = True,
        log_llm_calls: bool = True,
    ) -> None:
        super().__init__()
        self.log_tool_results = log_tool_results
        self.log_llm_calls = log_llm_calls
        self._log_entries: list[dict[str, Any]] = []

    async def on_start(self, app: Any) -> None:
        """Load config from app.config.plugins.logger if available."""
        cfg = getattr(getattr(app, "config", None), "plugins", None)
        if cfg:
            logger_cfg = getattr(cfg, "logger", None)
            if logger_cfg:
                self.log_tool_results = getattr(logger_cfg, "log_tool_results", self.log_tool_results)
                self.log_llm_calls = getattr(logger_cfg, "log_llm_calls", self.log_llm_calls)

    async def after_llm(self, context: "Context", response: "LLMResponse") -> "LLMResponse":
        """Log LLM call details."""
        if not self.log_llm_calls:
            return response

        entry = {
            "type": "llm_call",
            "timestamp": time.time(),
            "provider": getattr(response, "provider", "unknown"),
            "model": getattr(response, "model", "unknown"),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost": response.usage.cost,
            "duration_ms": response.duration_ms,
            "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
        }
        self._log_entries.append(entry)
        logger.info(
            "logger.llm provider=%s model=%s in=%d out=%d cost=$%.6f duration=%.0fms",
            entry["provider"],
            entry["model"],
            entry["input_tokens"],
            entry["output_tokens"],
            entry["cost"],
            entry["duration_ms"],
        )

        return response

    async def after_tool(
        self, tool_call: "ToolCall", result: "ToolResult"
    ) -> "ToolResult":
        """Log tool execution details."""
        if not self.log_tool_results:
            return result

        entry = {
            "type": "tool_call",
            "timestamp": time.time(),
            "name": tool_call.name,
            "args_summary": _summarize_args(tool_call.args),
            "result_summary": result.summary[:200] if result else "",
            "duration_ms": result.duration_ms if result else 0.0,
            "error": result.error if result else None,
        }
        self._log_entries.append(entry)
        logger.info(
            "logger.tool name=%s duration=%.0fms error=%s summary=%s",
            entry["name"],
            entry["duration_ms"],
            entry["error"],
            entry["result_summary"][:100],
        )

        return result

    @property
    def log_entries(self) -> list[dict[str, Any]]:
        """Get all log entries."""
        return list(self._log_entries)

    def clear(self) -> None:
        """Clear all log entries."""
        self._log_entries.clear()


def _summarize_args(args: dict[str, Any], max_len: int = 200) -> str:
    """Create a brief summary of tool arguments."""
    if not args:
        return "{}"
    parts = []
    for key, value in args.items():
        val_str = str(value)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        parts.append(f"{key}={val_str}")
    summary = ", ".join(parts)
    if len(summary) > max_len:
        summary = summary[: max_len - 3] + "..."
    return summary
