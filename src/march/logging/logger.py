"""Structured logging with subsystem tags, matching OpenClaw patterns.

Two separate log outputs:
  1. Text log (~/.march/logs/march.log) — structured JSONL, daily rotation, 7-day retention
     Console (stderr → journalctl) — human-readable [subsystem] tagged lines
  2. Metrics log (~/.march/logs/metrics.jsonl) — machine-readable metrics for visualization

Subsystems: [agent] [llm] [tools] [ws_proxy] [stream] [compaction] [metrics] [system]
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

# Fixed logging configuration — always JSON, always ~/.march/logs/march.log
LOG_DIR = Path.home() / ".march" / "logs"
LOG_FILE = LOG_DIR / "march.log"
LOG_LEVEL = logging.INFO
LOG_RETENTION_DAYS = 7

# Sentinel for whether we've already configured structlog
_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with subsystem-tagged output.

    Sets up:
      - Rotating file handler → JSONL with {time, level, subsystem, message, ...}
      - stderr handler → human-readable: TIMESTAMP [subsystem] message key=value
      - structlog processors for both outputs

    Args:
        level: Log level override (default: INFO).

    Safe to call multiple times — only configures on first call.
    """
    global _configured
    if _configured:
        return

    from march.logging.formatters import SubsystemConsoleRenderer, SubsystemJSONRenderer
    from march.logging.handlers import get_file_handler

    log_level = getattr(logging, level.upper(), logging.INFO)

    # ── stdlib logging setup (file rotation + stderr) ──────────────────

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Rotating file handler (daily, 7-day retention)
    file_handler = get_file_handler(LOG_FILE, retention_days=LOG_RETENTION_DAYS)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # stderr handler for journalctl visibility
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    root_logger.addHandler(stderr_handler)

    # ── structlog setup ────────────────────────────────────────────────

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # File gets JSONL: {"time":"...","level":"info","subsystem":"llm","message":"..."}
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=SubsystemJSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)

    # stderr gets human-readable: TIMESTAMP [subsystem] message key=value
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=SubsystemConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    stderr_handler.setFormatter(console_formatter)

    _configured = True


def reset_logging() -> None:
    """Reset logging configuration. Used in tests."""
    global _configured
    _configured = False
    structlog.reset_defaults()


def get_logger(name: str = "march", subsystem: str = "system", **initial_context: Any) -> Any:
    """Get a structlog logger with a subsystem tag.

    Args:
        name: Logger name.
        subsystem: Subsystem tag for [bracket] display.
        **initial_context: Key-value pairs bound to every log entry.

    Returns:
        A bound structlog logger with subsystem pre-bound.
    """
    return structlog.get_logger(name, subsystem=subsystem, **initial_context)


class MarchLogger:
    """High-level logging interface with subsystem-tagged output.

    Each method logs to the appropriate subsystem:
      - llm_call/llm_error → [llm]
      - tool_call/tool_error → [tools]
      - session/agent events → [agent]
      - plugin events → [system]
    """

    def __init__(self, session_id: str = "system") -> None:
        self._session_id = session_id
        # Subsystem-specific loggers
        self._llm = get_logger("march.llm", subsystem="llm", session_id=session_id)
        self._tools = get_logger("march.tools", subsystem="tools", session_id=session_id)
        self._agent = get_logger("march.agent", subsystem="agent", session_id=session_id)
        self._system = get_logger("march.system", subsystem="system", session_id=session_id)

    def bind(self, **kwargs: Any) -> MarchLogger:
        """Return a new MarchLogger with additional bound context."""
        new = MarchLogger.__new__(MarchLogger)
        new._session_id = kwargs.get("session_id", self._session_id)
        new._llm = self._llm.bind(**kwargs)
        new._tools = self._tools.bind(**kwargs)
        new._agent = self._agent.bind(**kwargs)
        new._system = self._system.bind(**kwargs)
        return new

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── LLM subsystem ─────────────────────────────────────────────────

    def llm_call(self, provider: str, model: str, input_tokens: int,
                 output_tokens: int, cost: float, duration_ms: float) -> None:
        self._llm.info("call completed",
                       provider=provider, model=model,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       cost_usd=cost, duration_ms=round(duration_ms, 1))

    def llm_error(self, provider: str, error: str, will_retry: bool,
                  attempt: int = 0, max_retries: int = 0, model: str = "") -> None:
        """Log LLM error with full troubleshooting context."""
        self._llm.error("call failed",
                        provider=provider, model=model, error=error,
                        will_retry=will_retry, attempt=attempt,
                        max_retries=max_retries,
                        action="retrying" if will_retry else "giving up")

    def llm_fallback(self, from_provider: str, to_provider: str, reason: str = "") -> None:
        self._llm.warning("provider fallback",
                          from_provider=from_provider, to_provider=to_provider,
                          reason=reason)

    def llm_stream_error(self, provider: str, model: str, error: str,
                         collected_length: int = 0) -> None:
        """Log streaming error with context about what was already collected."""
        self._llm.error("stream failed",
                        provider=provider, model=model, error=error,
                        collected_chars=collected_length,
                        action="aborting stream")

    # ── Tools subsystem ───────────────────────────────────────────────

    def tool_call(self, tool: str, args: dict[str, Any], result_summary: str,
                  duration_ms: float) -> None:
        # Summarize args for console readability
        args_summary = ", ".join(f"{k}={repr(v)[:50]}" for k, v in (args or {}).items())
        self._tools.info("executed",
                         tool=tool, args_summary=args_summary,
                         result_summary=result_summary[:100],
                         duration_ms=round(duration_ms, 1))

    def tool_error(self, tool: str, args: dict[str, Any], error: str) -> None:
        args_summary = ", ".join(f"{k}={repr(v)[:50]}" for k, v in (args or {}).items())
        self._tools.error("execution failed",
                          tool=tool, args_summary=args_summary, error=error,
                          action="returning error to LLM")

    def tool_blocked(self, tool: str, plugin: str, reason: str = "") -> None:
        self._tools.warning("blocked by plugin",
                            tool=tool, plugin=plugin, reason=reason)

    # ── Agent subsystem ───────────────────────────────────────────────

    def turn_start(self, session_id: str, message_length: int) -> None:
        self._agent.info("turn started",
                         session_id=session_id, message_length=message_length)

    def turn_complete(self, session_id: str, tool_calls: int, total_tokens: int,
                      total_cost: float, duration_ms: float) -> None:
        self._agent.info("turn complete",
                         session_id=session_id, tool_calls=tool_calls,
                         total_tokens=total_tokens, cost_usd=total_cost,
                         duration_ms=round(duration_ms, 1))

    def context_built(self, session_id: str, system_tokens: int, history_messages: int) -> None:
        self._agent.debug("context built",
                          session_id=session_id, system_tokens=system_tokens,
                          history_messages=history_messages)

    def max_iterations_reached(self, session_id: str, max_iterations: int,
                               tool_calls_made: int) -> None:
        self._agent.warning("max tool iterations reached",
                            session_id=session_id, max_iterations=max_iterations,
                            tool_calls_made=tool_calls_made,
                            action="returning partial response")

    # ── Plugin / system subsystem ─────────────────────────────────────

    def plugin_hook(self, plugin: str, hook: str, action: str, duration_ms: float) -> None:
        self._system.debug("plugin hook",
                           plugin=plugin, hook=hook, action=action,
                           duration_ms=round(duration_ms, 1))

    def plugin_error(self, plugin: str, hook: str, error: str) -> None:
        self._system.error("plugin error",
                           plugin=plugin, hook=hook, error=error)

    def subagent_spawn(self, agent_id: str, task: str, model: str) -> None:
        self._agent.info("subagent spawned",
                         agent_id=agent_id, task=task[:100], model=model)

    def subagent_complete(self, agent_id: str, result: str, duration_ms: float) -> None:
        self._agent.info("subagent completed",
                         agent_id=agent_id, result=result[:100],
                         duration_ms=round(duration_ms, 1))

    def subagent_error(self, agent_id: str, error: str) -> None:
        self._agent.error("subagent failed",
                          agent_id=agent_id, error=error)

    def security_blocked(self, action: str, reason: str, plugin: str) -> None:
        self._system.warning("security blocked",
                             action=action, reason=reason, plugin=plugin)

    def session_start(self, session_id: str, channel: str) -> None:
        self._agent.info("session started",
                         session_id=session_id, channel=channel)

    def session_end(self, session_id: str, channel: str) -> None:
        self._agent.info("session ended",
                         session_id=session_id, channel=channel)

    def memory_write(self, key: str, size_bytes: int) -> None:
        self._agent.info("memory written",
                         key=key, size_bytes=size_bytes)

    def config_loaded(self, path: str) -> None:
        self._system.info("config loaded", path=path)


class MetricsLogger:
    """Append-only JSONL metrics logger.

    Writes machine-readable metrics events to ~/.march/logs/metrics.jsonl.
    Events: llm.call, tool.call, turn.complete, message.received,
            message.complete, compaction, stream.draft_saved
    Thread-safe via a lock.
    """

    _instance: MetricsLogger | None = None
    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or "~/.march/logs/metrics.jsonl").expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    @classmethod
    def get(cls, path: str | Path | None = None) -> MetricsLogger:
        """Get or create the singleton MetricsLogger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for tests)."""
        cls._instance = None

    def _write(self, record: dict[str, Any]) -> None:
        """Append a JSON line to the metrics file."""
        record["ts"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._write_lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def llm_call(
        self,
        session_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        duration_ms: float,
    ) -> None:
        self._write({
            "event": "llm.call",
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6),
            "duration_ms": round(duration_ms, 1),
        })

    def tool_call(
        self,
        session_id: str,
        tool: str,
        duration_ms: float,
    ) -> None:
        self._write({
            "event": "tool.call",
            "session_id": session_id,
            "tool": tool,
            "duration_ms": round(duration_ms, 1),
        })

    def turn_complete(
        self,
        session_id: str,
        tool_calls: int,
        total_tokens: int,
        total_cost: float,
        duration_ms: float,
    ) -> None:
        self._write({
            "event": "turn.complete",
            "session_id": session_id,
            "tool_calls": tool_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "duration_ms": round(duration_ms, 1),
        })

    def message_received(
        self,
        session_id: str,
        content_length: int,
    ) -> None:
        self._write({
            "event": "message.received",
            "session_id": session_id,
            "content_length": content_length,
        })

    def message_complete(
        self,
        session_id: str,
        duration_ms: float,
        tool_calls: int = 0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> None:
        self._write({
            "event": "message.complete",
            "session_id": session_id,
            "duration_ms": round(duration_ms, 1),
            "tool_calls": tool_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
        })

    def compaction_done(
        self,
        session_id: str,
        messages_compacted: int,
        summary_length: int,
    ) -> None:
        self._write({
            "event": "compaction",
            "session_id": session_id,
            "messages_summarized": messages_compacted,
            "summary_length": summary_length,
        })

    def stream_draft_saved(
        self,
        session_id: str,
        chunks: int,
        content_length: int,
    ) -> None:
        self._write({
            "event": "stream.draft_saved",
            "session_id": session_id,
            "chunks": chunks,
            "content_length": content_length,
        })
