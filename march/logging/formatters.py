"""Log formatters — Console (subsystem-tagged) and JSON formatters for structlog.

Console format (stderr → journalctl):
    2026-03-05T22:41:00-08:00 [llm] call completed model=claude-opus-4-6 input=1500 cost=$0.03

File format (JSONL):
    {"time":"...","level":"info","subsystem":"llm","message":"call completed","model":"claude-opus-4-6",...}
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import structlog


# ── Fields that go into dedicated JSONL columns, not into the kv tail ─────────

_DEDICATED_FIELDS = frozenset({
    "timestamp", "level", "subsystem", "event",
    "_record", "_from_structlog",
})


def _format_value(v: Any) -> str:
    """Format a value for console key=value output."""
    if isinstance(v, float):
        # Cost gets $ prefix
        return f"{v:.4g}"
    if isinstance(v, dict):
        return json.dumps(v, default=str, ensure_ascii=False)
    return str(v)


# ── Console Renderer ──────────────────────────────────────────────────────────

class SubsystemConsoleRenderer:
    """Render log lines as: TIMESTAMP [subsystem] message key=value key=value

    Designed for stderr / journalctl readability. Matches OpenClaw's pattern.
    """

    # ANSI color codes for subsystem tags
    _COLORS = {
        "agent": "\033[36m",       # cyan
        "llm": "\033[33m",         # yellow
        "tools": "\033[32m",       # green
        "ws_proxy": "\033[35m",    # magenta
        "stream": "\033[34m",      # blue
        "compaction": "\033[31m",  # red
        "metrics": "\033[90m",     # gray
        "system": "\033[37m",      # white
    }
    _RESET = "\033[0m"

    def __init__(self, colors: bool = True) -> None:
        self._colors = colors

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> str:
        # Extract dedicated fields
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", method_name)
        subsystem = event_dict.pop("subsystem", "system")
        message = event_dict.pop("event", "")

        # Remove internal structlog fields
        event_dict.pop("_record", None)
        event_dict.pop("_from_structlog", None)

        # Build key=value pairs from remaining fields
        kv_parts: list[str] = []
        for k, v in event_dict.items():
            if k.startswith("_"):
                continue
            # Special formatting for known fields
            if k == "cost_usd":
                kv_parts.append(f"cost=${v:.4g}")
            elif k == "duration_ms":
                kv_parts.append(f"duration={round(v)}ms")
            elif k == "input_tokens":
                kv_parts.append(f"input={v}")
            elif k == "output_tokens":
                kv_parts.append(f"output={v}")
            elif k == "session_id":
                # Truncate session_id to 8 chars for console
                kv_parts.append(f"session={str(v)[:8]}")
            else:
                kv_parts.append(f"{k}={_format_value(v)}")

        kv_str = " ".join(kv_parts)

        # Build the line
        if self._colors:
            color = self._COLORS.get(subsystem, self._COLORS["system"])
            tag = f"{color}[{subsystem}]{self._RESET}"
        else:
            tag = f"[{subsystem}]"

        parts = [timestamp, tag, message]
        if kv_str:
            parts.append(kv_str)

        return " ".join(parts)


# ── JSON Renderer ─────────────────────────────────────────────────────────────

class SubsystemJSONRenderer:
    """Render log lines as JSONL with dedicated top-level fields.

    Output: {"time":"...","level":"info","subsystem":"llm","message":"call completed",...}
    """

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> str:
        # Build output dict with dedicated fields first (for readability)
        out: dict[str, Any] = {}

        out["time"] = event_dict.pop("timestamp", "")
        out["level"] = event_dict.pop("level", method_name)
        out["subsystem"] = event_dict.pop("subsystem", "system")
        out["message"] = event_dict.pop("event", "")

        # Remove internal structlog fields
        event_dict.pop("_record", None)
        event_dict.pop("_from_structlog", None)

        # Add remaining fields as metadata
        for k, v in event_dict.items():
            if not k.startswith("_"):
                out[k] = v

        return json.dumps(out, default=str, ensure_ascii=False)


# ── Legacy API (kept for backward compat) ─────────────────────────────────────

def get_console_processor() -> structlog.types.Processor:
    """Return the subsystem-aware console renderer."""
    return SubsystemConsoleRenderer(colors=True)


def get_json_processor() -> structlog.types.Processor:
    """Return the subsystem-aware JSON renderer."""
    return SubsystemJSONRenderer()


def format_for_audit(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Format an event dict for the SQLite audit trail."""
    timestamp = event_dict.get("timestamp", "")
    level = event_dict.get("level", "info")
    event = event_dict.get("event", "unknown")
    session_id = event_dict.get("session_id", "system")

    excluded_keys = {"timestamp", "level", "event", "session_id", "_record", "_from_structlog"}
    data = {k: v for k, v in event_dict.items() if k not in excluded_keys}

    return {
        "timestamp": timestamp,
        "level": level,
        "event": event,
        "session_id": session_id,
        "data": data,
    }
