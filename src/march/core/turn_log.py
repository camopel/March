"""Structured JSONL turn logger for debugging agent turns.

Writes one JSON line per event to ``~/.march/logs/turns.jsonl``.
Thread-safe, with automatic log rotation when the file exceeds 50 MB
(keeps up to 3 rotated files).

Events
------
- ``turn_start``   – user message received
- ``llm_call``     – one LLM provider call completed
- ``tool_call``    – one tool execution completed
- ``turn_complete``– turn finished successfully
- ``turn_cancelled``– turn cancelled by user
- ``turn_error``   – turn ended with an error
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_ROTATED = 3


class TurnLogger:
    """Append-only JSONL logger for agent turn events.

    Parameters
    ----------
    log_dir:
        Directory for the log file.  Defaults to ``~/.march/logs``.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._dir = log_dir or Path.home() / ".march" / "logs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "turns.jsonl"
        self._lock = threading.Lock()

    # ── public event methods ─────────────────────────────────────────

    def turn_start(
        self,
        turn_id: str,
        session_id: str,
        user_msg: str,
        source: str,
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="turn_start",
            user_msg=user_msg[:2000],  # cap to avoid huge log lines
            source=source,
        )

    def llm_call(
        self,
        turn_id: str,
        session_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration_ms: float,
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="llm_call",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            duration_ms=duration_ms,
        )

    def tool_call(
        self,
        turn_id: str,
        session_id: str,
        name: str,
        args: dict,
        duration_ms: float,
        status: str,
        summary: str,
        error: str = "",
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="tool_call",
            name=name,
            args=args,
            duration_ms=duration_ms,
            status=status,
            summary=summary[:500],
            error=error,
        )

    def turn_complete(
        self,
        turn_id: str,
        session_id: str,
        tool_calls: int,
        total_tokens: int,
        total_cost: float,
        duration_ms: float,
        final_reply_length: int,
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="turn_complete",
            tool_calls=tool_calls,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_ms=duration_ms,
            final_reply_length=final_reply_length,
        )

    def turn_cancelled(
        self,
        turn_id: str,
        session_id: str,
        partial_content_length: int,
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="turn_cancelled",
            partial_content_length=partial_content_length,
        )

    def turn_error(
        self,
        turn_id: str,
        session_id: str,
        error: str,
    ) -> None:
        self._write(
            turn_id=turn_id,
            session_id=session_id,
            event="turn_error",
            error=error,
        )

    # ── internal helpers ─────────────────────────────────────────────

    def _write(self, **fields: Any) -> None:
        """Serialize *fields* as a single JSON line and append to the log."""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        entry.update(fields)

        # Ensure args dicts are JSON-serialisable (convert non-serialisable
        # values to their repr).
        line = self._safe_dumps(entry) + "\n"

        with self._lock:
            self._maybe_rotate()
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def _maybe_rotate(self) -> None:
        """Rotate the log file if it exceeds ``_MAX_FILE_BYTES``."""
        try:
            if not self._path.exists():
                return
            if self._path.stat().st_size < _MAX_FILE_BYTES:
                return
        except OSError:
            return

        # Shift existing rotated files: .3 → delete, .2 → .3, .1 → .2
        for i in range(_MAX_ROTATED, 0, -1):
            src = self._path.with_suffix(f".jsonl.{i}")
            if i == _MAX_ROTATED:
                src.unlink(missing_ok=True)
            else:
                dst = self._path.with_suffix(f".jsonl.{i + 1}")
                if src.exists():
                    src.rename(dst)

        # Current → .1
        rotated = self._path.with_suffix(".jsonl.1")
        try:
            self._path.rename(rotated)
        except OSError:
            pass

    @staticmethod
    def _safe_dumps(obj: Any) -> str:
        """``json.dumps`` with a fallback for non-serialisable values."""
        def _default(o: Any) -> Any:
            return repr(o)

        return json.dumps(obj, default=_default, ensure_ascii=False)
