"""Audit trail — Security event tracking for March.

Provides a high-level interface for recording and querying security-sensitive
events: tool executions, blocked actions, config changes, cost thresholds.
All events are stored in SQLite for durability and queryability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from march.logging.handlers import SQLiteAuditHandler

# Default audit DB location
DEFAULT_AUDIT_DB = Path.home() / ".march" / "audit.db"


class AuditTrail:
    """High-level audit trail interface.

    Wraps the SQLiteAuditHandler for direct programmatic access to
    audit event recording and querying.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the audit trail.

        Args:
            db_path: Path to the SQLite database. Defaults to ~/.march/audit.db.
        """
        self._db_path = db_path or DEFAULT_AUDIT_DB
        self._handler = SQLiteAuditHandler(self._db_path)
        self._logger = logging.getLogger("march.audit")

    @property
    def db_path(self) -> Path:
        """Get the audit database path."""
        return self._db_path

    @property
    def handler(self) -> SQLiteAuditHandler:
        """Get the underlying SQLite handler (for attaching to loggers)."""
        return self._handler

    def record(
        self,
        event: str,
        *,
        session_id: str = "system",
        level: str = "INFO",
        **data: Any,
    ) -> None:
        """Record an audit event directly.

        This bypasses the logging system and writes directly to the audit DB.
        Use when you need guaranteed audit recording without depending on
        log level filters.

        Args:
            event: Event type (e.g. "security.blocked", "tool.call").
            session_id: Session identifier.
            level: Log level string.
            **data: Additional event data.
        """
        import datetime

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = logging.LogRecord(
            name="march.audit",
            level=getattr(logging, level, logging.INFO),
            pathname="",
            lineno=0,
            msg=json.dumps({
                "event": event,
                "timestamp": timestamp,
                "level": level,
                "session_id": session_id,
                **data,
            }),
            args=None,
            exc_info=None,
        )
        self._handler.emit(record)

    def record_tool_execution(
        self,
        tool: str,
        args: dict[str, Any],
        result_summary: str,
        duration_ms: float,
        session_id: str = "system",
    ) -> None:
        """Record a tool execution event.

        Args:
            tool: Tool name.
            args: Tool arguments.
            result_summary: Brief summary of the result.
            duration_ms: Execution duration in milliseconds.
            session_id: Session identifier.
        """
        self.record(
            "tool.call",
            session_id=session_id,
            tool=tool,
            args=args,
            result_summary=result_summary,
            duration_ms=duration_ms,
        )

    def record_blocked_action(
        self,
        action: str,
        reason: str,
        plugin: str,
        session_id: str = "system",
    ) -> None:
        """Record a blocked action (security event).

        Args:
            action: The action that was blocked.
            reason: Why it was blocked.
            plugin: Which plugin blocked it.
            session_id: Session identifier.
        """
        self.record(
            "security.blocked",
            session_id=session_id,
            level="WARNING",
            action=action,
            reason=reason,
            plugin=plugin,
        )

    def record_config_change(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        session_id: str = "system",
    ) -> None:
        """Record a configuration change.

        Args:
            key: Config key that changed.
            old_value: Previous value.
            new_value: New value.
            session_id: Session identifier.
        """
        self.record(
            "config.change",
            session_id=session_id,
            level="WARNING",
            key=key,
            old_value=str(old_value),
            new_value=str(new_value),
        )

    def query(
        self,
        event: str | None = None,
        session_id: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit trail entries.

        Args:
            event: Filter by event type.
            session_id: Filter by session ID.
            level: Filter by log level.
            limit: Maximum number of results.

        Returns:
            List of audit entries, newest first.
        """
        return self._handler.query(
            event=event,
            session_id=session_id,
            level=level,
            limit=limit,
        )

    def clear(self, before_days: int | None = None) -> int:
        """Clear audit trail entries.

        Args:
            before_days: Clear entries older than this. None = clear all.

        Returns:
            Number of entries cleared.
        """
        return self._handler.clear(before_days=before_days)
