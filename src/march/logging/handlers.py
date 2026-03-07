"""Log handlers — File rotation and SQLite audit trail."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


def get_file_handler(
    log_path: Path,
    retention_days: int = 7,
) -> logging.Handler:
    """Create a rotating file handler for JSON log output.

    Rotates daily at midnight, keeping `retention_days` backups.

    Args:
        log_path: Path to the log file.
        retention_days: Number of days of logs to retain.

    Returns:
        A configured TimedRotatingFileHandler.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d"
    return handler


class SQLiteAuditHandler(logging.Handler):
    """Logging handler that writes audit events to a SQLite database.

    Security-sensitive events (tool executions, blocked actions, config changes,
    cost threshold hits) are stored for later querying via the CLI.

    Thread-safe via a threading lock on write operations.
    """

    # Events that qualify for the audit trail
    AUDIT_EVENTS = frozenset({
        "security.blocked",
        "tool.call",
        "tool.error",
        "config.change",
        "llm.call",
        "subagent.spawn",
        "subagent.complete",
        "subagent.error",
        "session.start",
        "session.end",
    })

    def __init__(self, db_path: Path) -> None:
        """Initialize the SQLite audit handler.

        Args:
            db_path: Path to the SQLite database file.
        """
        super().__init__()
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new SQLite connection (one per call for thread safety)."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        """Create the audit table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT 'system',
                    data TEXT NOT NULL DEFAULT '{}',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_event
                ON audit_trail(event)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_trail(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_trail(session_id)
            """)
            conn.commit()
        finally:
            conn.close()

    def emit(self, record: logging.LogRecord) -> None:
        """Write an audit event to SQLite if it matches an audit event type.

        Args:
            record: The log record to potentially store.
        """
        try:
            # structlog attaches the event dict to the record message
            # We need to parse the event name from the record
            msg = self.format(record) if self.formatter else record.getMessage()

            # Try to parse as JSON to extract event type
            event_name = ""
            timestamp = ""
            session_id = "system"
            data: dict[str, Any] = {}

            try:
                parsed = json.loads(msg)
                event_name = parsed.get("event", "")
                timestamp = parsed.get("timestamp", "")
                session_id = parsed.get("session_id", "system")
                data = {
                    k: v for k, v in parsed.items()
                    if k not in ("event", "timestamp", "level", "session_id")
                }
            except (json.JSONDecodeError, TypeError):
                # Not JSON — try to extract event from the raw message
                event_name = getattr(record, "event", str(msg)[:100])
                timestamp = getattr(record, "timestamp", "")
                session_id = getattr(record, "session_id", "system")

            if event_name not in self.AUDIT_EVENTS:
                return

            level = record.levelname

            with self._lock:
                conn = self._get_connection()
                try:
                    conn.execute(
                        """
                        INSERT INTO audit_trail (timestamp, level, event, session_id, data)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            level,
                            event_name,
                            session_id,
                            json.dumps(data, default=str, ensure_ascii=False),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            # Never let audit logging crash the application
            self.handleError(record)

    def query(
        self,
        event: str | None = None,
        session_id: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit trail entries.

        Args:
            event: Filter by event type (e.g. "tool.call").
            session_id: Filter by session ID.
            level: Filter by log level.
            limit: Maximum number of results.

        Returns:
            List of audit trail entries as dicts.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if event:
            conditions.append("event = ?")
            params.append(event)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if level:
            conditions.append("level = ?")
            params.append(level)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query_sql = f"""
            SELECT id, timestamp, level, event, session_id, data, created_at
            FROM audit_trail
            {where}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query_sql, params)
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "level": row["level"],
                    "event": row["event"],
                    "session_id": row["session_id"],
                    "data": json.loads(row["data"]) if row["data"] else {},
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def clear(self, before_days: int | None = None) -> int:
        """Clear audit trail entries.

        Args:
            before_days: If set, only clear entries older than this many days.
                         If None, clear all entries.

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                if before_days is not None:
                    cursor = conn.execute(
                        """
                        DELETE FROM audit_trail
                        WHERE created_at < datetime('now', ?)
                        """,
                        (f"-{before_days} days",),
                    )
                else:
                    cursor = conn.execute("DELETE FROM audit_trail")
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
