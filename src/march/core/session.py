"""Session management for the March agent framework.

Sessions track conversation history, metadata, and provide persistence via SQLite.
Each session is tied to a source (terminal, Matrix room, IDE workspace, etc.).

Session identity is deterministic from the source:
  - Matrix room ID → consistent session UUID
  - VS Code workspace path → consistent session UUID
  - Terminal instance → new or resumed session

Sessions persist in SQLite across restarts. Resume on reconnect.
/reset only clears that source's session.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

from march.core.message import Message, Role


def deterministic_session_id(source_type: str, source_id: str) -> str:
    """Generate a deterministic session UUID from source identity.

    The same source_type + source_id always produces the same session ID.
    This ensures reconnecting to the same source resumes the same session.
    """
    key = f"{source_type}:{source_id}"
    hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=hash_bytes[:16]))


def _now() -> str:
    """ISO timestamp for DB storage."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    """A conversation session.

    Attributes:
        id: Unique session identifier (deterministic from source or random).
        source_type: Where the session originates (terminal, matrix, ws, acp).
        source_id: Source-specific identifier (room ID, workspace path, etc.).
        name: Human-readable session name.
        history: Ordered list of messages in this session.
        metadata: Session metadata (channel info, user prefs, etc.).
        created_at: Unix timestamp of session creation.
        last_active: Unix timestamp of last activity.
        state: Session state ("active" or "reset").
    """

    id: str = ""
    source_type: str = "terminal"
    source_id: str = ""
    name: str = ""
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_active: float = 0.0
    state: str = "active"
    # Compaction state
    compaction_summary: str = ""
    rolling_summary: str = ""
    backup_history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            # Deterministic ID from source if source_id is provided
            if self.source_id:
                self.id = deterministic_session_id(self.source_type, self.source_id)
            else:
                self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_active:
            self.last_active = self.created_at

    def add_message(self, message: Message) -> None:
        """Add a single message to the session history."""
        self.history.append(message)
        self.last_active = time.time()

    def add_exchange(self, user_message: str | list, assistant_content: str) -> None:
        """Add a user/assistant exchange to history."""
        self.history.append(Message.user(user_message))
        self.history.append(Message.assistant(assistant_content))
        self.last_active = time.time()

    def add_tool_exchange(
        self,
        assistant_message: Message,
        tool_result_message: Message,
    ) -> None:
        """Add an assistant tool-call + tool results to history."""
        self.history.append(assistant_message)
        self.history.append(tool_result_message)
        self.last_active = time.time()

    def clear(self) -> None:
        """Clear session history (reset). Clears backup too."""
        self.history.clear()
        self.backup_history.clear()
        self.compaction_summary = ""
        self.rolling_summary = ""
        self.last_active = time.time()

    def reset(self) -> None:
        """Reset the session: clear history, backup, summary, and mark as reset."""
        self.history.clear()
        self.backup_history.clear()
        self.compaction_summary = ""
        self.rolling_summary = ""
        self.state = "reset"
        self.last_active = time.time()

    def compact_history(self, summary: str, keep_recent: int = 10) -> int:
        """Move old messages to backup, replace with summary + recent.

        Args:
            summary: LLM-generated summary of the old messages.
            keep_recent: Number of recent messages to keep in active history.

        Returns:
            Number of messages moved to backup.
        """
        if len(self.history) <= keep_recent:
            return 0

        # Split: old messages → backup, recent → keep
        split_idx = len(self.history) - keep_recent
        old_messages = self.history[:split_idx]
        recent_messages = self.history[split_idx:]

        # Move old to backup
        self.backup_history.extend(old_messages)

        # Replace history with summary + recent
        self.compaction_summary = summary
        self.history = [Message.user(
            f"[Context Summary — {len(old_messages)} earlier messages compacted, "
            f"{len(self.backup_history)} total in backup]\n\n{summary}"
        )] + recent_messages

        self.last_active = time.time()
        return len(old_messages)

    def reactivate(self) -> None:
        """Reactivate a reset session."""
        self.state = "active"
        self.last_active = time.time()

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get all messages serialized for the LLM API.

        Flattens tool result messages into individual tool messages.
        """
        llm_messages: list[dict[str, Any]] = []
        for msg in self.history:
            llm_messages.extend(msg.to_llm_messages())
        return llm_messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict for persistence."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "name": self.name,
            "history": [msg.to_dict() for msg in self.history],
            "backup_history": [msg.to_dict() for msg in self.backup_history],
            "compaction_summary": self.compaction_summary,
            "rolling_summary": self.rolling_summary,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Deserialize session from dict."""
        history = [Message.from_dict(m) for m in data.get("history", [])]
        backup = [Message.from_dict(m) for m in data.get("backup_history", [])]
        return cls(
            id=data["id"],
            source_type=data.get("source_type", "terminal"),
            source_id=data.get("source_id", ""),
            name=data.get("name", ""),
            history=history,
            backup_history=backup,
            compaction_summary=data.get("compaction_summary", ""),
            rolling_summary=data.get("rolling_summary", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
            last_active=data.get("last_active", 0.0),
        )


# ── Unified SessionStore ─────────────────────────────────────────────────
#
# Single SQLite-backed store used by ALL channels (terminal, ACP, ws, matrix).
# No channel owns its own DB logic — everything goes through here.


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    rolling_summary TEXT DEFAULT '',
    compaction_summary TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_sessions_source
    ON sessions(source_type, source_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool', 'system')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT DEFAULT '[]',
    tool_results TEXT DEFAULT '[]',
    attachments TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
"""


class SessionStore:
    """Unified SQLite-backed session persistence.

    Used by ALL channels. Stores session metadata and message history.
    Attachments are stored as file path references (AttachmentRef dicts),
    never as base64 blobs.
    """

    def __init__(self, db_path: str | Path = "~/.march/march.db"):
        self.db_path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = sqlite3.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Check if tables exist and need migration vs fresh creation
        existing_tables = set()
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cur:
            async for row in cur:
                existing_tables.add(row["name"])

        if "sessions" in existing_tables:
            # Existing DB — run migrations to add new columns
            await self._run_migrations()
        else:
            # Fresh DB — create tables from scratch
            await self._db.executescript(_SCHEMA)

        await self._db.commit()

    async def _run_migrations(self) -> None:
        """Run schema migrations for compatibility with existing ws_proxy tables."""
        assert self._db is not None

        # Drop old agent-specific tables (never used, safe to remove)
        for old_table in ("agent_sessions", "agent_messages", "agent_backup_messages"):
            await self._db.execute(f"DROP TABLE IF EXISTS {old_table}")

        # ── Sessions table migrations ────────────────────────────────────
        existing_session_cols = set()
        async with self._db.execute("PRAGMA table_info(sessions)") as cur:
            async for row in cur:
                existing_session_cols.add(row["name"])

        session_migrations = {
            "source_type": "ALTER TABLE sessions ADD COLUMN source_type TEXT NOT NULL DEFAULT 'ws'",
            "source_id": "ALTER TABLE sessions ADD COLUMN source_id TEXT NOT NULL DEFAULT ''",
            "compaction_summary": "ALTER TABLE sessions ADD COLUMN compaction_summary TEXT DEFAULT ''",
            "rolling_summary": "ALTER TABLE sessions ADD COLUMN rolling_summary TEXT DEFAULT ''",
            "metadata": "ALTER TABLE sessions ADD COLUMN metadata TEXT DEFAULT '{}'",
        }
        for col, sql in session_migrations.items():
            if col not in existing_session_cols:
                try:
                    await self._db.execute(sql)
                except Exception:
                    pass

        # Create index if missing
        try:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source_type, source_id)"
            )
        except Exception:
            pass

        # ── Messages table migrations ────────────────────────────────────
        existing_msg_cols = set()
        async with self._db.execute("PRAGMA table_info(messages)") as cur:
            async for row in cur:
                existing_msg_cols.add(row["name"])

        msg_migrations = {
            "attachments": "ALTER TABLE messages ADD COLUMN attachments TEXT DEFAULT '[]'",
            "tool_results": "ALTER TABLE messages ADD COLUMN tool_results TEXT DEFAULT '[]'",
            "metadata": "ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT '{}'",
        }
        for col, sql in msg_migrations.items():
            if col not in existing_msg_cols:
                try:
                    await self._db.execute(sql)
                except Exception:
                    pass

        # Relax role CHECK constraint if needed (old schema only allowed user/assistant)
        # SQLite doesn't support ALTER CHECK, but new inserts with 'tool'/'system' will
        # work because SQLite CHECK is not enforced on existing rows by default.
        # For new DBs, the schema already includes all roles.

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ── Session CRUD ─────────────────────────────────────────────────────

    async def create_session(
        self,
        source_type: str,
        source_id: str,
        name: str = "",
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        """Create a new session and persist it."""
        assert self._db is not None

        session = Session(
            id=session_id or "",
            source_type=source_type,
            source_id=source_id,
            name=name,
            metadata=metadata or {},
        )
        now = _now()
        await self._db.execute(
            """INSERT INTO sessions (id, source_type, source_id, name, rolling_summary,
               compaction_summary, metadata, created_at, last_active, is_active)
               VALUES (?, ?, ?, ?, '', '', ?, ?, ?, 1)""",
            (session.id, source_type, source_id, name,
             json.dumps(session.metadata), now, now),
        )
        await self._db.commit()
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Load a session by ID (without messages — use get_messages for those)."""
        assert self._db is not None

        async with self._db.execute(
            "SELECT * FROM sessions WHERE id = ? AND is_active = 1", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None

        return self._row_to_session(row)

    async def get_or_create_session(
        self,
        source_type: str,
        source_id: str,
        name: str = "",
    ) -> Session:
        """Get existing session by source, or create a new one.

        Uses deterministic IDs — same source always maps to same session.
        """
        assert self._db is not None

        det_id = deterministic_session_id(source_type, source_id)

        async with self._db.execute(
            "SELECT * FROM sessions WHERE id = ? AND is_active = 1", (det_id,)
        ) as cur:
            row = await cur.fetchone()
            if row:
                session = self._row_to_session(row)
                # Load history into session
                messages = await self.get_messages(session.id)
                session.history = messages
                return session

        return await self.create_session(
            source_type, source_id, name, session_id=det_id
        )

    async def list_sessions(
        self,
        source_type: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """List sessions with optional filters."""
        assert self._db is not None

        conditions = []
        params: list[Any] = []

        if active_only:
            conditions.append("is_active = 1")
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM sessions {where} ORDER BY last_active DESC"

        results: list[dict[str, Any]] = []
        async with self._db.execute(query, params) as cur:
            async for row in cur:
                s = dict(row)
                s["is_active"] = bool(s["is_active"])
                # Get last message preview
                async with self._db.execute(
                    "SELECT content, role FROM messages WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (row["id"],),
                ) as msg_cur:
                    msg = await msg_cur.fetchone()
                    s["last_message"] = dict(msg) if msg else None
                results.append(s)
        return results

    async def update_session(self, session: Session) -> None:
        """Update session metadata (not messages)."""
        assert self._db is not None

        await self._db.execute(
            """UPDATE sessions SET name = ?, rolling_summary = ?, compaction_summary = ?,
               metadata = ?, last_active = ? WHERE id = ?""",
            (session.name, session.rolling_summary, session.compaction_summary,
             json.dumps(session.metadata), _now(), session.id),
        )
        await self._db.commit()

    async def delete_session(self, session_id: str) -> None:
        """Soft-delete a session."""
        assert self._db is not None

        await self._db.execute(
            "UPDATE sessions SET is_active = 0 WHERE id = ?", (session_id,)
        )
        await self._db.commit()

    async def clear_session(self, session_id: str) -> None:
        """Clear all messages and summaries for a session (reset)."""
        assert self._db is not None

        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._db.execute(
            "UPDATE sessions SET rolling_summary = '', compaction_summary = '' WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()

    # ── Message CRUD ─────────────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        message: Message,
        attachments: list[dict] | None = None,
    ) -> str:
        """Add a message to a session. Returns the message ID."""
        assert self._db is not None

        msg_id = str(uuid.uuid4())
        now = _now()

        # Serialize content — if it's a multimodal list, extract text-only for DB
        # (images are stored as attachment refs, not inline base64)
        content = message.content
        if isinstance(content, list):
            content = self._extract_text_content(content)

        tool_calls_json = json.dumps(
            [tc.to_dict() for tc in message.tool_calls] if message.tool_calls else []
        )
        tool_results_json = json.dumps(
            [tr.to_dict() for tr in message.tool_results] if message.tool_results else []
        )

        await self._db.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_calls,
               tool_results, attachments, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id, session_id,
                message.role.value if isinstance(message.role, Role) else message.role,
                content,
                tool_calls_json,
                tool_results_json,
                json.dumps(attachments or []),
                json.dumps(message.metadata),
                now,
            ),
        )
        await self._db.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?", (now, session_id)
        )
        await self._db.commit()
        return msg_id

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Get messages for a session, ordered by creation time."""
        assert self._db is not None

        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at"
        params: list[Any] = [session_id]

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        messages: list[Message] = []
        async with self._db.execute(query, params) as cur:
            async for row in cur:
                messages.append(self._row_to_message(row))
        return messages

    async def get_messages_raw(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages as raw dicts (for WebSocket/API responses)."""
        assert self._db is not None

        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at"
        params: list[Any] = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        results: list[dict[str, Any]] = []
        async with self._db.execute(query, params) as cur:
            async for row in cur:
                results.append(dict(row))
        return results

    async def get_message_count(self, session_id: str) -> int:
        """Get total message count for a session."""
        assert self._db is not None

        async with self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session."""
        assert self._db is not None

        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._db.commit()

    # ── Streaming Support (for dashboard) ────────────────────────────────

    async def save_draft(self, session_id: str, draft_id: str, content: str) -> None:
        """Save or update a draft (in-progress streaming) message."""
        assert self._db is not None

        now = _now()
        await self._db.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_calls,
               tool_results, attachments, metadata, created_at)
               VALUES (?, ?, 'assistant', ?, '[]', '[]', '[]', '{}', ?)
               ON CONFLICT(id) DO UPDATE SET content = excluded.content""",
            (draft_id, session_id, content, now),
        )
        await self._db.commit()

    async def finalize_draft(
        self,
        draft_id: str,
        content: str,
        tool_calls: list | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Finalize a draft message (streaming complete)."""
        assert self._db is not None

        await self._db.execute(
            "UPDATE messages SET content = ?, tool_calls = ?, metadata = ? WHERE id = ?",
            (content, json.dumps(tool_calls or []),
             json.dumps(metadata or {}), draft_id),
        )
        await self._db.commit()

    # ── Summaries ────────────────────────────────────────────────────────

    async def get_rolling_summary(self, session_id: str) -> str:
        """Get the rolling summary for a session."""
        assert self._db is not None

        async with self._db.execute(
            "SELECT rolling_summary FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return (row["rolling_summary"] or "") if row else ""

    async def update_rolling_summary(self, session_id: str, summary: str) -> None:
        """Update the rolling summary for a session."""
        assert self._db is not None

        await self._db.execute(
            "UPDATE sessions SET rolling_summary = ? WHERE id = ?", (summary, session_id)
        )
        await self._db.commit()

    async def update_compaction_summary(self, session_id: str, summary: str) -> None:
        """Update the compaction summary for a session."""
        assert self._db is not None

        await self._db.execute(
            "UPDATE sessions SET compaction_summary = ? WHERE id = ?", (summary, session_id)
        )
        await self._db.commit()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert a DB row to a Session object."""
        return Session(
            id=row["id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            name=row["name"],
            rolling_summary=row["rolling_summary"] or "",
            compaction_summary=row["compaction_summary"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            last_active=row["last_active"],
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert a DB row to a Message object."""
        from march.core.message import ToolCall, ToolResult

        tool_calls = None
        tc_raw = row["tool_calls"]
        if tc_raw and tc_raw != "[]":
            try:
                tool_calls = [ToolCall.from_dict(tc) for tc in json.loads(tc_raw)]
            except (json.JSONDecodeError, TypeError):
                pass

        tool_results = None
        tr_raw = row["tool_results"] if "tool_results" in row.keys() else None
        if tr_raw and tr_raw != "[]":
            try:
                tool_results = [ToolResult.from_dict(tr) for tr in json.loads(tr_raw)]
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = {}
        if "metadata" in row.keys() and row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Message(
            role=row["role"],
            content=row["content"],
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata=metadata,
        )

    @staticmethod
    def _extract_text_content(content: list) -> str:
        """Extract text from multimodal content blocks for DB storage.

        Images are NOT stored — only their text labels.
        The actual image data lives on disk via AttachmentManager.
        """
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Don't store base64 — just note that an image was here
                    parts.append("[image attachment]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
