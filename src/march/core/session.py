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
    """A conversation session with rolling context support.

    Attributes:
        id: Unique session identifier (deterministic from source or random).
        source_type: Where the session originates (terminal, matrix, ws, acp).
        source_id: Source-specific identifier (room ID, workspace path, etc.).
        name: Human-readable session name.
        rolling_summary: Carry-over summary from last compaction.
        messages: Messages since last compaction (sliding window).
        dirty_messages: Messages not yet flushed to DB.
        last_processed_seq: Sequence number of last processed message.
        metadata: Session metadata (channel info, user prefs, etc.).
        created_at: Unix timestamp of session creation.
        last_active: Unix timestamp of last activity.
        is_active: Whether the session is active.
    """

    id: str = ""
    source_type: str = "terminal"
    source_id: str = ""
    name: str = ""
    rolling_summary: str = ""          # Carry-over summary from last compaction
    messages: list[Message] = field(default_factory=list)  # Messages since last compaction (sliding window)
    dirty_messages: list[Message] = field(default_factory=list)  # Not yet flushed to DB
    last_processed_seq: int = 0        # Sequence number of last processed message
    _seq_counter: int = 0              # Internal sequence counter
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_active: float = 0.0
    is_active: bool = True
    _flush_timer: float = 0.0         # Last flush timestamp

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
        if not self._flush_timer:
            self._flush_timer = time.time()

    def add_message(self, message: Message) -> None:
        """Add a single message to the session."""
        self._seq_counter += 1
        message.metadata = dict(message.metadata) if message.metadata else {}
        message.metadata["seq"] = self._seq_counter
        self.messages.append(message)
        self.dirty_messages.append(message)
        self.last_active = time.time()

    def add_system_message(self, content: str) -> None:
        """Add a system message (e.g., sub-agent completion)."""
        self.add_message(Message.system(content))

    def add_exchange(self, user_message: str | list, assistant_content: str) -> None:
        """Add a user/assistant exchange."""
        self.add_message(Message.user(user_message))
        self.add_message(Message.assistant(assistant_content))

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get all messages serialized for the LLM API.

        Prepends rolling_summary as a user message if non-empty.
        Flattens tool result messages into individual tool messages.
        """
        llm_messages: list[dict[str, Any]] = []

        # Prepend rolling summary if available
        if self.rolling_summary:
            llm_messages.append({
                "role": "user",
                "content": f"[Context Summary — rolling context from previous compaction]\n\n{self.rolling_summary}",
            })

        for msg in self.messages:
            llm_messages.extend(msg.to_llm_messages())
        return llm_messages

    def compact(self, new_rolling_summary: str) -> None:
        """Compact the session: replace messages with a rolling summary.

        Args:
            new_rolling_summary: The new deduped rolling summary.
        """
        self.rolling_summary = new_rolling_summary
        self.messages = []
        self.dirty_messages = []
        self.last_processed_seq = self._seq_counter

    def needs_flush(self) -> bool:
        """Check if dirty messages should be flushed to DB.

        Returns True if 10+ dirty messages or 10+ seconds since last flush.
        """
        if len(self.dirty_messages) >= 10:
            return True
        if self.dirty_messages and (time.time() - self._flush_timer) >= 10:
            return True
        return False

    def flush(self) -> list[Message]:
        """Return dirty messages and clear the buffer.

        Returns:
            List of messages that need to be persisted.
        """
        flushed = list(self.dirty_messages)
        self.dirty_messages = []
        self._flush_timer = time.time()
        return flushed

    def clear(self) -> None:
        """Clear session (for /reset). Resets everything."""
        self.messages.clear()
        self.dirty_messages.clear()
        self.rolling_summary = ""
        self.last_processed_seq = 0
        self._seq_counter = 0
        self._flush_timer = time.time()
        self.last_active = time.time()

    def reset(self) -> None:
        """Reset the session: clear everything and mark as reset."""
        self.clear()
        self.is_active = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict for persistence."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "name": self.name,
            "rolling_summary": self.rolling_summary,
            "messages": [msg.to_dict() for msg in self.messages],
            "last_processed_seq": self.last_processed_seq,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Deserialize session from dict."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session = cls(
            id=data["id"],
            source_type=data.get("source_type", "terminal"),
            source_id=data.get("source_id", ""),
            name=data.get("name", ""),
            rolling_summary=data.get("rolling_summary", ""),
            messages=messages,
            last_processed_seq=data.get("last_processed_seq", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
            last_active=data.get("last_active", 0.0),
        )
        # Restore _seq_counter from messages
        max_seq = 0
        for msg in messages:
            seq = msg.metadata.get("seq", 0) if msg.metadata else 0
            if seq > max_seq:
                max_seq = seq
        session._seq_counter = max(max_seq, session.last_processed_seq)
        return session

    # ── Legacy compatibility properties ──────────────────────────────
    # These allow code that references session.history to still work
    # during the transition period. They map to session.messages.

    @property
    def history(self) -> list[Message]:
        """Legacy alias for messages (backward compatibility)."""
        return self.messages

    @history.setter
    def history(self, value: list[Message]) -> None:
        """Legacy setter for messages (backward compatibility)."""
        self.messages = value


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
    last_processed_seq INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_sessions_source
    ON sessions(source_type, source_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq INTEGER DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS idx_messages_seq
    ON messages(session_id, seq);
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
            # Existing DB — run migrations
            await self._run_migrations()
        else:
            # Fresh DB — create tables from scratch
            await self._db.executescript(_SCHEMA)

        await self._db.commit()

    async def _run_migrations(self) -> None:
        """Run schema migrations for rolling context support."""
        assert self._db is not None

        # Drop old agent-specific tables (never used, safe to remove)
        for old_table in ("agent_sessions", "agent_messages", "agent_backup_messages"):
            await self._db.execute(f"DROP TABLE IF EXISTS {old_table}")

        # ── Check if we need the rolling context migration ───────────
        existing_session_cols = set()
        async with self._db.execute("PRAGMA table_info(sessions)") as cur:
            async for row in cur:
                existing_session_cols.add(row["name"])

        # If compaction_summary column exists, we need the full migration
        if "compaction_summary" in existing_session_cols:
            # Force clean rebuild: delete all sessions and messages
            await self._db.execute("DELETE FROM messages")
            await self._db.execute("DELETE FROM sessions")

            # Recreate sessions table without compaction_summary
            await self._db.execute("DROP TABLE IF EXISTS sessions")
            await self._db.execute("DROP TABLE IF EXISTS messages")
            await self._db.executescript(_SCHEMA)
            return

        # ── Sessions table migrations ────────────────────────────────
        session_migrations = {
            "source_type": "ALTER TABLE sessions ADD COLUMN source_type TEXT NOT NULL DEFAULT 'ws'",
            "source_id": "ALTER TABLE sessions ADD COLUMN source_id TEXT NOT NULL DEFAULT ''",
            "rolling_summary": "ALTER TABLE sessions ADD COLUMN rolling_summary TEXT DEFAULT ''",
            "last_processed_seq": "ALTER TABLE sessions ADD COLUMN last_processed_seq INTEGER DEFAULT 0",
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

        # ── Messages table migrations ────────────────────────────────
        existing_msg_cols = set()
        async with self._db.execute("PRAGMA table_info(messages)") as cur:
            async for row in cur:
                existing_msg_cols.add(row["name"])

        msg_migrations = {
            "seq": "ALTER TABLE messages ADD COLUMN seq INTEGER DEFAULT 0",
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

        try:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_seq ON messages(session_id, seq)"
            )
        except Exception:
            pass

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ── Session CRUD ─────────────────────────────────────────────────

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
               last_processed_seq, metadata, created_at, last_active, is_active)
               VALUES (?, ?, ?, ?, '', 0, ?, ?, ?, 1)""",
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
                # Load messages after last_processed_seq to rebuild sliding window
                messages = await self.get_messages_after_seq(session.id, session.last_processed_seq)
                session.messages = messages
                # Restore _seq_counter
                max_seq = 0
                for msg in messages:
                    seq = msg.metadata.get("seq", 0) if msg.metadata else 0
                    if seq > max_seq:
                        max_seq = seq
                session._seq_counter = max(max_seq, session.last_processed_seq)
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
                # Get last message preview + count
                async with self._db.execute(
                    "SELECT content, role FROM messages WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (row["id"],),
                ) as msg_cur:
                    msg = await msg_cur.fetchone()
                    s["last_message"] = msg["content"][:100] if msg else None
                    s["last_message_role"] = msg["role"] if msg else None
                async with self._db.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (row["id"],),
                ) as cnt_cur:
                    cnt = await cnt_cur.fetchone()
                    s["message_count"] = cnt[0] if cnt else 0
                results.append(s)
        return results

    async def save_session(self, session: Session) -> None:
        """Save session metadata (rolling_summary, last_processed_seq, metadata)."""
        assert self._db is not None

        await self._db.execute(
            """UPDATE sessions SET name = ?, rolling_summary = ?,
               last_processed_seq = ?, metadata = ?, last_active = ? WHERE id = ?""",
            (session.name, session.rolling_summary, session.last_processed_seq,
             json.dumps(session.metadata), _now(), session.id),
        )
        await self._db.commit()

    async def update_session(self, session: Session) -> None:
        """Update session metadata (alias for save_session)."""
        await self.save_session(session)

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
            "UPDATE sessions SET rolling_summary = '', last_processed_seq = 0 WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()

    # ── Message CRUD ─────────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        message: Message,
        attachments: list[dict] | None = None,
    ) -> str:
        """Add a message to a session. Returns the message ID."""
        assert self._db is not None

        now = _now()

        # Serialize content — if it's a multimodal list, extract text-only for DB
        content = message.content
        if isinstance(content, list):
            content = self._extract_text_content(content)

        tool_calls_json = json.dumps(
            [tc.to_dict() for tc in message.tool_calls] if message.tool_calls else []
        )
        tool_results_json = json.dumps(
            [tr.to_dict() for tr in message.tool_results] if message.tool_results else []
        )

        seq = message.metadata.get("seq", 0) if message.metadata else 0

        async with self._db.execute(
            """INSERT INTO messages (session_id, seq, role, content, tool_calls,
               tool_results, attachments, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                seq,
                message.role.value if isinstance(message.role, Role) else message.role,
                content,
                tool_calls_json,
                tool_results_json,
                json.dumps(attachments or []),
                json.dumps(message.metadata or {}),
                now,
            ),
        ) as cur:
            msg_id = str(cur.lastrowid)

        await self._db.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?", (now, session_id)
        )
        await self._db.commit()
        return msg_id

    async def flush_messages(self, session_id: str, messages: list[Message]) -> None:
        """Batch insert dirty messages with their seq numbers."""
        assert self._db is not None

        if not messages:
            return

        now = _now()
        rows = []
        for message in messages:
            content = message.content
            if isinstance(content, list):
                content = self._extract_text_content(content)

            tool_calls_json = json.dumps(
                [tc.to_dict() for tc in message.tool_calls] if message.tool_calls else []
            )
            tool_results_json = json.dumps(
                [tr.to_dict() for tr in message.tool_results] if message.tool_results else []
            )
            seq = message.metadata.get("seq", 0) if message.metadata else 0

            rows.append((
                session_id,
                seq,
                message.role.value if isinstance(message.role, Role) else message.role,
                content,
                tool_calls_json,
                tool_results_json,
                json.dumps([]),
                json.dumps(message.metadata or {}),
                now,
            ))

        await self._db.executemany(
            """INSERT INTO messages (session_id, seq, role, content, tool_calls,
               tool_results, attachments, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self._db.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?", (now, session_id)
        )
        await self._db.commit()

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

    async def get_messages_after_seq(
        self,
        session_id: str,
        last_processed_seq: int,
    ) -> list[Message]:
        """Get messages with seq > last_processed_seq to rebuild sliding window."""
        assert self._db is not None

        query = "SELECT * FROM messages WHERE session_id = ? AND seq > ? ORDER BY seq"
        params: list[Any] = [session_id, last_processed_seq]

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

    # ── Summaries ────────────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert a DB row to a Session object."""
        cols = row.keys()
        last_processed_seq = row["last_processed_seq"] if "last_processed_seq" in cols else 0
        return Session(
            id=row["id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            name=row["name"],
            rolling_summary=row["rolling_summary"] or "",
            last_processed_seq=last_processed_seq,
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

        # Restore seq from DB
        if "seq" in row.keys():
            metadata["seq"] = row["seq"]

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
