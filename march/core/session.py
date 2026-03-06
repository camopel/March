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


@dataclass
class Session:
    """A conversation session.

    Attributes:
        id: Unique session identifier (deterministic from source or random).
        source_type: Where the session originates (terminal, matrix, vscode, acp).
        source_id: Source-specific identifier (room ID, workspace path, etc.).
        history: Ordered list of messages in this session.
        metadata: Session metadata (channel info, user prefs, etc.).
        created_at: Unix timestamp of session creation.
        last_active: Unix timestamp of last activity.
        state: Session state ("active" or "reset").
    """

    id: str = ""
    source_type: str = "terminal"
    source_id: str = ""
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_active: float = 0.0
    state: str = "active"
    # Compaction state
    compaction_summary: str = ""
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
        self.last_active = time.time()

    def reset(self) -> None:
        """Reset the session: clear history, backup, summary, and mark as reset."""
        self.history.clear()
        self.backup_history.clear()
        self.compaction_summary = ""
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
            "history": [msg.to_dict() for msg in self.history],
            "backup_history": [msg.to_dict() for msg in self.backup_history],
            "compaction_summary": self.compaction_summary,
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
            history=history,
            backup_history=backup,
            compaction_summary=data.get("compaction_summary", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
            last_active=data.get("last_active", 0.0),
        )


class SessionStore:
    """SQLite-backed session persistence.

    Stores session metadata and full conversation history.
    """

    def __init__(self, db_path: str | Path = "~/.march/march.db"):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and tables if they don't exist.

        NOTE: Agent-specific tables (agent_sessions, agent_messages, etc.)
        are disabled. The ws_proxy plugin owns sessions/messages tables directly.
        Re-enable if the core agent loop is used independently.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()

        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def save_session(self, session: Session) -> None:
        """Save or update a session and its messages."""
        assert self._db is not None, "SessionStore not initialized"

        await self._db.execute(
            """
            INSERT INTO agent_sessions (id, source_type, source_id, metadata, created_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                metadata = excluded.metadata,
                last_active = excluded.last_active
            """,
            (
                session.id,
                session.source_type,
                session.source_id,
                json.dumps(session.metadata),
                session.created_at,
                session.last_active,
            ),
        )

        # Delete existing messages and re-insert (simpler than diffing)
        await self._db.execute(
            "DELETE FROM agent_messages WHERE session_id = ?",
            (session.id,),
        )

        # If session was reset (no backup), clear backup too
        if not session.backup_history:
            await self._db.execute(
                "DELETE FROM agent_backup_messages WHERE session_id = ?",
                (session.id,),
            )

        for msg in session.history:
            tool_calls_json = (
                json.dumps([tc.to_dict() for tc in msg.tool_calls])
                if msg.tool_calls
                else None
            )
            tool_results_json = (
                json.dumps([tr.to_dict() for tr in msg.tool_results])
                if msg.tool_results
                else None
            )
            await self._db.execute(
                """
                INSERT INTO agent_messages
                    (session_id, role, content, tool_calls, tool_results, name, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    msg.role.value if isinstance(msg.role, Role) else msg.role,
                    json.dumps(msg.content) if isinstance(msg.content, list) else msg.content,
                    tool_calls_json,
                    tool_results_json,
                    msg.name,
                    json.dumps(msg.metadata),
                    time.time(),
                ),
            )

        # Save backup messages (append-only — don't delete existing backups)
        # Only insert new backup messages that aren't already stored
        existing_backup_count = 0
        async with self._db.execute(
            "SELECT COUNT(*) FROM agent_backup_messages WHERE session_id = ?",
            (session.id,),
        ) as cursor:
            row = await cursor.fetchone()
            existing_backup_count = row[0] if row else 0

        # Only insert backup messages beyond what's already stored
        new_backups = session.backup_history[existing_backup_count:]
        for msg in new_backups:
            tool_calls_json = (
                json.dumps([tc.to_dict() for tc in msg.tool_calls])
                if msg.tool_calls
                else None
            )
            tool_results_json = (
                json.dumps([tr.to_dict() for tr in msg.tool_results])
                if msg.tool_results
                else None
            )
            await self._db.execute(
                """
                INSERT INTO agent_backup_messages
                    (session_id, role, content, tool_calls, tool_results, name, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    msg.role.value if isinstance(msg.role, Role) else msg.role,
                    json.dumps(msg.content) if isinstance(msg.content, list) else msg.content,
                    tool_calls_json,
                    tool_results_json,
                    msg.name,
                    json.dumps(msg.metadata),
                    time.time(),
                ),
            )

        # Save compaction summary in session metadata
        if session.compaction_summary:
            meta = json.loads(
                (await (await self._db.execute(
                    "SELECT metadata FROM agent_sessions WHERE id = ?", (session.id,)
                )).fetchone())[0]
            )
            meta["compaction_summary"] = session.compaction_summary
            await self._db.execute(
                "UPDATE agent_sessions SET metadata = ? WHERE id = ?",
                (json.dumps(meta), session.id),
            )

        await self._db.commit()

    async def load_session(self, session_id: str) -> Session | None:
        """Load a session by ID, including all messages."""
        assert self._db is not None, "SessionStore not initialized"

        async with self._db.execute(
            "SELECT * FROM agent_sessions WHERE id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

        session = Session(
            id=row[0],
            source_type=row[1],
            source_id=row[2],
            metadata=json.loads(row[3]),
            created_at=row[4],
            last_active=row[5],
        )

        # Restore compaction summary from metadata
        session.compaction_summary = session.metadata.get("compaction_summary", "")

        # Load active messages
        async with self._db.execute(
            "SELECT role, content, tool_calls, tool_results, name, metadata FROM agent_messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ) as cursor:
            async for row in cursor:
                from march.core.message import ToolCall, ToolResult

                tool_calls = None
                if row[2]:
                    tool_calls = [ToolCall.from_dict(tc) for tc in json.loads(row[2])]
                tool_results = None
                if row[3]:
                    tool_results = [ToolResult.from_dict(tr) for tr in json.loads(row[3])]

                # Deserialize content: JSON list (multimodal) or plain string
                raw_content = row[1]
                if isinstance(raw_content, str) and raw_content.startswith("["):
                    try:
                        raw_content = json.loads(raw_content)
                    except (json.JSONDecodeError, TypeError):
                        pass

                msg = Message(
                    role=row[0],
                    content=raw_content,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    name=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                session.history.append(msg)

        # Load backup messages
        async with self._db.execute(
            "SELECT role, content, tool_calls, tool_results, name, metadata FROM agent_backup_messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ) as cursor:
            async for row in cursor:
                from march.core.message import ToolCall, ToolResult

                tool_calls = None
                if row[2]:
                    tool_calls = [ToolCall.from_dict(tc) for tc in json.loads(row[2])]
                tool_results = None
                if row[3]:
                    tool_results = [ToolResult.from_dict(tr) for tr in json.loads(row[3])]

                # Deserialize content: JSON list (multimodal) or plain string
                raw_content_b = row[1]
                if isinstance(raw_content_b, str) and raw_content_b.startswith("["):
                    try:
                        raw_content_b = json.loads(raw_content_b)
                    except (json.JSONDecodeError, TypeError):
                        pass

                msg = Message(
                    role=row[0],
                    content=raw_content_b,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    name=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                session.backup_history.append(msg)

        return session

    async def find_by_source(self, source_type: str, source_id: str) -> Session | None:
        """Find a session by its source type and ID."""
        assert self._db is not None, "SessionStore not initialized"

        async with self._db.execute(
            "SELECT id FROM agent_sessions WHERE source_type = ? AND source_id = ? AND state = 'active' "
            "ORDER BY last_active DESC LIMIT 1",
            (source_type, source_id),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return await self.load_session(row[0])

    async def list_sessions(
        self,
        source_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List sessions with optional filters."""
        assert self._db is not None, "SessionStore not initialized"

        query = "SELECT id, source_type, source_id, created_at, last_active, state FROM agent_sessions"
        params: list[Any] = []

        if source_type:
            query += " WHERE source_type = ?"
            params.append(source_type)

        query += " ORDER BY last_active DESC LIMIT ?"
        params.append(limit)

        results: list[dict[str, Any]] = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(
                    {
                        "id": row[0],
                        "source_type": row[1],
                        "source_id": row[2],
                        "created_at": row[3],
                        "last_active": row[4],
                        "state": row[5],
                    }
                )
        return results

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        assert self._db is not None, "SessionStore not initialized"

        cursor = await self._db.execute(
            "DELETE FROM agent_sessions WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_or_create_by_source(
        self,
        source_type: str,
        source_id: str,
    ) -> Session:
        """Get an existing session by source, or create a new one.

        Uses deterministic session IDs so the same source always maps
        to the same session, enabling resume on reconnect.
        """
        existing = await self.find_by_source(source_type, source_id)
        if existing:
            return existing

        session = Session(source_type=source_type, source_id=source_id)
        await self.save_session(session)
        return session

    async def reset_session(self, source_type: str, source_id: str) -> bool:
        """Reset a session by source. Clears history but keeps the session record.

        Returns True if a session was found and reset.
        """
        session = await self.find_by_source(source_type, source_id)
        if not session:
            return False

        session.reset()
        await self.save_session(session)
        return True
