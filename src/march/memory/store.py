"""Unified memory store for the March agent framework.

Two-tier facade:
  Tier 1: FileMemory (human-readable markdown files)
  Tier 2: SQLiteStore (structured data — sessions, messages, analytics)

MEMORY.md is loaded into context every turn and written to directly via /rmb.
Agent session logs serve as detailed memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from march.memory.file_memory import FileMemory
from march.memory.sqlite_store import SQLiteStore

logger = logging.getLogger("march.memory.store")

DEFAULT_DB_PATH = Path.home() / ".march" / "march.db"


class MemoryStore:
    """Unified memory interface (facade).

    Two-tier: FileMemory for human-readable markdown, SQLiteStore for
    structured data (sessions, messages, analytics).
    """

    def __init__(
        self,
        workspace: Path | None = None,
        config_dir: Path | None = None,
        system_rules_path: str = "SYSTEM.md",
        agent_profile_path: str = "AGENT.md",
        tool_rules_path: str = "TOOLS.md",
        memory_path: str = "MEMORY.md",
        db_path: Path | str | None = None,
    ):
        self.workspace = workspace or Path.cwd()

        # Tier 1: File memory
        self.files = FileMemory(
            workspace=self.workspace,
            config_dir=config_dir,
            system_rules_path=system_rules_path,
            agent_profile_path=agent_profile_path,
            tool_rules_path=tool_rules_path,
            memory_path=memory_path,
        )

        # Tier 2: SQLite store
        self.sqlite = SQLiteStore(
            db_path=Path(db_path) if db_path else DEFAULT_DB_PATH,
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory tiers."""
        if self._initialized:
            return

        # Load file memory (SYSTEM.md, AGENT.md, TOOLS.md, MEMORY.md)
        self.files.load_system_rules()
        self.files.load_agent_profile()
        self.files.load_tool_rules()
        self.files.load_long_term()
        self.files.load_today()

        # NOTE: SQLite store disabled — SessionStore (core/session.py) owns all persistence.
        # File-based memory (above) is sufficient for agent context.

        # Start file watching
        await self.files.start_watching()

        self._initialized = True
        logger.info("Memory store initialized: workspace=%s", self.workspace)

    async def close(self) -> None:
        """Clean up all resources."""
        await self.files.stop_watching()
        self._initialized = False

    # ── Tier 1: File Memory API ──

    async def load_system_rules(self) -> str:
        """Load SYSTEM.md — agent persona, voice, behavior rules."""
        return self.files.load_system_rules()

    async def load_agent_profile(self) -> str:
        """Load AGENT.md — agent specialization, role behavior."""
        return self.files.load_agent_profile()

    async def load_tool_rules(self) -> str:
        """Load TOOLS.md — tool usage guidance, safety rules."""
        return self.files.load_tool_rules()

    async def load_long_term(self) -> str:
        """Load MEMORY.md — curated long-term memory."""
        return self.files.load_long_term()

    async def load_session_memory(self, session_id: str) -> str:
        """Load all files recursively from ~/.march/memory/{session_id}/.

        Returns concatenated content of all .md/.txt files found,
        with filenames as headers. Returns empty string if directory
        doesn't exist or has no files.
        """
        memory_dir = Path.home() / ".march" / "memory" / session_id
        if not memory_dir.is_dir():
            return ""

        parts: list[str] = []
        for path in sorted(memory_dir.rglob("*")):
            if not path.is_file():
                continue
            # Only read text files
            if path.suffix.lower() not in (".md", ".txt", ".yaml", ".yml", ".json"):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    # Use relative path from memory dir as header
                    rel = path.relative_to(memory_dir)
                    parts.append(f"### {rel}\n\n{content}")
            except Exception:
                continue

        return "\n\n".join(parts)

    async def load_today(self) -> str:
        """Load today's daily memory file."""
        return self.files.load_today()

    async def save_daily(self, note: str) -> None:
        """Append a note to today's daily memory file."""
        self.files.save_daily(note)

    # ── /rmb: Append to MEMORY.md ──

    async def append_memory(self, content: str) -> None:
        """Append content to MEMORY.md."""
        self.files.save_memory(content)

    # ── /reset command: Clear session data ──

    async def reset_session(self, session_id: str) -> dict[str, int]:
        """Clear all session-specific data. Does NOT touch global memory.

        Args:
            session_id: The session to reset.

        Returns:
            Dict with counts of removed items.
        """
        sqlite_removed = await self.sqlite.delete_by_session(session_id)

        # Delete session memory files (facts.md, plan.md, etc.)
        session_memory_deleted = False
        try:
            from march.core.compaction import delete_session_memory
            session_memory_deleted = delete_session_memory(session_id)
        except Exception as e:
            logger.warning("Failed to delete session memory files for %s: %s", session_id, e)

        logger.info(
            "Reset session %s: %d sqlite entries removed, memory_files_deleted=%s",
            session_id, sqlite_removed, session_memory_deleted,
        )
        return {"sqlite_entries": sqlite_removed, "memory_deleted": int(session_memory_deleted)}

    # ── Analytics passthrough ──

    async def record_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        provider: str = "",
    ) -> None:
        """Record token usage for analytics."""
        await self.sqlite.record_usage(
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            provider=provider,
        )
