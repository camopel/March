"""SQLite structured store for the March memory system (Tier 2).

Minimal store — session/message persistence is owned by SessionStore (core/session.py).
This module provides the SQLiteStore interface for MemoryStore compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("march.memory.sqlite_store")

DEFAULT_DB_PATH = Path.home() / ".march" / "march.db"


class SQLiteStore:
    """Minimal SQLite store stub.

    Session and message persistence is handled by the unified SessionStore
    in core/session.py. This class exists for MemoryStore interface compatibility.
    Methods are no-ops or return safe defaults.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    @property
    def is_open(self) -> bool:
        return False

    async def initialize(self) -> None:
        """No-op — SessionStore owns the database."""
        pass

    async def close(self) -> None:
        """No-op."""
        pass

    async def delete_by_session(self, session_id: str) -> int:
        """No-op — session cleanup is handled by SessionStore."""
        return 0

    async def record_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        provider: str = "",
    ) -> None:
        """No-op — cost tracking is handled by the CostPlugin in-memory."""
        pass

    async def get_skill_state(self, skill_name: str, key: str) -> str | None:
        """No-op — skill state not currently used."""
        return None

    async def set_skill_state(self, skill_name: str, key: str, value: str) -> None:
        """No-op — skill state not currently used."""
        pass
