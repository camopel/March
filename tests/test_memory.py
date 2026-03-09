"""Tests for the March memory system.

Tests SQLite store and file memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── SQLite Store tests ──


class TestSQLiteStore:
    """Tests for SQLite structured store (minimal stub)."""

    @pytest.fixture
    async def store(self, tmp_path):
        from march.memory.sqlite_store import SQLiteStore

        db_path = tmp_path / "test.db"
        s = SQLiteStore(db_path=db_path)
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_is_stub(self, store):
        """SQLiteStore is a minimal stub — is_open returns False."""
        assert store.is_open is False

    @pytest.mark.asyncio
    async def test_delete_by_session_noop(self, store):
        result = await store.delete_by_session("nonexistent")
        assert result == 0

    @pytest.mark.asyncio
    async def test_record_usage_noop(self, store):
        # Should not raise
        await store.record_usage(
            session_id="s1", model="test-model",
            input_tokens=100, output_tokens=50, cost=0.01,
        )


# ── File Memory tests ──


class TestFileMemory:
    """Tests for file-based memory (Tier 1)."""

    def test_load_files(self, tmp_path):
        from march.memory.file_memory import FileMemory

        (tmp_path / "SYSTEM.md").write_text("# System\nBe helpful.")
        (tmp_path / "AGENT.md").write_text("# Agent\nCoder.")
        (tmp_path / "TOOLS.md").write_text("# Tools\nUse trash.")
        (tmp_path / "MEMORY.md").write_text("# Memory\nRemember stuff.")
        (tmp_path / "memory").mkdir()
        from datetime import date
        today = date.today().isoformat()
        (tmp_path / "memory" / f"{today}.md").write_text("Today's notes.")

        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        assert "Be helpful" in fm.load_system_rules()
        assert "Coder" in fm.load_agent_profile()
        assert "Use trash" in fm.load_tool_rules()
        assert "Remember stuff" in fm.load_long_term()
        assert "Today's notes" in fm.load_today()

    def test_missing_files_use_templates(self, tmp_path):
        from march.memory.file_memory import FileMemory

        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        # No files in workspace or config_dir, but package templates exist
        # so system_rules and agent_profile should have content from templates
        assert fm.load_system_rules() != ""
        assert fm.load_agent_profile() != ""
        # MEMORY.md is mutable-only — no template fallback for load_long_term
        assert fm.load_long_term() == ""

    def test_save_daily(self, tmp_path):
        from march.memory.file_memory import FileMemory

        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        fm.save_daily("Note 1")
        fm.save_daily("Note 2")
        content = fm.load_today()
        assert "Note 1" in content
        assert "Note 2" in content

    def test_get_all_watched_files(self, tmp_path):
        from march.memory.file_memory import FileMemory

        (tmp_path / "SYSTEM.md").write_text("system")
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "2025-01-01.md").write_text("day one")

        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        files = fm.get_all_watched_files()
        sources = {f[0] for f in files}
        assert "SYSTEM.md" in sources
        assert "memory/2025-01-01.md" in sources

    def test_check_needs_reindex(self, tmp_path):
        from march.memory.file_memory import FileMemory

        (tmp_path / "SYSTEM.md").write_text("original")
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        fm.load_system_rules()

        # Modify file
        (tmp_path / "SYSTEM.md").write_text("modified")
        changed = fm.check_needs_reindex()
        assert "SYSTEM.md" in changed
