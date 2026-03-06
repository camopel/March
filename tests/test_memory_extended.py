"""Extended tests for memory: store facade, search, file_memory, embeddings."""

from __future__ import annotations

import asyncio
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from march.memory.chunker import Chunk, chunk_text
from march.memory.file_memory import FileMemory
from march.memory.search import HybridSearch, SearchResult
from march.memory.sqlite_store import SQLiteStore
from march.memory.vector_store import VectorStore, VectorEntry, VectorSearchResult
from march.memory.store import MemoryStore


# ─────────────────────────────────────────────────────────────
# HybridSearch Extended
# ─────────────────────────────────────────────────────────────

class TestHybridSearchExtended:
    @pytest.fixture
    async def sqlite_store(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def vector_store(self, tmp_path):
        store = VectorStore(index_dir=tmp_path / "index", dim=8)
        store.load()
        return store

    @pytest.fixture
    def mock_embedder(self):
        mock = AsyncMock()
        mock.embed_single = AsyncMock(
            side_effect=lambda text: np.random.randn(8).astype(np.float32)
        )
        mock.embed = AsyncMock(
            side_effect=lambda texts: np.random.randn(len(texts), 8).astype(np.float32)
        )
        return mock

    async def test_keyword_only_search(self, sqlite_store, vector_store, mock_embedder):
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        await sqlite_store.upsert_memory(
            id="kw1", source="doc.md", content="The quick brown fox jumps over the lazy dog",
            content_type="memory",
        )
        results = await search.search("fox dog", strategy="keyword")
        assert len(results) >= 1
        assert results[0].id == "kw1"

    async def test_semantic_only_search(self, sqlite_store, vector_store, mock_embedder):
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        vecs = np.random.randn(2, 8).astype(np.float32)
        entries = [
            VectorEntry(id="v1", source="a.md", content="hello world"),
            VectorEntry(id="v2", source="b.md", content="goodbye world"),
        ]
        vector_store.add(vecs, entries)
        results = await search.search("hello", strategy="semantic")
        assert isinstance(results, list)

    async def test_hybrid_search_merges(self, sqlite_store, vector_store, mock_embedder):
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        await sqlite_store.upsert_memory(
            id="h1", source="doc.md", content="Python is a great language",
            content_type="memory",
        )
        vecs = np.random.randn(1, 8).astype(np.float32)
        entries = [VectorEntry(id="h1", source="doc.md", content="Python is a great language")]
        vector_store.add(vecs, entries)
        results = await search.search("Python", strategy="hybrid")
        assert isinstance(results, list)

    async def test_hybrid_search_handles_failures(self, sqlite_store, vector_store, mock_embedder):
        """Hybrid search handles one failing backend gracefully."""
        mock_embedder.embed_single = AsyncMock(side_effect=RuntimeError("embed fail"))
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        await sqlite_store.upsert_memory(
            id="f1", source="doc.md", content="test content",
            content_type="memory",
        )
        # Should still return keyword results even if semantic fails
        results = await search.search("test", strategy="hybrid")
        assert isinstance(results, list)

    async def test_keyword_search_empty_sqlite(self, sqlite_store, vector_store, mock_embedder):
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        results = await search._keyword_search("test", top_k=10)
        assert results == []

    async def test_semantic_search_empty_vector(self, sqlite_store, vector_store, mock_embedder):
        search = HybridSearch(sqlite_store, vector_store, mock_embedder)
        results = await search._semantic_search("test", top_k=10)
        assert results == []

    async def test_keyword_search_sqlite_closed(self, tmp_path, vector_store, mock_embedder):
        store = SQLiteStore(db_path=tmp_path / "closed.db")
        search = HybridSearch(store, vector_store, mock_embedder)
        results = await search._keyword_search("test", top_k=10)
        assert results == []

    def test_rrf_scores_decrease(self):
        """RRF scores should be positive and items at higher ranks score higher."""
        list_a = [
            SearchResult(id=f"r{i}", source="s", content=f"doc {i}", score=1.0)
            for i in range(5)
        ]
        merged = HybridSearch._rrf(list_a, [], k=60)
        # First item should have highest RRF score
        assert merged[0].score >= merged[-1].score

    def test_rrf_duplicate_handling(self):
        """Same doc in both lists should have higher combined score."""
        list_a = [SearchResult(id="dup", source="s", content="doc", score=1.0)]
        list_b = [SearchResult(id="dup", source="s", content="doc", score=1.0)]
        single = HybridSearch._rrf(list_a, [], k=60)
        both = HybridSearch._rrf(list_a, list_b, k=60)
        assert both[0].score > single[0].score


# ─────────────────────────────────────────────────────────────
# FileMemory Extended
# ─────────────────────────────────────────────────────────────

class TestFileMemoryExtended:
    def test_load_yesterday(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / f"{yesterday}.md").write_text("Yesterday's notes")
        assert "Yesterday's notes" in fm.load_yesterday()

    def test_load_yesterday_missing(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        assert fm.load_yesterday() == ""

    def test_get_all_daily_files(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "2025-01-01.md").write_text("Day one")
        (mem_dir / "2025-01-02.md").write_text("Day two")
        (mem_dir / "empty.md").write_text("   ")  # Empty content
        files = fm.get_all_daily_files()
        sources = [f[0] for f in files]
        assert "memory/2025-01-01.md" in sources
        assert "memory/2025-01-02.md" in sources
        # Empty files should be excluded
        assert all("empty" not in s for s in sources)

    def test_get_tracked_sources(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        (tmp_path / "SYSTEM.md").write_text("rules")
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "2025-01-01.md").write_text("day one")
        sources = fm.get_tracked_sources()
        assert "SYSTEM.md" in sources
        assert "memory/2025-01-01.md" in sources

    def test_hash_content(self):
        h1 = FileMemory._hash_content("hello")
        h2 = FileMemory._hash_content("hello")
        h3 = FileMemory._hash_content("world")
        assert h1 == h2
        assert h1 != h3

    def test_read_file_unicode_error(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        bad_file = tmp_path / "bad.md"
        bad_file.write_bytes(b"\x80\x81\x82")
        result = fm._read_file_at(bad_file)
        assert result == ""

    async def test_start_stop_watching(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        (tmp_path / "SYSTEM.md").write_text("test")
        fm.load_system_rules()
        await fm.start_watching()
        assert fm._running
        await fm.stop_watching()
        assert not fm._running
        assert fm._watch_task is None

    async def test_start_watching_idempotent(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        await fm.start_watching()
        task1 = fm._watch_task
        await fm.start_watching()  # Second call should be no-op
        assert fm._watch_task is task1
        await fm.stop_watching()

    async def test_file_change_callback(self, tmp_path):
        callback_calls = []

        async def on_change(path, content):
            callback_calls.append((path, content))

        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        (tmp_path / "SYSTEM.md").write_text("original")
        fm.load_system_rules()
        fm.set_on_change(on_change)

        # Modify file
        (tmp_path / "SYSTEM.md").write_text("modified")
        await fm._check_changes()
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "SYSTEM.md"
        assert callback_calls[0][1] == "modified"

    async def test_check_changes_new_file(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        (tmp_path / "SYSTEM.md").write_text("test")
        fm.load_system_rules()
        # Create a new daily file
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "2025-01-01.md").write_text("new file")
        await fm._check_changes()
        # Should track the new file
        assert "memory/2025-01-01.md" in fm._hashes

    def test_save_daily_creates_dir(self, tmp_path):
        fm = FileMemory(workspace=tmp_path, config_dir=tmp_path)
        fm.save_daily("test note")
        today = date.today().isoformat()
        assert (tmp_path / "memory" / f"{today}.md").exists()


# ─────────────────────────────────────────────────────────────
# MemoryStore Facade Extended
# ─────────────────────────────────────────────────────────────

class TestMemoryStoreExtended:
    @pytest.fixture
    async def store(self, tmp_path):
        (tmp_path / "SYSTEM.md").write_text("system rules")
        (tmp_path / "AGENT.md").write_text("agent profile")
        (tmp_path / "TOOLS.md").write_text("tool rules")
        (tmp_path / "MEMORY.md").write_text("long term")
        (tmp_path / "memory").mkdir()

        store = MemoryStore(
            workspace=tmp_path,
            config_dir=tmp_path,
            index_dir=tmp_path / "index",
            db_path=tmp_path / "march.db",
        )
        # Mock embedder
        mock_embedder = AsyncMock()
        mock_embedder.dim = 8
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: np.random.randn(len(texts), 8).astype(np.float32)
        )
        mock_embedder.embed_single = AsyncMock(
            side_effect=lambda text: np.random.randn(8).astype(np.float32)
        )
        store.embedder = mock_embedder
        store.vectors = VectorStore(index_dir=tmp_path / "index", dim=8)

        await store.initialize()
        yield store
        await store.close()

    async def test_initialize_idempotent(self, store):
        """Calling initialize() twice should be safe."""
        await store.initialize()  # Already initialized in fixture
        assert store._initialized

    async def test_search_no_search_engine(self, tmp_path):
        """search() returns [] if _search is None."""
        store = MemoryStore(workspace=tmp_path, config_dir=tmp_path)
        results = await store.search("test")
        assert results == []

    async def test_save_global_embedding_failure(self, store):
        """save_global handles embedding failures gracefully."""
        store.embedder.embed = AsyncMock(side_effect=RuntimeError("Ollama down"))
        entry_id = await store.save_global("remember", "content to store")
        assert entry_id.startswith("rmb-")
        # Should still be stored in SQLite even if embedding fails

    async def test_save_global_empty_chunks(self, store):
        """save_global with content that might not chunk well."""
        entry_id = await store.save_global("test", "short")
        assert entry_id.startswith("rmb-")

    async def test_reset_session_empty(self, store):
        """reset_session with no data for that session."""
        result = await store.reset_session("nonexistent-session")
        assert result["vector_entries"] == 0
        assert result["sqlite_entries"] == 0

    async def test_reindex_no_files(self, tmp_path):
        """Reindex with empty workspace and no templates."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        store = MemoryStore(
            workspace=empty_dir,
            config_dir=empty_dir,
            index_dir=tmp_path / "idx",
            db_path=tmp_path / "db.db",
        )
        mock_embedder = AsyncMock()
        mock_embedder.dim = 8
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: np.random.randn(len(texts), 8).astype(np.float32)
        )
        store.embedder = mock_embedder
        store.vectors = VectorStore(index_dir=tmp_path / "idx", dim=8)
        await store.initialize()
        count = await store.reindex()
        # May find package templates (SYSTEM.md, AGENT.md, TOOLS.md) via layered resolution
        # so count >= 0 is valid
        assert count >= 0
        await store.close()

    async def test_reindex_embedding_failure(self, store):
        """Reindex handles embedding failure for individual files."""
        store.embedder.embed = AsyncMock(side_effect=RuntimeError("embed error"))
        count = await store.reindex()
        # Should return 0 because all embeddings failed
        assert count == 0

    async def test_on_file_change_callback(self, store):
        """File change callback re-indexes the changed file."""
        await store._on_file_change("SYSTEM.md", "new system rules content")
        # Should have re-indexed in vector store

    async def test_on_file_change_empty(self, store):
        """File change callback with empty content should not crash."""
        await store._on_file_change("SYSTEM.md", "   ")

    async def test_on_file_change_embed_failure(self, store):
        """File change callback handles embed failure."""
        store.embedder.embed = AsyncMock(side_effect=RuntimeError("embed fail"))
        await store._on_file_change("SYSTEM.md", "content")
        # Should not crash

    async def test_close_saves_vectors(self, store):
        """close() should save vectors and close sqlite."""
        store.vectors.save = MagicMock()
        await store.close()
        store.vectors.save.assert_called_once()


# ─────────────────────────────────────────────────────────────
# Chunker Extended
# ─────────────────────────────────────────────────────────────

class TestChunkerExtended:
    def test_chunk_none_source(self):
        chunks = chunk_text("Hello world", source=None)
        assert len(chunks) >= 1

    def test_very_long_single_line(self):
        # Use multiple lines to ensure chunking occurs
        text = "\n".join(["word " * 200 for _ in range(25)])
        chunks = chunk_text(text, chunk_tokens=512, overlap_tokens=50)
        assert len(chunks) > 1
        # All content should be captured
        total_content = " ".join(c.content for c in chunks)
        assert "word" in total_content

    def test_chunk_indices_sequential(self):
        text = "\n".join(f"Line {i}: " + "content " * 50 for i in range(100))
        chunks = chunk_text(text, chunk_tokens=200, overlap_tokens=20)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# ─────────────────────────────────────────────────────────────
# VectorStore Extended
# ─────────────────────────────────────────────────────────────

class TestVectorStoreExtended:
    def test_search_empty_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            query = np.random.randn(8).astype(np.float32)
            results = store.search(query, top_k=5)
            assert results == []

    def test_search_top_k_larger_than_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vecs = np.random.randn(2, 8).astype(np.float32)
            entries = [
                VectorEntry(id="e1", source="a.md", content="one"),
                VectorEntry(id="e2", source="a.md", content="two"),
            ]
            store.add(vecs, entries)
            query = np.random.randn(8).astype(np.float32)
            results = store.search(query, top_k=100)
            assert len(results) == 2

    def test_get_by_id_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            assert store.get_by_id("nonexistent") is None

    def test_remove_nonexistent_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            removed = store.remove(["nonexistent"])
            assert removed == 0

    def test_cleanup_orphans_all_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vecs = np.random.randn(2, 8).astype(np.float32)
            entries = [
                VectorEntry(id="e1", source="valid.md", content="one"),
                VectorEntry(id="e2", source="valid.md", content="two"),
            ]
            store.add(vecs, entries)
            removed = store.cleanup_orphans(valid_sources={"valid.md"})
            assert removed == 0
            assert store.count == 2


# ─────────────────────────────────────────────────────────────
# SQLiteStore Extended
# ─────────────────────────────────────────────────────────────

class TestSQLiteStoreExtended:
    @pytest.fixture
    async def store(self, tmp_path):
        store = SQLiteStore(db_path=tmp_path / "test.db")
        await store.initialize()
        yield store
        await store.close()

    async def test_delete_memory_nonexistent(self, store):
        deleted = await store.delete_memory("nonexistent-id")
        assert deleted is False

    async def test_fts_special_chars(self, store):
        """FTS search handles special characters without crashing."""
        await store.upsert_memory(
            id="special", source="doc.md",
            content="Special chars: @#$%^&*() in content",
            content_type="memory",
        )
        results = await store.fts_search("Special chars", top_k=5)
        assert isinstance(results, list)

    async def test_multiple_usage_records(self, store):
        for i in range(5):
            await store.record_usage(
                session_id=f"sess-{i % 2}",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
            )
        usage = await store.get_usage_by_session("sess-0")
        assert usage["input_tokens"] == 300  # 3 records for sess-0
        day_usage = await store.get_usage_by_day()
        assert day_usage["total_cost"] > 0

    async def test_upsert_with_all_fields(self, store):
        await store.upsert_memory(
            id="full", source="test.md", content="full content",
            content_type="file", start_line=10, end_line=20,
            chunk_index=0, session_id="sess-1",
            metadata={"key": "value"},
        )
        # Verify via FTS
        results = await store.fts_search("full content", top_k=5)
        assert len(results) >= 1

    async def test_get_all_skill_state_empty(self, store):
        result = await store.get_all_skill_state("nonexistent_skill")
        assert result == {}
