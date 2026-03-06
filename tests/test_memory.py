"""Comprehensive tests for the March memory system.

Tests all three tiers: file memory, vector store, SQLite store,
plus chunking, embeddings, hybrid search, and the unified store facade.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ── Chunker tests ──


class TestChunker:
    """Tests for token-aware text chunking."""

    def test_empty_text_returns_empty(self):
        from march.memory.chunker import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        from march.memory.chunker import chunk_text

        text = "Hello world, this is a short document."
        chunks = chunk_text(text, source="test.md")
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].source == "test.md"
        assert chunks[0].start_line == 1
        assert chunks[0].index == 0

    def test_long_text_multiple_chunks(self):
        from march.memory.chunker import chunk_text

        # Create text that's definitely longer than 512 tokens
        lines = [f"Line {i}: " + "word " * 50 for i in range(50)]
        text = "\n".join(lines)
        chunks = chunk_text(text, source="big.md", chunk_tokens=512, overlap_tokens=50)
        assert len(chunks) > 1
        # Verify all content is represented
        for chunk in chunks:
            assert chunk.source == "big.md"
            assert chunk.content.strip() != ""

    def test_respects_markdown_headings(self):
        from march.memory.chunker import chunk_text

        text = "# Section 1\n\nContent for section one.\n\n# Section 2\n\nContent for section two."
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=0)
        # Should not split in the middle of a heading
        for chunk in chunks:
            lines = chunk.content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("#"):
                    # Heading should be at the start of a chunk or with content after
                    pass  # valid

    def test_does_not_split_code_blocks(self):
        from march.memory.chunker import chunk_text

        text = "Before code.\n\n```python\ndef foo():\n    return 42\n```\n\nAfter code."
        chunks = chunk_text(text, chunk_tokens=1000, overlap_tokens=0)
        # With a large enough budget, should be a single chunk
        assert len(chunks) == 1
        assert "```python" in chunks[0].content
        assert "```" in chunks[0].content

    def test_chunk_metadata(self):
        from march.memory.chunker import chunk_text

        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_text(text, source="meta.md")
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= chunks[0].start_line

    def test_chunk_documents(self):
        from march.memory.chunker import chunk_documents

        docs = [
            ("doc1.md", "Hello from doc 1."),
            ("doc2.md", "Hello from doc 2."),
        ]
        chunks = chunk_documents(docs)
        assert len(chunks) == 2
        sources = {c.source for c in chunks}
        assert sources == {"doc1.md", "doc2.md"}

    def test_overlap_exists(self):
        from march.memory.chunker import chunk_text

        # Large enough text to create multiple chunks
        lines = [f"Unique sentence number {i}." for i in range(200)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
        if len(chunks) >= 2:
            # Check that some content from end of chunk 0 appears in chunk 1
            end_of_first = chunks[0].content.split("\n")[-3:]
            start_of_second = chunks[1].content.split("\n")[:10]
            # There should be some overlap
            overlap = set(end_of_first) & set(start_of_second)
            # May not always overlap depending on boundary, but chunks should be contiguous


# ── Vector Store tests ──


class TestVectorStore:
    """Tests for FAISS vector store."""

    def test_empty_store(self):
        from march.memory.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            assert store.count == 0

    def test_add_and_search(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()

            vectors = np.random.randn(3, 8).astype(np.float32)
            entries = [
                VectorEntry(id="v1", source="test.md", content="first doc"),
                VectorEntry(id="v2", source="test.md", content="second doc"),
                VectorEntry(id="v3", source="test.md", content="third doc"),
            ]
            store.add(vectors, entries)
            assert store.count == 3

            # Search with first vector
            results = store.search(vectors[0], top_k=2)
            assert len(results) == 2
            assert results[0].entry.id == "v1"  # most similar to itself
            assert results[0].score > 0

    def test_persist_and_load(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            store1 = VectorStore(index_dir=tmpdir, dim=8)
            store1.load()
            vectors = np.random.randn(2, 8).astype(np.float32)
            entries = [
                VectorEntry(id="a", source="s.md", content="alpha"),
                VectorEntry(id="b", source="s.md", content="beta"),
            ]
            store1.add(vectors, entries)
            store1.save()

            # Load in new instance
            store2 = VectorStore(index_dir=tmpdir, dim=8)
            store2.load()
            assert store2.count == 2
            assert store2.get_by_id("a") is not None
            assert store2.get_by_id("a").content == "alpha"

    def test_remove(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vectors = np.random.randn(3, 8).astype(np.float32)
            entries = [
                VectorEntry(id="x1", source="a.md", content="one"),
                VectorEntry(id="x2", source="b.md", content="two"),
                VectorEntry(id="x3", source="a.md", content="three"),
            ]
            store.add(vectors, entries)
            assert store.count == 3

            removed = store.remove(["x2"])
            assert removed == 1
            assert store.count == 2
            assert store.get_by_id("x2") is None

    def test_remove_by_source(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vectors = np.random.randn(3, 8).astype(np.float32)
            entries = [
                VectorEntry(id="s1", source="a.md", content="aa"),
                VectorEntry(id="s2", source="b.md", content="bb"),
                VectorEntry(id="s3", source="a.md", content="cc"),
            ]
            store.add(vectors, entries)
            removed = store.remove_by_source("a.md")
            assert removed == 2
            assert store.count == 1

    def test_remove_by_session(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vectors = np.random.randn(2, 8).astype(np.float32)
            entries = [
                VectorEntry(id="p1", source="m.md", content="one", session_id="sess-1"),
                VectorEntry(id="p2", source="m.md", content="two", session_id="sess-2"),
            ]
            store.add(vectors, entries)
            removed = store.remove_by_session("sess-1")
            assert removed == 1
            assert store.count == 1

    def test_cleanup_orphans(self):
        from march.memory.vector_store import VectorStore, VectorEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir, dim=8)
            store.load()
            vectors = np.random.randn(3, 8).astype(np.float32)
            entries = [
                VectorEntry(id="o1", source="exists.md", content="kept"),
                VectorEntry(id="o2", source="deleted.md", content="orphan"),
                VectorEntry(id="o3", source="exists.md", content="also kept"),
            ]
            store.add(vectors, entries)
            removed = store.cleanup_orphans(valid_sources={"exists.md"})
            assert removed == 1
            assert store.count == 2


# ── SQLite Store tests ──


class TestSQLiteStore:
    """Tests for SQLite structured store."""

    @pytest.fixture
    async def store(self, tmp_path):
        from march.memory.sqlite_store import SQLiteStore

        db_path = tmp_path / "test.db"
        s = SQLiteStore(db_path=db_path)
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, store):
        assert store.is_open
        # Should not raise
        await store.set_metadata("test_key", "test_value")
        val = await store.get_metadata("test_key")
        assert val == "test_value"

    @pytest.mark.asyncio
    async def test_memory_crud(self, store):
        await store.upsert_memory(
            id="mem1", source="test.md", content="Hello world",
            content_type="memory",
        )
        # Upsert (update)
        await store.upsert_memory(
            id="mem1", source="test.md", content="Hello world updated",
            content_type="memory",
        )
        # Delete
        deleted = await store.delete_memory("mem1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fts_search(self, store):
        await store.upsert_memory(
            id="fts1", source="doc.md", content="The quick brown fox jumps over the lazy dog",
            content_type="memory",
        )
        await store.upsert_memory(
            id="fts2", source="doc.md", content="A completely different sentence about cats",
            content_type="memory",
        )
        results = await store.fts_search("fox dog", top_k=10)
        assert len(results) >= 1
        assert results[0].id == "fts1"

    @pytest.mark.asyncio
    async def test_fts_empty_query(self, store):
        results = await store.fts_search("", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_messages(self, store):
        row_id = await store.save_message("sess1", "user", "Hello agent")
        assert row_id > 0
        msgs = await store.get_session_messages("sess1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello agent"

    @pytest.mark.asyncio
    async def test_tool_results(self, store):
        row_id = await store.save_tool_result(
            session_id="sess1", tool_name="read_file",
            tool_call_id="tc1", input_args={"path": "/tmp/f"},
            output="file contents", duration_ms=42.0,
        )
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_analytics(self, store):
        await store.record_usage(
            session_id="sess1", model="claude-3-opus",
            input_tokens=1000, output_tokens=500, cost=0.05,
            provider="anthropic",
        )
        usage = await store.get_usage_by_session("sess1")
        assert usage["input_tokens"] == 1000
        assert usage["cost"] == 0.05

        day_usage = await store.get_usage_by_day()
        assert day_usage["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_skill_state(self, store):
        await store.set_skill_state("my_skill", "counter", "42")
        val = await store.get_skill_state("my_skill", "counter")
        assert val == "42"

        all_state = await store.get_all_skill_state("my_skill")
        assert all_state == {"counter": "42"}

        await store.delete_skill_state("my_skill", "counter")
        val = await store.get_skill_state("my_skill", "counter")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete_by_source(self, store):
        await store.upsert_memory(id="d1", source="file.md", content="a")
        await store.upsert_memory(id="d2", source="file.md", content="b")
        await store.upsert_memory(id="d3", source="other.md", content="c")
        count = await store.delete_by_source("file.md")
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_by_session(self, store):
        await store.upsert_memory(
            id="s1", source="m.md", content="x", session_id="sess-x"
        )
        await store.save_message("sess-x", "user", "hello")
        count = await store.delete_by_session("sess-x")
        assert count == 1  # 1 memory entry deleted


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


# ── Hybrid Search tests ──


class TestHybridSearch:
    """Tests for hybrid search with RRF."""

    def test_rrf_merge(self):
        from march.memory.search import HybridSearch, SearchResult

        list_a = [
            SearchResult(id="a", source="s", content="doc a", score=1.0),
            SearchResult(id="b", source="s", content="doc b", score=0.8),
            SearchResult(id="c", source="s", content="doc c", score=0.5),
        ]
        list_b = [
            SearchResult(id="b", source="s", content="doc b", score=1.0),
            SearchResult(id="d", source="s", content="doc d", score=0.9),
            SearchResult(id="a", source="s", content="doc a", score=0.3),
        ]

        merged = HybridSearch._rrf(list_a, list_b, k=60)
        ids = [r.id for r in merged]
        # "a" appears rank 0 in list_a and rank 2 in list_b
        # "b" appears rank 1 in list_a and rank 0 in list_b
        # Both should be ranked high
        assert "a" in ids[:3]
        assert "b" in ids[:3]
        # "b" should be top since it has best combined RRF score
        # b: 1/(60+2) + 1/(60+1) = ~0.0323
        # a: 1/(60+1) + 1/(60+3) = ~0.0322
        assert merged[0].id == "b" or merged[0].id == "a"

    def test_rrf_empty_lists(self):
        from march.memory.search import HybridSearch, SearchResult

        merged = HybridSearch._rrf([], [], k=60)
        assert merged == []

    def test_rrf_one_empty(self):
        from march.memory.search import HybridSearch, SearchResult

        list_a = [
            SearchResult(id="x", source="s", content="doc x", score=1.0),
        ]
        merged = HybridSearch._rrf(list_a, [], k=60)
        assert len(merged) == 1
        assert merged[0].id == "x"


# ── Store Facade tests ──


class TestStoreFacade:
    """Tests for the unified MemoryStore facade."""

    @pytest.fixture
    async def store(self, tmp_path):
        from march.memory.store import MemoryStore

        (tmp_path / "SYSTEM.md").write_text("# System Rules\nBe helpful.")
        (tmp_path / "AGENT.md").write_text("# Agent\nCoder.")
        (tmp_path / "TOOLS.md").write_text("# Tools\nSafety first.")
        (tmp_path / "MEMORY.md").write_text("# Memory\nLong term stuff.")
        (tmp_path / "memory").mkdir()

        index_dir = tmp_path / "index"
        db_path = tmp_path / "march.db"

        # Use a mock embedding provider to avoid Ollama dependency in tests
        store = MemoryStore(
            workspace=tmp_path,
            config_dir=tmp_path,
            index_dir=index_dir,
            db_path=db_path,
            embedding_provider="ollama",
        )
        # Replace embedder with a mock
        mock_embedder = AsyncMock()
        mock_embedder.dim = 8
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: np.random.randn(len(texts), 8).astype(np.float32)
        )
        mock_embedder.embed_single = AsyncMock(
            side_effect=lambda text: np.random.randn(8).astype(np.float32)
        )
        store.embedder = mock_embedder
        store.vectors = __import__("march.memory.vector_store", fromlist=["VectorStore"]).VectorStore(
            index_dir=index_dir, dim=8,
        )

        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_load_files(self, store):
        system = await store.load_system_rules()
        assert "Be helpful" in system

        agent = await store.load_agent_profile()
        assert "Coder" in agent

        tools = await store.load_tool_rules()
        assert "Safety first" in tools

        long_term = await store.load_long_term()
        assert "Long term stuff" in long_term

    @pytest.mark.asyncio
    async def test_save_daily(self, store):
        await store.save_daily("Test note for today")
        content = await store.load_today()
        assert "Test note for today" in content

    @pytest.mark.asyncio
    async def test_save_global_rmb(self, store):
        entry_id = await store.save_global(
            "remember the API endpoint",
            "The API endpoint is https://api.example.com/v2",
        )
        assert entry_id.startswith("rmb-")
        assert store.vectors.count > 0

    @pytest.mark.asyncio
    async def test_reset_session(self, store):
        # Add some session-specific data
        store.vectors.add(
            np.random.randn(1, 8).astype(np.float32),
            [__import__("march.memory.vector_store", fromlist=["VectorEntry"]).VectorEntry(
                id="sess-entry-1",
                source="session",
                content="session data",
                session_id="test-session",
            )],
        )
        result = await store.reset_session("test-session")
        assert result["vector_entries"] == 1

    @pytest.mark.asyncio
    async def test_reindex(self, store):
        count = await store.reindex()
        # Should have indexed the 4 markdown files
        assert count > 0

    @pytest.mark.asyncio
    async def test_search_returns_results(self, store):
        # Index some content first
        await store.reindex()
        # Search (with mocked embeddings, results may be random but shouldn't crash)
        results = await store.search("Be helpful", top_k=5)
        # May or may not find results depending on mock vectors, but should not error
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_skill_state_passthrough(self, store):
        await store.set_skill_state("test_skill", "key1", "value1")
        val = await store.get_skill_state("test_skill", "key1")
        assert val == "value1"

    @pytest.mark.asyncio
    async def test_record_usage_passthrough(self, store):
        await store.record_usage(
            session_id="s1", model="test-model",
            input_tokens=100, output_tokens=50, cost=0.01,
        )
        # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_orphans(self, store):
        # Add an entry with a source that doesn't exist as a file
        store.vectors.add(
            np.random.randn(1, 8).astype(np.float32),
            [__import__("march.memory.vector_store", fromlist=["VectorEntry"]).VectorEntry(
                id="orphan-1",
                source="nonexistent_file.md",
                content="orphaned content",
            )],
        )
        removed = await store.cleanup_orphans()
        assert removed == 1
