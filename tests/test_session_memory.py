"""Tests for session memory and compaction integration."""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from march.core.compaction import (
    COMPACTION_THRESHOLD,
    FACTS_BUDGET_RATIO,
    MIN_RECENT_KEEP,
    PLAN_BUDGET_RATIO,
    SAFETY_MARGIN,
    SUMMARY_BUDGET_RATIO,
    _compress_facts,
    _load_session_memory,
    build_summary_prompt,
    compact_messages,
    delete_session_memory,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    extract_session_memory,
    needs_compaction,
    split_for_compaction,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_messages(count: int, chars_each: int = 200) -> list[dict]:
    """Create N messages alternating user/assistant."""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"Message {i}: " + "x" * chars_each
        msgs.append({"role": role, "content": text})
    return msgs


def _make_session_dir(session_id: str) -> Path:
    """Create a session memory directory under ~/.march/memory/."""
    d = Path.home() / ".march" / "memory" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_session(session_id: str):
    d = Path.home() / ".march" / "memory" / session_id
    if d.exists():
        shutil.rmtree(d)


# ── Token Estimation ─────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 1  # min 1

    def test_estimate_tokens_proportional(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 1000)
        assert long > short

    def test_estimate_message_tokens_text(self):
        msg = {"role": "user", "content": "hello world"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_estimate_message_tokens_multimodal(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
        ]}
        tokens = estimate_message_tokens(msg)
        assert tokens >= 1600  # Image = 1600 tokens

    def test_estimate_message_tokens_tool_calls(self):
        msg = {"role": "assistant", "content": "ok", "tool_calls": [
            {"function": {"name": "read", "arguments": '{"path": "/tmp/test.py"}'}},
        ]}
        tokens = estimate_message_tokens(msg)
        assert tokens > estimate_message_tokens({"role": "assistant", "content": "ok"})


# ── Compaction Triggers ──────────────────────────────────────────────────

class TestCompactionTriggers:
    def test_no_compaction_small_context(self):
        msgs = _make_messages(5)
        assert not needs_compaction(msgs, context_window=100000, system_tokens=0)

    def test_compaction_needed_large_context(self):
        msgs = _make_messages(200, chars_each=500)
        # 200 msgs * ~125 tokens each = ~25000 tokens, window = 20000
        assert needs_compaction(msgs, context_window=20000, system_tokens=0)

    def test_no_compaction_few_messages(self):
        # Even if tokens are high, don't compact if <= MIN_RECENT_KEEP messages
        msgs = _make_messages(MIN_RECENT_KEEP)
        assert not needs_compaction(msgs, context_window=100, system_tokens=0)

    def test_split_preserves_recent(self):
        msgs = _make_messages(50, chars_each=400)
        old, recent = split_for_compaction(msgs, context_window=5000, system_tokens=0)
        assert len(recent) >= MIN_RECENT_KEEP
        assert len(old) + len(recent) == len(msgs)
        # Recent should be the last N messages
        assert recent[-1] == msgs[-1]
        if old:
            assert old[0] == msgs[0]


# ── Session Memory Loading ───────────────────────────────────────────────

class TestSessionMemoryLoading:
    def setup_method(self):
        self.session_id = "test-session-memory-loading"
        _cleanup_session(self.session_id)

    def teardown_method(self):
        _cleanup_session(self.session_id)

    def test_empty_dir(self):
        facts, plan = _load_session_memory(self.session_id)
        assert facts == ""
        assert plan == ""

    def test_facts_only(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("- fact 1\n- fact 2")
        facts, plan = _load_session_memory(self.session_id)
        assert "fact 1" in facts
        assert "fact 2" in facts
        assert plan == ""

    def test_plan_only(self):
        d = _make_session_dir(self.session_id)
        (d / "plan.md").write_text("1. step one\n2. step two")
        facts, plan = _load_session_memory(self.session_id)
        assert facts == ""
        assert "step one" in plan

    def test_both_facts_and_plan(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("- fact A")
        (d / "plan.md").write_text("- plan B")
        facts, plan = _load_session_memory(self.session_id)
        assert "fact A" in facts
        assert "plan B" in plan

    def test_extra_files_go_to_facts(self):
        d = _make_session_dir(self.session_id)
        (d / "notes.txt").write_text("some notes")
        facts, plan = _load_session_memory(self.session_id)
        assert "some notes" in facts
        assert plan == ""

    def test_nested_files(self):
        d = _make_session_dir(self.session_id)
        sub = d / "docs"
        sub.mkdir()
        (sub / "spec.md").write_text("spec content")
        facts, plan = _load_session_memory(self.session_id)
        assert "spec content" in facts

    def test_non_text_files_ignored(self):
        d = _make_session_dir(self.session_id)
        (d / "image.png").write_bytes(b"\x89PNG")
        (d / "facts.md").write_text("- real fact")
        facts, plan = _load_session_memory(self.session_id)
        assert "real fact" in facts
        assert "PNG" not in facts


# ── Session Memory Extraction ────────────────────────────────────────────

class TestSessionMemoryExtraction:
    def setup_method(self):
        self.session_id = "test-session-extraction"
        _cleanup_session(self.session_id)

    def teardown_method(self):
        _cleanup_session(self.session_id)

    @pytest.mark.asyncio
    async def test_extraction_creates_files(self):
        messages = [
            {"role": "user", "content": "We decided to use PostgreSQL for the database"},
            {"role": "assistant", "content": "Got it, PostgreSQL it is."},
            {"role": "user", "content": "Next step is to set up the schema"},
        ]

        async def fake_summarize(prompt):
            return "## Facts\n- Database: PostgreSQL\n\n## Plan\n- Set up the schema"

        await extract_session_memory(messages, self.session_id, fake_summarize)

        d = Path.home() / ".march" / "memory" / self.session_id
        assert (d / "facts.md").exists()
        assert (d / "plan.md").exists()
        assert "PostgreSQL" in (d / "facts.md").read_text()
        assert "schema" in (d / "plan.md").read_text()

    @pytest.mark.asyncio
    async def test_extraction_appends(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("- existing fact")

        async def fake_summarize(prompt):
            return "## Facts\n- new fact\n\n## Plan\nNone"

        await extract_session_memory(
            [{"role": "user", "content": "test"}],
            self.session_id, fake_summarize,
        )

        content = (d / "facts.md").read_text()
        assert "existing fact" in content
        assert "new fact" in content

    @pytest.mark.asyncio
    async def test_extraction_failure_nonfatal(self):
        async def failing_summarize(prompt):
            raise RuntimeError("LLM error")

        # Should not raise
        await extract_session_memory(
            [{"role": "user", "content": "test"}],
            self.session_id, failing_summarize,
        )


# ── Delete Session Memory ───────────────────────────────────────────────

class TestDeleteSessionMemory:
    def test_delete_existing(self):
        session_id = "test-delete-existing"
        d = _make_session_dir(session_id)
        (d / "facts.md").write_text("data")
        assert delete_session_memory(session_id) is True
        assert not d.exists()

    def test_delete_nonexistent(self):
        assert delete_session_memory("nonexistent-session-xyz") is False


# ── Facts Compression ────────────────────────────────────────────────────

class TestFactsCompression:
    @pytest.mark.asyncio
    async def test_compress_adds_reference(self):
        facts = "- fact 1\n- fact 2\n- fact 3"

        async def fake_summarize(prompt):
            return "- fact 1\n- fact 2\n- fact 3 (compressed)"

        result = await _compress_facts(facts, 100, fake_summarize, "/path/to/facts.md")
        assert "/path/to/facts.md" in result
        assert "read tool" in result

    @pytest.mark.asyncio
    async def test_compress_fallback_on_error(self):
        facts = "- fact 1\n- fact 2"

        async def failing_summarize(prompt):
            raise RuntimeError("fail")

        result = await _compress_facts(facts, 100, failing_summarize)
        assert "fact 1" in result  # Returns original on failure

    @pytest.mark.asyncio
    async def test_compress_truncates_on_error_if_too_large(self):
        facts = "x" * 10000

        async def failing_summarize(prompt):
            raise RuntimeError("fail")

        result = await _compress_facts(facts, 50, failing_summarize)  # 50 tokens = ~200 chars
        assert len(result) < len(facts)
        assert "truncated" in result


# ── Full Compaction Flow ─────────────────────────────────────────────────

class TestCompactMessages:
    @pytest.mark.asyncio
    async def test_no_compaction_when_not_needed(self):
        msgs = _make_messages(5)

        async def should_not_be_called(prompt):
            raise AssertionError("Should not summarize")

        result, summary = await compact_messages(
            msgs, context_window=100000, system_tokens=0,
            summarize_fn=should_not_be_called,
        )
        assert result == msgs

    @pytest.mark.asyncio
    async def test_compaction_produces_summary(self):
        # 50 msgs * ~250 tokens each = ~12500 tokens, window 5000 → triggers compaction
        msgs = _make_messages(50, chars_each=1000)

        async def fake_summarize(prompt):
            return "This is a summary of the conversation."

        result, summary = await compact_messages(
            msgs, context_window=5000, system_tokens=0,
            summarize_fn=fake_summarize,
        )
        assert len(result) < len(msgs)
        assert "summary" in summary.lower() or "compacted" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_compaction_with_session_memory(self):
        session_id = "test-compact-with-memory"
        try:
            d = _make_session_dir(session_id)
            (d / "facts.md").write_text("[2026-03-06 21:00 UTC]\n- DB is PostgreSQL")
            (d / "plan.md").write_text("[2026-03-06 21:00 UTC]\n1. Deploy to prod")

            msgs = _make_messages(50, chars_each=1000)

            async def fake_summarize(prompt):
                if "INDEX" in prompt or "index" in prompt:
                    return "- DB: PostgreSQL"
                if "memory curator" in prompt.lower():
                    return "## Facts\n- extracted fact\n\n## Plan\n- extracted plan"
                return "Conversation summary."

            result, summary = await compact_messages(
                msgs, context_window=5000, system_tokens=0,
                summarize_fn=fake_summarize,
                session_id=session_id,
            )
            # Summary should include session memory
            assert "PostgreSQL" in summary or "Preserved Session Memory" in summary

            # Original facts file should still exist (not cleared)
            assert (d / "facts.md").exists()
            assert "PostgreSQL" in (d / "facts.md").read_text()
        finally:
            _cleanup_session(session_id)

    @pytest.mark.asyncio
    async def test_compaction_fallback_on_summarize_failure(self):
        msgs = _make_messages(50, chars_each=1000)

        async def failing_summarize(prompt):
            raise RuntimeError("LLM down")

        result, summary = await compact_messages(
            msgs, context_window=5000, system_tokens=0,
            summarize_fn=failing_summarize,
        )
        # Should fall back to truncation (just recent messages)
        assert len(result) < len(msgs)


# ── Budget Constants ─────────────────────────────────────────────────────

class TestBudgetConstants:
    def test_facts_budget(self):
        assert FACTS_BUDGET_RATIO == 0.15

    def test_plan_budget(self):
        assert PLAN_BUDGET_RATIO == 0.05

    def test_summary_budget(self):
        assert SUMMARY_BUDGET_RATIO == 0.15

    def test_compaction_threshold(self):
        assert COMPACTION_THRESHOLD == 0.90

    def test_total_budget_reasonable(self):
        # Summary + facts + plans should leave room for recent messages
        total = SUMMARY_BUDGET_RATIO + FACTS_BUDGET_RATIO + PLAN_BUDGET_RATIO
        assert total <= 0.40  # Max 40% for compacted content
        assert total >= 0.30  # At least 30% for meaningful content


# ── Session Memory Tool ──────────────────────────────────────────────────

class TestSessionMemoryTool:
    def setup_method(self):
        self.session_id = "test-session-memory-tool"
        _cleanup_session(self.session_id)
        # Set contextvar so the tool can resolve session_id automatically
        from march.tools.context import current_session_id
        self._token = current_session_id.set(self.session_id)

    def teardown_method(self):
        _cleanup_session(self.session_id)
        # Reset contextvar
        from march.tools.context import current_session_id
        current_session_id.reset(self._token)

    @pytest.mark.asyncio
    async def test_save_facts(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="facts",
            content="- Python 3.12\n- Uses Bedrock",
        )
        assert "Saved" in result
        d = Path.home() / ".march" / "memory" / self.session_id
        content = (d / "facts.md").read_text()
        assert "Python 3.12" in content
        assert "Uses Bedrock" in content

    @pytest.mark.asyncio
    async def test_save_plan(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="plan",
            content="1. Build API\n2. Write tests",
        )
        assert "Saved" in result
        d = Path.home() / ".march" / "memory" / self.session_id
        assert "Build API" in (d / "plan.md").read_text()

    @pytest.mark.asyncio
    async def test_append_facts(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        await session_memory_tool(
            type="facts", content="- fact 1",
        )
        await session_memory_tool(
            type="facts", content="- fact 2",
        )
        d = Path.home() / ".march" / "memory" / self.session_id
        content = (d / "facts.md").read_text()
        assert "fact 1" in content
        assert "fact 2" in content

    @pytest.mark.asyncio
    async def test_auto_timestamp(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        await session_memory_tool(
            type="facts", content="- timestamped fact",
        )
        d = Path.home() / ".march" / "memory" / self.session_id
        content = (d / "facts.md").read_text()
        assert "UTC]" in content  # Timestamp present

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="invalid", content="test",
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="facts", content="",
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_no_session_id_in_context(self):
        """Tool should error when contextvar is not set."""
        from march.tools.context import current_session_id
        from march.tools.builtin.session_memory_tool import session_memory_tool
        # Reset to empty
        token = current_session_id.set("")
        try:
            result = await session_memory_tool(type="facts", content="test")
            assert "Error" in result
            assert "session_id" in result
        finally:
            current_session_id.reset(token)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
