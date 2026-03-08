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
    MIN_RECENT_KEEP,
    SAFETY_MARGIN,
    SUMMARY_BUDGET_RATIO,
    _load_session_memory,
    _parse_memory_sections,
    build_summary_prompt,
    compact_messages,
    dedup_session_memory,
    delete_session_memory,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    needs_compaction,
    split_for_compaction,
    write_session_memory,
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
        mem = _load_session_memory(self.session_id)
        assert mem["facts"] == ""
        assert mem["plan"] == ""
        assert set(mem.keys()) == {"facts", "plan"}

    def test_facts_only(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("- fact 1\n- fact 2")
        mem = _load_session_memory(self.session_id)
        assert "fact 1" in mem["facts"]
        assert "fact 2" in mem["facts"]
        assert mem["plan"] == ""

    def test_plan_only(self):
        d = _make_session_dir(self.session_id)
        (d / "plan.md").write_text("1. step one\n2. step two")
        mem = _load_session_memory(self.session_id)
        assert mem["facts"] == ""
        assert "step one" in mem["plan"]

    def test_both_facts_and_plan(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("- fact A")
        (d / "plan.md").write_text("- plan B")
        mem = _load_session_memory(self.session_id)
        assert "fact A" in mem["facts"]
        assert "plan B" in mem["plan"]

    def test_extra_files_go_to_facts(self):
        d = _make_session_dir(self.session_id)
        (d / "notes.txt").write_text("some notes")
        mem = _load_session_memory(self.session_id)
        assert "some notes" in mem["facts"]
        assert mem["plan"] == ""

    def test_nested_files(self):
        d = _make_session_dir(self.session_id)
        sub = d / "docs"
        sub.mkdir()
        (sub / "spec.md").write_text("spec content")
        mem = _load_session_memory(self.session_id)
        assert "spec content" in mem["facts"]

    def test_non_text_files_ignored(self):
        d = _make_session_dir(self.session_id)
        (d / "image.png").write_bytes(b"\x89PNG")
        (d / "facts.md").write_text("- real fact")
        mem = _load_session_memory(self.session_id)
        assert "real fact" in mem["facts"]
        assert "PNG" not in mem["facts"]


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
                if "deduplicat" in prompt.lower():
                    return "## Facts\n- DB is PostgreSQL\n\n## Plan\n1. Deploy to prod"
                return "Conversation summary."

            result, summary = await compact_messages(
                msgs, context_window=5000, system_tokens=0,
                summarize_fn=fake_summarize,
                session_id=session_id,
            )
            # Summary should include session memory (via merge)
            assert "PostgreSQL" in summary or "Deploy to prod" in summary

            # Facts file should be overwritten with merged content
            assert (d / "facts.md").exists()
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
    def test_summary_budget(self):
        assert SUMMARY_BUDGET_RATIO == 0.15

    def test_compaction_threshold(self):
        assert COMPACTION_THRESHOLD == 0.95


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
    async def test_save_checkpoint_writes_to_plan(self):
        """checkpoint type should write to plan.md, not checkpoint.md."""
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="checkpoint",
            content="## Phase 1 Complete\n- Schema deployed",
        )
        assert "Saved" in result
        d = Path.home() / ".march" / "memory" / self.session_id
        assert "Phase 1 Complete" in (d / "plan.md").read_text()
        assert not (d / "checkpoint.md").exists()

    @pytest.mark.asyncio
    async def test_save_progress_writes_to_plan(self):
        """progress type should write to plan.md, not progress.md."""
        from march.tools.builtin.session_memory_tool import session_memory_tool
        result = await session_memory_tool(
            type="progress",
            content="- ✅ Step 1 done\n- 🔄 Step 2 in progress",
        )
        assert "Saved" in result
        d = Path.home() / ".march" / "memory" / self.session_id
        assert "Step 1 done" in (d / "plan.md").read_text()
        assert not (d / "progress.md").exists()

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


# ── Dedup Session Memory ──────────────────────────────────────────────────

class TestDedupSessionMemory:
    """Tests for the dedup_session_memory function."""

    @pytest.mark.asyncio
    async def test_dedup_removes_duplicates(self):
        memory_dict = {
            "facts": "- DB is PostgreSQL\n- Python 3.12\n- DB is PostgreSQL",
            "plan": "1. Deploy to prod",
        }

        async def fake_summarize(prompt):
            assert "PostgreSQL" in prompt
            assert "Deploy to prod" in prompt
            return "## Facts\n- DB is PostgreSQL\n- Python 3.12\n\n## Plan\n1. Deploy to prod"

        result = await dedup_session_memory(
            memory_dict, fake_summarize, context_window=100000,
        )
        assert "PostgreSQL" in result["facts"]
        assert "Deploy to prod" in result["plan"]

    @pytest.mark.asyncio
    async def test_dedup_with_empty_memory(self):
        memory_dict = {"facts": "", "plan": ""}

        async def should_not_be_called(prompt):
            raise AssertionError("Should not call LLM for empty memory")

        result = await dedup_session_memory(
            memory_dict, should_not_be_called, context_window=100000,
        )
        assert result == memory_dict

    @pytest.mark.asyncio
    async def test_dedup_fallback_on_error(self):
        memory_dict = {
            "facts": "- important fact",
            "plan": "- important plan",
        }

        async def failing_summarize(prompt):
            raise RuntimeError("LLM down")

        result = await dedup_session_memory(
            memory_dict, failing_summarize, context_window=100000,
        )
        # Should return originals on failure
        assert result["facts"] == "- important fact"
        assert result["plan"] == "- important plan"

    @pytest.mark.asyncio
    async def test_dedup_size_target(self):
        """Verify the prompt includes a size target based on context window."""
        memory_dict = {
            "facts": "- fact",
            "plan": "",
        }
        captured_prompts = []

        async def capturing_summarize(prompt):
            captured_prompts.append(prompt)
            return "## Facts\n- fact"

        await dedup_session_memory(
            memory_dict, capturing_summarize, context_window=10000,
        )
        assert len(captured_prompts) == 1
        # Should mention target tokens (min of current size, 30% of 10000 = 3000)
        assert "token" in captured_prompts[0].lower()

    @pytest.mark.asyncio
    async def test_dedup_prompt_content(self):
        """Verify the prompt explicitly asks for deduplication, not compression."""
        memory_dict = {"facts": "- fact", "plan": ""}
        captured_prompts = []

        async def capturing_summarize(prompt):
            captured_prompts.append(prompt)
            return "## Facts\n- fact"

        await dedup_session_memory(
            memory_dict, capturing_summarize, context_window=100000,
        )
        prompt = captured_prompts[0].lower()
        assert "deduplicate" in prompt or "deduplicat" in prompt
        assert "do not compress" in prompt

    @pytest.mark.asyncio
    async def test_dedup_only_facts_and_plan(self):
        """Dedup should only process facts and plan sections."""
        memory_dict = {
            "facts": "- fact A",
            "plan": "- plan B",
        }

        async def fake_summarize(prompt):
            return (
                "## Facts\n- fact A\n\n"
                "## Plan\n- plan B"
            )

        result = await dedup_session_memory(
            memory_dict, fake_summarize, context_window=100000,
        )
        assert "fact A" in result["facts"]
        assert "plan B" in result["plan"]
        assert set(result.keys()) == {"facts", "plan"}


# ── Parse Memory Sections ────────────────────────────────────────────────

class TestParseMemorySections:
    def test_parse_facts_and_plan(self):
        text = (
            "## Facts\n- fact A\n- fact B\n\n"
            "## Plan\n1. step one"
        )
        result = _parse_memory_sections(text)
        assert "fact A" in result["facts"]
        assert "step one" in result["plan"]
        assert set(result.keys()) == {"facts", "plan"}

    def test_parse_partial_sections(self):
        text = "## Facts\n- only facts here"
        result = _parse_memory_sections(text)
        assert "only facts here" in result["facts"]
        assert result["plan"] == ""

    def test_parse_empty(self):
        result = _parse_memory_sections("")
        assert all(v == "" for v in result.values())


# ── Write Session Memory ─────────────────────────────────────────────────

class TestWriteSessionMemory:
    def setup_method(self):
        self.session_id = "test-write-session-mem"
        _cleanup_session(self.session_id)

    def teardown_method(self):
        _cleanup_session(self.session_id)

    def test_write_facts_and_plan(self):
        memory_dict = {
            "facts": "- fact A\n- fact B",
            "plan": "1. step one",
        }
        write_session_memory(self.session_id, memory_dict)

        d = Path.home() / ".march" / "memory" / self.session_id
        assert "fact A" in (d / "facts.md").read_text()
        assert "step one" in (d / "plan.md").read_text()
        # checkpoint.md and progress.md should NOT be created
        assert not (d / "checkpoint.md").exists()
        assert not (d / "progress.md").exists()

    def test_write_partial_does_not_clear_others(self):
        """Writing only facts should NOT clear existing plan file."""
        d = _make_session_dir(self.session_id)
        (d / "plan.md").write_text("existing plan")

        memory_dict = {
            "facts": "- new facts",
            "plan": "",  # empty — should NOT overwrite existing
        }
        write_session_memory(self.session_id, memory_dict)

        assert "new facts" in (d / "facts.md").read_text()
        # plan.md should still have old content (not cleared)
        assert "existing plan" in (d / "plan.md").read_text()

    def test_write_overwrites_existing(self):
        d = _make_session_dir(self.session_id)
        (d / "facts.md").write_text("old facts")

        memory_dict = {
            "facts": "- new facts",
            "plan": "",
        }
        write_session_memory(self.session_id, memory_dict)

        assert "new facts" in (d / "facts.md").read_text()
        assert "old facts" not in (d / "facts.md").read_text()

    def test_write_creates_dir(self):
        d = Path.home() / ".march" / "memory" / self.session_id
        assert not d.exists()

        write_session_memory(self.session_id, {"facts": "- data", "plan": ""})
        assert d.exists()
        assert "data" in (d / "facts.md").read_text()


# ── Configurable Compaction Threshold ────────────────────────────────────

class TestConfigurableCompaction:
    def test_needs_compaction_custom_threshold(self):
        """needs_compaction accepts a custom threshold parameter."""
        msgs = _make_messages(50, chars_each=400)
        # With very low threshold, should trigger compaction
        assert needs_compaction(msgs, context_window=50000, system_tokens=0, threshold=0.01)
        # With very high threshold, should not trigger
        assert not needs_compaction(msgs, context_window=50000, system_tokens=0, threshold=0.99)

    def test_split_for_compaction_custom_budget(self):
        """split_for_compaction accepts a custom summary_budget_ratio."""
        msgs = _make_messages(50, chars_each=400)
        # With larger summary budget, fewer messages should be kept
        _, recent_small = split_for_compaction(msgs, context_window=5000, system_tokens=0,
                                                summary_budget_ratio=0.05)
        _, recent_large = split_for_compaction(msgs, context_window=5000, system_tokens=0,
                                                summary_budget_ratio=0.40)
        # Larger summary budget means less room for recent messages
        assert len(recent_large) <= len(recent_small)

    @pytest.mark.asyncio
    async def test_compact_messages_custom_threshold(self):
        """compact_messages accepts custom threshold and budget params."""
        msgs = _make_messages(50, chars_each=1000)

        async def fake_summarize(prompt):
            return "Summary."

        # With threshold=0.99 and large window, should NOT compact
        result_high, _ = await compact_messages(
            msgs, context_window=100000, system_tokens=0,
            summarize_fn=fake_summarize, threshold=0.99,
        )
        assert result_high == msgs  # No compaction

        # With small window, default threshold (0.95) triggers compaction
        result_low, _ = await compact_messages(
            msgs, context_window=5000, system_tokens=0,
            summarize_fn=fake_summarize, threshold=0.95,
        )
        assert len(result_low) < len(msgs)  # Compacted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
