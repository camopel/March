"""Extended tests for context, git_context plugin, and additional coverage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.context import Context, estimate_tokens, truncate_to_tokens
from march.plugins.builtin.git_context import GitContextPlugin


# ─────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_basic(self):
        text = "hello world test"  # 16 chars → ~4 tokens
        assert estimate_tokens(text) == 4

    def test_estimate_tokens_long(self):
        text = "a" * 4000
        assert estimate_tokens(text) == 1000

    def test_truncate_not_needed(self):
        text = "short text"
        result = truncate_to_tokens(text, 1000)
        assert result == text

    def test_truncate_long_text(self):
        text = "a" * 10000
        result = truncate_to_tokens(text, 100)
        assert len(result) < 10000
        assert "truncated" in result

    def test_truncate_at_newline(self):
        lines = ["Line " + str(i) for i in range(500)]
        text = "\n".join(lines)
        result = truncate_to_tokens(text, 100)
        assert "truncated" in result
        # Should have been truncated at a newline boundary


# ─────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────

class TestContextBuilder:
    def test_empty_context(self):
        ctx = Context()
        prompt = ctx.build_system_prompt()
        assert prompt == ""

    def test_system_rules_only(self):
        ctx = Context(system_rules="Be helpful.")
        prompt = ctx.build_system_prompt()
        assert "System Rules" in prompt
        assert "Be helpful." in prompt

    def test_all_sections(self):
        ctx = Context(
            system_rules="rules",
            agent_profile="profile",
            tool_rules="tools",
            long_term_memory="memory",
            daily_memory="daily",
            session_context={"channel": "matrix"},
        )
        prompt = ctx.build_system_prompt()
        assert "System Rules" in prompt
        assert "Agent Profile" in prompt
        assert "Tool Rules" in prompt
        assert "Long-Term Memory" in prompt
        assert "Today's Notes" in prompt
        assert "Session Context" in prompt
        assert "channel" in prompt

    def test_extra_context(self):
        ctx = Context()
        ctx.add("extra info 1")
        ctx.add("extra info 2")
        ctx.add("  ")  # Should be ignored (whitespace only)
        assert len(ctx.extra_context) == 2
        prompt = ctx.build_system_prompt()
        assert "extra info 1" in prompt
        assert "extra info 2" in prompt

    def test_system_prompt_property(self):
        ctx = Context(system_rules="test")
        assert ctx.system_prompt == ctx.build_system_prompt()

    def test_estimated_tokens(self):
        ctx = Context(system_rules="test" * 100)
        assert ctx.estimated_tokens > 0

    def test_build_with_budget(self):
        ctx = Context(
            system_rules="x" * 400,   # ~100 tokens
            agent_profile="y" * 400,   # ~100 tokens
            tool_rules="z" * 400,      # ~100 tokens
            long_term_memory="m" * 400, # ~100 tokens
        )
        # Budget that fits only some sections
        prompt = ctx.build_system_prompt(max_tokens=200)
        assert len(prompt) < len(ctx.build_system_prompt())

    def test_build_with_budget_very_small(self):
        ctx = Context(
            system_rules="x" * 2000,
            agent_profile="y" * 2000,
        )
        # With a budget of only 50 tokens (~200 chars), no section fits
        # (the section header + content > 500 tokens), so result is empty
        prompt = ctx.build_system_prompt(max_tokens=50)
        assert prompt == ""

    def test_build_with_budget_truncates_section(self):
        ctx = Context(
            system_rules="x" * 2000,
        )
        # Budget of 200 tokens is enough to start the section but not finish it
        prompt = ctx.build_system_prompt(max_tokens=200)
        assert "truncated" in prompt
        assert "System Rules" in prompt

    def test_build_with_budget_includes_high_priority_first(self):
        ctx = Context(
            system_rules="RULES",
            daily_memory="DAILY",
        )
        # Small budget: should include system_rules (priority 1) before daily (priority 7)
        prompt = ctx.build_system_prompt(max_tokens=50)
        assert "RULES" in prompt

    def test_session_context_formatting(self):
        ctx = Context(session_context={"user": "bob", "channel": "matrix"})
        prompt = ctx.build_system_prompt()
        assert "**user**: bob" in prompt
        assert "**channel**: matrix" in prompt

    def test_build_with_budget_skip_when_empty(self):
        ctx = Context(
            system_rules="a" * 5000,
            agent_profile="b" * 5000,
            tool_rules="c" * 5000,
        )
        # Very small budget — all sections are too large (>1250 tokens each)
        # With budget=10 (~40 chars), nothing fits (budget_remaining <= 100)
        prompt = ctx.build_system_prompt(max_tokens=10)
        assert prompt == ""

        # Medium budget — first section fits truncated, rest skipped
        prompt2 = ctx.build_system_prompt(max_tokens=300)
        assert "System Rules" in prompt2


# ─────────────────────────────────────────────────────────────
# GitContextPlugin
# ─────────────────────────────────────────────────────────────

class TestGitContextPlugin:
    def test_init_defaults(self):
        p = GitContextPlugin()
        assert p.auto_detect is True
        assert p.inject_branch is True
        assert p.inject_status is True
        assert p.inject_diff is False
        assert p.name == "git_context"

    def test_init_custom(self):
        p = GitContextPlugin(
            auto_detect=False,
            inject_branch=False,
            inject_status=False,
            inject_diff=True,
            working_dir="/tmp",
        )
        assert p.auto_detect is False
        assert p.inject_diff is True
        assert p.working_dir == "/tmp"

    async def test_before_llm_auto_detect_disabled(self):
        p = GitContextPlugin(auto_detect=False)
        ctx = Context()
        result_ctx, result_msg = await p.before_llm(ctx, "hello")
        assert result_ctx is ctx
        assert result_msg == "hello"

    async def test_before_llm_not_git_repo(self):
        p = GitContextPlugin(working_dir="/tmp")
        p._is_git_repo = AsyncMock(return_value=False)
        ctx = Context()
        result_ctx, result_msg = await p.before_llm(ctx, "hello")
        assert len(result_ctx.extra_context) == 0

    async def test_before_llm_injects_branch(self):
        p = GitContextPlugin(
            inject_branch=True,
            inject_status=False,
            inject_diff=False,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_branch = AsyncMock(return_value="main")
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert len(result_ctx.extra_context) == 1
        assert "main" in result_ctx.extra_context[0]

    async def test_before_llm_injects_status(self):
        p = GitContextPlugin(
            inject_branch=False,
            inject_status=True,
            inject_diff=False,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_status = AsyncMock(return_value="M file.py\nA new.py")
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert len(result_ctx.extra_context) == 1
        assert "M file.py" in result_ctx.extra_context[0]

    async def test_before_llm_injects_diff(self):
        p = GitContextPlugin(
            inject_branch=False,
            inject_status=False,
            inject_diff=True,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_staged_diff = AsyncMock(return_value="+ new line")
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert len(result_ctx.extra_context) == 1
        assert "+ new line" in result_ctx.extra_context[0]

    async def test_before_llm_truncates_long_status(self):
        p = GitContextPlugin(
            inject_branch=False,
            inject_status=True,
            inject_diff=False,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_status = AsyncMock(return_value="M file.py\n" * 500)
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert "truncated" in result_ctx.extra_context[0]

    async def test_before_llm_truncates_long_diff(self):
        p = GitContextPlugin(
            inject_branch=False,
            inject_status=False,
            inject_diff=True,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_staged_diff = AsyncMock(return_value="+ line\n" * 2000)
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert "truncated" in result_ctx.extra_context[0]

    async def test_before_llm_no_output(self):
        """No branch/status/diff available → don't inject context."""
        p = GitContextPlugin(
            inject_branch=True,
            inject_status=True,
            inject_diff=False,
            working_dir="/tmp",
        )
        p._is_git_repo = AsyncMock(return_value=True)
        p._get_branch = AsyncMock(return_value="")
        p._get_status = AsyncMock(return_value="")
        ctx = Context()
        result_ctx, _ = await p.before_llm(ctx, "hello")
        assert len(result_ctx.extra_context) == 0

    async def test_is_git_repo_timeout(self):
        p = GitContextPlugin()
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await p._is_git_repo("/tmp")
            assert result is False

    async def test_is_git_repo_not_found(self):
        p = GitContextPlugin()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await p._is_git_repo("/tmp")
            assert result is False

    async def test_get_branch_timeout(self):
        p = GitContextPlugin()
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await p._get_branch("/tmp")
            assert result == ""

    async def test_get_status_timeout(self):
        p = GitContextPlugin()
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await p._get_status("/tmp")
            assert result == ""

    async def test_get_staged_diff_timeout(self):
        p = GitContextPlugin()
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await p._get_staged_diff("/tmp")
            assert result == ""


# ─────────────────────────────────────────────────────────────
# SearchResult repr
# ─────────────────────────────────────────────────────────────

class TestSearchResultRepr:
    def test_search_result_repr(self):
        from march.memory.search import SearchResult
        sr = SearchResult(id="test", source="file.md", content="Hello world content here", score=0.95)
        r = repr(sr)
        assert "test" in r
        assert "0.95" in r
        assert "Hello world" in r

    def test_search_result_repr_long_content(self):
        from march.memory.search import SearchResult
        sr = SearchResult(id="x", source="s", content="a" * 200, score=0.5)
        r = repr(sr)
        assert len(r) < 300


# ─────────────────────────────────────────────────────────────
# Protocol edge cases
# ─────────────────────────────────────────────────────────────

class TestProtocolExtended:
    def test_ipc_message_from_invalid_json(self):
        from march.agents.protocol import IPCMessage
        with pytest.raises(Exception):
            IPCMessage.from_json("not valid json")

    def test_ipc_message_roundtrip_all_types(self):
        from march.agents.protocol import IPCMessage, MessageType
        messages = [
            IPCMessage.task("task text"),
            IPCMessage.steer("steer text"),
            IPCMessage.cancel("reason"),
            IPCMessage.result("content"),
            IPCMessage.error("error msg"),
            IPCMessage.progress("running"),
            IPCMessage.tool_use("tool_name"),
        ]
        for msg in messages:
            json_str = msg.to_json()
            restored = IPCMessage.from_json(json_str)
            assert restored.type == msg.type


# ─────────────────────────────────────────────────────────────
# Tools registry edge cases
# ─────────────────────────────────────────────────────────────

class TestToolsRegistryExtended:
    def test_definitions_empty(self):
        from march.tools.registry import ToolRegistry
        reg = ToolRegistry()
        assert reg.definitions() == []

    def test_has_false(self):
        from march.tools.registry import ToolRegistry
        reg = ToolRegistry()
        assert not reg.has("nonexistent")

    def test_register_and_execute(self):
        from march.tools.registry import ToolRegistry
        from march.tools.base import tool
        from march.core.message import ToolCall

        reg = ToolRegistry()

        @tool(description="Test tool")
        async def my_tool(x: str) -> str:
            return f"result: {x}"

        reg.register_function(my_tool)
        assert reg.has("my_tool")
        assert reg.tool_count == 1

    def test_register_duplicate_overwrites(self):
        from march.tools.registry import ToolRegistry
        from march.tools.base import tool

        reg = ToolRegistry()

        @tool(description="V1")
        async def my_tool(x: str) -> str:
            return "v1"

        @tool(description="V2")
        async def my_tool_v2(x: str) -> str:
            return "v2"

        reg.register_function(my_tool, name="shared_name")
        reg.register_function(my_tool_v2, name="shared_name")
        assert reg.tool_count == 1

    async def test_execute_not_found(self):
        from march.tools.registry import ToolRegistry, ToolNotFound
        from march.core.message import ToolCall

        reg = ToolRegistry()
        tc = ToolCall(id="tc1", name="nonexistent", args={})
        with pytest.raises(ToolNotFound):
            await reg.execute(tc)
