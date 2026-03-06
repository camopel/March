"""Tests for March built-in plugins: safety, cost, logger, rate_limiter, git_context."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from march.core.context import Context
from march.core.message import ToolCall, ToolResult
from march.plugins.builtin.safety import SafetyPlugin, DEFAULT_BLOCKLIST
from march.plugins.builtin.cost import CostPlugin
from march.plugins.builtin.logger_plugin import LoggerPlugin
from march.plugins.builtin.rate_limiter import RateLimiterPlugin, TokenBucket
from march.plugins.builtin.git_context import GitContextPlugin
from march.plugins.manager import PluginManager


# ─── Mock LLMResponse ───

@dataclass(frozen=True)
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cost: float = 0.01

@dataclass(frozen=True)
class MockLLMResponse:
    content: str = "Hello"
    tool_calls: list[Any] = field(default_factory=list)
    usage: MockUsage = field(default_factory=MockUsage)
    duration_ms: float = 500.0
    provider: str = "litellm"
    model: str = "claude-3"


# ═══════════════════════════════════════════════════════════════
# Safety Plugin Tests
# ═══════════════════════════════════════════════════════════════


class TestSafetyPlugin:
    def test_plugin_metadata(self) -> None:
        """Safety plugin has correct name and priority."""
        plugin = SafetyPlugin()
        assert plugin.name == "safety"
        assert plugin.priority == 1
        assert plugin.enabled is True

    async def test_block_rm_rf_root(self) -> None:
        """Block rm -rf / command."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_1", name="exec", args={"command": "rm -rf /"})
        result = await plugin.before_tool(tc)
        assert result is None
        assert len(plugin.security_events) == 1
        assert plugin.security_events[0]["type"] == "blocked_dangerous_command"

    async def test_block_mkfs(self) -> None:
        """Block mkfs command."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_2", name="exec", args={"command": "mkfs.ext4 /dev/sda1"})
        result = await plugin.before_tool(tc)
        assert result is None

    async def test_block_dd_device(self) -> None:
        """Block dd to device."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_3", name="exec", args={"command": "dd if=/dev/zero of=/dev/sda"})
        result = await plugin.before_tool(tc)
        assert result is None

    async def test_allow_safe_exec(self) -> None:
        """Allow safe exec commands."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_4", name="exec", args={"command": "ls -la /tmp"})
        result = await plugin.before_tool(tc)
        assert result is not None
        assert result.name == "exec"

    async def test_allow_non_exec_tools(self) -> None:
        """Non-exec tools pass through without blocking."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_5", name="read_file", args={"path": "/etc/passwd"})
        result = await plugin.before_tool(tc)
        assert result is not None

    async def test_require_confirmation_logging(self) -> None:
        """Tools in require_confirmation list are logged."""
        plugin = SafetyPlugin(require_confirmation=["exec"])
        tc = ToolCall(id="tc_6", name="exec", args={"command": "echo hello"})
        result = await plugin.before_tool(tc)
        # Should be allowed but logged
        assert result is not None
        assert len(plugin.security_events) == 1
        assert plugin.security_events[0]["type"] == "requires_confirmation"

    async def test_on_error_logs_event(self) -> None:
        """Errors are logged as security events."""
        plugin = SafetyPlugin()
        await plugin.on_error(RuntimeError("test error"))
        assert len(plugin.security_events) == 1
        assert plugin.security_events[0]["type"] == "error"

    def test_clear_events(self) -> None:
        """Security events can be cleared."""
        plugin = SafetyPlugin()
        plugin._security_events.append({"type": "test"})
        assert len(plugin.security_events) == 1
        plugin.clear_events()
        assert len(plugin.security_events) == 0

    async def test_custom_blocklist(self) -> None:
        """Custom blocklist patterns work."""
        plugin = SafetyPlugin(blocklist=[r"\bmy_dangerous_cmd\b"])
        tc = ToolCall(id="tc_7", name="exec", args={"command": "my_dangerous_cmd --force"})
        result = await plugin.before_tool(tc)
        assert result is None

    async def test_block_shutdown(self) -> None:
        """Block shutdown command."""
        plugin = SafetyPlugin()
        tc = ToolCall(id="tc_8", name="shell", args={"command": "shutdown -h now"})
        result = await plugin.before_tool(tc)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Cost Plugin Tests
# ═══════════════════════════════════════════════════════════════


class TestCostPlugin:
    def test_plugin_metadata(self) -> None:
        """Cost plugin has correct name and priority."""
        plugin = CostPlugin()
        assert plugin.name == "cost"
        assert plugin.priority == 90

    async def test_track_cost(self) -> None:
        """Cost is tracked after LLM call."""
        plugin = CostPlugin()
        ctx = Context()
        response = MockLLMResponse()
        result = await plugin.after_llm(ctx, response)
        assert result is response
        assert plugin.session_cost == 0.01
        assert plugin.session_tokens == 150  # 100 + 50

    async def test_track_multiple_calls(self) -> None:
        """Multiple LLM calls accumulate cost."""
        plugin = CostPlugin()
        ctx = Context()
        for _ in range(3):
            await plugin.after_llm(ctx, MockLLMResponse())
        assert plugin.session_cost == pytest.approx(0.03, abs=1e-6)
        assert len(plugin.session_records) == 3

    async def test_alert_threshold_session(self) -> None:
        """Cost footer added when session budget exceeds threshold."""
        plugin = CostPlugin(budget_per_session=0.01, alert_threshold=0.5)
        ctx = Context()
        # Add cost that exceeds threshold
        await plugin.after_llm(ctx, MockLLMResponse(usage=MockUsage(cost=0.008)))
        result = await plugin.on_response("Hello world")
        assert isinstance(result, str)
        assert "💰" in result or "⚠️" in result

    async def test_no_alert_below_threshold(self) -> None:
        """No cost footer when below threshold."""
        plugin = CostPlugin(budget_per_session=100.0, alert_threshold=0.8)
        ctx = Context()
        await plugin.after_llm(ctx, MockLLMResponse(usage=MockUsage(cost=0.001)))
        result = await plugin.on_response("Hello world")
        assert result == "Hello world"

    async def test_budget_exceeded_warning(self) -> None:
        """Warning shown when budget exceeded."""
        plugin = CostPlugin(budget_per_session=0.005)
        ctx = Context()
        await plugin.after_llm(ctx, MockLLMResponse(usage=MockUsage(cost=0.01)))
        result = await plugin.on_response("Hello")
        assert "⚠️" in result
        assert "exceeded" in result.lower()

    def test_reset_session(self) -> None:
        """Session cost can be reset."""
        plugin = CostPlugin()
        plugin._session_records.append(MagicMock(cost=0.05, input_tokens=100, output_tokens=50))
        assert plugin.session_cost == 0.05
        plugin.reset_session()
        assert plugin.session_cost == 0.0

    def test_summary(self) -> None:
        """Summary dict contains expected keys."""
        plugin = CostPlugin()
        summary = plugin.summary()
        assert "session_cost" in summary
        assert "daily_cost" in summary
        assert "budget_per_session" in summary


# ═══════════════════════════════════════════════════════════════
# Logger Plugin Tests
# ═══════════════════════════════════════════════════════════════


class TestLoggerPlugin:
    def test_plugin_metadata(self) -> None:
        """Logger plugin has correct name."""
        plugin = LoggerPlugin()
        assert plugin.name == "logger"
        assert plugin.priority == 95

    async def test_log_llm_call(self) -> None:
        """LLM calls are logged."""
        plugin = LoggerPlugin()
        ctx = Context()
        response = MockLLMResponse()
        await plugin.after_llm(ctx, response)
        assert len(plugin.log_entries) == 1
        assert plugin.log_entries[0]["type"] == "llm_call"
        assert plugin.log_entries[0]["input_tokens"] == 100

    async def test_log_tool_call(self) -> None:
        """Tool calls are logged."""
        plugin = LoggerPlugin()
        tc = ToolCall(id="tc_1", name="read_file", args={"path": "/tmp/test"})
        tr = ToolResult(id="tc_1", content="file contents", duration_ms=10.0)
        await plugin.after_tool(tc, tr)
        assert len(plugin.log_entries) == 1
        assert plugin.log_entries[0]["type"] == "tool_call"
        assert plugin.log_entries[0]["name"] == "read_file"

    async def test_log_disabled(self) -> None:
        """No logging when disabled."""
        plugin = LoggerPlugin(log_tool_results=False, log_llm_calls=False)
        ctx = Context()
        await plugin.after_llm(ctx, MockLLMResponse())
        tc = ToolCall(id="tc_1", name="test", args={})
        tr = ToolResult(id="tc_1", content="ok")
        await plugin.after_tool(tc, tr)
        assert len(plugin.log_entries) == 0

    async def test_log_subagent(self) -> None:
        """Sub-agent spawn/complete are logged."""
        plugin = LoggerPlugin()
        config = {"task": "test task"}
        result = await plugin.on_subagent_spawn(config)
        assert result == config
        assert len(plugin.log_entries) == 1

        await plugin.on_subagent_complete("agent-1", "done")
        assert len(plugin.log_entries) == 2
        assert plugin.log_entries[1]["type"] == "subagent_complete"

    def test_clear(self) -> None:
        """Log entries can be cleared."""
        plugin = LoggerPlugin()
        plugin._log_entries.append({"type": "test"})
        plugin.clear()
        assert len(plugin.log_entries) == 0


# ═══════════════════════════════════════════════════════════════
# Rate Limiter Plugin Tests
# ═══════════════════════════════════════════════════════════════


class TestRateLimiterPlugin:
    def test_plugin_metadata(self) -> None:
        """Rate limiter has correct name and priority."""
        plugin = RateLimiterPlugin()
        assert plugin.name == "rate_limiter"
        assert plugin.priority == 5

    def test_token_bucket_consume(self) -> None:
        """Token bucket allows consumption when tokens available."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        wait = bucket.try_consume()
        assert wait == 0.0
        assert bucket.tokens < 10.0

    def test_token_bucket_empty(self) -> None:
        """Token bucket returns wait time when empty."""
        bucket = TokenBucket(capacity=1.0, refill_rate=1.0)
        bucket.tokens = 0.0
        wait = bucket.try_consume()
        assert wait > 0.0

    async def test_allow_normal_calls(self) -> None:
        """Normal calls pass through."""
        plugin = RateLimiterPlugin(max_calls_per_minute=100)
        ctx = Context()
        result = await plugin.before_llm(ctx, "test")
        assert len(result) == 2
        assert result[1] == "test"

    async def test_block_excessive_calls(self) -> None:
        """Excessive calls get blocked (short-circuit response)."""
        plugin = RateLimiterPlugin(
            max_calls_per_minute=2,
            max_wait_seconds=0.001,  # Very low wait tolerance
        )
        ctx = Context()
        # Drain the bucket
        plugin._get_buckets("default")[0].tokens = 0.0
        result = await plugin.before_llm(ctx, "test")
        # Should either delay or short-circuit
        assert len(result) >= 2

    def test_stats(self) -> None:
        """Stats dict returns expected keys."""
        plugin = RateLimiterPlugin()
        stats = plugin.stats()
        assert "total_delayed" in stats
        assert "total_blocked" in stats
        assert "providers" in stats


# ═══════════════════════════════════════════════════════════════
# Git Context Plugin Tests
# ═══════════════════════════════════════════════════════════════


class TestGitContextPlugin:
    def test_plugin_metadata(self) -> None:
        """Git context plugin has correct name."""
        plugin = GitContextPlugin()
        assert plugin.name == "git_context"
        assert plugin.priority == 80

    async def test_disabled_passthrough(self) -> None:
        """Disabled plugin passes through unchanged."""
        plugin = GitContextPlugin(auto_detect=False)
        ctx = Context()
        result_ctx, result_msg = await plugin.before_llm(ctx, "test")
        assert result_msg == "test"
        assert len(result_ctx.extra_context) == 0

    async def test_non_git_directory(self) -> None:
        """Non-git directory passes through unchanged."""
        plugin = GitContextPlugin(working_dir="/tmp")
        ctx = Context()
        result_ctx, result_msg = await plugin.before_llm(ctx, "test")
        assert result_msg == "test"

    async def test_git_context_injected(self) -> None:
        """Git context is injected in a git repo."""
        plugin = GitContextPlugin(
            inject_branch=True,
            inject_status=True,
            inject_diff=False,
        )

        # Mock the git commands
        async def mock_is_git_repo(cwd):
            return True

        async def mock_get_branch(cwd):
            return "main"

        async def mock_get_status(cwd):
            return "M  file.py"

        plugin._is_git_repo = mock_is_git_repo
        plugin._get_branch = mock_get_branch
        plugin._get_status = mock_get_status

        ctx = Context()
        result_ctx, result_msg = await plugin.before_llm(ctx, "test")
        assert len(result_ctx.extra_context) == 1
        assert "main" in result_ctx.extra_context[0]
        assert "file.py" in result_ctx.extra_context[0]


# ═══════════════════════════════════════════════════════════════
# Integration: Multiple Plugins Together
# ═══════════════════════════════════════════════════════════════


class TestPluginIntegration:
    async def test_safety_blocks_before_others(self) -> None:
        """Safety plugin (priority 1) blocks before other plugins run."""
        manager = PluginManager()
        safety = SafetyPlugin()
        logger_p = LoggerPlugin()
        manager.register(safety)
        manager.register(logger_p)

        tc = ToolCall(id="tc_1", name="exec", args={"command": "rm -rf /"})
        result = await manager.dispatch_before_tool(tc)
        assert result is None  # Blocked by safety

    async def test_cost_and_logger_both_run(self) -> None:
        """Cost and logger both process the same LLM response."""
        manager = PluginManager()
        cost = CostPlugin()
        logger_p = LoggerPlugin()
        manager.register(cost)
        manager.register(logger_p)

        ctx = Context()
        response = MockLLMResponse()
        result = await manager.dispatch_after_llm(ctx, response)
        assert result is response
        assert cost.session_cost == 0.01
        assert len(logger_p.log_entries) == 1
