"""Tests for the March logging system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import structlog

from march.logging.formatters import format_for_audit, get_console_processor, get_json_processor
from march.logging.handlers import SQLiteAuditHandler, get_file_handler
from march.logging.logger import (
    MarchLogger,
    configure_logging,
    get_logger,
    reset_logging,
)


# ─── Fixtures ───


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state before and after each test."""
    reset_logging()
    # Clear all handlers from root logger
    root = logging.getLogger()
    root.handlers.clear()
    yield
    reset_logging()
    root = logging.getLogger()
    root.handlers.clear()


@pytest.fixture
def audit_db(tmp_path: Path) -> Path:
    """Provide a temporary audit database path."""
    return tmp_path / "audit.db"


@pytest.fixture
def audit_handler(audit_db: Path) -> SQLiteAuditHandler:
    """Create a SQLiteAuditHandler with a temp database."""
    return SQLiteAuditHandler(audit_db)


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Provide a temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


# ─── Formatter Tests ───


class TestFormatters:
    """Test log formatters."""

    def test_json_processor_is_callable(self):
        proc = get_json_processor()
        assert callable(proc)

    def test_console_processor_is_callable(self):
        proc = get_console_processor()
        assert callable(proc)

    def test_format_for_audit_extracts_fields(self):
        event_dict = {
            "timestamp": "2026-03-04T12:00:00Z",
            "level": "info",
            "event": "tool.call",
            "session_id": "abc-123",
            "tool": "exec",
            "args": {"command": "ls"},
        }
        result = format_for_audit(event_dict)
        assert result["timestamp"] == "2026-03-04T12:00:00Z"
        assert result["level"] == "info"
        assert result["event"] == "tool.call"
        assert result["session_id"] == "abc-123"
        assert result["data"]["tool"] == "exec"
        assert "timestamp" not in result["data"]
        assert "level" not in result["data"]

    def test_format_for_audit_handles_missing_fields(self):
        result = format_for_audit({})
        assert result["timestamp"] == ""
        assert result["level"] == "info"
        assert result["event"] == "unknown"
        assert result["session_id"] == "system"
        assert result["data"] == {}


# ─── Handler Tests ───


class TestFileHandler:
    """Test file rotation handler."""

    def test_creates_log_directory(self, tmp_path: Path):
        log_path = tmp_path / "deep" / "nested" / "app.log"
        handler = get_file_handler(log_path, retention_days=3)
        assert log_path.parent.exists()
        assert isinstance(handler, logging.Handler)
        handler.close()

    def test_handler_writes_to_file(self, log_dir: Path):
        log_path = log_dir / "test.log"
        handler = get_file_handler(log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=None, exc_info=None,
        )
        handler.emit(record)
        handler.flush()
        handler.close()
        assert log_path.exists()
        content = log_path.read_text()
        assert "hello world" in content


class TestSQLiteAuditHandler:
    """Test SQLite audit handler."""

    def test_creates_database(self, audit_db: Path):
        handler = SQLiteAuditHandler(audit_db)
        assert audit_db.exists()
        handler._get_connection().close()

    def test_audit_events_filter(self, audit_handler: SQLiteAuditHandler):
        """Only audit events should be stored."""
        assert "tool.call" in SQLiteAuditHandler.AUDIT_EVENTS
        assert "security.blocked" in SQLiteAuditHandler.AUDIT_EVENTS
        assert "llm.call" in SQLiteAuditHandler.AUDIT_EVENTS
        assert "some.random.event" not in SQLiteAuditHandler.AUDIT_EVENTS

    def test_emit_stores_audit_event(self, audit_handler: SQLiteAuditHandler):
        """Emitting an audit event should store it in SQLite."""
        formatter = logging.Formatter("%(message)s")
        audit_handler.setFormatter(formatter)

        event_data = {
            "event": "tool.call",
            "timestamp": "2026-03-04T12:00:00Z",
            "level": "INFO",
            "session_id": "test-session",
            "tool": "exec",
            "args": {"command": "ls"},
        }
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=json.dumps(event_data), args=None, exc_info=None,
        )
        audit_handler.emit(record)

        results = audit_handler.query(event="tool.call")
        assert len(results) == 1
        assert results[0]["event"] == "tool.call"
        assert results[0]["session_id"] == "test-session"
        assert results[0]["data"]["tool"] == "exec"

    def test_emit_ignores_non_audit_event(self, audit_handler: SQLiteAuditHandler):
        """Non-audit events should not be stored."""
        formatter = logging.Formatter("%(message)s")
        audit_handler.setFormatter(formatter)

        event_data = {
            "event": "debug.trace",
            "timestamp": "2026-03-04T12:00:00Z",
        }
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0,
            msg=json.dumps(event_data), args=None, exc_info=None,
        )
        audit_handler.emit(record)

        results = audit_handler.query()
        assert len(results) == 0

    def test_query_with_filters(self, audit_handler: SQLiteAuditHandler):
        """Test querying with different filters."""
        formatter = logging.Formatter("%(message)s")
        audit_handler.setFormatter(formatter)

        events = [
            {"event": "tool.call", "timestamp": "2026-03-04T12:00:00Z",
             "session_id": "s1", "tool": "exec"},
            {"event": "tool.call", "timestamp": "2026-03-04T12:01:00Z",
             "session_id": "s2", "tool": "read"},
            {"event": "security.blocked", "timestamp": "2026-03-04T12:02:00Z",
             "session_id": "s1", "action": "rm -rf /"},
        ]
        for event_data in events:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=json.dumps(event_data), args=None, exc_info=None,
            )
            audit_handler.emit(record)

        # All events
        all_results = audit_handler.query()
        assert len(all_results) == 3

        # Filter by event type
        tool_results = audit_handler.query(event="tool.call")
        assert len(tool_results) == 2

        # Filter by session
        s1_results = audit_handler.query(session_id="s1")
        assert len(s1_results) == 2

        # Filter by event + session
        filtered = audit_handler.query(event="tool.call", session_id="s1")
        assert len(filtered) == 1

        # Limit
        limited = audit_handler.query(limit=1)
        assert len(limited) == 1

    def test_clear_all(self, audit_handler: SQLiteAuditHandler):
        """Test clearing all audit entries."""
        formatter = logging.Formatter("%(message)s")
        audit_handler.setFormatter(formatter)

        event_data = {
            "event": "tool.call", "timestamp": "2026-03-04T12:00:00Z",
            "session_id": "test",
        }
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=json.dumps(event_data), args=None, exc_info=None,
        )
        audit_handler.emit(record)
        assert len(audit_handler.query()) == 1

        count = audit_handler.clear()
        assert count == 1
        assert len(audit_handler.query()) == 0


# ─── Logger Tests ───


class TestConfigureLogging:
    """Test structlog configuration."""

    def test_configure_json(self, log_dir: Path):
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_console(self, log_dir: Path):
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_both(self, log_dir: Path):
        configure_logging()
        logger = get_logger("test")
        assert logger is not None


class TestMarchLogger:
    """Test the high-level MarchLogger interface."""

    def test_create_logger(self):
        logger = MarchLogger(session_id="test-session")
        assert logger is not None

    def test_bind_returns_new_logger(self):
        logger = MarchLogger(session_id="test")
        bound = logger.bind(channel="matrix")
        assert bound is not logger

    def test_llm_call_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        # Should not raise
        logger.llm_call(
            provider="bedrock",
            model="claude-opus",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
            duration_ms=1500.0,
        )

    def test_llm_error_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.llm_error(provider="openai", error="rate limit", will_retry=True)

    def test_llm_fallback_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.llm_fallback(from_provider="openai", to_provider="bedrock")

    def test_tool_call_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.tool_call(
            tool="exec",
            args={"command": "ls"},
            result_summary="3 files listed",
            duration_ms=50.0,
        )

    def test_tool_error_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.tool_error(tool="exec", args={"command": "bad"}, error="permission denied")

    def test_plugin_hook_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.plugin_hook(
            plugin="safety", hook="before_tool", action="allowed", duration_ms=1.0,
        )

    def test_subagent_events_do_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.subagent_spawn(agent_id="child-1", task="fix bug", model="claude")
        logger.subagent_complete(agent_id="child-1", result="done", duration_ms=5000.0)
        logger.subagent_error(agent_id="child-1", error="timeout")

    def test_security_blocked_does_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.security_blocked(
            action="rm -rf /", reason="dangerous command", plugin="safety",
        )

    def test_session_events_do_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.session_start(session_id="s1", channel="terminal")
        logger.session_end(session_id="s1", channel="terminal")

    def test_memory_events_do_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.memory_write(key="MEMORY.md", size_bytes=1024)

    def test_config_events_do_not_raise(self, log_dir: Path):
        configure_logging()
        logger = MarchLogger(session_id="test")
        logger.config_loaded(path="/home/user/.march/config.yaml")
