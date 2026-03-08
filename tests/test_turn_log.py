"""Comprehensive tests for TurnLogger (src/march/core/turn_log.py)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from march.core.turn_log import TurnLogger, _MAX_FILE_BYTES, _MAX_ROTATED

# Default session_id used by most tests
_TEST_SESSION = "test-session"


# ── helpers ──────────────────────────────────────────────────────────

def _read_lines(path: Path) -> list[dict]:
    """Read a JSONL file and return parsed dicts."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _make_logger(tmp_path: Path) -> TurnLogger:
    log_dir = tmp_path / "logs"
    return TurnLogger(log_dir=log_dir)


def _log_path(logger: TurnLogger, session_id: str = _TEST_SESSION) -> Path:
    """Return the resolved log file path for a given session_id."""
    return logger._resolve_path(session_id)


# ── TestTurnLogger ───────────────────────────────────────────────────

class TestTurnLogger:

    def test_turn_start_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.turn_start(
            turn_id="t1",
            session_id="s1",
            user_msg="hello",
            source="cli",
        )
        log_file = _log_path(logger, "s1")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "turn_start"
        assert entry["turn_id"] == "t1"
        assert entry["session_id"] == "s1"
        assert entry["user_msg"] == "hello"
        assert entry["source"] == "cli"
        assert "ts" in entry

    def test_turn_complete_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.turn_complete(
            turn_id="t2",
            session_id="s2",
            tool_calls=3,
            total_tokens=1500,
            total_cost=0.05,
            duration_ms=1234.5,
            final_reply_length=200,
        )
        log_file = _log_path(logger, "s2")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "turn_complete"
        assert entry["tool_calls"] == 3
        assert entry["total_tokens"] == 1500
        assert entry["total_cost"] == 0.05
        assert entry["duration_ms"] == 1234.5
        assert entry["final_reply_length"] == 200

    def test_llm_call_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.llm_call(
            turn_id="t3",
            session_id="s3",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
            duration_ms=500.0,
        )
        log_file = _log_path(logger, "s3")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "llm_call"
        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-4"
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 50
        assert entry["cost"] == 0.01
        assert entry["duration_ms"] == 500.0

    def test_tool_call_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.tool_call(
            turn_id="t4",
            session_id="s4",
            name="web_search",
            args={"query": "test"},
            duration_ms=300.0,
            status="ok",
            summary="Found 10 results",
        )
        log_file = _log_path(logger, "s4")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "tool_call"
        assert entry["name"] == "web_search"
        assert entry["args"] == {"query": "test"}
        assert entry["duration_ms"] == 300.0
        assert entry["status"] == "ok"
        assert entry["summary"] == "Found 10 results"

    def test_turn_cancelled_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.turn_cancelled(
            turn_id="t5",
            session_id="s5",
            partial_content_length=42,
        )
        log_file = _log_path(logger, "s5")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "turn_cancelled"
        assert entry["partial_content_length"] == 42

    def test_turn_error_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.turn_error(
            turn_id="t6",
            session_id="s6",
            error="Connection timeout",
        )
        log_file = _log_path(logger, "s6")
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        assert entry["event"] == "turn_error"
        assert entry["error"] == "Connection timeout"

    def test_log_rotation_50mb(self, tmp_path: Path) -> None:
        """Create file >50MB, verify rotation to .jsonl.1 (keep max 3)."""
        logger = _make_logger(tmp_path)
        session_id = "srot"
        log_path = _log_path(logger, session_id)

        # Pre-fill the log file to just over 50 MB
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            # Write ~51 MB of data
            chunk = "x" * (1024 * 1024) + "\n"  # ~1 MB per line
            for _ in range(51):
                fh.write(chunk)

        assert log_path.stat().st_size > _MAX_FILE_BYTES

        # Next write should trigger rotation
        logger.turn_start(
            turn_id="rot1",
            session_id=session_id,
            user_msg="after rotation",
            source="test",
        )

        # Old file should have been rotated to .jsonl.1
        rotated = log_path.with_suffix(".jsonl.1")
        assert rotated.exists(), "Rotated file .jsonl.1 should exist"
        assert rotated.stat().st_size > _MAX_FILE_BYTES

        # New log file should contain only the new entry
        lines = _read_lines(log_path)
        assert len(lines) == 1
        assert lines[0]["turn_id"] == "rot1"

        # Verify max 3 rotated files: trigger more rotations
        for i in range(2, 5):
            # Bloat the current log again
            with open(log_path, "w", encoding="utf-8") as fh:
                for _ in range(51):
                    fh.write(chunk)
            logger.turn_start(
                turn_id=f"rot{i}",
                session_id=session_id,
                user_msg="rotation",
                source="test",
            )

        # .jsonl.1, .jsonl.2, .jsonl.3 should exist; .jsonl.4 should NOT
        for n in range(1, _MAX_ROTATED + 1):
            assert log_path.with_suffix(f".jsonl.{n}").exists(), f".jsonl.{n} should exist"
        assert not log_path.with_suffix(f".jsonl.{_MAX_ROTATED + 1}").exists(), (
            f".jsonl.{_MAX_ROTATED + 1} should NOT exist"
        )

    def test_thread_safety(self, tmp_path: Path) -> None:
        """Concurrent writes from 10 threads, verify no corruption."""
        logger = _make_logger(tmp_path)
        session_id = "s-thread"
        n_threads = 10
        writes_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def _writer(tid: int) -> None:
            barrier.wait()
            for i in range(writes_per_thread):
                logger.turn_start(
                    turn_id=f"t-{tid}-{i}",
                    session_id=session_id,
                    user_msg=f"msg-{tid}-{i}",
                    source="thread",
                )

        threads = [threading.Thread(target=_writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        assert len(lines) == n_threads * writes_per_thread

        # Every line should be valid JSON (no corruption / interleaving)
        turn_ids = {entry["turn_id"] for entry in lines}
        expected = {f"t-{tid}-{i}" for tid in range(n_threads) for i in range(writes_per_thread)}
        assert turn_ids == expected

    def test_json_serialization_non_serializable(self, tmp_path: Path) -> None:
        """Pass non-serializable args, verify graceful handling."""
        logger = _make_logger(tmp_path)
        session_id = "s-ns"

        class Custom:
            pass

        obj = Custom()
        logger.tool_call(
            turn_id="t-ns",
            session_id=session_id,
            name="custom_tool",
            args={"obj": obj, "normal": 42},
            duration_ms=10.0,
            status="ok",
            summary="handled",
        )
        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        assert len(lines) == 1
        entry = lines[0]
        # The non-serializable value should have been converted via repr()
        assert "Custom" in entry["args"]["obj"]
        assert entry["args"]["normal"] == 42

    def test_user_msg_truncation(self, tmp_path: Path) -> None:
        """Long user msg (>2000 chars) truncated in log."""
        logger = _make_logger(tmp_path)
        session_id = "s-trunc"
        long_msg = "A" * 5000
        logger.turn_start(
            turn_id="t-trunc",
            session_id=session_id,
            user_msg=long_msg,
            source="test",
        )
        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        assert len(lines) == 1
        assert len(lines[0]["user_msg"]) == 2000

    def test_summary_truncation(self, tmp_path: Path) -> None:
        """Long summary (>500 chars) truncated."""
        logger = _make_logger(tmp_path)
        session_id = "s-sum"
        long_summary = "B" * 1000
        logger.tool_call(
            turn_id="t-sum",
            session_id=session_id,
            name="tool",
            args={},
            duration_ms=1.0,
            status="ok",
            summary=long_summary,
        )
        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        assert len(lines) == 1
        assert len(lines[0]["summary"]) == 500

    def test_log_directory_creation(self, tmp_path: Path) -> None:
        """TurnLogger creates dir if not exists."""
        nested = tmp_path / "deep" / "nested" / "logs"
        assert not nested.exists()
        logger = TurnLogger(log_dir=nested)
        assert nested.exists()
        assert nested.is_dir()

    def test_multiple_turns_sequential(self, tmp_path: Path) -> None:
        """Write 5 turns, verify all 5 in file."""
        logger = _make_logger(tmp_path)
        session_id = "s-seq"
        for i in range(5):
            logger.turn_start(
                turn_id=f"seq-{i}",
                session_id=session_id,
                user_msg=f"msg {i}",
                source="test",
            )
        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        assert len(lines) == 5
        for i, entry in enumerate(lines):
            assert entry["turn_id"] == f"seq-{i}"
            assert entry["user_msg"] == f"msg {i}"

    def test_iso_timestamp_format(self, tmp_path: Path) -> None:
        """Verify ts field is valid ISO 8601."""
        logger = _make_logger(tmp_path)
        session_id = "s-ts"
        logger.turn_start(
            turn_id="t-ts",
            session_id=session_id,
            user_msg="timestamp check",
            source="test",
        )
        log_file = _log_path(logger, session_id)
        lines = _read_lines(log_file)
        ts_str = lines[0]["ts"]
        # Should parse without error
        parsed = datetime.fromisoformat(ts_str)
        # Should have timezone info (UTC)
        assert parsed.tzinfo is not None


# ── TestLogDateRotation ──────────────────────────────────────────────

class TestLogDateRotation:

    def test_log_per_date_file(self, tmp_path: Path) -> None:
        """Each date should get its own log file under {session_id}/ subdirectory."""
        logger = _make_logger(tmp_path)
        session_id = "sd"
        logger.turn_start(
            turn_id="d1",
            session_id=session_id,
            user_msg="date test",
            source="test",
        )
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_file = logger._dir / session_id / f"{today}.jsonl"
        assert date_file.exists(), f"Expected per-date log file at {date_file}"
        lines = _read_lines(date_file)
        assert len(lines) == 1
        assert lines[0]["turn_id"] == "d1"
