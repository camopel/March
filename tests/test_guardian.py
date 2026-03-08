"""Comprehensive Guardian tests.

Covers:
  - Guardian lifecycle (start, stop, crash detection)
  - Config backup / rollback / rotation
  - Sub-agent & PID registration, deregistration, timeout notifications
  - Recovery (completion notification, crash handling, stale log detection)
  - Health-check loop behaviour
  - Event persistence (events.jsonl) and notification file (notifications.jsonl)

All tests run without external services. Uses pytest + pytest-asyncio.
Each test is independent — no shared mutable state between tests.
Does NOT duplicate tests from test_session_reset.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from march.agents.guardian import (
    CONFIG_BACKUP_DIR,
    DEFAULT_STALE_THRESHOLD,
    EVENTS_FILE,
    GUARDIAN_STATE_DIR,
    NOTIFICATIONS_FILE,
    GuardianEvent,
    REGISTRY_FILE,
    Guardian,
    GuardianConfig,
    WatchEntry,
    MAX_PERSISTED_EVENTS,
    run_guardian,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def tmp_guardian_dirs(tmp_path, monkeypatch):
    """Redirect all guardian state/config paths into a temp directory."""
    state_dir = tmp_path / "guardian"
    backup_dir = state_dir / "config_backups"
    registry = state_dir / "watched.json"
    events_file = state_dir / "events.jsonl"
    notifications_file = state_dir / "notifications.jsonl"
    state_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)

    monkeypatch.setattr("march.agents.guardian.GUARDIAN_STATE_DIR", state_dir)
    monkeypatch.setattr("march.agents.guardian.CONFIG_BACKUP_DIR", backup_dir)
    monkeypatch.setattr("march.agents.guardian.REGISTRY_FILE", registry)
    monkeypatch.setattr("march.agents.guardian.EVENTS_FILE", events_file)
    monkeypatch.setattr("march.agents.guardian.NOTIFICATIONS_FILE", notifications_file)

    return state_dir, backup_dir, registry


@pytest.fixture()
def guardian(tmp_guardian_dirs):
    """Return a Guardian with paths redirected to tmp."""
    _, backup_dir, _ = tmp_guardian_dirs
    cfg = GuardianConfig(
        check_interval=1,
        config_backup_count=3,
        march_config_path=str(backup_dir.parent / "config.yaml"),
    )
    g = Guardian(config=cfg)
    return g


@pytest.fixture()
def config_file(tmp_guardian_dirs):
    """Create a fake march config.yaml in the tmp dir."""
    state_dir, _, _ = tmp_guardian_dirs
    cfg_path = state_dir / "config.yaml"
    cfg_path.write_text("llm:\n  default: litellm\n")
    return cfg_path


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianLifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianLifecycle:
    """Guardian start, stop, and crash-detection behaviour."""

    def test_guardian_starts_with_main(self):
        """Verify _start_subprocess is called for guardian when --no-guardian is NOT set.

        The start command calls _start_subprocess("guardian", "start") which
        spawns the guardian as a background process.
        """
        with patch("march.cli.start._start_subprocess", return_value=12345) as mock_sp, \
             patch("march.cli.start._find_march_pids", return_value=[]), \
             patch("march.cli.start._ensure_templates"), \
             patch("march.app.MarchApp") as MockApp:

            app_instance = MagicMock()
            app_instance.config.channels.terminal.enabled = True
            app_instance.config.channels.matrix.enabled = False
            app_instance.run = MagicMock()  # prevent actual run
            MockApp.return_value = app_instance

            from click.testing import CliRunner
            from march.cli.start import start

            runner = CliRunner()
            result = runner.invoke(start, [])

            # _start_subprocess should have been called with "guardian", "start"
            guardian_calls = [
                c for c in mock_sp.call_args_list
                if c.args and c.args[0] == "guardian"
            ]
            assert len(guardian_calls) >= 1, (
                f"Expected _start_subprocess('guardian', ...) call, got: {mock_sp.call_args_list}"
            )

    def test_guardian_not_started_with_no_guardian_flag(self):
        """When --no-guardian is passed, guardian subprocess should NOT be spawned."""
        with patch("march.cli.start._start_subprocess", return_value=12345) as mock_sp, \
             patch("march.cli.start._find_march_pids", return_value=[]), \
             patch("march.cli.start._ensure_templates"), \
             patch("march.app.MarchApp") as MockApp:

            app_instance = MagicMock()
            app_instance.config.channels.terminal.enabled = True
            app_instance.config.channels.matrix.enabled = False
            app_instance.run = MagicMock()
            MockApp.return_value = app_instance

            from click.testing import CliRunner
            from march.cli.start import start

            runner = CliRunner()
            result = runner.invoke(start, ["--no-guardian"])

            guardian_calls = [
                c for c in mock_sp.call_args_list
                if c.args and c.args[0] == "guardian"
            ]
            assert len(guardian_calls) == 0, (
                f"Guardian should NOT be started with --no-guardian, but got: {mock_sp.call_args_list}"
            )

    @pytest.mark.asyncio
    async def test_guardian_stops_on_shutdown(self, guardian, tmp_guardian_dirs):
        """When stop() is called, the run_loop exits."""
        await guardian.initialize()

        # Start the loop in a task
        loop_task = asyncio.create_task(guardian.run_loop())
        # Give the event loop a chance to start the task
        await asyncio.sleep(0.1)
        assert guardian._running is True

        # Stop it
        await guardian.stop()
        assert guardian._running is False

        # The loop task should finish within a reasonable time
        try:
            await asyncio.wait_for(loop_task, timeout=3)
        except asyncio.TimeoutError:
            loop_task.cancel()
            pytest.fail("Guardian run_loop did not exit after stop()")

    @pytest.mark.asyncio
    async def test_guardian_detects_main_crash(self, guardian, tmp_guardian_dirs):
        """If a watched PID dies, guardian detects it and writes notification."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        # Register a fake PID that doesn't exist
        entry = WatchEntry(
            id="main-proc",
            pid=999999,  # almost certainly not running
            command="march main",
            target="room:!test:example.com",
        )
        guardian.register(entry)

        await guardian._check_entries()

        # Should have written a notification
        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "pid_died" in data["event_type"]
        assert "march main" in data["message"]

        # Entry should have been removed (dead PID auto-removed)
        assert "main-proc" not in guardian._entries

    def test_is_pid_alive_with_current_process(self):
        """_is_pid_alive returns True for the current process."""
        assert Guardian._is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_with_dead_pid(self):
        """_is_pid_alive returns False for a non-existent PID."""
        assert Guardian._is_pid_alive(999999) is False

    def test_is_pid_alive_with_zero_or_negative(self):
        """_is_pid_alive returns False for pid <= 0."""
        assert Guardian._is_pid_alive(0) is False
        assert Guardian._is_pid_alive(-1) is False


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianConfigBackup
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianConfigBackup:
    """Config backup, rollback, and rotation on restart."""

    def test_config_backup_on_restart(self, guardian, config_file, tmp_guardian_dirs):
        """register_restart() copies config.yaml into config_backups/."""
        guardian.config.march_config_path = str(config_file)
        _, backup_dir, _ = tmp_guardian_dirs

        backup_path = guardian.register_restart()

        assert backup_path != ""
        assert Path(backup_path).exists()
        assert Path(backup_path).parent == backup_dir
        # Content should match original
        assert Path(backup_path).read_text() == config_file.read_text()

    def test_config_backup_returns_empty_if_no_config(self, guardian, tmp_guardian_dirs):
        """register_restart() returns '' if config.yaml doesn't exist."""
        guardian.config.march_config_path = "/nonexistent/config.yaml"
        assert guardian.register_restart() == ""

    @pytest.mark.asyncio
    async def test_config_rollback_on_failure(self, guardian, config_file, tmp_guardian_dirs):
        """If restart fails, recover_from_failed_restart tries config backups."""
        guardian.config.march_config_path = str(config_file)
        _, backup_dir, _ = tmp_guardian_dirs
        state_dir = tmp_guardian_dirs[0]
        notifications_file = state_dir / "notifications.jsonl"

        # Create a backup
        guardian.register_restart()
        backups_before = list(backup_dir.glob("*.yaml"))
        assert len(backups_before) == 1

        # Now corrupt the config
        config_file.write_text("INVALID YAML !!!")

        # Mock subprocess.run to simulate recovery success on backup
        with patch("subprocess.run") as mock_run, \
             patch.object(guardian, "verify_after_restart", new_callable=AsyncMock, return_value=True):

            mock_run.return_value = MagicMock(returncode=0)
            result = await guardian.recover_from_failed_restart()

        assert result is True
        # Config should have been restored from backup
        restored_content = config_file.read_text()
        assert "INVALID" not in restored_content
        assert "litellm" in restored_content

        # Should have written a restart_recovered notification
        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert any("restart_recovered" in line for line in lines)

    @pytest.mark.asyncio
    async def test_config_rollback_all_fail(self, guardian, config_file, tmp_guardian_dirs):
        """If all backups fail, recover_from_failed_restart returns False."""
        guardian.config.march_config_path = str(config_file)
        state_dir = tmp_guardian_dirs[0]
        notifications_file = state_dir / "notifications.jsonl"

        # Create a backup
        guardian.register_restart()

        # Mock everything to fail
        with patch("subprocess.run", side_effect=Exception("fail")):
            result = await guardian.recover_from_failed_restart()

        assert result is False
        # Should have written a restart_failed notification
        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert last["event_type"] == "restart_failed"
        assert "failed" in last["message"].lower()

    def test_backup_rotation(self, guardian, config_file, tmp_guardian_dirs):
        """Old backups beyond config_backup_count are pruned."""
        guardian.config.march_config_path = str(config_file)
        guardian.config.config_backup_count = 3
        _, backup_dir, _ = tmp_guardian_dirs

        # Create more backups than the limit
        for i in range(6):
            # Ensure unique timestamps
            with patch("march.agents.guardian.time") as mock_time:
                mock_time.time.return_value = 1000000 + i
                guardian.register_restart()

        backups = list(backup_dir.glob("*.yaml"))
        assert len(backups) <= 3, f"Expected at most 3 backups, got {len(backups)}"

    def test_backup_preserves_newest(self, guardian, config_file, tmp_guardian_dirs):
        """After rotation, the newest backups are kept."""
        guardian.config.march_config_path = str(config_file)
        guardian.config.config_backup_count = 2
        _, backup_dir, _ = tmp_guardian_dirs

        # Create 4 backups with distinct timestamps
        paths = []
        for i in range(4):
            with patch("march.agents.guardian.time") as mock_time:
                mock_time.time.return_value = 1000000 + i * 100
                p = guardian.register_restart()
                paths.append(p)

        remaining = sorted(backup_dir.glob("*.yaml"), reverse=True)
        assert len(remaining) == 2
        # The two newest should survive
        remaining_names = {r.name for r in remaining}
        assert Path(paths[-1]).name in remaining_names
        assert Path(paths[-2]).name in remaining_names


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianSubAgentRegistration
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianSubAgentRegistration:
    """Register / deregister sub-agents and PIDs; timeout notifications."""

    @pytest.mark.asyncio
    async def test_register_subagent(self, guardian, tmp_guardian_dirs):
        """Register a sub-agent with session_key → appears in watched.json."""
        await guardian.initialize()
        _, _, registry = tmp_guardian_dirs

        entry = WatchEntry(
            id="subagent-abc",
            session_key="session:abc:123",
            command="coding task",
            target="room:!test:example.com",
            timeout=300,
        )
        guardian.register(entry)

        assert "subagent-abc" in guardian._entries
        # Verify persisted to disk
        data = json.loads(registry.read_text())
        assert "subagent-abc" in data
        assert data["subagent-abc"]["session_key"] == "session:abc:123"
        assert data["subagent-abc"]["command"] == "coding task"

    @pytest.mark.asyncio
    async def test_register_pid(self, guardian, tmp_guardian_dirs):
        """Register a PID → appears in watched.json."""
        await guardian.initialize()
        _, _, registry = tmp_guardian_dirs

        entry = WatchEntry(
            id="bg-task-1",
            pid=54321,
            command="background build",
            target="room:!build:example.com",
            log_path="/tmp/build.log",
        )
        guardian.register(entry)

        assert "bg-task-1" in guardian._entries
        data = json.loads(registry.read_text())
        assert data["bg-task-1"]["pid"] == 54321
        assert data["bg-task-1"]["log_path"] == "/tmp/build.log"

    @pytest.mark.asyncio
    async def test_deregister_on_completion(self, guardian, tmp_guardian_dirs):
        """When remove() is called, entry is removed from watched.json."""
        await guardian.initialize()
        _, _, registry = tmp_guardian_dirs

        entry = WatchEntry(id="task-done", pid=11111, command="finished task")
        guardian.register(entry)
        assert "task-done" in guardian._entries

        result = guardian.remove("task-done")
        assert result is True
        assert "task-done" not in guardian._entries

        # Verify disk state
        data = json.loads(registry.read_text())
        assert "task-done" not in data

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_returns_false(self, guardian, tmp_guardian_dirs):
        """Removing a non-existent entry returns False."""
        await guardian.initialize()
        assert guardian.remove("nonexistent") is False

    @pytest.mark.asyncio
    async def test_timeout_notification(self, guardian, tmp_guardian_dirs):
        """If a registered task's log goes stale beyond timeout, guardian writes notification."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        # Create a stale log file (modified long ago)
        log_file = state_dir / "stale_task.log"
        log_file.write_text("some output")
        # Set mtime to 10 minutes ago
        old_time = time.time() - 600
        os.utime(log_file, (old_time, old_time))

        entry = WatchEntry(
            id="stale-task",
            pid=os.getpid(),  # use our own PID so it's "alive"
            log_path=str(log_file),
            command="long running task",
            target="room:!alerts:example.com",
            timeout=60,  # 60s threshold
        )
        guardian.register(entry)

        await guardian._check_entries()

        # Notification should be written
        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "log_stale"
        assert "frozen" in data["message"].lower() or "stale" in data["message"].lower()

        # Stale log entries are NOT auto-removed (might recover)
        assert "stale-task" in guardian._entries

    @pytest.mark.asyncio
    async def test_notification_target(self, guardian, tmp_guardian_dirs):
        """Events include the correct entry_id for the triggering entry."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        entry = WatchEntry(
            id="targeted-task",
            pid=999999,  # dead PID
            command="targeted process",
            target="room:!specific-room:matrix.org",
        )
        guardian.register(entry)

        await guardian._check_entries()

        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entry_id"] == "targeted-task"
        assert data["event_type"] == "pid_died"

    @pytest.mark.asyncio
    async def test_multiple_entries_independent(self, guardian, tmp_guardian_dirs):
        """Multiple registered entries are checked independently."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        # One alive, one dead
        guardian.register(WatchEntry(
            id="alive-task", pid=os.getpid(), command="alive"
        ))
        guardian.register(WatchEntry(
            id="dead-task", pid=999999, command="dead"
        ))

        await guardian._check_entries()

        # Alive should remain, dead should be removed
        assert "alive-task" in guardian._entries
        assert "dead-task" not in guardian._entries

        # Should have one notification for the dead task
        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entry_id"] == "dead-task"


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianRecovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianRecovery:
    """Recovery: completion notification, crash handling, stale log detection."""

    @pytest.mark.asyncio
    async def test_parent_receives_completion(self, guardian, tmp_guardian_dirs):
        """When a watched PID dies (completes), guardian writes notification."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        entry = WatchEntry(
            id="completed-subagent",
            pid=999999,
            command="coding task",
            target="room:!parent:example.com",
        )
        guardian.register(entry)

        await guardian._check_entries()

        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "pid_died"
        assert data["entry_id"] == "completed-subagent"

    @pytest.mark.asyncio
    async def test_parent_continues_after_subagent_crash(self, guardian, tmp_guardian_dirs):
        """If a sub-agent dies, the entry is removed so the parent can handle failure."""
        await guardian.initialize()

        guardian.register(WatchEntry(
            id="crashed-agent", pid=999999, command="crashed task",
            target="room:!parent:example.com",
        ))
        guardian.register(WatchEntry(
            id="healthy-agent", pid=os.getpid(), command="healthy task",
            target="room:!parent:example.com",
        ))

        await guardian._check_entries()

        # Crashed entry removed, healthy remains
        assert "crashed-agent" not in guardian._entries
        assert "healthy-agent" in guardian._entries

    @pytest.mark.asyncio
    async def test_stale_log_detection(self, guardian, tmp_guardian_dirs):
        """If a registered task's log goes stale, guardian detects and writes notification."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        log_file = state_dir / "stale.log"
        log_file.write_text("output")
        old_time = time.time() - 1000
        os.utime(log_file, (old_time, old_time))

        entry = WatchEntry(
            id="stale-log-task",
            pid=os.getpid(),  # alive PID
            log_path=str(log_file),
            command="stale log task",
            target="room:!logs:example.com",
            timeout=60,
        )
        guardian.register(entry)

        await guardian._check_entries()

        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "log_stale"
        assert data["entry_id"] == "stale-log-task"
        assert "frozen" in data["message"].lower() or "stale" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_fresh_log_not_flagged(self, guardian, tmp_guardian_dirs):
        """A recently-modified log should NOT trigger stale detection."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        log_file = state_dir / "fresh.log"
        log_file.write_text("recent output")
        # mtime is now (fresh)

        entry = WatchEntry(
            id="fresh-log-task",
            pid=os.getpid(),
            log_path=str(log_file),
            command="fresh log task",
            target="room:!logs:example.com",
            timeout=300,
        )
        guardian.register(entry)

        await guardian._check_entries()

        # No notification should be written
        if notifications_file.exists():
            assert notifications_file.read_text().strip() == ""
        assert "fresh-log-task" in guardian._entries

    @pytest.mark.asyncio
    async def test_missing_log_treated_as_stale(self, guardian, tmp_guardian_dirs):
        """If the log file doesn't exist, it's treated as stale."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"

        entry = WatchEntry(
            id="missing-log",
            pid=os.getpid(),
            log_path="/nonexistent/path/task.log",
            command="missing log task",
            target="room:!logs:example.com",
            timeout=60,
        )
        guardian.register(entry)

        await guardian._check_entries()

        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "log_stale"

        # Stale log entries are NOT auto-removed (might recover)
        assert "missing-log" in guardian._entries

    @pytest.mark.asyncio
    async def test_stale_log_not_auto_removed(self, guardian, tmp_guardian_dirs):
        """Stale log entries stay registered — they might recover."""
        await guardian.initialize()
        state_dir, _, _ = tmp_guardian_dirs

        log_file = state_dir / "maybe_stale.log"
        log_file.write_text("output")
        old_time = time.time() - 1000
        os.utime(log_file, (old_time, old_time))

        entry = WatchEntry(
            id="recoverable",
            pid=os.getpid(),
            log_path=str(log_file),
            command="recoverable task",
            timeout=60,
        )
        guardian.register(entry)

        await guardian._check_entries()

        # Entry should still be registered
        assert "recoverable" in guardian._entries

    @pytest.mark.asyncio
    async def test_dead_pid_auto_removed(self, guardian, tmp_guardian_dirs):
        """Dead PID entries are auto-removed — no point watching a gone process."""
        await guardian.initialize()

        guardian.register(WatchEntry(
            id="gone-process", pid=999999, command="dead task"
        ))

        await guardian._check_entries()

        assert "gone-process" not in guardian._entries


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianHealthCheck
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianHealthCheck:
    """Health-check loop: periodic runs, logging, and memory pressure detection."""

    @pytest.mark.asyncio
    async def test_health_check_runs_periodically(self, guardian, tmp_guardian_dirs):
        """Guardian runs _check_entries at each check_interval tick."""
        await guardian.initialize()
        guardian.config.check_interval = 1  # 1 second

        check_count = 0
        original_check = guardian._check_entries

        async def counting_check():
            nonlocal check_count
            check_count += 1
            await original_check()

        guardian._check_entries = counting_check

        # Run the loop for ~2.5 seconds
        loop_task = asyncio.create_task(guardian.run_loop())
        await asyncio.sleep(2.5)
        await guardian.stop()

        try:
            await asyncio.wait_for(loop_task, timeout=3)
        except asyncio.TimeoutError:
            loop_task.cancel()

        # Should have run at least 2 checks in 2.5 seconds with 1s interval
        assert check_count >= 2, f"Expected >=2 checks, got {check_count}"

    @pytest.mark.asyncio
    async def test_health_check_logs_status(self, guardian, tmp_guardian_dirs):
        """Guardian logs info when it starts (status indicator)."""
        await guardian.initialize()

        with patch("march.agents.guardian.logger") as mock_logger:
            loop_task = asyncio.create_task(guardian.run_loop())
            await asyncio.sleep(0.1)
            await guardian.stop()
            try:
                await asyncio.wait_for(loop_task, timeout=3)
            except asyncio.TimeoutError:
                loop_task.cancel()

            # Should have logged the startup message
            mock_logger.info.assert_any_call(
                "guardian started, check_interval=%ds", guardian.config.check_interval
            )

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Memory pressure detection not yet implemented in Guardian. "
               "Guardian currently only monitors PID liveness and log staleness. "
               "A future enhancement would add psutil-based memory checks.",
        strict=False,
    )
    async def test_health_check_detects_memory_pressure(self, guardian, tmp_guardian_dirs):
        """If process memory exceeds threshold, guardian logs warning.

        Not yet implemented — Guardian currently monitors PIDs and logs only.
        """
        await guardian.initialize()

        # Register our own PID
        entry = WatchEntry(
            id="memory-hog",
            pid=os.getpid(),
            command="memory intensive task",
            target="room:!test:example.com",
        )
        guardian.register(entry)

        # This would require a memory_threshold config and psutil integration
        assert hasattr(guardian.config, "memory_threshold"), (
            "GuardianConfig should have a memory_threshold field"
        )

        with patch("march.agents.guardian.logger") as mock_logger:
            await guardian._check_entries()
            # Should log a warning about memory
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "memory" in str(c).lower()
            ]
            assert len(warning_calls) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianEventPersistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianEventPersistence:
    """Event persistence (events.jsonl) and notification file (notifications.jsonl)."""

    @pytest.mark.asyncio
    async def test_send_to_main_agent_persists_event(self, guardian, tmp_guardian_dirs):
        """_send_to_main_agent writes to events.jsonl."""
        state_dir, _, _ = tmp_guardian_dirs
        events_file = state_dir / "events.jsonl"
        await guardian.initialize()

        event = GuardianEvent(
            ts=time.time(),
            level="warning",
            event_type="pid_died",
            message="disk event",
            entry_id="d1",
        )
        await guardian._send_to_main_agent(event)

        assert events_file.exists()
        lines = events_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "pid_died"
        assert data["entry_id"] == "d1"
        assert data["message"] == "disk event"

    @pytest.mark.asyncio
    async def test_send_to_main_agent_writes_notification(self, guardian, tmp_guardian_dirs):
        """_send_to_main_agent writes to notifications.jsonl."""
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"
        await guardian.initialize()

        event = GuardianEvent(
            ts=time.time(),
            level="warning",
            event_type="pid_died",
            message="test notification",
            entry_id="n1",
        )
        await guardian._send_to_main_agent(event)

        assert notifications_file.exists()
        lines = notifications_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "pid_died"
        assert data["entry_id"] == "n1"
        assert "[Guardian]" in data["message"]

    @pytest.mark.asyncio
    async def test_multiple_events_persisted(self, guardian, tmp_guardian_dirs):
        """Multiple events are all persisted to events.jsonl and notifications.jsonl."""
        state_dir, _, _ = tmp_guardian_dirs
        events_file = state_dir / "events.jsonl"
        notifications_file = state_dir / "notifications.jsonl"
        await guardian.initialize()

        for i in range(3):
            event = GuardianEvent(
                ts=time.time(),
                level="warning",
                event_type="pid_died",
                message=f"event-{i}",
                entry_id=f"e{i}",
            )
            await guardian._send_to_main_agent(event)

        # All persisted to events.jsonl
        lines = events_file.read_text().strip().splitlines()
        assert len(lines) == 3

        # All written to notifications.jsonl
        nlines = notifications_file.read_text().strip().splitlines()
        assert len(nlines) == 3

    @pytest.mark.asyncio
    async def test_event_rotation(self, guardian, tmp_guardian_dirs):
        """Events file is rotated to keep at most MAX_PERSISTED_EVENTS."""
        state_dir, _, _ = tmp_guardian_dirs
        events_file = state_dir / "events.jsonl"
        await guardian.initialize()

        # Write more than MAX_PERSISTED_EVENTS
        for i in range(MAX_PERSISTED_EVENTS + 20):
            event = GuardianEvent(
                ts=time.time(),
                level="warning",
                event_type="pid_died",
                message=f"event-{i}",
                entry_id=f"e{i}",
            )
            await guardian._send_to_main_agent(event)

        lines = events_file.read_text().strip().splitlines()
        assert len(lines) <= MAX_PERSISTED_EVENTS

        # The newest events should be kept
        last_data = json.loads(lines[-1])
        assert last_data["message"] == f"event-{MAX_PERSISTED_EVENTS + 19}"

    @pytest.mark.asyncio
    async def test_guardian_event_dataclass(self):
        """GuardianEvent serializes and deserializes correctly."""
        event = GuardianEvent(
            ts=1234567890.0,
            level="warning",
            event_type="pid_died",
            message="test",
            entry_id="e1",
        )
        d = event.to_dict()
        restored = GuardianEvent.from_dict(d)
        assert restored.ts == event.ts
        assert restored.level == event.level
        assert restored.event_type == event.event_type
        assert restored.message == event.message
        assert restored.entry_id == event.entry_id

    @pytest.mark.asyncio
    async def test_no_callback_pattern(self):
        """Guardian no longer accepts on_event callback — uses file-based notifications."""
        import inspect
        sig = inspect.signature(Guardian.__init__)
        param_names = list(sig.parameters.keys())
        assert "on_event" not in param_names, "Guardian should not accept on_event parameter"

    @pytest.mark.asyncio
    async def test_check_interval_default_is_30(self):
        """GuardianConfig.check_interval defaults to 30 seconds."""
        cfg = GuardianConfig()
        assert cfg.check_interval == 30

    @pytest.mark.asyncio
    async def test_main_session_id_field_exists(self):
        """GuardianConfig has main_session_id field."""
        cfg = GuardianConfig()
        assert hasattr(cfg, "main_session_id")
        assert cfg.main_session_id == ""


# ═══════════════════════════════════════════════════════════════════════════════
# TestDrainNotifications
# ═══════════════════════════════════════════════════════════════════════════════


class TestDrainNotifications:
    """Test Guardian.drain_notifications() static method."""

    @pytest.mark.asyncio
    async def test_drain_returns_notifications(self, guardian, tmp_guardian_dirs):
        """drain_notifications reads and clears notifications.jsonl."""
        state_dir, _, _ = tmp_guardian_dirs
        notifications_file = state_dir / "notifications.jsonl"
        await guardian.initialize()

        # Write some notifications
        for i in range(3):
            event = GuardianEvent(
                ts=time.time(),
                level="warning",
                event_type="pid_died",
                message=f"msg-{i}",
                entry_id=f"e{i}",
            )
            await guardian._send_to_main_agent(event)

        # Drain them
        notifications = Guardian.drain_notifications()
        assert len(notifications) == 3
        assert notifications[0]["entry_id"] == "e0"
        assert notifications[2]["entry_id"] == "e2"
        assert "[Guardian]" in notifications[0]["message"]

        # File should be empty after drain
        assert notifications_file.read_text().strip() == ""

    @pytest.mark.asyncio
    async def test_drain_empty_file(self, guardian, tmp_guardian_dirs):
        """drain_notifications returns empty list when no notifications."""
        await guardian.initialize()
        notifications = Guardian.drain_notifications()
        assert notifications == []

    @pytest.mark.asyncio
    async def test_drain_nonexistent_file(self, tmp_guardian_dirs, monkeypatch):
        """drain_notifications returns empty list when file doesn't exist."""
        state_dir, _, _ = tmp_guardian_dirs
        # Make sure the file doesn't exist
        nf = state_dir / "notifications.jsonl"
        if nf.exists():
            nf.unlink()
        notifications = Guardian.drain_notifications()
        assert notifications == []

    @pytest.mark.asyncio
    async def test_drain_clears_file(self, guardian, tmp_guardian_dirs):
        """After drain, subsequent drain returns empty."""
        state_dir, _, _ = tmp_guardian_dirs
        await guardian.initialize()

        event = GuardianEvent(
            ts=time.time(), level="warning", event_type="pid_died",
            message="once", entry_id="x",
        )
        await guardian._send_to_main_agent(event)

        first = Guardian.drain_notifications()
        assert len(first) == 1

        second = Guardian.drain_notifications()
        assert len(second) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestWatchEntry
# ═══════════════════════════════════════════════════════════════════════════════


class TestWatchEntry:
    """WatchEntry data class serialization and defaults."""

    def test_to_dict_roundtrip(self):
        """WatchEntry can be serialized and deserialized."""
        entry = WatchEntry(
            id="test-entry",
            pid=12345,
            session_key="session:key:123",
            log_path="/tmp/test.log",
            target="room:!test:example.com",
            timeout=120,
            command="test command",
        )
        d = entry.to_dict()
        restored = WatchEntry.from_dict(d)

        assert restored.id == entry.id
        assert restored.pid == entry.pid
        assert restored.session_key == entry.session_key
        assert restored.log_path == entry.log_path
        assert restored.target == entry.target
        assert restored.timeout == entry.timeout
        assert restored.command == entry.command
        assert restored.registered_at == entry.registered_at

    def test_registered_at_auto_set(self):
        """registered_at is automatically set to current time if not provided."""
        before = time.time()
        entry = WatchEntry(id="auto-time", command="test")
        after = time.time()

        assert before <= entry.registered_at <= after

    def test_from_dict_ignores_extra_keys(self):
        """from_dict ignores unknown keys without error."""
        d = {"id": "test", "pid": 1, "unknown_field": "ignored", "command": "x"}
        entry = WatchEntry.from_dict(d)
        assert entry.id == "test"
        assert entry.pid == 1
        assert not hasattr(entry, "unknown_field")

    def test_defaults(self):
        """WatchEntry has sensible defaults."""
        entry = WatchEntry(id="defaults-test")
        assert entry.pid == 0
        assert entry.session_key == ""
        assert entry.log_path == ""
        assert entry.target == ""
        assert entry.timeout == 0
        assert entry.command == ""


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianState
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianState:
    """State persistence (save/load to watched.json)."""

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, guardian, tmp_guardian_dirs):
        """Entries survive save + load cycle."""
        await guardian.initialize()

        guardian.register(WatchEntry(id="persist-1", pid=111, command="task 1"))
        guardian.register(WatchEntry(id="persist-2", pid=222, command="task 2"))

        # Create a new guardian that loads from the same state dir
        _, _, registry = tmp_guardian_dirs
        g2 = Guardian(config=guardian.config)
        await g2.initialize()

        assert "persist-1" in g2._entries
        assert "persist-2" in g2._entries
        assert g2._entries["persist-1"].pid == 111
        assert g2._entries["persist-2"].command == "task 2"

    @pytest.mark.asyncio
    async def test_load_state_handles_corrupt_json(self, tmp_guardian_dirs):
        """If watched.json is corrupt, guardian starts with empty entries."""
        state_dir, _, registry = tmp_guardian_dirs
        registry.write_text("NOT VALID JSON {{{")

        g = Guardian()
        # Redirect to tmp
        with patch("march.agents.guardian.REGISTRY_FILE", registry), \
             patch("march.agents.guardian.GUARDIAN_STATE_DIR", state_dir):
            await g.initialize()

        assert len(g._entries) == 0

    @pytest.mark.asyncio
    async def test_load_state_handles_missing_file(self, tmp_guardian_dirs):
        """If watched.json doesn't exist, guardian starts with empty entries."""
        state_dir, _, registry = tmp_guardian_dirs
        if registry.exists():
            registry.unlink()

        g = Guardian()
        with patch("march.agents.guardian.REGISTRY_FILE", registry), \
             patch("march.agents.guardian.GUARDIAN_STATE_DIR", state_dir):
            await g.initialize()

        assert len(g._entries) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianStatus
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianStatus:
    """Guardian status reporting."""

    @pytest.mark.asyncio
    async def test_status_empty(self, guardian, tmp_guardian_dirs):
        """Status with no entries."""
        await guardian.initialize()
        status = guardian.status()

        assert status["running"] is False  # not started yet
        assert status["entries"] == []
        assert isinstance(status["config_backups"], int)

    @pytest.mark.asyncio
    async def test_status_with_entries(self, guardian, tmp_guardian_dirs):
        """Status reports alive/dead for registered PIDs."""
        await guardian.initialize()

        guardian.register(WatchEntry(
            id="alive", pid=os.getpid(), command="self"
        ))
        guardian.register(WatchEntry(
            id="dead", pid=999999, command="ghost"
        ))

        status = guardian.status()
        entries = {e["id"]: e for e in status["entries"]}

        assert entries["alive"]["alive"] is True
        assert entries["dead"]["alive"] is False
        assert entries["alive"]["command"] == "self"
        assert entries["dead"]["command"] == "ghost"

    @pytest.mark.asyncio
    async def test_status_counts_backups(self, guardian, config_file, tmp_guardian_dirs):
        """Status reports the number of config backups."""
        guardian.config.march_config_path = str(config_file)
        await guardian.initialize()

        # Use distinct timestamps so backups don't overwrite each other
        with patch("march.agents.guardian.time") as mock_time:
            mock_time.time.return_value = 1000000
            guardian.register_restart()
        with patch("march.agents.guardian.time") as mock_time:
            mock_time.time.return_value = 1000001
            guardian.register_restart()

        status = guardian.status()
        assert status["config_backups"] >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# TestRunGuardian
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunGuardian:
    """Test the run_guardian() entry point."""

    @pytest.mark.asyncio
    async def test_run_guardian_initializes_and_runs(self, tmp_guardian_dirs):
        """run_guardian creates a Guardian, initializes it, and starts the loop."""
        cfg = GuardianConfig(check_interval=1)

        with patch("march.agents.guardian.Guardian") as MockGuardian:
            instance = AsyncMock()
            instance.run_loop = AsyncMock()
            instance.stop = AsyncMock()
            instance.initialize = AsyncMock()
            MockGuardian.return_value = instance

            # We need to handle signal handlers — mock the event loop
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop_instance = MagicMock()
                mock_loop.return_value = mock_loop_instance

                await run_guardian(cfg)

            instance.initialize.assert_called_once()
            instance.run_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_after_restart_success(self, guardian):
        """verify_after_restart returns True when march --status succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = await guardian.verify_after_restart()
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_after_restart_failure(self, guardian):
        """verify_after_restart returns False when march --status fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = await guardian.verify_after_restart()
            assert result is False

    @pytest.mark.asyncio
    async def test_verify_after_restart_timeout(self, guardian):
        """verify_after_restart returns False on timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("march", 30)):
            result = await guardian.verify_after_restart()
            assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# TestGuardianCLI
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardianCLI:
    """Test the CLI commands (guardian_cmd.py) via Click's test runner."""

    def test_guardian_status_no_registry(self, tmp_path):
        """'guardian status' with no registry file prints appropriate message."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()

        from click.testing import CliRunner
        from march.cli.guardian_cmd import guardian_status

        # Patch Path.home() to return our fake home (no .march/guardian dir)
        with patch("pathlib.Path.home", return_value=fake_home):
            runner = CliRunner()
            result = runner.invoke(guardian_status)
            assert result.exit_code == 0
            assert "no registry" in result.output.lower()

    def test_guardian_status_with_entries(self, tmp_path):
        """'guardian status' with entries shows them."""
        fake_home = tmp_path / "fakehome"
        guardian_dir = fake_home / ".march" / "guardian"
        guardian_dir.mkdir(parents=True)
        registry = guardian_dir / "watched.json"

        data = {
            "task-1": {
                "id": "task-1",
                "pid": os.getpid(),
                "session_key": "",
                "log_path": "",
                "target": "",
                "timeout": 0,
                "command": "test task",
                "registered_at": time.time(),
            }
        }
        registry.write_text(json.dumps(data))

        from click.testing import CliRunner
        from march.cli.guardian_cmd import guardian_status

        with patch("pathlib.Path.home", return_value=fake_home):
            runner = CliRunner()
            result = runner.invoke(guardian_status)
            assert result.exit_code == 0
            assert "1 watched" in result.output or "task-1" in result.output

    def test_guardian_status_corrupt_registry(self, tmp_path):
        """'guardian status' with corrupt JSON prints error message."""
        fake_home = tmp_path / "fakehome"
        guardian_dir = fake_home / ".march" / "guardian"
        guardian_dir.mkdir(parents=True)
        registry = guardian_dir / "watched.json"
        registry.write_text("NOT VALID JSON {{{")

        from click.testing import CliRunner
        from march.cli.guardian_cmd import guardian_status

        with patch("pathlib.Path.home", return_value=fake_home):
            runner = CliRunner()
            result = runner.invoke(guardian_status)
            assert result.exit_code == 0
            assert "corrupted" in result.output.lower()

    def test_guardian_status_empty_registry(self, tmp_path):
        """'guardian status' with empty registry shows 0 entries."""
        fake_home = tmp_path / "fakehome"
        guardian_dir = fake_home / ".march" / "guardian"
        guardian_dir.mkdir(parents=True)
        registry = guardian_dir / "watched.json"
        registry.write_text("{}")

        from click.testing import CliRunner
        from march.cli.guardian_cmd import guardian_status

        with patch("pathlib.Path.home", return_value=fake_home):
            runner = CliRunner()
            result = runner.invoke(guardian_status)
            assert result.exit_code == 0
            assert "0 watched" in result.output
