"""Guardian process for March.

A lightweight process that survives March restarts. Handles:
1. PID watching: Monitor registered PIDs for liveness + log staleness
2. Restart protection: Backup config, verify health after restart,
   revert on failure

The guardian communicates with the main agent by writing notifications
to ``~/.march/guardian/notifications.jsonl``.  The Orchestrator checks
this file at the start of each turn and injects pending notifications
as system messages.  Events are also persisted to ``events.jsonl`` for
audit trail.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from march.logging import get_logger

logger = get_logger("march.guardian")

GUARDIAN_STATE_DIR = Path.home() / ".march" / "guardian"
CONFIG_BACKUP_DIR = GUARDIAN_STATE_DIR / "config_backups"
REGISTRY_FILE = GUARDIAN_STATE_DIR / "watched.json"
EVENTS_FILE = GUARDIAN_STATE_DIR / "events.jsonl"
NOTIFICATIONS_FILE = GUARDIAN_STATE_DIR / "notifications.jsonl"
DEFAULT_STALE_THRESHOLD = 300  # seconds
MAX_PERSISTED_EVENTS = 100


@dataclass
class WatchEntry:
    """A registered PID or session to watch."""

    id: str
    pid: int = 0
    session_key: str = ""
    log_path: str = ""
    target: str = ""  # notification target (channel/room)
    timeout: int = 0  # max seconds before considering stale
    command: str = ""  # description of what this process does
    registered_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.registered_at:
            self.registered_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GuardianEvent:
    """Event from guardian to parent agent."""

    ts: float
    level: str  # "warning", "critical"
    event_type: str  # "pid_died", "log_stale", "restart_failed", "restart_recovered"
    message: str
    entry_id: str = ""  # which watch entry triggered this

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardianEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GuardianConfig:
    """Guardian configuration."""

    log_stale_threshold: int = DEFAULT_STALE_THRESHOLD
    config_backup_count: int = 5
    check_interval: int = 30  # seconds between PID checks
    march_config_path: str = str(Path.home() / ".march" / "config.yaml")
    main_session_id: str = ""  # main agent's session ID (informational)


class Guardian:
    """Guardian process — monitors PIDs and protects restarts.

    This is designed to be run as a standalone process that outlives
    the main March runtime. Events are persisted to events.jsonl for
    audit trail and written to notifications.jsonl for the main agent
    to pick up on its next turn.
    """

    def __init__(
        self,
        config: GuardianConfig | None = None,
    ) -> None:
        self.config = config or GuardianConfig()
        self._entries: dict[str, WatchEntry] = {}
        self._running = False

    async def initialize(self) -> None:
        """Set up directories and load state."""
        GUARDIAN_STATE_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def register(self, entry: WatchEntry) -> None:
        """Register a PID or session to watch."""
        self._entries[entry.id] = entry
        self._save_state()
        logger.info("guardian.register id=%s pid=%d cmd=%s", entry.id, entry.pid, entry.command[:60])

    def remove(self, entry_id: str) -> bool:
        """Remove a watch entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save_state()
            return True
        return False

    def status(self) -> dict[str, Any]:
        """Get guardian status."""
        entries = []
        for entry in self._entries.values():
            alive = self._is_pid_alive(entry.pid) if entry.pid else None
            log_stale = self._is_log_stale(entry) if entry.log_path else None
            entries.append({
                "id": entry.id,
                "pid": entry.pid,
                "alive": alive,
                "log_stale": log_stale,
                "command": entry.command,
                "age_seconds": time.time() - entry.registered_at,
            })
        return {
            "running": self._running,
            "entries": entries,
            "config_backups": len(list(CONFIG_BACKUP_DIR.glob("*.yaml"))),
        }

    async def run_loop(self) -> None:
        """Main guardian monitoring loop."""
        self._running = True
        logger.info("guardian started, check_interval=%ds", self.config.check_interval)

        while self._running:
            await self._check_entries()
            await asyncio.sleep(self.config.check_interval)

    async def stop(self) -> None:
        """Stop the guardian loop."""
        self._running = False

    async def _check_entries(self) -> None:
        """Check all registered entries for liveness."""
        dead_pids = []
        for entry_id, entry in list(self._entries.items()):
            if entry.pid:
                if not self._is_pid_alive(entry.pid):
                    await self._send_to_main_agent(GuardianEvent(
                        ts=time.time(),
                        level="warning",
                        event_type="pid_died",
                        message=f"Sub-agent '{entry.command}' (PID {entry.pid}) died",
                        entry_id=entry.id,
                    ))
                    dead_pids.append(entry_id)
                    continue

            if entry.log_path and self._is_log_stale(entry):
                threshold = entry.timeout or self.config.log_stale_threshold
                log_path = Path(entry.log_path)
                if log_path.exists():
                    stale_seconds = int(time.time() - log_path.stat().st_mtime)
                else:
                    stale_seconds = threshold
                await self._send_to_main_agent(GuardianEvent(
                    ts=time.time(),
                    level="warning",
                    event_type="log_stale",
                    message=f"Sub-agent '{entry.command}' may be frozen (log stale {stale_seconds}s)",
                    entry_id=entry.id,
                ))
                # Don't auto-remove stale logs — process might recover

        # Auto-remove dead PIDs (they're gone, no point watching)
        for entry_id in dead_pids:
            self.remove(entry_id)

    # ── Restart Protection ───────────────────────────────────────────────

    def register_restart(self) -> str:
        """Called before a March restart. Backs up current config.

        Returns the backup file path.
        """
        config_path = Path(self.config.march_config_path)
        if not config_path.exists():
            return ""

        # Create timestamped backup
        timestamp = int(time.time())
        backup_path = CONFIG_BACKUP_DIR / f"config_{timestamp}.yaml"
        shutil.copy2(config_path, backup_path)

        # Prune old backups
        backups = sorted(CONFIG_BACKUP_DIR.glob("*.yaml"), reverse=True)
        for old in backups[self.config.config_backup_count:]:
            old.unlink()

        logger.info("guardian.register_restart backup=%s", backup_path)
        return str(backup_path)

    async def verify_after_restart(self) -> bool:
        """Verify March is healthy after a restart.

        Returns True if healthy, False if needs recovery.
        """
        try:
            result = subprocess.run(
                ["march", "--status"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def recover_from_failed_restart(self) -> bool:
        """Try config backups newest-to-oldest to recover from failed restart.

        Returns True if recovery succeeded.
        """
        config_path = Path(self.config.march_config_path)
        backups = sorted(CONFIG_BACKUP_DIR.glob("*.yaml"), reverse=True)

        for backup in backups:
            logger.info("guardian.recover trying backup=%s", backup.name)
            try:
                shutil.copy2(backup, config_path)

                # Attempt restart
                result = subprocess.run(
                    ["march", "serve", "--daemon"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    # Verify health
                    await asyncio.sleep(2)
                    if await self.verify_after_restart():
                        await self._send_to_main_agent(GuardianEvent(
                            ts=time.time(),
                            level="warning",
                            event_type="restart_recovered",
                            message=f"March recovered using config backup: {backup.name}",
                        ))
                        return True
            except Exception as e:
                logger.error("guardian.recover backup=%s failed: %s", backup.name, e)
                continue

        # All backups failed
        await self._send_to_main_agent(GuardianEvent(
            ts=time.time(),
            level="critical",
            event_type="restart_failed",
            message="March recovery failed — all config backups exhausted. Manual intervention required.",
        ))
        return False

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a process is alive."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _is_log_stale(self, entry: WatchEntry) -> bool:
        """Check if a log file hasn't been modified within threshold."""
        log_path = Path(entry.log_path)
        if not log_path.exists():
            return True
        threshold = entry.timeout or self.config.log_stale_threshold
        mtime = log_path.stat().st_mtime
        return (time.time() - mtime) > threshold

    async def _send_to_main_agent(self, event: GuardianEvent) -> None:
        """Send event as notification to main agent.

        1. Persist to events.jsonl for audit trail.
        2. Log it.
        3. Write to notifications.jsonl for the Orchestrator to pick up.
        """
        # Persist to events.jsonl for audit
        self._persist_event(event)
        # Log it
        logger.warning("guardian: [%s] %s", event.event_type, event.message)
        # Write to notifications.jsonl for main agent
        self._write_notification(event)

    def _persist_event(self, event: GuardianEvent) -> None:
        """Append event to disk (events.jsonl) for audit trail."""
        try:
            GUARDIAN_STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(EVENTS_FILE, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            # Rotate if over max
            self._rotate_events_file()
        except OSError as e:
            logger.error("guardian._persist_event failed: %s", e)

    def _write_notification(self, event: GuardianEvent) -> None:
        """Append event to notifications.jsonl for the Orchestrator to read."""
        try:
            GUARDIAN_STATE_DIR.mkdir(parents=True, exist_ok=True)
            notification = {
                "ts": event.ts,
                "event_type": event.event_type,
                "message": f"[Guardian] {event.event_type}: {event.message}",
                "entry_id": event.entry_id,
            }
            with open(NOTIFICATIONS_FILE, "a") as f:
                f.write(json.dumps(notification) + "\n")
        except OSError as e:
            logger.error("guardian._write_notification failed: %s", e)

    def _rotate_events_file(self) -> None:
        """Keep at most MAX_PERSISTED_EVENTS lines in events.jsonl."""
        try:
            if not EVENTS_FILE.exists():
                return
            lines = EVENTS_FILE.read_text().strip().splitlines()
            if len(lines) > MAX_PERSISTED_EVENTS:
                # Keep only the newest events
                lines = lines[-MAX_PERSISTED_EVENTS:]
                EVENTS_FILE.write_text("\n".join(lines) + "\n")
        except OSError as e:
            logger.error("guardian._rotate_events_file failed: %s", e)

    def _save_state(self) -> None:
        """Persist watch entries to disk."""
        try:
            GUARDIAN_STATE_DIR.mkdir(parents=True, exist_ok=True)
            data = {eid: e.to_dict() for eid, e in self._entries.items()}
            REGISTRY_FILE.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.error("guardian.save_state failed: %s", e)

    def _load_state(self) -> None:
        """Load watch entries from disk."""
        if not REGISTRY_FILE.exists():
            return
        try:
            data = json.loads(REGISTRY_FILE.read_text())
            for eid, edata in data.items():
                self._entries[eid] = WatchEntry.from_dict(edata)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("guardian.load_state failed: %s", e)

    @staticmethod
    def drain_notifications() -> list[dict[str, Any]]:
        """Read and clear all pending notifications from notifications.jsonl.

        Called by the Orchestrator at the start of each turn to inject
        guardian notifications as system messages.

        Returns a list of notification dicts, each with keys:
        ts, event_type, message, entry_id.
        """
        if not NOTIFICATIONS_FILE.exists():
            return []
        try:
            text = NOTIFICATIONS_FILE.read_text().strip()
            if not text:
                return []
            notifications = []
            for line in text.splitlines():
                try:
                    notifications.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            # Clear the file after reading
            NOTIFICATIONS_FILE.write_text("")
            return notifications
        except OSError:
            return []


async def run_guardian(config: GuardianConfig | None = None) -> None:
    """Entry point for running the guardian as a standalone process."""
    guardian = Guardian(config)
    await guardian.initialize()

    # Handle SIGTERM/SIGINT gracefully
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(guardian.stop()))

    await guardian.run_loop()
