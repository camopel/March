"""Sub-agent registry for March.

Tracks all sub-agent runs. Persists to disk (~/.march/agents/) as JSON
for crash recovery. On startup, restores pending runs, cleans orphans,
and retries interrupted announcements.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from march.logging import get_logger

logger = get_logger("march.registry")

DEFAULT_PERSIST_DIR = Path.home() / ".march" / "agents"


@dataclass
class RunOutcome:
    """Outcome of a sub-agent run."""

    status: str  # "ok", "error", "timeout", "cancelled"
    error: str = ""
    output: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunOutcome":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RunRecord:
    """A single sub-agent run record.

    Attributes:
        run_id: Unique identifier for this run.
        child_key: Session key of the child agent.
        requester_key: Session key of the parent that spawned this.
        requester_origin: Channel/source of the parent (for announce delivery).
        task: Description of the task assigned to the child.
        started_at: Unix timestamp when the run started.
        ended_at: Unix timestamp when the run ended (0 = still running).
        mode: "run" (one-shot, cleanup after) or "session" (keep alive).
        cleanup: "delete" or "keep".
        outcome: Run outcome (set on completion).
        cleanup_done: Whether post-completion cleanup has been performed.
    """

    run_id: str
    child_key: str
    requester_key: str
    requester_origin: str = ""
    task: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    mode: str = "run"
    cleanup: str = "delete"
    outcome: RunOutcome | None = None
    cleanup_done: bool = False

    @property
    def is_active(self) -> bool:
        return self.ended_at == 0.0

    @property
    def duration_seconds(self) -> float:
        if self.ended_at:
            return self.ended_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.outcome:
            d["outcome"] = self.outcome.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        outcome = None
        if data.get("outcome"):
            outcome = RunOutcome.from_dict(data["outcome"])
        return cls(
            run_id=data["run_id"],
            child_key=data["child_key"],
            requester_key=data["requester_key"],
            requester_origin=data.get("requester_origin", ""),
            task=data.get("task", ""),
            started_at=data.get("started_at", 0.0),
            ended_at=data.get("ended_at", 0.0),
            mode=data.get("mode", "run"),
            cleanup=data.get("cleanup", "delete"),
            outcome=outcome,
            cleanup_done=data.get("cleanup_done", False),
        )


class SubagentRegistry:
    """Track all sub-agent runs. Persisted to disk for crash recovery.

    Each run is stored as a separate JSON file in the persist directory
    for atomic writes and easy cleanup.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._persist_dir = persist_dir or DEFAULT_PERSIST_DIR
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create persist directory if needed."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def register(self, record: RunRecord) -> None:
        """Register a new sub-agent run."""
        self._runs[record.run_id] = record
        self._persist_record(record)
        logger.info(
            "registry.register run_id=%s child=%s requester=%s task=%s",
            record.run_id, record.child_key, record.requester_key,
            record.task[:80],
        )

    def complete(self, run_id: str, outcome: RunOutcome) -> RunRecord | None:
        """Mark a run as complete with the given outcome.

        Returns the updated record, or None if not found.
        """
        record = self._runs.get(run_id)
        if not record:
            logger.warning("registry.complete run_id=%s not found", run_id)
            return None

        record.ended_at = time.time()
        record.outcome = outcome
        self._persist_record(record)
        logger.info(
            "registry.complete run_id=%s status=%s duration=%.1fs",
            run_id, outcome.status, record.duration_seconds,
        )
        return record

    def mark_cleanup_done(self, run_id: str) -> None:
        """Mark that announce delivery is done.

        NOTE: This does NOT mean the session is deleted. Sub-agent sessions
        persist until the parent does /reset. This flag only tracks whether
        the completion announcement was successfully delivered.
        """
        record = self._runs.get(run_id)
        if record:
            record.cleanup_done = True
            self._persist_record(record)

    def get(self, run_id: str) -> RunRecord | None:
        """Get a run record by ID."""
        return self._runs.get(run_id)

    def get_by_child_key(self, child_key: str) -> RunRecord | None:
        """Find a run record by child session key."""
        for record in self._runs.values():
            if record.child_key == child_key:
                return record
        return None

    def count_active(self, requester_key: str) -> int:
        """Count active (not yet completed) runs for a given requester."""
        return sum(
            1 for r in self._runs.values()
            if r.requester_key == requester_key and r.is_active
        )

    def list_active(self) -> list[RunRecord]:
        """List all active (running) sub-agent records."""
        return [r for r in self._runs.values() if r.is_active]

    def list_all(self) -> list[RunRecord]:
        """List all records (active and completed)."""
        return list(self._runs.values())

    def list_for_requester(self, requester_key: str) -> list[RunRecord]:
        """List all runs spawned by a specific requester."""
        return [r for r in self._runs.values() if r.requester_key == requester_key]

    def remove(self, run_id: str) -> None:
        """Remove a run record entirely."""
        self._runs.pop(run_id, None)
        record_path = self._persist_dir / f"{run_id}.json"
        if record_path.exists():
            record_path.unlink()

    def cleanup_old(self, max_age_seconds: float = 3600) -> int:
        """Remove completed records older than max_age_seconds.

        This is a safety net for truly orphaned records — records whose
        parent session no longer exists. Normal cleanup happens via
        AgentManager.reset_children() when the parent does /reset.

        Only removes records where cleanup_done=True (announce delivered)
        AND the record is older than max_age_seconds.

        Returns the number of records removed.
        """
        now = time.time()
        to_remove = [
            run_id
            for run_id, record in self._runs.items()
            if record.ended_at > 0
            and record.cleanup_done
            and (now - record.ended_at) > max_age_seconds
        ]
        for run_id in to_remove:
            self.remove(run_id)
        return len(to_remove)

    def restore_on_startup(self) -> list[RunRecord]:
        """Restore records from disk on startup.

        Returns records that need attention:
        - Runs that were interrupted (no ended_at) → marked as error
        - Runs that completed but cleanup wasn't done → need retry
        """
        needs_attention: list[RunRecord] = []

        if not self._persist_dir.exists():
            return needs_attention

        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                record = RunRecord.from_dict(data)
                self._runs[record.run_id] = record

                if record.is_active:
                    # Run was interrupted by restart
                    record.ended_at = time.time()
                    record.outcome = RunOutcome(
                        status="error",
                        error="interrupted by restart",
                    )
                    self._persist_record(record)
                    needs_attention.append(record)
                elif not record.cleanup_done:
                    # Completed but announce was interrupted
                    needs_attention.append(record)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("registry.restore failed for %s: %s", path.name, e)
                continue

        logger.info(
            "registry.restore loaded=%d needs_attention=%d",
            len(self._runs), len(needs_attention),
        )
        return needs_attention

    def _persist_record(self, record: RunRecord) -> None:
        """Write a single record to disk."""
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            path = self._persist_dir / f"{record.run_id}.json"
            path.write_text(json.dumps(record.to_dict(), indent=2))
        except OSError as e:
            logger.error("registry.persist failed: %s", e)
