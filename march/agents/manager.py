"""Agent Manager for March.

Spawn, monitor, steer, and kill sub-agents. Integrates with the
lane-based task queue and sub-agent registry for crash recovery.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from march.agents.announce import SubagentAnnouncer
from march.agents.registry import RunOutcome, RunRecord, SubagentRegistry
from march.agents.task_queue import TaskQueue
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.core.session import Session, SessionStore

logger = get_logger("march.agent_manager")


@dataclass
class SpawnParams:
    """Parameters for spawning a sub-agent."""

    task: str
    agent_id: str = ""
    model: str = ""
    tools: list[str] | None = None
    timeout: int = 0
    mode: str = "run"  # "run" or "session"
    cleanup: str = "keep"  # "keep" (default) or "delete" — sub-agent sessions persist until parent /reset
    label: str = ""

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = f"subagent-{uuid4().hex[:8]}"


@dataclass
class SpawnContext:
    """Context for the spawn request."""

    requester_session: str  # session key of the parent
    origin: str = ""  # channel/source of the parent
    caller_depth: int = 0  # current spawn depth


@dataclass
class SpawnResult:
    """Result of a spawn request."""

    status: str  # "accepted", "forbidden", "error"
    child_key: str = ""
    run_id: str = ""
    error: str = ""
    note: str = ""


@dataclass
class AgentStatus:
    """Status of a running sub-agent."""

    run_id: str
    child_key: str
    task: str
    started_at: float
    duration_seconds: float
    status: str  # "running", "completed", "error"
    requester_key: str = ""
    outcome: RunOutcome | None = None


@dataclass
class AgentManagerConfig:
    """Configuration for the agent manager."""

    max_spawn_depth: int = 1
    max_children_per_agent: int = 5
    max_concurrent_subagents: int = 8
    run_timeout_seconds: int = 0  # 0 = no timeout
    archive_after_minutes: int = 60
    announce_timeout_seconds: int = 60


class AgentManager:
    """Spawn, monitor, steer, and kill sub-agents.

    Integrates with:
    - TaskQueue: lanes for concurrent execution
    - SubagentRegistry: persistent tracking and crash recovery
    - SubagentAnnouncer: push-based completion delivery
    """

    def __init__(
        self,
        config: AgentManagerConfig | None = None,
        task_queue: TaskQueue | None = None,
        registry: SubagentRegistry | None = None,
        announcer: SubagentAnnouncer | None = None,
        agent_factory: Any = None,  # Callable to create child Agent instances
        session_store: Any = None,  # SessionStore for managing child sessions
    ) -> None:
        self.config = config or AgentManagerConfig()
        self.task_queue = task_queue or TaskQueue()
        self.registry = registry or SubagentRegistry()
        self.announcer = announcer or SubagentAnnouncer()
        self._agent_factory = agent_factory
        self._session_store = session_store

        # Active child tasks keyed by run_id for cancellation
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}

        # Steer messages keyed by child_key
        self._steer_queues: dict[str, asyncio.Queue[str]] = {}

    async def initialize(self) -> None:
        """Initialize the manager: set up registry, restore from crash."""
        await self.registry.initialize()

        # Configure subagent lane
        self.task_queue.configure_lane("subagent", self.config.max_concurrent_subagents)

        # Restore from crash
        needs_attention = self.registry.restore_on_startup()
        for record in needs_attention:
            if record.outcome and not record.cleanup_done:
                # Retry announcement
                asyncio.create_task(
                    self._retry_announce(record)
                )

    async def spawn(self, params: SpawnParams, ctx: SpawnContext) -> SpawnResult:
        """Spawn a new sub-agent.

        Args:
            params: What to spawn (task, model, tools, etc.).
            ctx: Who is spawning (requester session, depth).

        Returns:
            SpawnResult with status and child session key.
        """
        # Validate depth limits
        if ctx.caller_depth >= self.config.max_spawn_depth:
            return SpawnResult(
                status="forbidden",
                error=f"max spawn depth ({self.config.max_spawn_depth}) reached",
            )

        # Validate max children
        active_children = self.registry.count_active(ctx.requester_session)
        if active_children >= self.config.max_children_per_agent:
            return SpawnResult(
                status="forbidden",
                error=f"max children ({self.config.max_children_per_agent}) reached for this agent",
            )

        # Create child session key
        run_id = str(uuid4())
        child_key = f"agent:{params.agent_id}:subagent:{uuid4().hex[:12]}"

        # Register in registry (persists to disk)
        record = RunRecord(
            run_id=run_id,
            child_key=child_key,
            requester_key=ctx.requester_session,
            requester_origin=ctx.origin,
            task=params.task,
            started_at=time.time(),
            mode=params.mode,
            cleanup=params.cleanup,
        )
        self.registry.register(record)

        # Create steer queue for this child
        self._steer_queues[child_key] = asyncio.Queue()

        # Fire into subagent lane (non-blocking)
        task = asyncio.create_task(
            self.task_queue.enqueue(
                "subagent",
                lambda rid=run_id, ck=child_key, p=params, c=ctx: self._execute_child(
                    rid, ck, p, c
                ),
            )
        )
        self._active_tasks[run_id] = task

        logger.info(
            "spawn run_id=%s child=%s requester=%s task=%s",
            run_id, child_key, ctx.requester_session, params.task[:80],
        )

        return SpawnResult(
            status="accepted",
            child_key=child_key,
            run_id=run_id,
            note="auto-announces on completion, do not poll",
        )

    async def list(self) -> list[AgentStatus]:
        """List all sub-agent runs (active and recent completed)."""
        statuses = []
        for record in self.registry.list_all():
            status = "running" if record.is_active else "completed"
            if record.outcome and record.outcome.status == "error":
                status = "error"

            statuses.append(AgentStatus(
                run_id=record.run_id,
                child_key=record.child_key,
                task=record.task,
                started_at=record.started_at,
                duration_seconds=record.duration_seconds,
                status=status,
                requester_key=record.requester_key,
                outcome=record.outcome,
            ))
        return statuses

    async def kill(self, agent_id: str) -> bool:
        """Kill a sub-agent by run_id or child_key.

        Returns True if the agent was found and killed.
        """
        # Find by run_id first, then child_key
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record:
            return False

        # Cancel the asyncio task
        task = self._active_tasks.get(record.run_id)
        if task and not task.done():
            task.cancel()

        # Mark as completed
        self.registry.complete(
            record.run_id,
            RunOutcome(status="cancelled", error="killed by user"),
        )

        # Clean up steer queue
        self._steer_queues.pop(record.child_key, None)
        self._active_tasks.pop(record.run_id, None)

        logger.info("kill run_id=%s child=%s", record.run_id, record.child_key)
        return True

    async def send(self, agent_id: str, message: str) -> bool:
        """Send a steering message to a running sub-agent.

        Returns True if the message was queued for delivery.
        """
        # Find by run_id or child_key
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record or not record.is_active:
            return False

        queue = self._steer_queues.get(record.child_key)
        if queue:
            await queue.put(message)
            logger.info("steer sent to %s: %s", record.child_key, message[:80])
            return True
        return False

    async def logs(self, agent_id: str, tail: int = 50) -> list[str]:
        """Get recent log entries from a sub-agent.

        Currently returns basic status info. Full log integration
        requires the structured logging system.
        """
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record:
            return [f"No agent found with id: {agent_id}"]

        lines = [
            f"Run ID: {record.run_id}",
            f"Child Key: {record.child_key}",
            f"Task: {record.task}",
            f"Started: {time.ctime(record.started_at)}",
            f"Status: {'running' if record.is_active else 'completed'}",
            f"Duration: {record.duration_seconds:.1f}s",
        ]
        if record.outcome:
            lines.append(f"Outcome: {record.outcome.status}")
            if record.outcome.error:
                lines.append(f"Error: {record.outcome.error}")
        return lines[-tail:]

    async def _execute_child(
        self,
        run_id: str,
        child_key: str,
        params: SpawnParams,
        ctx: SpawnContext,
    ) -> None:
        """Execute a child agent run in the subagent lane."""
        outcome: RunOutcome
        try:
            # For now, simulate execution. When agent_factory is set,
            # this will create a real child agent and run it.
            if self._agent_factory:
                result = await self._run_real_child(run_id, child_key, params, ctx)
                outcome = RunOutcome(status="ok", output=result)
            else:
                # Placeholder: no agent factory configured
                outcome = RunOutcome(
                    status="error",
                    error="agent_factory not configured — cannot run sub-agents yet",
                )

            # Apply timeout if configured
        except asyncio.CancelledError:
            outcome = RunOutcome(status="cancelled", error="task was cancelled")
        except asyncio.TimeoutError:
            outcome = RunOutcome(status="timeout", error="run exceeded timeout")
        except Exception as e:
            outcome = RunOutcome(status="error", error=str(e))

        # Complete in registry
        record = self.registry.complete(run_id, outcome)

        # Clean up
        self._active_tasks.pop(run_id, None)
        self._steer_queues.pop(child_key, None)

        # Announce to parent (session is NOT deleted — persists until parent /reset)
        if record:
            try:
                await asyncio.wait_for(
                    self.announcer.announce_completion(record, outcome),
                    timeout=self.config.announce_timeout_seconds or 60,
                )
                self.registry.mark_cleanup_done(run_id)
            except asyncio.TimeoutError:
                logger.warning("announce timed out for run_id=%s", run_id)
            except Exception as e:
                logger.error("announce failed for run_id=%s: %s", run_id, e)

    async def _run_real_child(
        self,
        run_id: str,
        child_key: str,
        params: SpawnParams,
        ctx: SpawnContext,
    ) -> str:
        """Run a real child agent (when agent_factory is available)."""
        # This would create a child Agent, Session, and run the task.
        # For now, delegate to the factory callable.
        result = await self._agent_factory(
            task=params.task,
            model=params.model,
            tools=params.tools,
            child_key=child_key,
            parent_key=ctx.requester_session,
        )
        return str(result)

    async def _retry_announce(self, record: RunRecord) -> None:
        """Retry announcing a completed run that was interrupted."""
        if record.outcome:
            try:
                await self.announcer.announce_completion(record, record.outcome)
                self.registry.mark_cleanup_done(record.run_id)
            except Exception as e:
                logger.error("retry announce failed for %s: %s", record.run_id, e)

    async def cleanup(self) -> int:
        """Clean up old completed records.

        NOTE: This only removes records that have been explicitly marked
        for cleanup (cleanup_done=True). Normal completed runs persist
        until the parent calls reset_children().
        """
        max_age = self.config.archive_after_minutes * 60
        return self.registry.cleanup_old(max_age)

    async def reset_children(self, parent_session_key: str) -> int:
        """Clean up all sub-agent sessions for a parent session.

        Called when the parent session does /reset. Removes completed
        sub-agent runs and deletes their sessions.

        Active (still running) children are killed first.

        Args:
            parent_session_key: The requester_key of the parent session.

        Returns:
            Number of child records cleaned up.
        """
        records = self.registry.list_for_requester(parent_session_key)
        cleaned = 0

        for record in records:
            # Kill if still running
            if record.is_active:
                await self.kill(record.run_id)

            # Delete child session if session store available
            if self._session_store:
                try:
                    await self._session_store.delete_session(record.child_key)
                except Exception as e:
                    logger.warning(
                        "reset_children: failed to delete session %s: %s",
                        record.child_key, e,
                    )

            # Remove from registry (and disk)
            self.registry.remove(record.run_id)
            cleaned += 1

        logger.info(
            "reset_children parent=%s cleaned=%d",
            parent_session_key, cleaned,
        )
        return cleaned

    def get_child_sessions(self, parent_session_key: str) -> list[RunRecord]:
        """Get all sub-agent records (active and completed) for a parent.

        The parent can use this to access sub-agent session history
        and memory after the sub-agent finishes.

        Args:
            parent_session_key: The requester_key of the parent session.

        Returns:
            List of RunRecords for all children of this parent.
        """
        return self.registry.list_for_requester(parent_session_key)
