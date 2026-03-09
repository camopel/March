"""Agent Manager for March.

Spawn, monitor, steer, and kill agents (mtAgent and mpAgent). Integrates
with the lane-based task queue and agent registry.

mtAgent: asyncio tasks in the main process (default, low overhead).
mpAgent: isolated child processes with IPC heartbeat monitoring.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from march.agents.announce import AgentAnnouncer
from march.agents.registry import RunOutcome, RunRecord, AgentRegistry
from march.agents.task_queue import TaskQueue
from march.config.schema import MpConfig as MpSchemaConfig, MtConfig as MtSchemaConfig
from march.logging import get_logger

# Backward-compat aliases (used by older imports)
SubagentAnnouncer = AgentAnnouncer
SubagentRegistry = AgentRegistry

logger = get_logger("march.agent_manager")


@dataclass
class SpawnParams:
    """Parameters for spawning an agent."""

    task: str
    agent_id: str = ""
    model: str = ""
    tools: list[str] | None = None
    timeout: int = 0
    mode: str = "run"  # "run" or "session"
    cleanup: str = "keep"  # "keep" (default) or "delete"
    label: str = ""
    execution: str = "mt"  # "mt" (asyncio) or "mp" (multiprocess)

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = f"agent-{uuid4().hex[:8]}"


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
    """Status of a running agent."""

    run_id: str
    child_key: str
    task: str
    started_at: float
    duration_seconds: float
    status: str  # "running", "completed", "error"
    requester_key: str = ""
    outcome: RunOutcome | None = None
    execution: str = "mt"
    heartbeat: dict | None = None  # Latest heartbeat (mpAgent only)


@dataclass
class AgentManagerConfig:
    """Configuration for the agent manager."""

    max_spawn_depth: int = 1
    reset_after_complete_minutes: int = 60
    announce_timeout_seconds: int = 60


class AgentManager:
    """Spawn, monitor, steer, and kill agents (mt and mp).

    Integrates with:
    - TaskQueue: lanes for concurrent execution (mt lane, mp lane)
    - AgentRegistry: in-memory tracking
    - AgentAnnouncer: push-based completion delivery
    """

    def __init__(
        self,
        config: AgentManagerConfig | None = None,
        mt_config: MtSchemaConfig | None = None,
        mp_config: MpSchemaConfig | None = None,
        task_queue: TaskQueue | None = None,
        registry: AgentRegistry | None = None,
        announcer: AgentAnnouncer | None = None,
        agent_factory: Any = None,  # Callable to create child Agent instances
        session_store: Any = None,  # SessionStore for managing child sessions
    ) -> None:
        self.config = config or AgentManagerConfig()
        self._mt_config = mt_config or MtSchemaConfig()
        self._mp_config = mp_config or MpSchemaConfig()
        self.task_queue = task_queue or TaskQueue()
        self.registry = registry or AgentRegistry()
        self.announcer = announcer or AgentAnnouncer()
        self._agent_factory = agent_factory
        self._session_store = session_store

        # Active child tasks keyed by run_id for cancellation (mtAgent)
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}

        # Active MpRunner instances keyed by run_id (mpAgent)
        self._active_runners: dict[str, Any] = {}  # run_id → MpRunner

        # Steer messages keyed by child_key (mtAgent only)
        self._steer_queues: dict[str, asyncio.Queue[str]] = {}

    async def initialize(self) -> None:
        """Initialize the manager: configure lanes."""
        # Configure mt and mp lanes with their respective concurrency limits
        self.task_queue.configure_lane("mt", self._mt_config.max_concurrent)
        self.task_queue.configure_lane("mp", self._mp_config.max_concurrent)

    async def spawn(self, params: SpawnParams, ctx: SpawnContext) -> SpawnResult:
        """Spawn a new agent (mt or mp).

        Args:
            params: What to spawn (task, model, tools, execution mode, etc.).
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

        execution = params.execution or "mt"

        # Enforce execution consistency: if parent is a sub-agent,
        # child must use the same execution mode as parent.
        parent_record = self.registry.get_by_child_key(ctx.requester_session)
        if parent_record:
            # Parent is a sub-agent — enforce consistency
            if execution != parent_record.execution:
                logger.warning(
                    "spawn: forcing execution=%s→%s to match parent %s",
                    execution, parent_record.execution, parent_record.child_key,
                )
                execution = parent_record.execution

        run_id = str(uuid4())

        # Session key format: {agent_id}:mtagent:{uuid[:12]} or {agent_id}:mpagent:{uuid[:12]}
        suffix = "mtagent" if execution == "mt" else "mpagent"
        child_key = f"{params.agent_id}:{suffix}:{uuid4().hex[:12]}"

        # Determine log path
        log_path = str(Path.home() / ".march" / "logs" / child_key)

        # Register in registry
        record = RunRecord(
            run_id=run_id,
            child_key=child_key,
            requester_key=ctx.requester_session,
            requester_origin=ctx.origin,
            task=params.task,
            started_at=time.time(),
            mode=params.mode,
            cleanup=params.cleanup,
            execution=execution,
            log_path=log_path,
        )
        self.registry.register(record)

        if execution == "mp":
            # mpAgent path: enqueue into mp lane
            task = asyncio.create_task(
                self.task_queue.enqueue(
                    "mp",
                    lambda rid=run_id, ck=child_key, p=params, c=ctx: self._execute_child_mp(
                        rid, ck, p, c
                    ),
                )
            )
            self._active_tasks[run_id] = task
        else:
            # mtAgent path: enqueue into mt lane (existing asyncio logic)
            self._steer_queues[child_key] = asyncio.Queue()
            task = asyncio.create_task(
                self.task_queue.enqueue(
                    "mt",
                    lambda rid=run_id, ck=child_key, p=params, c=ctx: self._execute_child(
                        rid, ck, p, c
                    ),
                )
            )
            self._active_tasks[run_id] = task

        logger.info(
            "spawn run_id=%s child=%s execution=%s requester=%s task=%s",
            run_id, child_key, execution, ctx.requester_session, params.task[:80],
        )

        return SpawnResult(
            status="accepted",
            child_key=child_key,
            run_id=run_id,
            note="auto-announces on completion, do not poll",
        )

    async def list(self) -> list[AgentStatus]:
        """List all agent runs (active and recent completed)."""
        statuses = []
        for record in self.registry.list_all():
            status = "running" if record.is_active else "completed"
            if record.outcome and record.outcome.status == "error":
                status = "error"

            # For mpAgent, include latest heartbeat
            heartbeat = None
            if record.execution == "mp" and record.is_active:
                runner = self._active_runners.get(record.run_id)
                if runner:
                    heartbeat = runner.get_latest_heartbeat()

            statuses.append(AgentStatus(
                run_id=record.run_id,
                child_key=record.child_key,
                task=record.task,
                started_at=record.started_at,
                duration_seconds=record.duration_seconds,
                status=status,
                requester_key=record.requester_key,
                outcome=record.outcome,
                execution=record.execution,
                heartbeat=heartbeat,
            ))
        return statuses

    async def kill(self, agent_id: str) -> bool:
        """Kill an agent by run_id or child_key.

        Returns True if the agent was found and killed.
        """
        # Find by run_id first, then child_key
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record:
            return False

        if record.execution == "mp":
            # mpAgent: kill via MpRunner (killpg entire process group)
            runner = self._active_runners.get(record.run_id)
            if runner:
                await runner.kill()
                self._active_runners.pop(record.run_id, None)
        else:
            # mtAgent: cancel the asyncio task
            task = self._active_tasks.get(record.run_id)
            if task and not task.done():
                task.cancel()

        # Mark as completed
        self.registry.complete(
            record.run_id,
            RunOutcome(status="cancelled", error="killed by user"),
        )

        # Clean up
        self._steer_queues.pop(record.child_key, None)
        self._active_tasks.pop(record.run_id, None)

        logger.info("kill run_id=%s child=%s execution=%s", record.run_id, record.child_key, record.execution)
        return True

    async def kill_recursive(self, agent_id: str) -> int:
        """Recursively kill an agent and all its descendants.

        Depth-first: kills the deepest descendants first, then works up.
        This avoids orphaned children when a parent is killed first.

        Args:
            agent_id: run_id or child_key of the root agent.

        Returns:
            Total number of agents killed.
        """
        # Find the root record
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record:
            return 0

        killed = 0

        # Get all descendants (depth-first order)
        descendants = self.registry.get_subtree(record.child_key)

        # Kill in reverse order (deepest first)
        for desc in reversed(descendants):
            if desc.is_active:
                await self.kill(desc.run_id)
                killed += 1

        # Kill the root itself
        if record.is_active:
            await self.kill(record.run_id)
            killed += 1

        return killed

    async def send(self, agent_id: str, message: str) -> bool:
        """Send a steering message to a running agent.

        Returns True if the message was queued for delivery.
        """
        # Find by run_id or child_key
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record or not record.is_active:
            return False

        if record.execution == "mp":
            # mpAgent: send steer via MpRunner IPC
            runner = self._active_runners.get(record.run_id)
            if runner:
                return await runner.send_steer(message)
            return False
        else:
            # mtAgent: queue for delivery
            queue = self._steer_queues.get(record.child_key)
            if queue:
                await queue.put(message)
                logger.info("steer sent to %s: %s", record.child_key, message[:80])
                return True
            return False

    async def logs(self, agent_id: str, tail: int = 50) -> list[str]:
        """Get recent log entries from an agent.

        For mpAgent, includes latest heartbeat info and log path.
        """
        record = self.registry.get(agent_id)
        if not record:
            record = self.registry.get_by_child_key(agent_id)
        if not record:
            return [f"No agent found with id: {agent_id}"]

        lines = [
            f"Run ID: {record.run_id}",
            f"Child Key: {record.child_key}",
            f"Execution: {record.execution}",
            f"Task: {record.task}",
            f"Started: {time.ctime(record.started_at)}",
            f"Status: {'running' if record.is_active else 'completed'}",
            f"Duration: {record.duration_seconds:.1f}s",
        ]

        if record.execution == "mp":
            if record.pid:
                lines.append(f"PID: {record.pid}")
            if record.log_path:
                lines.append(f"Log Path: {record.log_path}")

            # Include latest heartbeat for active mpAgent
            if record.is_active:
                runner = self._active_runners.get(record.run_id)
                if runner:
                    hb = runner.get_latest_heartbeat()
                    if hb:
                        lines.append("--- Latest Heartbeat ---")
                        lines.append(f"  Memory RSS: {hb.get('memory_rss_mb', '?')} MB")
                        lines.append(f"  Elapsed: {hb.get('elapsed_seconds', '?')}s")
                        lines.append(f"  Tokens: {hb.get('tokens_used', '?')}")
                        lines.append(f"  Cost: ${hb.get('total_cost', 0):.4f}")
                        lines.append(f"  Tool Calls: {hb.get('tool_calls_made', '?')}")
                        lines.append(f"  Summary: {hb.get('summary', '')}")
                        current = hb.get("current_tool")
                        if current:
                            lines.append(f"  Current Tool: {current}")
                            lines.append(f"  Detail: {hb.get('current_tool_detail', '')}")
                    else:
                        lines.append("No heartbeat received yet.")

        if record.outcome:
            lines.append(f"Outcome: {record.outcome.status}")
            if record.outcome.error:
                lines.append(f"Error: {record.outcome.error}")

        return lines[-tail:]

    # ── mtAgent execution ────────────────────────────────────────────

    async def _execute_child(
        self,
        run_id: str,
        child_key: str,
        params: SpawnParams,
        ctx: SpawnContext,
    ) -> None:
        """Execute a child agent run in the mt lane (asyncio task)."""
        outcome: RunOutcome
        try:
            if self._agent_factory:
                result = await self._run_real_child(run_id, child_key, params, ctx)
                outcome = RunOutcome(status="ok", output=result)
            else:
                outcome = RunOutcome(
                    status="error",
                    error="agent_factory not configured — cannot run agents yet",
                )
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

        # Announce to parent
        if record:
            await self._announce(record, outcome)

    async def _run_real_child(
        self,
        run_id: str,
        child_key: str,
        params: SpawnParams,
        ctx: SpawnContext,
    ) -> str:
        """Run a real child agent (when agent_factory is available)."""
        result = await self._agent_factory(
            task=params.task,
            model=params.model,
            tools=params.tools,
            child_key=child_key,
            parent_key=ctx.requester_session,
        )
        return str(result)

    # ── mpAgent execution ────────────────────────────────────────────

    async def _execute_child_mp(
        self,
        run_id: str,
        child_key: str,
        params: SpawnParams,
        ctx: SpawnContext,
    ) -> None:
        """Execute a child agent run in the mp lane (isolated process)."""
        from march.agents.mp_runner import MpRunner, MpConfig as RunnerMpConfig

        outcome: RunOutcome
        runner: MpRunner | None = None

        try:
            # Build MpRunner config from schema config
            runner_config = RunnerMpConfig(
                heartbeat_interval_seconds=self._mp_config.heartbeat_interval_seconds,
                heartbeat_timeout_seconds=self._mp_config.heartbeat_timeout_seconds,
                kill_grace_seconds=self._mp_config.kill_grace_seconds,
                log_dir=str(Path.home() / ".march" / "logs" / child_key),
            )

            # Determine config path for the child process
            config_path = str(Path.home() / ".march" / "config.yaml")

            # Build spawn proxy handlers for grandchild delegation
            async def _handle_spawn_request(
                task: str, agent_id: str, model: str, timeout: int, request_id: str
            ) -> tuple[str, str, str, str]:
                spawn_params = SpawnParams(
                    task=task,
                    agent_id=agent_id or "",
                    model=model,
                    timeout=timeout,
                    execution="mp",  # Force mp — no mixed execution modes
                )
                child_ctx = SpawnContext(
                    requester_session=child_key,  # The mpAgent is the requester
                    origin=ctx.origin,
                    caller_depth=ctx.caller_depth + 1,
                )
                result = await self.spawn(spawn_params, child_ctx)
                return (result.status, result.child_key, result.run_id, result.error)

            async def _handle_steer(grandchild_key: str, message: str) -> bool:
                return await self.send(grandchild_key, message)

            async def _handle_kill(grandchild_key: str) -> bool:
                return await self.kill(grandchild_key)

            # Create and spawn runner with spawn proxy handlers
            runner = MpRunner(
                spawn_handler=_handle_spawn_request,
                steer_handler=_handle_steer,
                kill_handler=_handle_kill,
            )
            self._active_runners[run_id] = runner

            handle = await runner.spawn(
                task=params.task,
                session_id=child_key,
                config_path=config_path,
                mp_config=runner_config,
            )

            # Record PID in registry
            record = self.registry.get(run_id)
            if record:
                record.pid = handle.pid

            logger.info(
                "mpAgent spawned: run_id=%s pid=%d child=%s",
                run_id, handle.pid, child_key,
            )

            # Wait for result (guaranteed to return, never hangs)
            outcome = await runner.wait_result()

        except asyncio.CancelledError:
            outcome = RunOutcome(status="cancelled", error="mp task was cancelled")
            if runner:
                await runner.kill()
        except Exception as e:
            outcome = RunOutcome(status="error", error=f"mpAgent spawn/run error: {e}")
            logger.error("mpAgent error for %s: %s", child_key, e, exc_info=True)

        # Clean up runner reference
        self._active_runners.pop(run_id, None)

        # Complete in registry
        record = self.registry.complete(run_id, outcome)

        # Clean up task reference
        self._active_tasks.pop(run_id, None)

        # Parent process writes to SessionStore if needed (mpAgent child doesn't)
        if record and self._session_store and outcome.output:
            try:
                # Create a session record for the mpAgent run so the parent
                # can reference it later
                await self._session_store.create_session(
                    source_type="mpagent",
                    source_id=child_key,
                    name=f"mpAgent {child_key[:20]}",
                    session_id=child_key,
                )
                # Store the result as an assistant message
                from march.core.session import Message, Role
                msg = Message(role=Role.ASSISTANT, content=outcome.output)
                await self._session_store.add_message(child_key, msg)
            except Exception as e:
                logger.warning(
                    "Failed to persist mpAgent result to SessionStore: %s", e,
                )

        # Announce to parent
        if record:
            await self._announce(record, outcome)

    # ── Shared helpers ───────────────────────────────────────────────

    async def _announce(self, record: RunRecord, outcome: RunOutcome) -> None:
        """Announce completion to the parent session.

        If the parent is an mpAgent (requester_key matches an active mpAgent's
        child_key), also notify via IPC so the child process can resolve
        its wait_child() future.
        """
        # Check if the parent is an active mpAgent — notify via IPC
        await self._notify_mp_parent(record, outcome)

        try:
            await asyncio.wait_for(
                self.announcer.announce_completion(record, outcome),
                timeout=self.config.announce_timeout_seconds or 60,
            )
            self.registry.mark_cleanup_done(record.run_id)
        except asyncio.TimeoutError:
            logger.warning("announce timed out for run_id=%s", record.run_id)
        except Exception as e:
            logger.error("announce failed for run_id=%s: %s", record.run_id, e)

    async def _notify_mp_parent(self, record: RunRecord, outcome: RunOutcome) -> None:
        """If the requester is an active mpAgent, send child_completed via IPC.

        This allows the mpAgent child process to resolve its wait_child() future
        when a grandchild completes.
        """
        requester_key = record.requester_key

        # Find the parent's run record by child_key matching requester_key
        parent_record = self.registry.get_by_child_key(requester_key)
        if not parent_record or parent_record.execution != "mp" or not parent_record.is_active:
            return

        # Find the MpRunner for the parent
        runner = self._active_runners.get(parent_record.run_id)
        if not runner:
            return

        try:
            await runner.notify_child_completed(
                child_key=record.child_key,
                status=outcome.status,
                output=outcome.output,
                error=outcome.error,
            )
            logger.debug(
                "Notified mpAgent %s that grandchild %s completed",
                requester_key, record.child_key,
            )
        except Exception as e:
            logger.warning(
                "Failed to notify mpAgent %s of grandchild completion: %s",
                requester_key, e,
            )

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
        max_age = self.config.reset_after_complete_minutes * 60
        return self.registry.cleanup_old(max_age)

    async def reset_children(self, parent_session_key: str) -> int:
        """Clean up all agent sessions for a parent session.

        Called when the parent session does /reset. Removes completed
        agent runs and deletes their sessions.

        Active (still running) children are killed first.

        Args:
            parent_session_key: The requester_key of the parent session.

        Returns:
            Number of child records cleaned up.
        """
        from march.core.compaction import delete_session_memory

        records = self.registry.list_for_requester(parent_session_key)
        cleaned = 0

        for record in records:
            # Recursively kill if still running (kills descendants first)
            if record.is_active:
                await self.kill_recursive(record.run_id)

            # Delete child's session memory files on disk
            try:
                delete_session_memory(record.child_key)
            except Exception as e:
                logger.warning(
                    "reset_children: failed to delete session memory for %s: %s",
                    record.child_key, e,
                )

            # Delete child session if session store available
            if self._session_store:
                try:
                    await self._session_store.delete_session(record.child_key)
                except Exception as e:
                    logger.warning(
                        "reset_children: failed to delete session %s: %s",
                        record.child_key, e,
                    )

            # Remove from registry
            self.registry.remove(record.run_id)
            cleaned += 1

        logger.info(
            "reset_children parent=%s cleaned=%d",
            parent_session_key, cleaned,
        )
        return cleaned

    def get_child_sessions(self, parent_session_key: str) -> list[RunRecord]:
        """Get all agent records (active and completed) for a parent.

        The parent can use this to access agent session history
        and memory after the agent finishes.

        Args:
            parent_session_key: The requester_key of the parent session.

        Returns:
            List of RunRecords for all children of this parent.
        """
        return self.registry.list_for_requester(parent_session_key)
