"""Lane-based async task queue for March.

Three lanes with independent concurrency:
  - main: User-facing sessions (maxConcurrent: auto = CPU cores)
  - subagent: Sub-agent runs (maxConcurrent: 8)
  - cron: Scheduled jobs (maxConcurrent: 1)

Each lane has its own queue and active task tracking. Tasks are enqueued
and await completion. The internal drain loop pumps queued tasks up to
max_concurrent for each lane.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from march.logging import get_logger

logger = get_logger("march.task_queue")

T = TypeVar("T")

_next_task_id = 0


def _gen_task_id() -> str:
    global _next_task_id
    _next_task_id += 1
    return f"task-{_next_task_id}"


@dataclass
class QueueEntry(Generic[T]):
    """A queued task waiting to execute."""

    task: Callable[[], Awaitable[T]]
    future: asyncio.Future[T] = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    task_id: str = field(default_factory=_gen_task_id)


@dataclass
class LaneState:
    """State for a single task lane."""

    name: str
    max_concurrent: int
    queue: deque[QueueEntry[Any]] = field(default_factory=deque)
    active: set[str] = field(default_factory=set)


class TaskQueue:
    """Lane-based async task queue with independent concurrency per lane.

    Usage:
        tq = TaskQueue()
        tq.configure_lane("main", max_concurrent=4)
        tq.configure_lane("subagent", max_concurrent=8)
        tq.configure_lane("cron", max_concurrent=1)

        result = await tq.enqueue("subagent", my_coroutine_fn)
    """

    def __init__(self) -> None:
        self._lanes: dict[str, LaneState] = {}
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Set up default lanes per spec."""
        cpu_count = os.cpu_count() or 4
        self.configure_lane("main", max_concurrent=cpu_count)
        self.configure_lane("subagent", max_concurrent=8)
        self.configure_lane("cron", max_concurrent=1)

    def configure_lane(self, name: str, max_concurrent: int) -> None:
        """Configure or update a lane's concurrency limit."""
        if name in self._lanes:
            self._lanes[name].max_concurrent = max(1, max_concurrent)
        else:
            self._lanes[name] = LaneState(
                name=name,
                max_concurrent=max(1, max_concurrent),
            )

    def _get_lane(self, name: str) -> LaneState:
        """Get or create a lane by name."""
        if name not in self._lanes:
            self._lanes[name] = LaneState(name=name, max_concurrent=4)
        return self._lanes[name]

    async def enqueue(self, lane: str, task: Callable[[], Awaitable[T]]) -> T:
        """Enqueue a task in the specified lane. Returns when task completes.

        Args:
            lane: Lane name (main, subagent, cron).
            task: An async callable (zero-arg) to execute.

        Returns:
            The result of the task.

        Raises:
            Exception: If the task raises an exception, it's re-raised here.
        """
        state = self._get_lane(lane)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        entry = QueueEntry(task=task, future=future)
        state.queue.append(entry)

        logger.debug("enqueue lane=%s task_id=%s queue_depth=%d", lane, entry.task_id, len(state.queue))
        self._drain(lane)

        return await future

    def enqueue_fire_and_forget(self, lane: str, task: Callable[[], Awaitable[Any]]) -> str:
        """Enqueue a task without waiting for completion. Returns task ID.

        Args:
            lane: Lane name.
            task: An async callable to execute.

        Returns:
            The task ID for tracking.
        """
        state = self._get_lane(lane)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        entry = QueueEntry(task=task, future=future)
        state.queue.append(entry)

        self._drain(lane)
        return entry.task_id

    def _drain(self, lane: str) -> None:
        """Pump queued tasks up to max_concurrent for the lane."""
        state = self._get_lane(lane)
        while len(state.active) < state.max_concurrent and state.queue:
            entry = state.queue.popleft()
            state.active.add(entry.task_id)
            asyncio.create_task(self._run(lane, entry))

    async def _run(self, lane: str, entry: QueueEntry[Any]) -> None:
        """Execute a single task and resolve its future."""
        state = self._get_lane(lane)
        try:
            result = await entry.task()
            if not entry.future.done():
                entry.future.set_result(result)
        except asyncio.CancelledError:
            if not entry.future.done():
                entry.future.cancel()
        except Exception as e:
            logger.error("task failed lane=%s task_id=%s error=%s", lane, entry.task_id, e)
            if not entry.future.done():
                entry.future.set_exception(e)
        finally:
            state.active.discard(entry.task_id)
            self._drain(lane)  # Pump next waiting task

    # ── Introspection ────────────────────────────────────────────────────

    def lane_stats(self, lane: str) -> dict[str, Any]:
        """Get stats for a lane."""
        state = self._get_lane(lane)
        return {
            "name": state.name,
            "max_concurrent": state.max_concurrent,
            "active": len(state.active),
            "queued": len(state.queue),
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all lanes."""
        return {name: self.lane_stats(name) for name in self._lanes}

    @property
    def total_active(self) -> int:
        """Total active tasks across all lanes."""
        return sum(len(s.active) for s in self._lanes.values())

    @property
    def total_queued(self) -> int:
        """Total queued tasks across all lanes."""
        return sum(len(s.queue) for s in self._lanes.values())
