"""Tests for March agents: task queue, registry, manager, announcer, protocol."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from march.agents.task_queue import TaskQueue
from march.agents.registry import SubagentRegistry, RunRecord, RunOutcome
from march.agents.manager import AgentManager, AgentManagerConfig, SpawnParams, SpawnContext, SpawnResult
from march.agents.announce import SubagentAnnouncer
from march.agents.protocol import IPCMessage, MessageType


# ── Task Queue Tests ─────────────────────────────────────────────────────

class TestTaskQueue:
    """Tests for the lane-based task queue."""

    def test_default_lanes(self):
        tq = TaskQueue()
        stats = tq.all_stats()
        assert "main" in stats
        assert "subagent" in stats
        assert "cron" in stats
        assert stats["subagent"]["max_concurrent"] == 8
        assert stats["cron"]["max_concurrent"] == 1

    def test_configure_lane(self):
        tq = TaskQueue()
        tq.configure_lane("subagent", max_concurrent=16)
        assert tq.lane_stats("subagent")["max_concurrent"] == 16

    @pytest.mark.asyncio
    async def test_enqueue_basic(self):
        tq = TaskQueue()

        async def coro():
            return 42

        result = await tq.enqueue("main", coro)
        assert result == 42

    @pytest.mark.asyncio
    async def test_enqueue_exception(self):
        tq = TaskQueue()

        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await tq.enqueue("main", failing)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        tq = TaskQueue()
        tq.configure_lane("test", max_concurrent=2)

        order = []

        async def task(n, delay):
            order.append(f"start-{n}")
            await asyncio.sleep(delay)
            order.append(f"end-{n}")
            return n

        # Run 3 tasks with concurrency=2
        results = await asyncio.gather(
            tq.enqueue("test", lambda: task(1, 0.05)),
            tq.enqueue("test", lambda: task(2, 0.05)),
            tq.enqueue("test", lambda: task(3, 0.05)),
        )
        assert sorted(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_fire_and_forget(self):
        tq = TaskQueue()
        completed = asyncio.Event()

        async def bg_task():
            completed.set()

        task_id = tq.enqueue_fire_and_forget("subagent", bg_task)
        assert task_id.startswith("task-")
        await asyncio.wait_for(completed.wait(), timeout=2.0)

    def test_total_stats(self):
        tq = TaskQueue()
        assert tq.total_active == 0
        assert tq.total_queued == 0


# ── Registry Tests ───────────────────────────────────────────────────────

class TestSubagentRegistry:
    """Tests for the sub-agent registry."""

    @pytest.fixture
    def tmp_persist_dir(self, tmp_path):
        return tmp_path / "agents"

    @pytest.fixture
    def registry(self, tmp_persist_dir):
        return SubagentRegistry(persist_dir=tmp_persist_dir)

    def test_register_and_get(self, registry):
        record = RunRecord(
            run_id="run-1",
            child_key="agent:test:subagent:abc",
            requester_key="main-session",
            task="test task",
            started_at=time.time(),
        )
        registry.register(record)
        assert registry.get("run-1") is record

    def test_complete(self, registry):
        record = RunRecord(
            run_id="run-2",
            child_key="agent:test:subagent:def",
            requester_key="main-session",
            task="test task",
            started_at=time.time(),
        )
        registry.register(record)
        assert record.is_active

        outcome = RunOutcome(status="ok", output="done")
        updated = registry.complete("run-2", outcome)
        assert updated is not None
        assert not updated.is_active
        assert updated.outcome.status == "ok"

    def test_count_active(self, registry):
        for i in range(3):
            registry.register(RunRecord(
                run_id=f"run-{i}",
                child_key=f"child-{i}",
                requester_key="parent-1",
                task=f"task {i}",
                started_at=time.time(),
            ))
        assert registry.count_active("parent-1") == 3

        registry.complete("run-0", RunOutcome(status="ok"))
        assert registry.count_active("parent-1") == 2

    def test_get_by_child_key(self, registry):
        record = RunRecord(
            run_id="run-x",
            child_key="agent:x:subagent:123",
            requester_key="parent",
            task="find me",
            started_at=time.time(),
        )
        registry.register(record)
        found = registry.get_by_child_key("agent:x:subagent:123")
        assert found is record

    def test_list_active(self, registry):
        registry.register(RunRecord(run_id="a", child_key="c-a", requester_key="p", task="t1", started_at=time.time()))
        registry.register(RunRecord(run_id="b", child_key="c-b", requester_key="p", task="t2", started_at=time.time()))
        registry.complete("a", RunOutcome(status="ok"))

        active = registry.list_active()
        assert len(active) == 1
        assert active[0].run_id == "b"

    def test_persist_and_restore(self, tmp_persist_dir):
        reg1 = SubagentRegistry(persist_dir=tmp_persist_dir)
        reg1.register(RunRecord(
            run_id="persist-1",
            child_key="child-p",
            requester_key="parent-p",
            task="persist test",
            started_at=time.time(),
        ))

        # New instance should restore
        reg2 = SubagentRegistry(persist_dir=tmp_persist_dir)
        needs_attention = reg2.restore_on_startup()

        # Active run interrupted by restart → needs attention
        assert len(needs_attention) == 1
        assert needs_attention[0].run_id == "persist-1"
        assert needs_attention[0].outcome.status == "error"

    def test_cleanup_old(self, registry):
        record = RunRecord(
            run_id="old-1",
            child_key="c-old",
            requester_key="p",
            task="old",
            started_at=time.time() - 7200,
        )
        registry.register(record)
        registry.complete("old-1", RunOutcome(status="ok"))
        registry.mark_cleanup_done("old-1")

        # Manually set ended_at in the past so cleanup considers it old
        record.ended_at = time.time() - 7200
        registry._persist_record(record)

        removed = registry.cleanup_old(max_age_seconds=60)
        assert removed == 1
        assert registry.get("old-1") is None

    def test_remove(self, registry, tmp_persist_dir):
        registry.register(RunRecord(
            run_id="rm-1",
            child_key="c-rm",
            requester_key="p",
            task="remove me",
            started_at=time.time(),
        ))
        assert (tmp_persist_dir / "rm-1.json").exists()

        registry.remove("rm-1")
        assert registry.get("rm-1") is None
        assert not (tmp_persist_dir / "rm-1.json").exists()


# ── Announcer Tests ──────────────────────────────────────────────────────

class TestSubagentAnnouncer:
    """Tests for the push-based completion announcer."""

    @pytest.mark.asyncio
    async def test_announce_steer_delivery(self):
        steer_called = False

        async def mock_steer(key, msg):
            nonlocal steer_called
            steer_called = True
            return True

        announcer = SubagentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="ann-1",
            child_key="child-ann",
            requester_key="parent",
            task="test",
            started_at=time.time(),
        )
        outcome = RunOutcome(status="ok", output="result text")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered
        assert steer_called

    @pytest.mark.asyncio
    async def test_announce_fallback_to_queue(self):
        queue_called = False

        async def mock_steer(key, msg):
            return False  # steer fails

        async def mock_queue(key, msg):
            nonlocal queue_called
            queue_called = True
            return True

        announcer = SubagentAnnouncer(try_steer=mock_steer, try_queue=mock_queue)
        record = RunRecord(
            run_id="ann-2",
            child_key="child-ann2",
            requester_key="parent",
            task="test",
            started_at=time.time(),
        )
        outcome = RunOutcome(status="error", error="something broke")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered
        assert queue_called

    @pytest.mark.asyncio
    async def test_announce_pending_fallback(self):
        announcer = SubagentAnnouncer()  # No delivery methods configured
        record = RunRecord(
            run_id="ann-3",
            child_key="child-ann3",
            requester_key="parent-key",
            task="test",
            started_at=time.time(),
        )
        outcome = RunOutcome(status="ok", output="pending result")
        delivered = await announcer.announce_completion(record, outcome)
        assert not delivered
        assert announcer.pending_count == 1

        # Retrieve pending
        pending = announcer.get_pending("parent-key")
        assert len(pending) == 1
        assert "pending result" in pending[0]
        assert announcer.pending_count == 0


# ── Protocol Tests ───────────────────────────────────────────────────────

class TestIPCProtocol:
    """Tests for parent-child IPC protocol."""

    def test_task_message(self):
        msg = IPCMessage.task("Build feature X", model="claude-3", timeout=300)
        assert msg.type == MessageType.TASK
        assert msg.payload["task"] == "Build feature X"
        assert msg.payload["model"] == "claude-3"
        assert msg.payload["timeout"] == 300

    def test_steer_message(self):
        msg = IPCMessage.steer("Focus on the tests")
        assert msg.type == MessageType.STEER
        assert msg.payload["message"] == "Focus on the tests"

    def test_cancel_message(self):
        msg = IPCMessage.cancel("user requested")
        assert msg.type == MessageType.CANCEL
        assert msg.payload["reason"] == "user requested"

    def test_result_message(self):
        msg = IPCMessage.result("Task completed", metadata={"tokens": 1000})
        assert msg.type == MessageType.RESULT
        assert msg.payload["content"] == "Task completed"
        assert msg.payload["metadata"]["tokens"] == 1000

    def test_error_message(self):
        msg = IPCMessage.error("Something broke", traceback="Traceback...")
        assert msg.type == MessageType.ERROR
        assert msg.payload["error"] == "Something broke"

    def test_json_roundtrip(self):
        original = IPCMessage.task("Test task", model="test-model")
        json_str = original.to_json()
        restored = IPCMessage.from_json(json_str)
        assert restored.type == MessageType.TASK
        assert restored.payload["task"] == "Test task"
        assert restored.payload["model"] == "test-model"

    def test_progress_message(self):
        msg = IPCMessage.progress("running", detail="50% done", percent=50.0)
        assert msg.type == MessageType.PROGRESS
        assert msg.payload["percent"] == 50.0

    def test_tool_use_message(self):
        msg = IPCMessage.tool_use("web_search", args={"query": "test"}, result_summary="3 results")
        assert msg.type == MessageType.TOOL_USE
        assert msg.payload["tool"] == "web_search"


# ── Manager Tests ────────────────────────────────────────────────────────

class TestAgentManager:
    """Tests for the agent manager."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AgentManagerConfig(
            max_spawn_depth=2,
            max_concurrent_subagents=4,
        )
        registry = SubagentRegistry(persist_dir=tmp_path / "agents")
        tq = TaskQueue()
        return AgentManager(config=config, task_queue=tq, registry=registry)

    @pytest.mark.asyncio
    async def test_spawn_accepted(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="test task"),
            SpawnContext(requester_session="parent-1", caller_depth=0),
        )
        assert result.status == "accepted"
        assert result.child_key
        assert result.run_id

    @pytest.mark.asyncio
    async def test_spawn_depth_limit(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="deep task"),
            SpawnContext(requester_session="parent-1", caller_depth=2),  # depth == max
        )
        assert result.status == "forbidden"
        assert "max spawn depth" in result.error

    @pytest.mark.asyncio
    async def test_spawn_many_children_allowed(self, manager):
        """After removing max_children_per_agent, spawning many children is allowed
        (limited only by max_concurrent in the task queue lane)."""
        await manager.initialize()
        for i in range(4):
            result = await manager.spawn(
                SpawnParams(task=f"task {i}"),
                SpawnContext(requester_session="parent-1", caller_depth=0),
            )
            assert result.status == "accepted"

    @pytest.mark.asyncio
    async def test_list_agents(self, manager):
        await manager.initialize()
        await manager.spawn(
            SpawnParams(task="list test"),
            SpawnContext(requester_session="parent-1", caller_depth=0),
        )
        agents = await manager.list()
        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_kill_agent(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="kill test"),
            SpawnContext(requester_session="parent-1", caller_depth=0),
        )
        killed = await manager.kill(result.run_id)
        assert killed

    @pytest.mark.asyncio
    async def test_kill_nonexistent(self, manager):
        await manager.initialize()
        killed = await manager.kill("nonexistent-id")
        assert not killed

    @pytest.mark.asyncio
    async def test_logs(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="logs test task"),
            SpawnContext(requester_session="parent-1", caller_depth=0),
        )
        logs = await manager.logs(result.run_id)
        assert any("logs test task" in line for line in logs)


# ── Sub-agent Session Persistence Tests ──────────────────────────────────

class TestSubagentSessionPersistence:
    """Tests for sub-agent sessions persisting after completion.

    Design rule: sub-agent sessions and registry records survive completion
    and remain accessible until the parent session does /reset.
    """

    @pytest.fixture
    def manager(self, tmp_path):
        config = AgentManagerConfig(
            max_spawn_depth=2,
            max_concurrent_subagents=8,
        )
        registry = SubagentRegistry(persist_dir=tmp_path / "agents")
        tq = TaskQueue()
        return AgentManager(config=config, task_queue=tq, registry=registry)

    def test_spawn_params_default_cleanup_is_keep(self):
        """SpawnParams default cleanup should be 'keep', not 'delete'."""
        params = SpawnParams(task="some task")
        assert params.cleanup == "keep"

    def test_spawn_params_mode_run_keeps_session(self):
        """mode='run' should still default to cleanup='keep'."""
        params = SpawnParams(task="task", mode="run")
        assert params.cleanup == "keep"

    @pytest.mark.asyncio
    async def test_completed_run_stays_in_registry(self, manager):
        """After a sub-agent completes, its record remains in the registry."""
        await manager.initialize()

        result = await manager.spawn(
            SpawnParams(task="persist after complete"),
            SpawnContext(requester_session="parent-sess", caller_depth=0),
        )

        # Simulate completion
        outcome = RunOutcome(status="ok", output="task done")
        manager.registry.complete(result.run_id, outcome)

        # Record should still be in registry
        record = manager.registry.get(result.run_id)
        assert record is not None
        assert not record.is_active
        assert record.outcome.status == "ok"

    @pytest.mark.asyncio
    async def test_get_child_sessions_after_completion(self, manager):
        """Parent can list completed child sessions."""
        await manager.initialize()

        # Spawn two children
        r1 = await manager.spawn(
            SpawnParams(task="child 1"),
            SpawnContext(requester_session="parent-A", caller_depth=0),
        )
        r2 = await manager.spawn(
            SpawnParams(task="child 2"),
            SpawnContext(requester_session="parent-A", caller_depth=0),
        )

        # Complete one
        manager.registry.complete(r1.run_id, RunOutcome(status="ok", output="result 1"))

        # Both should be accessible
        children = manager.get_child_sessions("parent-A")
        assert len(children) == 2
        child_ids = {c.run_id for c in children}
        assert r1.run_id in child_ids
        assert r2.run_id in child_ids

    @pytest.mark.asyncio
    async def test_reset_children_clears_all(self, manager):
        """reset_children removes all child records for a parent."""
        await manager.initialize()

        # Spawn children for parent-B
        r1 = await manager.spawn(
            SpawnParams(task="child B1"),
            SpawnContext(requester_session="parent-B", caller_depth=0),
        )
        r2 = await manager.spawn(
            SpawnParams(task="child B2"),
            SpawnContext(requester_session="parent-B", caller_depth=0),
        )

        # Complete both
        manager.registry.complete(r1.run_id, RunOutcome(status="ok"))
        manager.registry.complete(r2.run_id, RunOutcome(status="ok"))

        # Verify they exist
        assert len(manager.get_child_sessions("parent-B")) == 2

        # Reset
        cleaned = await manager.reset_children("parent-B")
        assert cleaned == 2
        assert len(manager.get_child_sessions("parent-B")) == 0

    @pytest.mark.asyncio
    async def test_reset_children_kills_active(self, manager):
        """reset_children kills still-running children before cleanup."""
        await manager.initialize()

        result = await manager.spawn(
            SpawnParams(task="still running"),
            SpawnContext(requester_session="parent-C", caller_depth=0),
        )

        # Child is still active (not completed)
        record = manager.registry.get(result.run_id)
        assert record.is_active

        # Reset should kill and clean up
        cleaned = await manager.reset_children("parent-C")
        assert cleaned == 1
        assert manager.registry.get(result.run_id) is None

    @pytest.mark.asyncio
    async def test_reset_children_isolates_parents(self, manager):
        """reset_children only affects the specified parent's children."""
        await manager.initialize()

        # Parent X's child
        rx = await manager.spawn(
            SpawnParams(task="X's child"),
            SpawnContext(requester_session="parent-X", caller_depth=0),
        )
        # Parent Y's child
        ry = await manager.spawn(
            SpawnParams(task="Y's child"),
            SpawnContext(requester_session="parent-Y", caller_depth=0),
        )

        manager.registry.complete(rx.run_id, RunOutcome(status="ok"))
        manager.registry.complete(ry.run_id, RunOutcome(status="ok"))

        # Reset only parent-X
        await manager.reset_children("parent-X")

        # X's children gone, Y's still there
        assert len(manager.get_child_sessions("parent-X")) == 0
        assert len(manager.get_child_sessions("parent-Y")) == 1

    @pytest.mark.asyncio
    async def test_announce_does_not_delete_session(self):
        """Announcer should NOT delete sessions on completion."""
        deleted_keys: list[str] = []

        async def mock_delete(key):
            deleted_keys.append(key)

        async def mock_steer(key, msg):
            return True

        announcer = SubagentAnnouncer(
            try_steer=mock_steer,
            delete_session=mock_delete,
        )

        record = RunRecord(
            run_id="no-delete-1",
            child_key="child-nd",
            requester_key="parent",
            task="test",
            started_at=time.time(),
            cleanup="keep",  # default
        )
        outcome = RunOutcome(status="ok", output="done")
        await announcer.announce_completion(record, outcome)

        # Session should NOT have been deleted
        assert deleted_keys == []

    @pytest.mark.asyncio
    async def test_announce_does_not_delete_even_with_old_cleanup_delete(self):
        """Even if cleanup='delete' is explicitly set, announcer no longer deletes."""
        deleted_keys: list[str] = []

        async def mock_delete(key):
            deleted_keys.append(key)

        async def mock_steer(key, msg):
            return True

        announcer = SubagentAnnouncer(
            try_steer=mock_steer,
            delete_session=mock_delete,
        )

        record = RunRecord(
            run_id="no-delete-2",
            child_key="child-nd2",
            requester_key="parent",
            task="test",
            started_at=time.time(),
            cleanup="delete",  # explicitly set — announcer should still NOT delete
        )
        outcome = RunOutcome(status="ok", output="done")
        await announcer.announce_completion(record, outcome)

        # Announcer no longer calls delete_session at all
        assert deleted_keys == []

    @pytest.mark.asyncio
    async def test_completed_children_persist_on_disk(self, tmp_path):
        """Completed sub-agent records persist on disk until parent resets."""
        persist_dir = tmp_path / "agents"
        registry = SubagentRegistry(persist_dir=persist_dir)

        registry.register(RunRecord(
            run_id="disk-persist-1",
            child_key="child-dp1",
            requester_key="parent-dp",
            task="persist on disk",
            started_at=time.time(),
        ))
        registry.complete("disk-persist-1", RunOutcome(status="ok", output="result"))
        registry.mark_cleanup_done("disk-persist-1")

        # File should still exist on disk
        assert (persist_dir / "disk-persist-1.json").exists()

        # Load from a fresh registry instance — record should be there
        reg2 = SubagentRegistry(persist_dir=persist_dir)
        reg2.restore_on_startup()
        record = reg2.get("disk-persist-1")
        assert record is not None
        assert record.outcome.status == "ok"
