"""Extended tests for agents module: task queue edge cases, registry, manager, announcer."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from march.agents.task_queue import TaskQueue
from march.agents.registry import SubagentRegistry, RunRecord, RunOutcome
from march.agents.manager import (
    AgentManager, AgentManagerConfig, SpawnParams, SpawnContext, SpawnResult, AgentStatus,
)
from march.agents.announce import SubagentAnnouncer


# ─────────────────────────────────────────────────────────────
# TaskQueue Extended
# ─────────────────────────────────────────────────────────────

class TestTaskQueueExtended:
    def test_configure_lane_min_concurrent(self):
        tq = TaskQueue()
        tq.configure_lane("test", max_concurrent=0)
        stats = tq.lane_stats("test")
        assert stats["max_concurrent"] == 1  # min is 1

    def test_get_lane_auto_creates(self):
        tq = TaskQueue()
        state = tq._get_lane("new_lane")
        assert state.name == "new_lane"
        assert state.max_concurrent == 4  # default

    async def test_enqueue_cron_lane(self):
        tq = TaskQueue()
        results = []

        async def task1():
            results.append(1)
            return 1

        async def task2():
            results.append(2)
            return 2

        # Cron lane has max_concurrent=1, so tasks should run sequentially
        r1, r2 = await asyncio.gather(
            tq.enqueue("cron", task1),
            tq.enqueue("cron", task2),
        )
        assert set(results) == {1, 2}

    async def test_fire_and_forget_error_handling(self):
        tq = TaskQueue()

        async def failing_task():
            raise ValueError("bg task failed")

        task_id = tq.enqueue_fire_and_forget("main", failing_task)
        assert task_id.startswith("task-")
        await asyncio.sleep(0.1)  # Let task run and fail
        # Should not crash the queue

    async def test_many_tasks_queued(self):
        tq = TaskQueue()
        tq.configure_lane("serial", max_concurrent=1)
        count = 0

        async def increment():
            nonlocal count
            count += 1
            return count

        results = await asyncio.gather(*[
            tq.enqueue("serial", increment) for _ in range(20)
        ])
        assert len(results) == 20
        assert count == 20

    def test_all_stats(self):
        tq = TaskQueue()
        stats = tq.all_stats()
        assert "main" in stats
        assert "mt" in stats
        assert "mp" in stats
        assert "cron" in stats
        for lane in stats.values():
            assert "active" in lane
            assert "queued" in lane
            assert "max_concurrent" in lane


# ─────────────────────────────────────────────────────────────
# SubagentRegistry Extended
# ─────────────────────────────────────────────────────────────

class TestSubagentRegistryExtended:
    @pytest.fixture
    def registry(self):
        return SubagentRegistry()

    def test_list_all(self, registry):
        for i in range(3):
            registry.register(RunRecord(
                run_id=f"r{i}", child_key=f"c{i}", requester_key="p",
                task=f"task {i}", started_at=time.time(),
            ))
        assert len(registry.list_all()) == 3

    def test_list_for_requester(self, registry):
        registry.register(RunRecord(
            run_id="r1", child_key="c1", requester_key="parent-A",
            task="t1", started_at=time.time(),
        ))
        registry.register(RunRecord(
            run_id="r2", child_key="c2", requester_key="parent-B",
            task="t2", started_at=time.time(),
        ))
        assert len(registry.list_for_requester("parent-A")) == 1
        assert len(registry.list_for_requester("parent-B")) == 1
        assert len(registry.list_for_requester("parent-C")) == 0

    def test_complete_nonexistent(self, registry):
        result = registry.complete("nonexistent", RunOutcome(status="ok"))
        assert result is None

    def test_mark_cleanup_done_nonexistent(self, registry):
        registry.mark_cleanup_done("nonexistent")  # Should not crash

    def test_run_record_duration_active(self, registry):
        record = RunRecord(
            run_id="dur", child_key="c", requester_key="p",
            task="t", started_at=time.time() - 10,
        )
        assert record.is_active
        assert record.duration_seconds >= 9.5

    def test_run_record_duration_completed(self):
        record = RunRecord(
            run_id="dur2", child_key="c", requester_key="p",
            task="t", started_at=100.0, ended_at=110.0,
        )
        assert not record.is_active
        assert record.duration_seconds == 10.0

    def test_run_record_serialization(self):
        record = RunRecord(
            run_id="ser1", child_key="c1", requester_key="p1",
            requester_origin="matrix:room", task="serialize test",
            started_at=1000.0, ended_at=1010.0,
            mode="session", cleanup="keep",
            outcome=RunOutcome(status="ok", output="done", duration_ms=5000.0),
            cleanup_done=True,
        )
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.run_id == "ser1"
        assert restored.outcome.status == "ok"
        assert restored.cleanup_done

    def test_run_outcome_serialization(self):
        outcome = RunOutcome(status="error", error="boom", output="partial", duration_ms=42.0)
        d = outcome.to_dict()
        restored = RunOutcome.from_dict(d)
        assert restored.status == "error"
        assert restored.error == "boom"

    def test_cleanup_old_skips_incomplete(self, registry):
        """cleanup_old should not remove records that haven't been cleaned up."""
        record = RunRecord(
            run_id="no-cleanup", child_key="c", requester_key="p",
            task="t", started_at=time.time() - 7200,
        )
        registry.register(record)
        registry.complete("no-cleanup", RunOutcome(status="ok"))
        # cleanup_done is still False
        record.ended_at = time.time() - 7200
        removed = registry.cleanup_old(max_age_seconds=60)
        assert removed == 0  # Should NOT be removed

    def test_registry_is_pure_in_memory(self):
        """Registry is pure in-memory, no disk persistence."""
        reg = SubagentRegistry()
        reg.register(RunRecord(
            run_id="mem-1", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        ))
        # No file system involved — just check it's in memory
        assert reg.get("mem-1") is not None

    def test_complete_and_get_outcome(self):
        """Completed but unannounced records remain accessible."""
        reg = SubagentRegistry()
        record = RunRecord(
            run_id="complete-unclean", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        reg.register(record)
        reg.complete("complete-unclean", RunOutcome(status="ok", output="done"))
        result = reg.get("complete-unclean")
        assert result is not None
        assert result.outcome.status == "ok"
        assert not result.cleanup_done

    def test_registry_no_init_needed(self):
        """Registry doesn't need async initialization — pure in-memory."""
        reg = SubagentRegistry()
        # Can register immediately without any init call
        reg.register(RunRecord(
            run_id="no-init", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        ))
        assert reg.get("no-init") is not None


# ─────────────────────────────────────────────────────────────
# SubagentAnnouncer Extended
# ─────────────────────────────────────────────────────────────

class TestSubagentAnnouncerExtended:
    async def test_announce_direct_delivery(self):
        direct_calls = []

        async def mock_direct(key, origin, msg):
            direct_calls.append((key, origin, msg))

        async def mock_steer(key, msg):
            return False

        async def mock_queue(key, msg):
            return False

        announcer = SubagentAnnouncer(
            try_steer=mock_steer,
            try_queue=mock_queue,
            send_direct=mock_direct,
        )
        record = RunRecord(
            run_id="direct1", child_key="c", requester_key="parent",
            requester_origin="matrix:room", task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="ok", output="result")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered
        assert len(direct_calls) == 1

    async def test_announce_read_child_output(self):
        async def mock_read(key):
            return "child output text"

        async def mock_steer(key, msg):
            return True

        announcer = SubagentAnnouncer(
            read_child_output=mock_read,
            try_steer=mock_steer,
        )
        record = RunRecord(
            run_id="read1", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="ok")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered

    async def test_announce_read_child_failure(self):
        async def mock_read(key):
            raise RuntimeError("read failed")

        async def mock_steer(key, msg):
            assert "failed to read" in msg.lower()
            return True

        announcer = SubagentAnnouncer(
            read_child_output=mock_read,
            try_steer=mock_steer,
        )
        record = RunRecord(
            run_id="readfail", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="ok")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered

    async def test_announce_steer_exception(self):
        async def mock_steer(key, msg):
            raise RuntimeError("steer failed")

        async def mock_queue(key, msg):
            return True

        announcer = SubagentAnnouncer(try_steer=mock_steer, try_queue=mock_queue)
        record = RunRecord(
            run_id="steer-exc", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="ok", output="done")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered  # Fell back to queue

    async def test_announce_queue_exception(self):
        async def mock_queue(key, msg):
            raise RuntimeError("queue failed")

        announcer = SubagentAnnouncer(try_queue=mock_queue)
        record = RunRecord(
            run_id="q-exc", child_key="c", requester_key="parent-key",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="ok")
        delivered = await announcer.announce_completion(record, outcome)
        assert not delivered
        assert announcer.pending_count == 1

    async def test_announce_direct_exception(self):
        async def mock_direct(key, origin, msg):
            raise RuntimeError("direct failed")

        announcer = SubagentAnnouncer(send_direct=mock_direct)
        record = RunRecord(
            run_id="d-exc", child_key="c", requester_key="parent-key",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="error", error="something broke")
        delivered = await announcer.announce_completion(record, outcome)
        assert not delivered
        assert announcer.pending_count == 1

    async def test_announce_timeout_status(self):
        async def mock_steer(key, msg):
            assert "timed out" in msg
            return True

        announcer = SubagentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="timeout1", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="timeout", error="exceeded limit")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered

    async def test_announce_cancelled_status(self):
        async def mock_steer(key, msg):
            assert "cancelled" in msg.lower()
            return True

        announcer = SubagentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="cancel1", child_key="c", requester_key="p",
            task="t", started_at=time.time(),
        )
        outcome = RunOutcome(status="cancelled", error="user requested")
        delivered = await announcer.announce_completion(record, outcome)
        assert delivered

    def test_get_pending_clears(self):
        announcer = SubagentAnnouncer()
        announcer._pending_queue["key1"] = ["msg1", "msg2"]
        pending = announcer.get_pending("key1")
        assert len(pending) == 2
        assert announcer.get_pending("key1") == []

    def test_get_pending_nonexistent(self):
        announcer = SubagentAnnouncer()
        assert announcer.get_pending("nonexistent") == []


# ─────────────────────────────────────────────────────────────
# AgentManager Extended
# ─────────────────────────────────────────────────────────────

class TestAgentManagerExtended:
    @pytest.fixture
    def manager(self, tmp_path):
        config = AgentManagerConfig(
            max_spawn_depth=2,
        )
        registry = SubagentRegistry()
        tq = TaskQueue()
        return AgentManager(config=config, task_queue=tq, registry=registry)

    async def test_send_message_to_active(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="steer test"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        sent = await manager.send(result.child_key, "Focus on tests")
        assert sent

    async def test_send_message_nonexistent(self, manager):
        await manager.initialize()
        sent = await manager.send("nonexistent", "message")
        assert not sent

    async def test_send_message_completed(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="complete test"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        manager.registry.complete(result.run_id, RunOutcome(status="ok"))
        sent = await manager.send(result.run_id, "message")
        assert not sent

    async def test_logs_nonexistent(self, manager):
        await manager.initialize()
        logs = await manager.logs("nonexistent-id")
        assert any("No agent found" in line for line in logs)

    async def test_cleanup_old_records(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="old task"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        manager.registry.complete(result.run_id, RunOutcome(status="ok"))
        manager.registry.mark_cleanup_done(result.run_id)
        # Manually set ended_at in the past
        record = manager.registry.get(result.run_id)
        record.ended_at = time.time() - 7200

        cleaned = await manager.cleanup()
        assert cleaned >= 1

    async def test_spawn_params_defaults(self):
        params = SpawnParams(task="test")
        assert params.agent_id.startswith("agent-")
        assert params.mode == "run"
        assert params.cleanup == "keep"
        assert params.timeout == 0
        assert params.execution == "mt"

    async def test_spawn_result_fields(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="test", label="my-label"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        assert result.status == "accepted"
        assert result.note  # Should have auto-announce note

    async def test_agent_status_fields(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="status test"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        statuses = await manager.list()
        assert len(statuses) >= 1
        status = statuses[0]
        assert isinstance(status, AgentStatus)
        assert status.task == "status test"
        assert status.status == "running"

    async def test_kill_by_child_key(self, manager):
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="kill by key"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        killed = await manager.kill(result.child_key)
        assert killed

    async def test_execute_child_no_factory(self, manager):
        """Without agent_factory, child execution returns error outcome."""
        await manager.initialize()
        result = await manager.spawn(
            SpawnParams(task="no factory"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        # Wait for execution to complete
        await asyncio.sleep(0.2)
        record = manager.registry.get(result.run_id)
        if record and record.outcome:
            assert record.outcome.status == "error"

    async def test_execute_child_with_factory(self, tmp_path):
        """With agent_factory, child execution runs the factory."""
        factory_calls = []

        async def mock_factory(**kwargs):
            factory_calls.append(kwargs)
            return "factory result"

        config = AgentManagerConfig(max_spawn_depth=2)
        registry = SubagentRegistry()
        tq = TaskQueue()
        manager = AgentManager(
            config=config, task_queue=tq, registry=registry,
            agent_factory=mock_factory,
        )
        await manager.initialize()

        result = await manager.spawn(
            SpawnParams(task="factory task", model="test-model"),
            SpawnContext(requester_session="parent", caller_depth=0),
        )
        # Wait for execution
        await asyncio.sleep(0.3)
        assert len(factory_calls) >= 1
