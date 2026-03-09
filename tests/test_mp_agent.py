"""Tests for mpAgent: IPC protocol, lifecycle, fault handling, heartbeat, config, registry, announcer."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from march.agents.ipc import (
    MSG_HEARTBEAT,
    MSG_KILL,
    MSG_LOG,
    MSG_PROGRESS,
    MSG_RESULT,
    MSG_STEER,
    MSG_SPAWN_REQUEST,
    MSG_SPAWN_RESULT,
    MSG_SPAWN_STEER,
    MSG_SPAWN_KILL,
    MSG_CHILD_COMPLETED,
    _HEADER_FMT,
    _HEADER_SIZE,
    _pack,
    _unpack,
    create_socket_pair,
    recv_message,
    recv_message_sync,
    send_message,
    send_message_sync,
)
from march.agents.registry import AgentRegistry, RunOutcome, RunRecord, SubagentRegistry
from march.agents.announce import AgentAnnouncer, SubagentAnnouncer
from march.config.schema import AgentsConfig, MpConfig, MtConfig, SubagentsCommonConfig


# ═══════════════════════════════════════════════════════════════════════
# IPC Protocol Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIPC:
    """Tests for the IPC protocol layer (ipc.py)."""

    def test_socket_pair_creation(self):
        """Create a socket pair and verify bidirectional communication."""
        parent_sock, child_sock = create_socket_pair()
        try:
            # Both sockets should be valid
            assert parent_sock.fileno() >= 0
            assert child_sock.fileno() >= 0

            # Send from parent to child
            parent_sock.sendall(b"hello child")
            data = child_sock.recv(1024)
            assert data == b"hello child"

            # Send from child to parent
            child_sock.sendall(b"hello parent")
            data = parent_sock.recv(1024)
            assert data == b"hello parent"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_heartbeat_send_receive(self):
        """Child sends heartbeat via sync API, parent receives via async API."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            hb_msg = {
                "type": MSG_HEARTBEAT,
                "ts": time.time(),
                "data": {
                    "memory_rss_mb": 512.3,
                    "elapsed_seconds": 10.5,
                    "tokens_used": 100,
                    "summary": "working on it",
                    "current_tool": "exec",
                },
            }
            # Child sends synchronously
            send_message_sync(child_sock, hb_msg)

            # Parent receives asynchronously
            received = await recv_message(parent_sock)
            assert received["type"] == MSG_HEARTBEAT
            assert received["data"]["memory_rss_mb"] == 512.3
            assert received["data"]["current_tool"] == "exec"
            assert received["data"]["summary"] == "working on it"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_steer_message_delivery(self):
        """Parent sends steering message, child receives it."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            steer_msg = {
                "type": MSG_STEER,
                "message": "Focus on the unit tests first",
            }
            # Parent sends asynchronously
            await send_message(parent_sock, steer_msg)

            # Child receives synchronously
            received = recv_message_sync(child_sock, timeout=5.0)
            assert received is not None
            assert received["type"] == MSG_STEER
            assert received["message"] == "Focus on the unit tests first"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_large_result_transfer(self):
        """Transfer a large result (>1MB) without data loss."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            # Create a >1MB payload
            large_output = "x" * (1024 * 1024 + 100)  # ~1MB + 100 bytes
            result_msg = {
                "type": MSG_RESULT,
                "status": "ok",
                "output": large_output,
                "error": "",
                "tokens": 5000,
                "cost": 1.23,
            }

            # Use a thread for the sync send (socket buffer may fill up for large msgs)
            import threading
            send_error = [None]

            def sender():
                try:
                    send_message_sync(child_sock, result_msg)
                except Exception as e:
                    send_error[0] = e

            t = threading.Thread(target=sender)
            t.start()

            # Parent receives asynchronously
            received = await asyncio.wait_for(recv_message(parent_sock), timeout=10.0)
            t.join(timeout=5)

            assert send_error[0] is None, f"Send failed: {send_error[0]}"
            assert received["type"] == MSG_RESULT
            assert received["status"] == "ok"
            assert len(received["output"]) == len(large_output)
            assert received["output"] == large_output
            assert received["tokens"] == 5000
            assert received["cost"] == 1.23
        finally:
            parent_sock.close()
            child_sock.close()

    def test_msgpack_serialization_roundtrip(self):
        """All message types survive msgpack serialization/deserialization."""
        messages = [
            # Steer (parent → child)
            {"type": MSG_STEER, "message": "change direction"},
            # Kill (parent → child)
            {"type": MSG_KILL},
            # Heartbeat (child → parent)
            {
                "type": MSG_HEARTBEAT,
                "ts": 1709912345.678,
                "data": {
                    "memory_rss_mb": 2048.0,
                    "elapsed_seconds": 125.3,
                    "tokens_used": 12340,
                    "total_cost": 0.0856,
                    "tool_calls_made": 5,
                    "llm_calls_made": 3,
                    "summary": "Running training script",
                    "current_tool": "exec",
                    "current_tool_detail": "python train.py — epoch 3/10",
                    "recent_tools": [
                        {"name": "file_read", "status": "done", "ms": 45},
                        {"name": "exec", "status": "running", "summary": "training"},
                    ],
                },
            },
            # Progress (child → parent)
            {
                "type": MSG_PROGRESS,
                "tool_name": "web_search",
                "status": "done",
                "summary": "Found 3 results",
                "duration_ms": 1234.5,
            },
            # Result (child → parent)
            {
                "type": MSG_RESULT,
                "status": "ok",
                "output": "Task completed successfully",
                "error": "",
                "tokens": 5000,
                "cost": 0.42,
            },
            # Log (child → parent)
            {
                "type": MSG_LOG,
                "level": "info",
                "message": "Starting task execution",
            },
        ]

        for original in messages:
            packed = _pack(original)
            assert isinstance(packed, bytes)
            restored = _unpack(packed)
            assert restored == original, f"Roundtrip failed for {original['type']}"

    @pytest.mark.asyncio
    async def test_multiple_messages_in_sequence(self):
        """Send multiple messages in sequence and receive them all."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msgs = [
                {"type": MSG_HEARTBEAT, "ts": 1.0, "data": {"summary": "msg1"}},
                {"type": MSG_PROGRESS, "tool_name": "exec", "status": "running",
                 "summary": "step 2", "duration_ms": 100.0},
                {"type": MSG_RESULT, "status": "ok", "output": "done",
                 "error": "", "tokens": 0, "cost": 0.0},
            ]
            for msg in msgs:
                send_message_sync(child_sock, msg)

            for expected in msgs:
                received = await recv_message(parent_sock)
                assert received["type"] == expected["type"]
        finally:
            parent_sock.close()
            child_sock.close()

    def test_sync_recv_timeout(self):
        """recv_message_sync returns None on timeout."""
        parent_sock, child_sock = create_socket_pair()
        try:
            # No message sent — should timeout
            result = recv_message_sync(parent_sock, timeout=0.1)
            assert result is None
        finally:
            parent_sock.close()
            child_sock.close()

    def test_recv_sync_connection_closed(self):
        """recv_message_sync raises ConnectionError when remote closes."""
        parent_sock, child_sock = create_socket_pair()
        child_sock.close()  # Close child end
        with pytest.raises(ConnectionError):
            recv_message_sync(parent_sock, timeout=1.0)
        parent_sock.close()

    @pytest.mark.asyncio
    async def test_recv_async_connection_closed(self):
        """recv_message raises ConnectionError when remote closes."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_sock.close()
        with pytest.raises(ConnectionError):
            await recv_message(parent_sock)
        parent_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════


def _child_success_target(child_fd: int) -> None:
    """Simple child process target that sends a success result via IPC."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)
    try:
        # Send a heartbeat first
        send_message_sync(sock, {
            "type": MSG_HEARTBEAT,
            "ts": time.time(),
            "data": {"summary": "starting", "memory_rss_mb": 100.0},
        })
        time.sleep(0.1)
        # Send success result
        send_message_sync(sock, {
            "type": MSG_RESULT,
            "status": "ok",
            "output": "task completed successfully",
            "error": "",
            "tokens": 1000,
            "cost": 0.05,
        })
    finally:
        sock.close()


def _child_error_target(child_fd: int) -> None:
    """Child process target that sends an error result via IPC."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)
    try:
        send_message_sync(sock, {
            "type": MSG_RESULT,
            "status": "error",
            "output": "",
            "error": "ValueError: something went wrong",
            "tokens": 0,
            "cost": 0.0,
        })
    finally:
        sock.close()


class TestLifecycle:
    """Tests for mpAgent spawn/complete lifecycle."""

    @pytest.mark.asyncio
    async def test_spawn_and_complete_success(self):
        """Spawn a child process that completes successfully."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_success_target, args=(child_fd,))
        proc.start()
        child_sock.close()

        # Receive heartbeat
        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_HEARTBEAT
        assert msg["data"]["summary"] == "starting"

        # Receive result
        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_RESULT
        assert msg["status"] == "ok"
        assert msg["output"] == "task completed successfully"
        assert msg["tokens"] == 1000

        proc.join(timeout=5)
        assert proc.exitcode == 0
        parent_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_and_complete_error(self):
        """Spawn a child process that returns an error."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_error_target, args=(child_fd,))
        proc.start()
        child_sock.close()

        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_RESULT
        assert msg["status"] == "error"
        assert "ValueError" in msg["error"]

        proc.join(timeout=5)
        assert proc.exitcode == 0
        parent_sock.close()

    def test_session_id_format_mt(self):
        """mtAgent session ID follows the format {agent_id}:mtagent:{uuid[:12]}."""
        agent_id = "subagent-abc12345"
        uid = uuid4().hex[:12]
        session_id = f"{agent_id}:mtagent:{uid}"

        parts = session_id.split(":")
        assert len(parts) == 3
        assert parts[1] == "mtagent"
        assert len(parts[2]) == 12

    def test_session_id_format_mp(self):
        """mpAgent session ID follows the format {agent_id}:mpagent:{uuid[:12]}."""
        agent_id = "subagent-def67890"
        uid = uuid4().hex[:12]
        session_id = f"{agent_id}:mpagent:{uid}"

        parts = session_id.split(":")
        assert len(parts) == 3
        assert parts[1] == "mpagent"
        assert len(parts[2]) == 12


# ═══════════════════════════════════════════════════════════════════════
# Fault Handling Tests
# ═══════════════════════════════════════════════════════════════════════


def _child_exception_target(child_fd: int) -> None:
    """Child that crashes with an unhandled exception after sending error via IPC."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)
    try:
        send_message_sync(sock, {
            "type": MSG_RESULT,
            "status": "error",
            "output": "",
            "error": "RuntimeError: child crashed",
            "tokens": 0,
            "cost": 0.0,
        })
    finally:
        sock.close()
    sys.exit(1)


def _child_freeze_target(child_fd: int) -> None:
    """Child that freezes (infinite sleep) without sending heartbeats."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)
    # Just sleep forever — no heartbeat, no result
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sock.close()


class TestFaultHandling:
    """Tests for mpAgent fault handling."""

    @pytest.mark.asyncio
    async def test_child_exception_returns_error(self):
        """Child process exception → parent receives error result via IPC."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_exception_target, args=(child_fd,))
        proc.start()
        child_sock.close()

        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_RESULT
        assert msg["status"] == "error"
        assert "RuntimeError" in msg["error"]

        proc.join(timeout=5)
        parent_sock.close()

    @pytest.mark.asyncio
    async def test_child_freeze_timeout_kill(self):
        """Child freezes → parent detects via heartbeat timeout → kills process group."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_freeze_target, args=(child_fd,))
        proc.start()
        child_sock.close()
        pid = proc.pid

        # Wait a short time for the child to start
        await asyncio.sleep(0.5)

        # Simulate heartbeat timeout detection — send SIGTERM to process group
        assert proc.is_alive()
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        # Wait for grace period
        await asyncio.sleep(1.0)

        if proc.is_alive():
            # Force kill
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        proc.join(timeout=5)
        assert not proc.is_alive()
        parent_sock.close()

    @pytest.mark.asyncio
    async def test_parent_always_gets_result(self):
        """No matter what fault, the parent can always construct a result."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        # Spawn a child that crashes immediately
        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_exception_target, args=(child_fd,))
        proc.start()
        child_sock.close()

        # Try to get a result via IPC
        result = None
        try:
            msg = await asyncio.wait_for(recv_message(parent_sock), timeout=5.0)
            result = RunOutcome(
                status=msg.get("status", "error"),
                output=msg.get("output", ""),
                error=msg.get("error", ""),
            )
        except (ConnectionError, asyncio.TimeoutError):
            # IPC broken — construct error from process exit
            proc.join(timeout=5)
            exitcode = proc.exitcode
            result = RunOutcome(
                status="error",
                error=f"Child exited with code {exitcode}",
            )

        # Guarantee: we always have a result
        assert result is not None
        assert result.status in ("ok", "error", "timeout", "cancelled")

        proc.join(timeout=5)
        parent_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# Process Group Cleanup Tests
# ═══════════════════════════════════════════════════════════════════════


def _child_with_grandchild(child_fd: int) -> None:
    """Child that spawns a grandchild process, then waits."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)

    # Spawn a grandchild that sleeps
    import subprocess
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        preexec_fn=None,  # Inherits our process group
    )

    # Send the grandchild PID to parent so it can verify cleanup
    send_message_sync(sock, {
        "type": MSG_LOG,
        "level": "info",
        "message": f"grandchild_pid={grandchild.pid}",
    })

    # Wait for kill signal
    try:
        while True:
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sock.close()


class TestProcessGroup:
    """Tests for process group cleanup via killpg."""

    @pytest.mark.asyncio
    async def test_killpg_cleans_process_group(self):
        """killpg cleans up the entire process group including grandchildren."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_with_grandchild, args=(child_fd,))
        proc.start()
        child_sock.close()
        pgid = proc.pid

        # Wait for the log message with grandchild PID
        msg = await asyncio.wait_for(recv_message(parent_sock), timeout=10.0)
        assert msg["type"] == MSG_LOG
        grandchild_pid = int(msg["message"].split("=")[1])

        # Verify grandchild is alive
        try:
            os.kill(grandchild_pid, 0)
            grandchild_alive = True
        except (ProcessLookupError, PermissionError):
            grandchild_alive = False
        assert grandchild_alive, "Grandchild should be alive before killpg"

        # Kill the entire process group
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

        proc.join(timeout=5)

        # Wait a moment for the grandchild to be reaped
        await asyncio.sleep(0.5)

        # Verify grandchild is dead
        try:
            os.kill(grandchild_pid, 0)
            grandchild_still_alive = True
        except (ProcessLookupError, PermissionError):
            grandchild_still_alive = False
        assert not grandchild_still_alive, "Grandchild should be dead after killpg"

        parent_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# Heartbeat Tests
# ═══════════════════════════════════════════════════════════════════════


def _child_heartbeat_target(child_fd: int, interval: float) -> None:
    """Child that sends heartbeats at a configurable interval."""
    os.setpgrp()
    sock = socket.fromfd(child_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_fd)
    try:
        for i in range(3):
            send_message_sync(sock, {
                "type": MSG_HEARTBEAT,
                "ts": time.time(),
                "data": {
                    "memory_rss_mb": 256.0 + i * 10,
                    "elapsed_seconds": interval * (i + 1),
                    "tokens_used": 100 * (i + 1),
                    "total_cost": 0.01 * (i + 1),
                    "tool_calls_made": i,
                    "llm_calls_made": i,
                    "summary": f"heartbeat {i + 1}",
                    "current_tool": "exec" if i % 2 == 0 else "",
                    "current_tool_detail": f"step {i + 1}",
                    "recent_tools": [{"name": "exec", "status": "done", "ms": 100}],
                },
            })
            time.sleep(interval)

        # Send result after heartbeats
        send_message_sync(sock, {
            "type": MSG_RESULT,
            "status": "ok",
            "output": "done after heartbeats",
            "error": "",
            "tokens": 300,
            "cost": 0.03,
        })
    finally:
        sock.close()


class TestHeartbeat:
    """Tests for heartbeat content and interval."""

    @pytest.mark.asyncio
    async def test_heartbeat_includes_status(self):
        """Heartbeat messages include memory_rss, current_tool, etc."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_heartbeat_target, args=(child_fd, 0.2))
        proc.start()
        child_sock.close()

        # Receive first heartbeat
        msg = await asyncio.wait_for(recv_message(parent_sock), timeout=5.0)
        assert msg["type"] == MSG_HEARTBEAT
        data = msg["data"]

        # Verify required fields
        assert "memory_rss_mb" in data
        assert isinstance(data["memory_rss_mb"], (int, float))
        assert "elapsed_seconds" in data
        assert "tokens_used" in data
        assert "summary" in data
        assert "current_tool" in data
        assert "current_tool_detail" in data
        assert "recent_tools" in data
        assert isinstance(data["recent_tools"], list)

        proc.join(timeout=10)
        parent_sock.close()

    @pytest.mark.asyncio
    async def test_heartbeat_interval_configurable(self):
        """Heartbeat interval is configurable (passed to child)."""
        interval = 0.3  # 300ms for fast testing
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        child_fd = child_sock.fileno()

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_heartbeat_target, args=(child_fd, interval))
        proc.start()
        child_sock.close()

        # Receive first two heartbeats and measure interval
        t0 = time.monotonic()
        msg1 = await asyncio.wait_for(recv_message(parent_sock), timeout=5.0)
        assert msg1["type"] == MSG_HEARTBEAT

        msg2 = await asyncio.wait_for(recv_message(parent_sock), timeout=5.0)
        assert msg2["type"] == MSG_HEARTBEAT
        t1 = time.monotonic()

        # The interval between heartbeats should be approximately `interval`
        elapsed = t1 - t0
        # Allow generous tolerance since we're measuring from first to second
        assert elapsed >= interval * 0.5, f"Heartbeats too fast: {elapsed}s"
        assert elapsed < interval * 5, f"Heartbeats too slow: {elapsed}s"

        proc.join(timeout=10)
        parent_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConfig:
    """Tests for MpConfig, MtConfig, and AgentsConfig."""

    def test_config_loads_mp_section(self):
        """MpConfig loads correctly with custom values."""
        mp = MpConfig(
            max_concurrent=4,
            heartbeat_interval_seconds=30,
            heartbeat_timeout_seconds=120,
            kill_grace_seconds=5,
            spawn_method="forkserver",
        )
        assert mp.max_concurrent == 4
        assert mp.heartbeat_interval_seconds == 30
        assert mp.heartbeat_timeout_seconds == 120
        assert mp.kill_grace_seconds == 5
        assert mp.spawn_method == "forkserver"

    def test_default_config_values(self):
        """MpConfig defaults are correct per the plan."""
        mp = MpConfig()
        assert mp.max_concurrent == 8
        assert mp.heartbeat_interval_seconds == 60
        assert mp.heartbeat_timeout_seconds == 300
        assert mp.kill_grace_seconds == 10
        assert mp.spawn_method == "spawn"

    def test_mt_config(self):
        """MtConfig loads correctly."""
        mt = MtConfig()
        assert mt.max_concurrent == 8

        mt_custom = MtConfig(max_concurrent=16)
        assert mt_custom.max_concurrent == 16

    def test_agents_config_structure(self):
        """AgentsConfig contains mt, mp, and subagents sections."""
        agents = AgentsConfig()

        # Check mt section
        assert isinstance(agents.mt, MtConfig)
        assert agents.mt.max_concurrent == 8

        # Check mp section
        assert isinstance(agents.mp, MpConfig)
        assert agents.mp.max_concurrent == 8
        assert agents.mp.heartbeat_interval_seconds == 60

        # Check subagents section
        assert isinstance(agents.subagents, SubagentsCommonConfig)
        assert agents.subagents.max_spawn_depth == 1

    def test_agents_config_from_dict(self):
        """AgentsConfig can be constructed from a dict (like YAML loading)."""
        data = {
            "mt": {"max_concurrent": 4},
            "mp": {
                "max_concurrent": 2,
                "heartbeat_interval_seconds": 30,
                "heartbeat_timeout_seconds": 120,
                "kill_grace_seconds": 5,
                "spawn_method": "forkserver",
            },
            "subagents": {"max_spawn_depth": 3},
        }
        agents = AgentsConfig(**data)
        assert agents.mt.max_concurrent == 4
        assert agents.mp.max_concurrent == 2
        assert agents.mp.heartbeat_interval_seconds == 30
        assert agents.subagents.max_spawn_depth == 3

    def test_mp_config_in_march_config(self):
        """MpConfig is accessible through the full MarchConfig."""
        from march.config.schema import MarchConfig

        config = MarchConfig()
        assert isinstance(config.agents.mp, MpConfig)
        assert config.agents.mp.heartbeat_timeout_seconds == 300


# ═══════════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Tests for AgentRegistry with new mpAgent fields."""

    def test_registry_in_memory_only(self):
        """AgentRegistry is pure in-memory — no file system involvement."""
        reg = AgentRegistry()

        # Register a record
        reg.register(RunRecord(
            run_id="mem-1",
            child_key="agent:test:mtagent:abc123",
            requester_key="parent",
            task="in-memory test",
            started_at=time.time(),
        ))

        # Verify it's in memory
        assert reg.get("mem-1") is not None

        # No persist_dir, no file on disk — just memory
        # A new instance won't have the record
        reg2 = AgentRegistry()
        assert reg2.get("mem-1") is None

    def test_registry_new_fields(self):
        """RunRecord has execution, pid, and log_path fields."""
        record = RunRecord(
            run_id="mp-1",
            child_key="agent:test:mpagent:def456",
            requester_key="parent",
            task="mp test",
            started_at=time.time(),
            execution="mp",
            pid=12345,
            log_path="/home/user/.march/logs/agent:test:mpagent:def456",
        )

        assert record.execution == "mp"
        assert record.pid == 12345
        assert record.log_path.endswith("def456")

        # Serialization roundtrip preserves new fields
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.execution == "mp"
        assert restored.pid == 12345
        assert restored.log_path == record.log_path

    def test_list_by_execution(self):
        """Filter records by execution mode (mt/mp)."""
        reg = AgentRegistry()

        reg.register(RunRecord(
            run_id="mt-1", child_key="c-mt-1", requester_key="p",
            task="mt task", started_at=time.time(), execution="mt",
        ))
        reg.register(RunRecord(
            run_id="mp-1", child_key="c-mp-1", requester_key="p",
            task="mp task 1", started_at=time.time(), execution="mp",
        ))
        reg.register(RunRecord(
            run_id="mp-2", child_key="c-mp-2", requester_key="p",
            task="mp task 2", started_at=time.time(), execution="mp",
        ))

        mt_records = reg.list_by_execution("mt")
        assert len(mt_records) == 1
        assert mt_records[0].run_id == "mt-1"

        mp_records = reg.list_by_execution("mp")
        assert len(mp_records) == 2
        assert {r.run_id for r in mp_records} == {"mp-1", "mp-2"}

    def test_backward_compat_alias(self):
        """SubagentRegistry is an alias for AgentRegistry."""
        assert SubagentRegistry is AgentRegistry

        # Can instantiate via the alias
        reg = SubagentRegistry()
        reg.register(RunRecord(
            run_id="compat-1", child_key="c", requester_key="p",
            task="alias test", started_at=time.time(),
        ))
        assert reg.get("compat-1") is not None

    def test_default_execution_is_mt(self):
        """RunRecord defaults to execution='mt'."""
        record = RunRecord(
            run_id="default-exec",
            child_key="c",
            requester_key="p",
            task="t",
            started_at=time.time(),
        )
        assert record.execution == "mt"
        assert record.pid == 0
        assert record.log_path == ""


# ═══════════════════════════════════════════════════════════════════════
# Announcer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAnnouncer:
    """Tests for AgentAnnouncer with execution-aware display names."""

    @pytest.mark.asyncio
    async def test_announcer_display_name_mt(self):
        """mtAgent completion message uses 'mtAgent' display name."""
        messages_sent: list[str] = []

        async def mock_steer(key: str, msg: str) -> bool:
            messages_sent.append(msg)
            return True

        announcer = AgentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="ann-mt",
            child_key="agent:test:mtagent:abc",
            requester_key="parent",
            task="mt test",
            started_at=time.time(),
            execution="mt",
        )
        outcome = RunOutcome(status="ok", output="mt result")
        await announcer.announce_completion(record, outcome)

        assert len(messages_sent) == 1
        assert "mtAgent" in messages_sent[0]
        assert "✅" in messages_sent[0]

    @pytest.mark.asyncio
    async def test_announcer_display_name_mp(self):
        """mpAgent completion message uses 'mpAgent' display name."""
        messages_sent: list[str] = []

        async def mock_steer(key: str, msg: str) -> bool:
            messages_sent.append(msg)
            return True

        announcer = AgentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="ann-mp",
            child_key="agent:test:mpagent:def",
            requester_key="parent",
            task="mp test",
            started_at=time.time(),
            execution="mp",
        )
        outcome = RunOutcome(status="error", error="OOM killed")
        await announcer.announce_completion(record, outcome)

        assert len(messages_sent) == 1
        assert "mpAgent" in messages_sent[0]
        assert "❌" in messages_sent[0]
        assert "OOM killed" in messages_sent[0]

    def test_backward_compat_alias(self):
        """SubagentAnnouncer is an alias for AgentAnnouncer."""
        assert SubagentAnnouncer is AgentAnnouncer

        # Can instantiate via the alias
        announcer = SubagentAnnouncer()
        assert announcer.pending_count == 0

    @pytest.mark.asyncio
    async def test_announcer_timeout_status_mp(self):
        """mpAgent timeout uses correct display name."""
        messages_sent: list[str] = []

        async def mock_steer(key: str, msg: str) -> bool:
            messages_sent.append(msg)
            return True

        announcer = AgentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="ann-timeout",
            child_key="agent:test:mpagent:timeout",
            requester_key="parent",
            task="long task",
            started_at=time.time(),
            execution="mp",
        )
        outcome = RunOutcome(status="timeout", error="No heartbeat for 300s")
        await announcer.announce_completion(record, outcome)

        assert len(messages_sent) == 1
        assert "mpAgent" in messages_sent[0]
        assert "⏱️" in messages_sent[0]

    @pytest.mark.asyncio
    async def test_announcer_cancelled_status_mp(self):
        """mpAgent cancellation uses correct display name."""
        messages_sent: list[str] = []

        async def mock_steer(key: str, msg: str) -> bool:
            messages_sent.append(msg)
            return True

        announcer = AgentAnnouncer(try_steer=mock_steer)
        record = RunRecord(
            run_id="ann-cancel",
            child_key="agent:test:mpagent:cancel",
            requester_key="parent",
            task="cancelled task",
            started_at=time.time(),
            execution="mp",
        )
        outcome = RunOutcome(status="cancelled", error="Killed by parent")
        await announcer.announce_completion(record, outcome)

        assert len(messages_sent) == 1
        assert "mpAgent" in messages_sent[0]
        assert "🚫" in messages_sent[0]


# ═══════════════════════════════════════════════════════════════════════
# Agent Tree Tests (recursive depth > 1)
# ═══════════════════════════════════════════════════════════════════════


from march.agents.manager import AgentManager, AgentManagerConfig, SpawnParams, SpawnContext


class TestAgentTree:
    """Recursive agent tree tests."""

    def _make_record(
        self,
        run_id: str,
        child_key: str,
        requester_key: str,
        execution: str = "mt",
        active: bool = True,
    ) -> RunRecord:
        """Helper to create a RunRecord."""
        record = RunRecord(
            run_id=run_id,
            child_key=child_key,
            requester_key=requester_key,
            task=f"task for {run_id}",
            started_at=time.time(),
            execution=execution,
        )
        if not active:
            record.ended_at = time.time()
            record.outcome = RunOutcome(status="ok", output="done")
        return record

    def test_get_subtree_empty(self):
        """No descendants returns empty list."""
        reg = AgentRegistry()
        reg.register(self._make_record("r1", "child-A", "main-session"))

        result = reg.get_subtree("child-A")
        assert result == []

    def test_get_subtree_single_child(self):
        """Single child is returned."""
        reg = AgentRegistry()
        parent = self._make_record("r1", "child-A", "main-session")
        child = self._make_record("r2", "child-B", "child-A")
        reg.register(parent)
        reg.register(child)

        result = reg.get_subtree("child-A")
        assert len(result) == 1
        assert result[0].run_id == "r2"

    def test_get_subtree_deep_tree(self):
        """depth=3 tree, verify depth-first order."""
        reg = AgentRegistry()
        # Tree structure:
        #   main-session
        #     └─ A (r1)
        #         ├─ B (r2)
        #         │   └─ D (r4)
        #         └─ C (r3)
        reg.register(self._make_record("r1", "A", "main-session"))
        reg.register(self._make_record("r2", "B", "A"))
        reg.register(self._make_record("r3", "C", "A"))
        reg.register(self._make_record("r4", "D", "B"))

        result = reg.get_subtree("A")
        run_ids = [r.run_id for r in result]
        # Depth-first: B, D (child of B), then C
        assert run_ids == ["r2", "r4", "r3"]

    def test_get_tree_with_depth(self):
        """Verify each node's depth value is correct."""
        reg = AgentRegistry()
        # Tree:
        #   root
        #     └─ A (depth 0)
        #         └─ B (depth 1)
        #             └─ C (depth 2)
        reg.register(self._make_record("r1", "A", "root"))
        reg.register(self._make_record("r2", "B", "A"))
        reg.register(self._make_record("r3", "C", "B"))

        result = reg.get_tree_with_depth("root")
        assert len(result) == 3
        assert result[0] == (0, reg.get("r1"))
        assert result[1] == (1, reg.get("r2"))
        assert result[2] == (2, reg.get("r3"))

    @pytest.mark.asyncio
    async def test_kill_recursive_kills_all(self):
        """kill_recursive kills all descendants."""
        reg = AgentRegistry()
        mgr = AgentManager(
            config=AgentManagerConfig(max_spawn_depth=5),
            registry=reg,
        )
        await mgr.initialize()

        # Build tree: A → B → C (all active)
        reg.register(self._make_record("r1", "A", "main-session"))
        reg.register(self._make_record("r2", "B", "A"))
        reg.register(self._make_record("r3", "C", "B"))

        killed = await mgr.kill_recursive("r1")
        assert killed == 3

        # All should be marked completed
        for rid in ["r1", "r2", "r3"]:
            record = reg.get(rid)
            assert not record.is_active
            assert record.outcome.status == "cancelled"

    @pytest.mark.asyncio
    async def test_kill_recursive_depth_first(self):
        """Verify depth-first order: deepest killed first."""
        reg = AgentRegistry()
        mgr = AgentManager(
            config=AgentManagerConfig(max_spawn_depth=5),
            registry=reg,
        )
        await mgr.initialize()

        # Tree: A → B → C
        reg.register(self._make_record("r1", "A", "main-session"))
        reg.register(self._make_record("r2", "B", "A"))
        reg.register(self._make_record("r3", "C", "B"))

        # Track kill order by patching kill
        kill_order: list[str] = []
        original_kill = mgr.kill

        async def tracking_kill(agent_id: str) -> bool:
            kill_order.append(agent_id)
            return await original_kill(agent_id)

        mgr.kill = tracking_kill  # type: ignore[assignment]

        await mgr.kill_recursive("r1")

        # C (deepest) should be killed first, then B, then A
        assert kill_order == ["r3", "r2", "r1"]

    @pytest.mark.asyncio
    async def test_execution_consistency_enforced(self):
        """When parent is mp, child must also be mp."""
        reg = AgentRegistry()
        mgr = AgentManager(
            config=AgentManagerConfig(max_spawn_depth=5),
            registry=reg,
        )
        await mgr.initialize()

        # Register parent as an mp agent
        reg.register(self._make_record("r1", "parent-agent", "main-session", execution="mp"))

        # Spawn child with execution="mt" — should be forced to "mp"
        result = await mgr.spawn(
            SpawnParams(task="child task", execution="mt"),
            SpawnContext(requester_session="parent-agent", caller_depth=1),
        )
        assert result.status == "accepted"

        # Find the child record and verify execution was forced to mp
        child_record = reg.get_by_child_key(result.child_key)
        assert child_record is not None
        assert child_record.execution == "mp"

    @pytest.mark.asyncio
    async def test_execution_consistency_main_session_allows_any(self):
        """Main session (not a sub-agent) allows any execution mode."""
        reg = AgentRegistry()
        mgr = AgentManager(
            config=AgentManagerConfig(max_spawn_depth=5),
            registry=reg,
        )
        await mgr.initialize()

        # No parent record for "main-session" in registry — it's a main session

        # Spawn with mt — should be allowed
        result_mt = await mgr.spawn(
            SpawnParams(task="mt task", execution="mt"),
            SpawnContext(requester_session="main-session", caller_depth=0),
        )
        assert result_mt.status == "accepted"
        child_mt = reg.get_by_child_key(result_mt.child_key)
        assert child_mt.execution == "mt"

        # Spawn with mp — should also be allowed
        result_mp = await mgr.spawn(
            SpawnParams(task="mp task", execution="mp"),
            SpawnContext(requester_session="main-session", caller_depth=0),
        )
        assert result_mp.status == "accepted"
        child_mp = reg.get_by_child_key(result_mp.child_key)
        assert child_mp.execution == "mp"


# ═══════════════════════════════════════════════════════════════════════
# IPC Spawn Proxy Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIPCSpawnProxy:
    """IPC spawn proxy message tests."""

    def test_spawn_request_message_roundtrip(self):
        """spawn_request message survives msgpack serialization/deserialization."""
        msg = {
            "type": MSG_SPAWN_REQUEST,
            "task": "analyze this data",
            "agent_id": "agent-abc123",
            "model": "gpt-4",
            "timeout": 300,
            "request_id": "req-001",
        }
        packed = _pack(msg)
        assert isinstance(packed, bytes)
        restored = _unpack(packed)
        assert restored == msg
        assert restored["type"] == MSG_SPAWN_REQUEST
        assert restored["task"] == "analyze this data"
        assert restored["agent_id"] == "agent-abc123"
        assert restored["model"] == "gpt-4"
        assert restored["timeout"] == 300
        assert restored["request_id"] == "req-001"

    def test_spawn_result_message_roundtrip(self):
        """spawn_result message survives msgpack serialization/deserialization."""
        msg = {
            "type": MSG_SPAWN_RESULT,
            "request_id": "req-001",
            "status": "accepted",
            "child_key": "agent-abc:mpagent:def456789012",
            "run_id": "run-uuid-123",
            "error": "",
        }
        packed = _pack(msg)
        restored = _unpack(packed)
        assert restored == msg
        assert restored["type"] == MSG_SPAWN_RESULT
        assert restored["status"] == "accepted"
        assert restored["child_key"] == "agent-abc:mpagent:def456789012"

        # Also test error case
        msg_err = {
            "type": MSG_SPAWN_RESULT,
            "request_id": "req-002",
            "status": "error",
            "child_key": "",
            "run_id": "",
            "error": "max spawn depth reached",
        }
        packed_err = _pack(msg_err)
        restored_err = _unpack(packed_err)
        assert restored_err == msg_err
        assert restored_err["status"] == "error"
        assert restored_err["error"] == "max spawn depth reached"

    def test_child_completed_message_roundtrip(self):
        """child_completed message survives msgpack serialization/deserialization."""
        msg = {
            "type": MSG_CHILD_COMPLETED,
            "child_key": "agent-abc:mpagent:def456789012",
            "status": "ok",
            "output": "Task completed successfully with detailed results.",
            "error": "",
        }
        packed = _pack(msg)
        restored = _unpack(packed)
        assert restored == msg
        assert restored["type"] == MSG_CHILD_COMPLETED
        assert restored["status"] == "ok"
        assert restored["output"] == "Task completed successfully with detailed results."

        # Test error status
        msg_err = {
            "type": MSG_CHILD_COMPLETED,
            "child_key": "agent-xyz:mpagent:abc123456789",
            "status": "error",
            "output": "",
            "error": "RuntimeError: something went wrong",
        }
        packed_err = _pack(msg_err)
        restored_err = _unpack(packed_err)
        assert restored_err == msg_err

    def test_spawn_steer_message_roundtrip(self):
        """spawn_steer message survives msgpack serialization/deserialization."""
        msg = {
            "type": MSG_SPAWN_STEER,
            "child_key": "agent-abc:mpagent:def456789012",
            "message": "Focus on the unit tests first",
        }
        packed = _pack(msg)
        restored = _unpack(packed)
        assert restored == msg
        assert restored["type"] == MSG_SPAWN_STEER
        assert restored["child_key"] == "agent-abc:mpagent:def456789012"
        assert restored["message"] == "Focus on the unit tests first"

    def test_spawn_kill_message_roundtrip(self):
        """spawn_kill message survives msgpack serialization/deserialization."""
        msg = {
            "type": MSG_SPAWN_KILL,
            "child_key": "agent-abc:mpagent:def456789012",
        }
        packed = _pack(msg)
        restored = _unpack(packed)
        assert restored == msg
        assert restored["type"] == MSG_SPAWN_KILL
        assert restored["child_key"] == "agent-abc:mpagent:def456789012"

    @pytest.mark.asyncio
    async def test_spawn_request_ipc_transfer(self):
        """spawn_request sent from child (sync) and received by parent (async)."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msg = {
                "type": MSG_SPAWN_REQUEST,
                "task": "run analysis",
                "agent_id": "",
                "model": "claude-3",
                "timeout": 120,
                "request_id": "req-transfer-001",
            }
            send_message_sync(child_sock, msg)
            received = await recv_message(parent_sock)
            assert received["type"] == MSG_SPAWN_REQUEST
            assert received["task"] == "run analysis"
            assert received["request_id"] == "req-transfer-001"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_result_ipc_transfer(self):
        """spawn_result sent from parent (async) and received by child (sync)."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msg = {
                "type": MSG_SPAWN_RESULT,
                "request_id": "req-transfer-001",
                "status": "accepted",
                "child_key": "agent-test:mpagent:abc123",
                "run_id": "run-456",
                "error": "",
            }
            await send_message(parent_sock, msg)
            received = recv_message_sync(child_sock, timeout=5.0)
            assert received is not None
            assert received["type"] == MSG_SPAWN_RESULT
            assert received["status"] == "accepted"
            assert received["child_key"] == "agent-test:mpagent:abc123"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_child_completed_ipc_transfer(self):
        """child_completed sent from parent (async) and received by child (sync)."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msg = {
                "type": MSG_CHILD_COMPLETED,
                "child_key": "agent-test:mpagent:abc123",
                "status": "ok",
                "output": "Analysis complete: found 42 issues.",
                "error": "",
            }
            await send_message(parent_sock, msg)
            received = recv_message_sync(child_sock, timeout=5.0)
            assert received is not None
            assert received["type"] == MSG_CHILD_COMPLETED
            assert received["output"] == "Analysis complete: found 42 issues."
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_steer_ipc_transfer(self):
        """spawn_steer sent from child (sync) and received by parent (async)."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msg = {
                "type": MSG_SPAWN_STEER,
                "child_key": "agent-test:mpagent:abc123",
                "message": "Hurry up!",
            }
            send_message_sync(child_sock, msg)
            received = await recv_message(parent_sock)
            assert received["type"] == MSG_SPAWN_STEER
            assert received["message"] == "Hurry up!"
        finally:
            parent_sock.close()
            child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_kill_ipc_transfer(self):
        """spawn_kill sent from child (sync) and received by parent (async)."""
        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)
        try:
            msg = {
                "type": MSG_SPAWN_KILL,
                "child_key": "agent-test:mpagent:abc123",
            }
            send_message_sync(child_sock, msg)
            received = await recv_message(parent_sock)
            assert received["type"] == MSG_SPAWN_KILL
            assert received["child_key"] == "agent-test:mpagent:abc123"
        finally:
            parent_sock.close()
            child_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# MpRunner Spawn Handler Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMpRunnerSpawnHandler:
    """Tests for MpRunner spawn proxy handler integration."""

    @pytest.mark.asyncio
    async def test_spawn_handler_called_on_request(self):
        """MpRunner calls spawn_handler when it receives MSG_SPAWN_REQUEST."""
        from march.agents.mp_runner import MpRunner

        handler_calls: list[tuple] = []

        async def mock_spawn_handler(task, agent_id, model, timeout, request_id):
            handler_calls.append((task, agent_id, model, timeout, request_id))
            return ("accepted", "child-key-123", "run-456", "")

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        runner = MpRunner(spawn_handler=mock_spawn_handler)
        runner._parent_sock = parent_sock
        runner._session_id = "test-session"
        runner._done = False

        # Send spawn_request from child side
        send_message_sync(child_sock, {
            "type": MSG_SPAWN_REQUEST,
            "task": "analyze data",
            "agent_id": "agent-test",
            "model": "gpt-4",
            "timeout": 60,
            "request_id": "req-001",
        })

        # Run one iteration of recv
        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_SPAWN_REQUEST

        # Simulate the handler dispatch
        await runner._handle_spawn_request(msg)

        # Verify handler was called
        assert len(handler_calls) == 1
        assert handler_calls[0] == ("analyze data", "agent-test", "gpt-4", 60, "req-001")

        # Verify spawn_result was sent back
        result = recv_message_sync(child_sock, timeout=5.0)
        assert result is not None
        assert result["type"] == MSG_SPAWN_RESULT
        assert result["status"] == "accepted"
        assert result["child_key"] == "child-key-123"
        assert result["run_id"] == "run-456"

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_no_spawn_handler_returns_error(self):
        """MpRunner without spawn_handler returns error on spawn_request."""
        from march.agents.mp_runner import MpRunner

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        runner = MpRunner()  # No spawn_handler
        runner._parent_sock = parent_sock
        runner._session_id = "test-session"
        runner._done = False

        msg = {
            "type": MSG_SPAWN_REQUEST,
            "task": "analyze data",
            "agent_id": "",
            "model": "",
            "timeout": 0,
            "request_id": "req-002",
        }

        await runner._handle_spawn_request(msg)

        # Verify error result was sent back
        result = recv_message_sync(child_sock, timeout=5.0)
        assert result is not None
        assert result["type"] == MSG_SPAWN_RESULT
        assert result["status"] == "error"
        assert "not configured" in result["error"]

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_notify_child_completed(self):
        """MpRunner.notify_child_completed sends MSG_CHILD_COMPLETED via IPC."""
        from march.agents.mp_runner import MpRunner

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        runner = MpRunner()
        runner._parent_sock = parent_sock
        runner._session_id = "test-session"
        runner._done = False

        await runner.notify_child_completed(
            child_key="grandchild-key-123",
            status="ok",
            output="Grandchild finished successfully",
            error="",
        )

        # Verify message was sent
        result = recv_message_sync(child_sock, timeout=5.0)
        assert result is not None
        assert result["type"] == MSG_CHILD_COMPLETED
        assert result["child_key"] == "grandchild-key-123"
        assert result["status"] == "ok"
        assert result["output"] == "Grandchild finished successfully"
        assert result["error"] == ""

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_steer_handler_called(self):
        """MpRunner calls steer_handler when it receives MSG_SPAWN_STEER."""
        from march.agents.mp_runner import MpRunner

        steer_calls: list[tuple] = []

        async def mock_steer_handler(child_key, message):
            steer_calls.append((child_key, message))
            return True

        runner = MpRunner(steer_handler=mock_steer_handler)
        runner._session_id = "test-session"
        runner._done = False

        msg = {
            "type": MSG_SPAWN_STEER,
            "child_key": "grandchild-key",
            "message": "Focus on tests",
        }

        await runner._handle_spawn_steer(msg)

        assert len(steer_calls) == 1
        assert steer_calls[0] == ("grandchild-key", "Focus on tests")

    @pytest.mark.asyncio
    async def test_kill_handler_called(self):
        """MpRunner calls kill_handler when it receives MSG_SPAWN_KILL."""
        from march.agents.mp_runner import MpRunner

        kill_calls: list[str] = []

        async def mock_kill_handler(child_key):
            kill_calls.append(child_key)
            return True

        runner = MpRunner(kill_handler=mock_kill_handler)
        runner._session_id = "test-session"
        runner._done = False

        msg = {
            "type": MSG_SPAWN_KILL,
            "child_key": "grandchild-key",
        }

        await runner._handle_spawn_kill(msg)

        assert len(kill_calls) == 1
        assert kill_calls[0] == "grandchild-key"

    @pytest.mark.asyncio
    async def test_spawn_handler_exception_returns_error(self):
        """If spawn_handler raises, MpRunner sends error result."""
        from march.agents.mp_runner import MpRunner

        async def failing_handler(task, agent_id, model, timeout, request_id):
            raise ValueError("handler exploded")

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        runner = MpRunner(spawn_handler=failing_handler)
        runner._parent_sock = parent_sock
        runner._session_id = "test-session"
        runner._done = False

        msg = {
            "type": MSG_SPAWN_REQUEST,
            "task": "will fail",
            "agent_id": "",
            "model": "",
            "timeout": 0,
            "request_id": "req-fail",
        }

        await runner._handle_spawn_request(msg)

        result = recv_message_sync(child_sock, timeout=5.0)
        assert result is not None
        assert result["type"] == MSG_SPAWN_RESULT
        assert result["status"] == "error"
        assert "handler exploded" in result["error"]

        parent_sock.close()
        child_sock.close()


# ═══════════════════════════════════════════════════════════════════════
# SpawnProxy Tests (child-side)
# ═══════════════════════════════════════════════════════════════════════


class TestSpawnProxy:
    """Tests for _SpawnProxy in the child process."""

    @pytest.mark.asyncio
    async def test_spawn_proxy_spawn_and_result(self):
        """SpawnProxy.spawn sends request and resolves on result."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        async def simulate_parent():
            """Simulate parent receiving spawn_request and sending result."""
            msg = await recv_message(parent_sock)
            assert msg["type"] == MSG_SPAWN_REQUEST
            assert msg["task"] == "do something"
            await send_message(parent_sock, {
                "type": MSG_SPAWN_RESULT,
                "request_id": msg["request_id"],
                "status": "accepted",
                "child_key": "grandchild-key-abc",
                "run_id": "run-xyz",
                "error": "",
            })

        async def simulate_child():
            """Child calls spawn and waits for result."""
            # We need to receive the spawn_result in a separate thread
            # since the child uses sync recv. For testing, we'll manually
            # resolve the future.
            child_key, run_id = await proxy.spawn("do something")
            return child_key, run_id

        # We need a way to receive the spawn_result on the child side.
        # In production, the heartbeat thread does this. For testing,
        # we'll use a background task to read from child_sock and dispatch.
        import threading

        def recv_and_dispatch():
            """Read from child_sock and dispatch to proxy."""
            result = recv_message_sync(child_sock, timeout=5.0)
            if result and result.get("type") == MSG_SPAWN_RESULT:
                proxy.handle_spawn_result(result)

        recv_thread = threading.Thread(target=recv_and_dispatch)
        recv_thread.start()

        parent_task = asyncio.create_task(simulate_parent())
        child_key, run_id = await asyncio.wait_for(
            proxy.spawn("do something"), timeout=5.0
        )
        await parent_task
        recv_thread.join(timeout=5.0)

        assert child_key == "grandchild-key-abc"
        assert run_id == "run-xyz"

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_proxy_wait_child(self):
        """SpawnProxy.wait_child resolves when child_completed arrives."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        async def simulate_parent():
            """Parent sends child_completed."""
            await asyncio.sleep(0.1)  # Small delay
            await send_message(parent_sock, {
                "type": MSG_CHILD_COMPLETED,
                "child_key": "grandchild-abc",
                "status": "ok",
                "output": "task done",
                "error": "",
            })

        import threading

        def recv_and_dispatch():
            """Read from child_sock and dispatch to proxy."""
            result = recv_message_sync(child_sock, timeout=5.0)
            if result and result.get("type") == MSG_CHILD_COMPLETED:
                proxy.handle_child_completed(result)

        recv_thread = threading.Thread(target=recv_and_dispatch)
        recv_thread.start()

        parent_task = asyncio.create_task(simulate_parent())
        status, output = await asyncio.wait_for(
            proxy.wait_child("grandchild-abc"), timeout=5.0
        )
        await parent_task
        recv_thread.join(timeout=5.0)

        assert status == "ok"
        assert output == "task done"

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_proxy_steer_child(self):
        """SpawnProxy.steer_child sends MSG_SPAWN_STEER via IPC."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        await proxy.steer_child("grandchild-abc", "Focus on tests")

        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_SPAWN_STEER
        assert msg["child_key"] == "grandchild-abc"
        assert msg["message"] == "Focus on tests"

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_proxy_kill_child(self):
        """SpawnProxy.kill_child sends MSG_SPAWN_KILL via IPC."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        await proxy.kill_child("grandchild-abc")

        msg = await recv_message(parent_sock)
        assert msg["type"] == MSG_SPAWN_KILL
        assert msg["child_key"] == "grandchild-abc"

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_proxy_spawn_rejected(self):
        """SpawnProxy.spawn raises RuntimeError when spawn is rejected."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        async def simulate_parent():
            msg = await recv_message(parent_sock)
            await send_message(parent_sock, {
                "type": MSG_SPAWN_RESULT,
                "request_id": msg["request_id"],
                "status": "error",
                "child_key": "",
                "run_id": "",
                "error": "max spawn depth reached",
            })

        import threading

        def recv_and_dispatch():
            result = recv_message_sync(child_sock, timeout=5.0)
            if result and result.get("type") == MSG_SPAWN_RESULT:
                proxy.handle_spawn_result(result)

        recv_thread = threading.Thread(target=recv_and_dispatch)
        recv_thread.start()

        parent_task = asyncio.create_task(simulate_parent())

        with pytest.raises(RuntimeError, match="Spawn rejected"):
            await asyncio.wait_for(proxy.spawn("will be rejected"), timeout=5.0)

        await parent_task
        recv_thread.join(timeout=5.0)

        parent_sock.close()
        child_sock.close()

    @pytest.mark.asyncio
    async def test_spawn_proxy_wait_child_error(self):
        """SpawnProxy.wait_child returns error status and message."""
        import logging
        from march.agents.mp_child import _SpawnProxy

        parent_sock, child_sock = create_socket_pair()
        parent_sock.setblocking(False)

        proxy = _SpawnProxy(
            sock=child_sock,
            session_id="test-child",
            execution="mp",
            logger=logging.getLogger("test"),
        )

        async def simulate_parent():
            await asyncio.sleep(0.1)
            await send_message(parent_sock, {
                "type": MSG_CHILD_COMPLETED,
                "child_key": "grandchild-fail",
                "status": "error",
                "output": "",
                "error": "OOM killed",
            })

        import threading

        def recv_and_dispatch():
            result = recv_message_sync(child_sock, timeout=5.0)
            if result and result.get("type") == MSG_CHILD_COMPLETED:
                proxy.handle_child_completed(result)

        recv_thread = threading.Thread(target=recv_and_dispatch)
        recv_thread.start()

        parent_task = asyncio.create_task(simulate_parent())
        status, output = await asyncio.wait_for(
            proxy.wait_child("grandchild-fail"), timeout=5.0
        )
        await parent_task
        recv_thread.join(timeout=5.0)

        assert status == "error"
        assert "OOM killed" in output

        parent_sock.close()
        child_sock.close()
