"""mpAgent parent-side runner.

``MpRunner`` manages the full lifecycle of a single mpAgent child process:
spawn, heartbeat monitoring, steering, kill, and result collection.

Key guarantee: ``wait_result()`` **always** returns a ``RunOutcome`` — it never
hangs, regardless of child crashes, OOM kills, IPC failures, or timeouts.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from march.agents.ipc import (
    MSG_HEARTBEAT,
    MSG_KILL,
    MSG_LOG,
    MSG_PROGRESS,
    MSG_RESULT,
    MSG_STEER,
    create_socket_pair,
    recv_message,
    send_message,
)
from march.agents.registry import RunOutcome
from march.logging import get_logger

logger = get_logger("march.mp_runner", subsystem="agent")


# ── Configuration dataclass ──────────────────────────────────────────


@dataclass
class MpConfig:
    """Configuration for an mpAgent run, passed from the caller."""

    heartbeat_interval_seconds: float = 60.0
    heartbeat_timeout_seconds: float = 300.0
    kill_grace_seconds: float = 10.0
    log_dir: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MpConfig":
        return cls(
            heartbeat_interval_seconds=data.get("heartbeat_interval_seconds", 60.0),
            heartbeat_timeout_seconds=data.get("heartbeat_timeout_seconds", 300.0),
            kill_grace_seconds=data.get("kill_grace_seconds", 10.0),
            log_dir=data.get("log_dir", ""),
        )


# ── Run handle (returned to caller) ─────────────────────────────────


@dataclass
class MpRunHandle:
    """Handle to a running mpAgent child process.

    Returned by ``MpRunner.spawn()``. The caller uses this to interact
    with the child.
    """

    pid: int
    session_id: str
    runner: "MpRunner"


# ── MpRunner ─────────────────────────────────────────────────────────


class MpRunner:
    """Manages a single mpAgent child process lifecycle.

    Usage::

        runner = MpRunner()
        handle = await runner.spawn(task, session_id, config_path, mp_config)
        outcome = await runner.wait_result()
        # outcome is always a RunOutcome, never None
    """

    def __init__(self) -> None:
        self._process: multiprocessing.Process | None = None
        self._parent_sock: socket.socket | None = None
        self._child_sock: socket.socket | None = None
        self._pid: int | None = None
        self._pgid: int | None = None
        self._session_id: str = ""
        self._mp_config: MpConfig = MpConfig()

        # Latest heartbeat data (thread-safe via asyncio — single writer)
        self._latest_heartbeat: dict | None = None
        self._last_heartbeat_time: float = 0.0

        # Result storage
        self._result_future: asyncio.Future[RunOutcome] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._done = False

    # ── Spawn ────────────────────────────────────────────────────────

    async def spawn(
        self,
        task: str,
        session_id: str,
        config_path: str,
        mp_config: MpConfig | None = None,
    ) -> MpRunHandle:
        """Spawn the child process and start monitoring.

        Args:
            task: The task string for the child agent.
            session_id: Unique session ID for this mpAgent run.
            config_path: Path to the March config YAML.
            mp_config: mpAgent configuration (timeouts, intervals, log dir).

        Returns:
            An ``MpRunHandle`` for interacting with the child.
        """
        self._mp_config = mp_config or MpConfig()
        self._session_id = session_id

        # Determine log directory
        log_dir = self._mp_config.log_dir
        if not log_dir:
            from pathlib import Path
            log_dir = str(Path.home() / ".march" / "logs" / session_id)

        # Create IPC socketpair
        self._parent_sock, self._child_sock = create_socket_pair()

        # Set parent socket to non-blocking for asyncio
        self._parent_sock.setblocking(False)

        # Spawn child process
        # We pass the child socket's fd — the child will reconstruct the socket
        child_fd = self._child_sock.fileno()

        from march.agents.mp_child import mp_child_main

        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=mp_child_main,
            args=(
                child_fd,
                config_path,
                task,
                session_id,
                log_dir,
                self._mp_config.heartbeat_interval_seconds,
            ),
            name=f"mpagent-{session_id[:12]}",
            daemon=False,  # Not daemon — we manage lifecycle explicitly
        )
        self._process.start()
        self._pid = self._process.pid

        # Close the child end in the parent process
        self._child_sock.close()
        self._child_sock = None

        # Try to get the child's process group (it calls setpgrp, so pgid == pid)
        self._pgid = self._pid

        logger.info(
            "mpAgent spawned: pid=%d session=%s task=%s",
            self._pid,
            session_id,
            task[:80],
        )

        # Initialize timing
        self._last_heartbeat_time = time.monotonic()

        # Create result future
        loop = asyncio.get_running_loop()
        self._result_future = loop.create_future()

        # Start background tasks
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name=f"mp-recv-{session_id[:12]}"
        )
        self._monitor_task = asyncio.create_task(
            self._monitor(), name=f"mp-monitor-{session_id[:12]}"
        )

        return MpRunHandle(pid=self._pid, session_id=session_id, runner=self)

    # ── IPC receive loop ─────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        """Continuously receive messages from the child via IPC.

        Dispatches heartbeats, progress, logs, and the final result.
        """
        assert self._parent_sock is not None

        try:
            while not self._done:
                try:
                    msg = await recv_message(self._parent_sock)
                except (ConnectionError, OSError) as exc:
                    if self._done:
                        return
                    logger.warning(
                        "IPC recv error for %s: %s", self._session_id, exc
                    )
                    # IPC broken — the child is gone or dying
                    self._resolve_result(RunOutcome(
                        status="error",
                        error=f"IPC connection lost: {exc}",
                    ))
                    return
                except ValueError as exc:
                    logger.error("IPC message error for %s: %s", self._session_id, exc)
                    continue

                msg_type = msg.get("type")

                if msg_type == MSG_HEARTBEAT:
                    self._latest_heartbeat = msg.get("data", {})
                    self._last_heartbeat_time = time.monotonic()
                    logger.debug(
                        "heartbeat from %s: %s",
                        self._session_id,
                        self._latest_heartbeat.get("summary", ""),
                    )

                elif msg_type == MSG_PROGRESS:
                    logger.debug(
                        "progress from %s: tool=%s status=%s",
                        self._session_id,
                        msg.get("tool_name"),
                        msg.get("status"),
                    )

                elif msg_type == MSG_LOG:
                    level = msg.get("level", "info").upper()
                    log_msg = msg.get("message", "")
                    logger.log(
                        getattr(logger, "level", lambda: 20)
                        if not hasattr(logger, level.lower())
                        else 20,
                        "child log [%s] %s: %s",
                        self._session_id,
                        level,
                        log_msg,
                    )

                elif msg_type == MSG_RESULT:
                    status = msg.get("status", "error")
                    outcome = RunOutcome(
                        status=status,
                        output=msg.get("output", ""),
                        error=msg.get("error", ""),
                    )
                    # Attach token/cost info to the output if available
                    tokens = msg.get("tokens", 0)
                    cost = msg.get("cost", 0.0)
                    if tokens or cost:
                        outcome.output = (
                            outcome.output
                            or f"tokens={tokens} cost={cost:.4f}"
                        )
                    self._resolve_result(outcome)
                    return

                else:
                    logger.warning(
                        "Unknown IPC message type from %s: %s",
                        self._session_id,
                        msg_type,
                    )

        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                "Unexpected error in recv loop for %s: %s",
                self._session_id,
                exc,
                exc_info=True,
            )
            self._resolve_result(RunOutcome(
                status="error",
                error=f"Internal recv loop error: {exc}",
            ))

    # ── Heartbeat monitor ────────────────────────────────────────────

    async def _monitor(self) -> None:
        """Monitor heartbeats and detect timeouts.

        If no heartbeat arrives within ``heartbeat_timeout_seconds``, sends
        SIGTERM to the process group. After ``kill_grace_seconds``, sends SIGKILL.
        """
        timeout = self._mp_config.heartbeat_timeout_seconds
        check_interval = min(timeout / 4, 15.0)  # Check at least every 15s

        try:
            while not self._done:
                await asyncio.sleep(check_interval)

                if self._done:
                    return

                # Check if process is still alive
                if self._process and not self._process.is_alive():
                    # Process exited — let _handle_process_exit deal with it
                    await self._handle_process_exit()
                    return

                # Check heartbeat timeout
                elapsed = time.monotonic() - self._last_heartbeat_time
                if elapsed > timeout:
                    logger.warning(
                        "Heartbeat timeout for %s: %.1fs > %.1fs",
                        self._session_id,
                        elapsed,
                        timeout,
                    )
                    # SIGTERM the process group
                    await self._kill_process_group(graceful=True)
                    self._resolve_result(RunOutcome(
                        status="timeout",
                        error=(
                            f"No heartbeat for {elapsed:.0f}s "
                            f"(timeout: {timeout:.0f}s)"
                        ),
                    ))
                    return

        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                "Monitor error for %s: %s", self._session_id, exc, exc_info=True
            )

    async def _handle_process_exit(self) -> None:
        """Handle the child process having exited (detected by monitor)."""
        if self._done:
            return

        exitcode = self._process.exitcode if self._process else None
        logger.info(
            "mpAgent process exited: pid=%s exitcode=%s session=%s",
            self._pid,
            exitcode,
            self._session_id,
        )

        if exitcode == 0:
            # Normal exit — result should have been sent via IPC already.
            # If we get here without a result, it means IPC was lost.
            # Give the recv loop a moment to process any final message.
            await asyncio.sleep(0.5)
            if not self._done:
                self._resolve_result(RunOutcome(
                    status="error",
                    error="Child exited normally but no result received via IPC",
                ))
        elif exitcode == -9:  # SIGKILL — likely OOM
            self._resolve_result(RunOutcome(
                status="error",
                error="Child process killed by SIGKILL (likely OOM)",
            ))
        elif exitcode is not None and exitcode < 0:
            sig_num = -exitcode
            sig_name = _signal_name(sig_num)
            self._resolve_result(RunOutcome(
                status="error",
                error=f"Child process killed by signal {sig_name} ({sig_num})",
            ))
        else:
            self._resolve_result(RunOutcome(
                status="error",
                error=f"Child process exited with code {exitcode}",
            ))

    # ── Public API ───────────────────────────────────────────────────

    async def send_steer(self, message: str) -> bool:
        """Send a steering message to the child process.

        Args:
            message: The steering text to inject.

        Returns:
            True if sent successfully, False if IPC is unavailable.
        """
        if not self._parent_sock or self._done:
            return False

        try:
            await send_message(self._parent_sock, {
                "type": MSG_STEER,
                "message": message,
            })
            logger.info("Steer sent to %s: %s", self._session_id, message[:80])
            return True
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            logger.warning("Failed to send steer to %s: %s", self._session_id, exc)
            return False

    async def kill(self) -> None:
        """Immediately kill the child process group.

        Sends SIGKILL to the entire process group (``os.killpg``).
        """
        logger.info("Kill requested for %s (pid=%s)", self._session_id, self._pid)

        # Try graceful kill message first
        if self._parent_sock and not self._done:
            try:
                await send_message(self._parent_sock, {"type": MSG_KILL})
            except (BrokenPipeError, ConnectionError, OSError):
                pass

        await self._kill_process_group(graceful=False)
        self._resolve_result(RunOutcome(
            status="cancelled",
            error="Killed by parent",
        ))

    async def wait_result(self) -> RunOutcome:
        """Wait for the child to complete and return the outcome.

        **Guarantee:** This method always returns a ``RunOutcome`` and never
        hangs indefinitely. All failure modes (crash, OOM, timeout, IPC loss)
        are handled and produce an appropriate ``RunOutcome``.

        Returns:
            ``RunOutcome`` with status and details.
        """
        if self._result_future is None:
            return RunOutcome(status="error", error="Runner was never spawned")

        try:
            # Add a generous safety timeout so we never truly hang.
            # The monitor task handles the real timeout logic.
            safety_timeout = (
                self._mp_config.heartbeat_timeout_seconds
                + self._mp_config.kill_grace_seconds
                + 60  # Extra buffer
            )
            outcome = await asyncio.wait_for(
                asyncio.shield(self._result_future),
                timeout=safety_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Safety timeout reached for %s — force killing",
                self._session_id,
            )
            await self._kill_process_group(graceful=False)
            outcome = RunOutcome(
                status="timeout",
                error="Safety timeout reached — process force-killed",
            )
        except asyncio.CancelledError:
            await self._kill_process_group(graceful=False)
            outcome = RunOutcome(
                status="cancelled",
                error="wait_result was cancelled",
            )
        finally:
            await self._cleanup()

        # Compute duration
        if self._latest_heartbeat:
            elapsed = self._latest_heartbeat.get("elapsed_seconds", 0)
            outcome.duration_ms = elapsed * 1000

        return outcome

    def get_latest_heartbeat(self) -> dict | None:
        """Return the most recent heartbeat data from the child.

        Returns:
            Heartbeat data dict, or ``None`` if no heartbeat received yet.
        """
        return self._latest_heartbeat

    # ── Internal helpers ─────────────────────────────────────────────

    def _resolve_result(self, outcome: RunOutcome) -> None:
        """Set the result future if not already resolved (idempotent)."""
        if self._done:
            return
        self._done = True

        if self._result_future and not self._result_future.done():
            self._result_future.set_result(outcome)

        logger.info(
            "mpAgent result for %s: status=%s error=%s",
            self._session_id,
            outcome.status,
            outcome.error[:100] if outcome.error else "",
        )

    async def _kill_process_group(self, graceful: bool = True) -> None:
        """Kill the child's process group.

        Args:
            graceful: If True, send SIGTERM first and wait ``kill_grace_seconds``
                before SIGKILL. If False, send SIGKILL immediately.
        """
        pgid = self._pgid
        if pgid is None:
            return

        if graceful:
            try:
                os.killpg(pgid, signal.SIGTERM)
                logger.info("SIGTERM sent to pgid=%d", pgid)
            except (ProcessLookupError, PermissionError):
                return  # Already gone

            # Wait for grace period
            grace = self._mp_config.kill_grace_seconds
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if self._process and not self._process.is_alive():
                    return  # Exited during grace period
                await asyncio.sleep(0.5)

        # SIGKILL
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.info("SIGKILL sent to pgid=%d", pgid)
        except (ProcessLookupError, PermissionError):
            pass  # Already gone

        # Wait for process to be reaped
        if self._process:
            self._process.join(timeout=5.0)

    async def _cleanup(self) -> None:
        """Clean up resources after the child has finished."""
        # Cancel background tasks
        for task in (self._recv_task, self._monitor_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close parent socket
        if self._parent_sock:
            try:
                self._parent_sock.close()
            except OSError:
                pass
            self._parent_sock = None

        # Join process
        if self._process and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                # Last resort
                try:
                    self._process.kill()
                except OSError:
                    pass

        logger.info("mpAgent cleanup complete for %s", self._session_id)


# ── Utilities ────────────────────────────────────────────────────────


def _signal_name(sig_num: int) -> str:
    """Get a human-readable signal name from a signal number."""
    try:
        return signal.Signals(sig_num).name
    except (ValueError, AttributeError):
        return f"SIG{sig_num}"
