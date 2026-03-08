"""IPC protocol for mpAgent communication.

Provides a full-duplex Unix socketpair-based IPC layer between the parent
(MpRunner) and child (mp_child_main) processes. Messages are serialized with
msgpack (with JSON fallback) and framed with a 4-byte big-endian length prefix.

Message types:
  Parent → Child: steer, kill
  Child → Parent: heartbeat, progress, result, log
"""

from __future__ import annotations

import asyncio
import socket
import struct
from dataclasses import dataclass, field
from typing import Any, TypedDict

from march.logging import get_logger

logger = get_logger("march.ipc", subsystem="agent")

# ── Serialization backend ────────────────────────────────────────────

try:
    import msgpack

    def _pack(obj: dict) -> bytes:
        return msgpack.packb(obj, use_bin_type=True)

    def _unpack(data: bytes) -> dict:
        return msgpack.unpackb(data, raw=False)

except ImportError:
    import json as _json

    logger.warning("msgpack not installed, falling back to JSON for IPC")

    def _pack(obj: dict) -> bytes:  # type: ignore[misc]
        return _json.dumps(obj).encode("utf-8")

    def _unpack(data: bytes) -> dict:  # type: ignore[misc]
        return _json.loads(data.decode("utf-8"))


# ── Message type constants ───────────────────────────────────────────

# Parent → Child
MSG_STEER = "steer"
MSG_KILL = "kill"

# Child → Parent
MSG_HEARTBEAT = "heartbeat"
MSG_PROGRESS = "progress"
MSG_RESULT = "result"
MSG_LOG = "log"

# Length-prefix format: 4-byte unsigned big-endian
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MB safety limit


# ── Message schemas (TypedDict) ─────────────────────────────────────


class SteerMessage(TypedDict):
    """Parent → Child: inject a steering message."""

    type: str  # MSG_STEER
    message: str


class KillMessage(TypedDict):
    """Parent → Child: request graceful shutdown."""

    type: str  # MSG_KILL


class HeartbeatData(TypedDict, total=False):
    """Payload for heartbeat messages."""

    memory_rss_mb: float
    elapsed_seconds: float
    tokens_used: int
    total_cost: float
    tool_calls_made: int
    llm_calls_made: int
    summary: str
    current_tool: str
    current_tool_detail: str
    recent_tools: list[dict[str, Any]]


class HeartbeatMessage(TypedDict):
    """Child → Parent: periodic heartbeat with progress info."""

    type: str  # MSG_HEARTBEAT
    ts: float
    data: HeartbeatData


class ProgressMessage(TypedDict):
    """Child → Parent: tool execution progress."""

    type: str  # MSG_PROGRESS
    tool_name: str
    status: str
    summary: str
    duration_ms: float


class ResultMessage(TypedDict):
    """Child → Parent: final run result."""

    type: str  # MSG_RESULT
    status: str  # "ok" or "error"
    output: str
    error: str
    tokens: int
    cost: float


class LogMessage(TypedDict):
    """Child → Parent: log entry."""

    type: str  # MSG_LOG
    level: str
    message: str


# ── Socket pair creation ─────────────────────────────────────────────


def create_socket_pair() -> tuple[socket.socket, socket.socket]:
    """Create a Unix socketpair for IPC between parent and child processes.

    Returns:
        (parent_sock, child_sock) — each end is a connected AF_UNIX SOCK_STREAM socket.
        Both sockets are set to inheritable so they survive across ``multiprocessing.Process``
        with start_method="spawn".
    """
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    # Mark inheritable for spawn-based multiprocessing
    parent_sock.set_inheritable(True)
    child_sock.set_inheritable(True)
    return parent_sock, child_sock


# ── Async send / recv (parent process, asyncio event loop) ──────────


async def send_message(sock: socket.socket, msg: dict) -> None:
    """Send a length-prefixed msgpack message asynchronously.

    Uses the running asyncio event loop's ``sock_sendall``.

    Args:
        sock: A connected socket (non-blocking is handled by asyncio).
        msg: Dictionary to serialize and send.

    Raises:
        BrokenPipeError: If the remote end has closed.
        OSError: On other socket errors.
    """
    loop = asyncio.get_running_loop()
    payload = _pack(msg)
    header = struct.pack(_HEADER_FMT, len(payload))
    await loop.sock_sendall(sock, header + payload)


async def recv_message(sock: socket.socket) -> dict:
    """Receive a length-prefixed msgpack message asynchronously.

    Blocks (async) until a complete message is available.

    Args:
        sock: A connected socket.

    Returns:
        The deserialized message dictionary.

    Raises:
        ConnectionError: If the remote end has closed (EOF on header read).
        ValueError: If the message exceeds the safety size limit.
    """
    loop = asyncio.get_running_loop()

    # Read the 4-byte length header
    header = await _recv_exact_async(loop, sock, _HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        raise ConnectionError("IPC connection closed (incomplete header)")

    (length,) = struct.unpack(_HEADER_FMT, header)
    if length > _MAX_MESSAGE_SIZE:
        raise ValueError(
            f"IPC message too large: {length} bytes (max {_MAX_MESSAGE_SIZE})"
        )

    # Read the payload
    payload = await _recv_exact_async(loop, sock, length)
    if len(payload) < length:
        raise ConnectionError("IPC connection closed (incomplete payload)")

    return _unpack(payload)


async def _recv_exact_async(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    n: int,
) -> bytes:
    """Read exactly *n* bytes from *sock* using asyncio, or return fewer on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = await loop.sock_recv(sock, n - len(buf))
        if not chunk:
            break  # EOF
        buf.extend(chunk)
    return bytes(buf)


# ── Synchronous send / recv (child process, heartbeat thread) ───────


def send_message_sync(sock: socket.socket, msg: dict) -> None:
    """Send a length-prefixed msgpack message synchronously.

    Intended for use in the child process heartbeat thread (no asyncio).

    Args:
        sock: A connected socket (blocking mode).
        msg: Dictionary to serialize and send.

    Raises:
        BrokenPipeError: If the remote end has closed (orphan detection).
        OSError: On other socket errors.
    """
    payload = _pack(msg)
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def recv_message_sync(sock: socket.socket, timeout: float | None = None) -> dict | None:
    """Receive a length-prefixed msgpack message synchronously.

    Args:
        sock: A connected socket (blocking mode).
        timeout: Socket timeout in seconds. ``None`` means block forever.
            Returns ``None`` on timeout.

    Returns:
        The deserialized message dictionary, or ``None`` on timeout.

    Raises:
        ConnectionError: If the remote end has closed.
        ValueError: If the message exceeds the safety size limit.
    """
    old_timeout = sock.gettimeout()
    try:
        sock.settimeout(timeout)

        # Read header
        header = _recv_exact_sync(sock, _HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ConnectionError("IPC connection closed (incomplete header)")

        (length,) = struct.unpack(_HEADER_FMT, header)
        if length > _MAX_MESSAGE_SIZE:
            raise ValueError(
                f"IPC message too large: {length} bytes (max {_MAX_MESSAGE_SIZE})"
            )

        # Read payload
        payload = _recv_exact_sync(sock, length)
        if len(payload) < length:
            raise ConnectionError("IPC connection closed (incomplete payload)")

        return _unpack(payload)

    except socket.timeout:
        return None
    finally:
        sock.settimeout(old_timeout)


def _recv_exact_sync(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock* synchronously, or return fewer on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break  # EOF
        buf.extend(chunk)
    return bytes(buf)
