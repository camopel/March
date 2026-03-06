"""Parent ↔ child IPC protocol for March sub-agents.

JSON lines over stdin/stdout. Each line is a complete JSON message with a "type" field.

Parent → Child message types:
  - task: Initial task assignment
  - steer: Inject guidance/message mid-run
  - cancel: Request graceful cancellation

Child → Parent message types:
  - progress: Status update during execution
  - tool_use: Notification of tool execution
  - result: Final result (success)
  - error: Final result (failure)
  - request: Child requests something from parent (e.g., resource access)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class MessageType(str, Enum):
    """IPC message types."""

    # Parent → Child
    TASK = "task"
    STEER = "steer"
    CANCEL = "cancel"

    # Child → Parent
    PROGRESS = "progress"
    TOOL_USE = "tool_use"
    RESULT = "result"
    ERROR = "error"
    REQUEST = "request"


@dataclass
class IPCMessage:
    """A single IPC message between parent and child agents.

    Attributes:
        type: The message type (task, steer, cancel, progress, etc.).
        payload: The message data (type-specific).
        id: Optional message ID for request/response correlation.
        timestamp: Unix timestamp (set automatically if not provided).
    """

    type: MessageType | str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            try:
                self.type = MessageType(self.type)
            except ValueError:
                pass  # Allow unknown types for forward compatibility
        if not self.timestamp:
            import time
            self.timestamp = time.time()

    def to_json(self) -> str:
        """Serialize to a JSON line (no newlines in output)."""
        data = {
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
            "payload": self.payload,
        }
        if self.id:
            data["id"] = self.id
        data["timestamp"] = self.timestamp
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "IPCMessage":
        """Deserialize from a JSON line."""
        data = json.loads(line.strip())
        return cls(
            type=data["type"],
            payload=data.get("payload", {}),
            id=data.get("id", ""),
            timestamp=data.get("timestamp", 0.0),
        )

    # ── Factory Methods ──────────────────────────────────────────────────

    @classmethod
    def task(
        cls,
        task: str,
        model: str = "",
        tools: list[str] | None = None,
        timeout: int = 0,
        context: dict[str, Any] | None = None,
    ) -> "IPCMessage":
        """Create a task assignment message (parent → child)."""
        payload: dict[str, Any] = {"task": task}
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
        if timeout:
            payload["timeout"] = timeout
        if context:
            payload["context"] = context
        return cls(type=MessageType.TASK, payload=payload)

    @classmethod
    def steer(cls, message: str, priority: str = "normal") -> "IPCMessage":
        """Create a steering message (parent → child)."""
        return cls(
            type=MessageType.STEER,
            payload={"message": message, "priority": priority},
        )

    @classmethod
    def cancel(cls, reason: str = "") -> "IPCMessage":
        """Create a cancellation request (parent → child)."""
        return cls(type=MessageType.CANCEL, payload={"reason": reason})

    @classmethod
    def progress(cls, status: str, detail: str = "", percent: float = -1) -> "IPCMessage":
        """Create a progress update (child → parent)."""
        payload: dict[str, Any] = {"status": status}
        if detail:
            payload["detail"] = detail
        if percent >= 0:
            payload["percent"] = percent
        return cls(type=MessageType.PROGRESS, payload=payload)

    @classmethod
    def tool_use(cls, tool_name: str, args: dict[str, Any] | None = None, result_summary: str = "") -> "IPCMessage":
        """Create a tool use notification (child → parent)."""
        payload: dict[str, Any] = {"tool": tool_name}
        if args:
            payload["args"] = args
        if result_summary:
            payload["result_summary"] = result_summary
        return cls(type=MessageType.TOOL_USE, payload=payload)

    @classmethod
    def result(cls, content: str, metadata: dict[str, Any] | None = None) -> "IPCMessage":
        """Create a final result message (child → parent)."""
        payload: dict[str, Any] = {"content": content}
        if metadata:
            payload["metadata"] = metadata
        return cls(type=MessageType.RESULT, payload=payload)

    @classmethod
    def error(cls, error: str, traceback: str = "") -> "IPCMessage":
        """Create an error message (child → parent)."""
        payload: dict[str, Any] = {"error": error}
        if traceback:
            payload["traceback"] = traceback
        return cls(type=MessageType.ERROR, payload=payload)

    @classmethod
    def request(cls, action: str, params: dict[str, Any] | None = None, msg_id: str = "") -> "IPCMessage":
        """Create a request from child to parent."""
        payload: dict[str, Any] = {"action": action}
        if params:
            payload["params"] = params
        return cls(type=MessageType.REQUEST, payload=payload, id=msg_id)


class IPCWriter:
    """Write IPC messages to an asyncio StreamWriter (or stdout)."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def send(self, message: IPCMessage) -> None:
        """Send a message as a JSON line."""
        line = message.to_json() + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

    def close(self) -> None:
        """Close the writer."""
        self._writer.close()


class IPCReader:
    """Read IPC messages from an asyncio StreamReader (or stdin)."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def receive(self) -> IPCMessage | None:
        """Read the next message. Returns None on EOF."""
        try:
            line = await self._reader.readline()
            if not line:
                return None
            return IPCMessage.from_json(line.decode("utf-8"))
        except (json.JSONDecodeError, KeyError, ConnectionError):
            return None

    async def stream(self) -> AsyncIterator[IPCMessage]:
        """Stream messages until EOF."""
        while True:
            msg = await self.receive()
            if msg is None:
                break
            yield msg
