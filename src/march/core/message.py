"""Message types for the March agent framework.

Defines the core data structures for messages, tool calls, and tool results
used throughout the agent loop and serialized for LLM APIs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        args: Arguments to pass to the tool.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, args: dict[str, Any] | None = None) -> "ToolCall":
        """Create a ToolCall with an auto-generated ID."""
        return cls(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            args=args or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for LLM APIs."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.args,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Deserialize from dict (supports both flat and nested formats)."""
        if "function" in data:
            func = data["function"]
            return cls(
                id=data.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                name=func["name"],
                args=func.get("arguments", {}),
            )
        return cls(
            id=data.get("id", f"call_{uuid.uuid4().hex[:12]}"),
            name=data["name"],
            args=data.get("args", {}),
        )


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        id: The tool_call ID this result corresponds to.
        content: The result content (string output).
        error: Error message if execution failed.
        duration_ms: Execution time in milliseconds.
    """

    id: str
    content: str = ""
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def is_error(self) -> bool:
        """Whether this result represents an error."""
        return self.error is not None

    @property
    def summary(self) -> str:
        """Short summary of the result for logging."""
        if self.error:
            return f"ERROR: {self.error[:200]}"
        if len(self.content) > 200:
            return self.content[:197] + "..."
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "duration_ms": self.duration_ms,
        }
        if self.error is not None:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            content=data.get("content", ""),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class Message:
    """A single message in a conversation.

    Supports all roles: user, assistant, system, tool.
    Assistant messages can contain tool_calls. Tool messages contain tool_results.

    Attributes:
        role: The role of the message sender.
        content: Text content of the message.
        tool_calls: Tool calls requested by the assistant (assistant role only).
        tool_results: Results from tool execution (tool role only).
        name: Optional name for the message sender.
        metadata: Additional metadata (timestamps, tokens, etc.).
    """

    role: Role | str
    content: str | list = ""
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize role to Role enum."""
        if isinstance(self.role, str):
            self.role = Role(self.role)

    @property
    def has_tool_calls(self) -> bool:
        """Whether this message contains tool calls."""
        return bool(self.tool_calls)

    @property
    def has_tool_results(self) -> bool:
        """Whether this message contains tool results."""
        return bool(self.tool_results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for LLM APIs.

        Produces a format compatible with OpenAI/Anthropic message schemas.
        """
        result: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, Role) else self.role,
        }

        if self.content:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_results:
            # For LLM APIs, tool results are typically sent as individual tool messages.
            # When serializing the full message (e.g. for persistence), include all results.
            result["tool_results"] = [tr.to_dict() for tr in self.tool_results]

        if self.name:
            result["name"] = self.name

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_llm_messages(self) -> list[dict[str, Any]]:
        """Convert to one or more LLM API messages.

        An assistant message with tool calls becomes one message.
        A tool result message becomes one message per result (OpenAI format).
        """
        if self.role == Role.TOOL and self.tool_results:
            # Each tool result becomes a separate tool message
            messages: list[dict[str, Any]] = []
            for tr in self.tool_results:
                content = tr.content if not tr.is_error else f"Error: {tr.error}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.id,
                        "content": content,
                    }
                )
            return messages

        if self.role == Role.ASSISTANT and self.tool_calls:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": self.content or None,
                "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            }
            return [msg]

        return [
            {
                "role": self.role.value if isinstance(self.role, Role) else self.role,
                "content": self.content,
            }
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize from dict."""
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        tool_results = None
        if "tool_results" in data:
            tool_results = [ToolResult.from_dict(tr) for tr in data["tool_results"]]

        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_results=tool_results,
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def user(cls, content: str | list) -> "Message":
        """Create a user message. Content can be str or multimodal list."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool(cls, results: list[ToolResult]) -> "Message":
        """Create a tool results message."""
        return cls(role=Role.TOOL, tool_results=results)
