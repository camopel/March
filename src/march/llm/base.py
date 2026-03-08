"""Abstract LLM provider interface and shared data types.

This module defines the contracts that all LLM providers must implement,
plus shared data types used across the LLM layer and consumed by the
core agent loop.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# ─── Tool Types ──────────────────────────────────────────────────────────────


@dataclass
class ToolParameter:
    """A single parameter in a tool definition."""

    name: str
    type: str  # JSON Schema type: string, integer, number, boolean, array, object
    description: str = ""
    required: bool = False
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # For array types
    properties: dict[str, Any] | None = None  # For object types
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema for LLM tool_use format (OpenAI-compatible)."""
        return self.to_openai_schema()

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.items is not None:
                prop["items"] = param.items
            if param.properties is not None:
                prop["properties"] = param.properties
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            schema["function"]["parameters"]["required"] = required
        return schema

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool-use format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.items is not None:
                prop["items"] = param.items
            if param.properties is not None:
                prop["properties"] = param.properties
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            input_schema["required"] = required

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema,
        }

    def to_bedrock_schema(self) -> dict[str, Any]:
        """Convert to AWS Bedrock Converse tool format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.items is not None:
                prop["items"] = param.items
            if param.properties is not None:
                prop["properties"] = param.properties
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            input_schema["required"] = required

        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": input_schema},
            }
        }

    def to_ollama_schema(self) -> dict[str, Any]:
        """Convert to Ollama tool format (OpenAI-compatible)."""
        return self.to_openai_schema()


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def arguments(self) -> dict[str, Any]:
        """Alias for args — some code uses 'arguments'."""
        return self.args


# ─── Usage & Response Types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMUsage:
    """Token usage and cost information for an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    duration_ms: float = 0.0
    stop_reason: str = ""
    model: str = ""
    provider: str = ""

    @property
    def finish_reason(self) -> str:
        """Alias for stop_reason — some code uses 'finish_reason'."""
        return self.stop_reason

    def to_message(self) -> dict[str, Any]:
        """Convert to a message dict suitable for adding back to conversation."""
        msg: dict[str, Any] = {"role": "assistant"}
        if self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.args,
                }
                for tc in self.tool_calls
            ]
        return msg


# ─── Streaming Types ────────────────────────────────────────────────────────


@dataclass
class DeltaToolCall:
    """Partial tool call accumulator for streaming."""

    index: int = 0
    id: str = ""
    name: str = ""
    arguments_json: str = ""


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Uses 'delta' for text and 'tool_call_delta' for tool call fragments,
    matching the agent loop's expected interface.
    """

    delta: str = ""
    delta_text: str = ""  # Alias — providers can use either
    tool_call_delta: dict[str, Any] | None = None
    delta_tool_call: DeltaToolCall | None = None  # Structured form used by providers
    tool_progress: dict[str, Any] | None = None  # Tool execution progress info
    usage: LLMUsage | None = None
    is_final: bool = False
    finish_reason: str | None = None
    stop_reason: str = ""

    def __post_init__(self) -> None:
        # Sync delta/delta_text — whichever is set
        if self.delta_text and not self.delta:
            object.__setattr__(self, "delta", self.delta_text)
        elif self.delta and not self.delta_text:
            object.__setattr__(self, "delta_text", self.delta)
        # Sync finish_reason/stop_reason
        if self.finish_reason and not self.stop_reason:
            object.__setattr__(self, "stop_reason", self.finish_reason)
        elif self.stop_reason and not self.finish_reason:
            object.__setattr__(self, "finish_reason", self.stop_reason)


# ─── Exceptions ──────────────────────────────────────────────────────────────


class ProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class RateLimitError(ProviderError):
    """Rate limit hit (HTTP 429). Always retryable."""

    def __init__(self, message: str, provider: str = "", retry_after: float | None = None):
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Authentication failed. Retryable only for expired tokens (after credential refresh)."""

    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message, provider=provider, retryable=retryable)


class ContextLengthError(ProviderError):
    """Input too long for model context window. Never retryable."""

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider=provider, retryable=False)


# ─── Utilities ───────────────────────────────────────────────────────────────


class _Timer:
    """Simple context manager for timing operations in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000


# ─── Provider Base Class ────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Every provider must normalize its responses into the common LLMResponse/StreamChunk
    format. Providers handle their own retry logic for transient errors.
    """

    name: str = ""
    model: str = ""
    input_price: float = 0.0   # USD per million input tokens
    output_price: float = 0.0  # USD per million output tokens
    timeout: float = 120.0     # seconds

    @abstractmethod
    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages and get a complete response.

        Args:
            messages: Conversation messages in [{role, content}] format.
            system: Optional system prompt.
            tools: Optional tool definitions (ToolDefinition objects or raw dicts).
            temperature: Optional temperature override.
            max_tokens: Optional max output tokens override.

        Returns:
            Normalized LLMResponse.
        """
        ...

    @abstractmethod
    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming version of converse. Yields text deltas and tool call fragments.

        Args:
            messages: Conversation messages.
            system: Optional system prompt.
            tools: Optional tool definitions.
            temperature: Optional temperature override.
            max_tokens: Optional max output tokens override.

        Yields:
            StreamChunk objects. The last chunk has is_final=True with usage data.
        """
        ...
        # Make this an async generator
        yield StreamChunk()  # type: ignore[misc]

    async def health_check(self) -> bool:
        """Check if this provider is currently available.

        Default implementation returns True. Providers can override to ping
        their endpoint, verify credentials, etc.
        """
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a given token count.

        Uses the provider's configured per-million-token prices.
        """
        return (
            input_tokens * self.input_price + output_tokens * self.output_price
        ) / 1_000_000

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMProvider":
        """Create a provider instance from a config dict.

        Subclasses should override this if they need custom initialization logic.
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_config()")
