"""Anthropic Messages API provider via the official anthropic Python SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from march.llm.base import (
    AuthenticationError,
    ContextLengthError,
    DeltaToolCall,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderError,
    RateLimitError,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    _Timer,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


def _backoff_delay(attempt: int, base: float = _BASE_DELAY, cap: float = _MAX_DELAY) -> float:
    import random
    delay = min(base * (2 ** attempt), cap)
    return delay * (0.5 + random.random() * 0.5)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider (direct, not via Bedrock).

    Supports Claude models with tool use and streaming via the official SDK.
    """

    name: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
        input_price: float = 3.0,
        output_price: float = 15.0,
    ) -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install march[anthropic]"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.input_price = input_price
        self.output_price = output_price

        kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": 0,  # We handle retries ourselves
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.AsyncAnthropic(**kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnthropicProvider":
        return cls(
            model=config.get("model", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url") or config.get("url"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120.0),
            input_price=config.get("cost", {}).get("input", 3.0),
            output_price=config.get("cost", {}).get("output", 15.0),
        )

    def _format_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert generic messages to Anthropic format.

        Anthropic requires alternating user/assistant roles.
        System messages are handled separately via the system= parameter.
        Tool results are sent as user messages with tool_result content blocks.
        """
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                continue

            if role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    args = tc.get("arguments", tc.get("args", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc.get("name", ""),
                        "input": args,
                    })
                if content_blocks:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })

            elif role == "user":
                if isinstance(content, list):
                    # Tool results
                    content_blocks_user: list[dict[str, Any]] = []
                    for item in content:
                        if hasattr(item, "id"):
                            error_val = getattr(item, "error", None)
                            content_val = getattr(item, "content", None)
                            block: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": item.id,
                            }
                            if error_val:
                                block["content"] = str(error_val)
                                block["is_error"] = True
                            else:
                                block["content"] = str(content_val or "")
                            content_blocks_user.append(block)
                        elif isinstance(item, dict) and "id" in item:
                            block_d: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": item["id"],
                            }
                            if item.get("error"):
                                block_d["content"] = str(item["error"])
                                block_d["is_error"] = True
                            else:
                                block_d["content"] = str(item.get("content", ""))
                            content_blocks_user.append(block_d)
                        elif isinstance(item, dict) and item.get("type") == "image":
                            # Image content block for vision
                            content_blocks_user.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            content_blocks_user.append({
                                "type": "text",
                                "text": item["text"],
                            })
                        else:
                            content_blocks_user.append({
                                "type": "text",
                                "text": str(item),
                            })
                    if content_blocks_user:
                        anthropic_messages.append({
                            "role": "user",
                            "content": content_blocks_user,
                        })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": str(content),
                    })
            else:
                anthropic_messages.append({
                    "role": "user",
                    "content": str(content),
                })

        return anthropic_messages

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        result = []
        for t in tools:
            if isinstance(t, dict):
                result.append(t)
            else:
                result.append(t.to_anthropic_schema())
        return result

    def _handle_error(self, error: anthropic.APIError) -> None:
        """Convert anthropic errors to typed ProviderError."""
        message = str(error)
        status = getattr(error, "status_code", None)

        if isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError(
                f"Anthropic auth: {message}", provider=self.name
            )
        elif isinstance(error, anthropic.RateLimitError):
            retry_after = None
            headers = getattr(error, "headers", None) or {}
            if hasattr(headers, "get"):
                ra = headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except (ValueError, TypeError):
                        pass
            raise RateLimitError(
                f"Anthropic rate limit: {message}",
                provider=self.name,
                retry_after=retry_after,
            )
        elif status == 400 and "context" in message.lower():
            raise ContextLengthError(
                f"Anthropic context length: {message}", provider=self.name
            )
        else:
            retryable = isinstance(
                error,
                (anthropic.APIConnectionError, anthropic.InternalServerError),
            ) or (status is not None and status >= 500)
            raise ProviderError(
                f"Anthropic error: {message}",
                provider=self.name,
                retryable=retryable,
            )

    def _parse_response(
        self, response: Any, duration_ms: float
    ) -> LLMResponse:
        """Parse Anthropic Message response into LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=block.input if isinstance(block.input, dict) else {},
                ))

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        cache_read = 0
        cache_write = 0
        if hasattr(response.usage, "cache_read_input_tokens"):
            cache_read = response.usage.cache_read_input_tokens or 0
        if hasattr(response.usage, "cache_creation_input_tokens"):
            cache_write = response.usage.cache_creation_input_tokens or 0

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=self.estimate_cost(input_tokens, output_tokens),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            duration_ms=duration_ms,
            stop_reason=response.stop_reason or "",
            model=self.model,
            provider=self.name,
        )

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        anthropic_messages = self._format_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with _Timer() as timer:
                    response = await self._client.messages.create(**kwargs)
                return self._parse_response(response, timer.elapsed_ms)
            except anthropic.APIError as e:
                try:
                    self._handle_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "Anthropic retryable error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"Anthropic unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "Anthropic: max retries exceeded", provider=self.name
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        anthropic_messages = self._format_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        stream_ctx = None
        for attempt in range(_MAX_RETRIES):
            try:
                stream_ctx = self._client.messages.stream(**kwargs)
                break
            except anthropic.APIError as e:
                try:
                    self._handle_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "Anthropic stream retryable error (attempt %d/%d), "
                            "retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"Anthropic stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        if stream_ctx is None:
            raise last_error or ProviderError(
                "Anthropic stream: max retries exceeded", provider=self.name
            )

        current_tool_call: DeltaToolCall | None = None
        tool_index = 0

        async with stream_ctx as stream:
            async for event in stream:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_call = DeltaToolCall(
                            index=tool_index,
                            id=block.id,
                            name=block.name,
                            arguments_json="",
                        )
                        tool_index += 1
                        yield StreamChunk(delta_tool_call=current_tool_call)

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamChunk(delta_text=delta.text)
                    elif delta.type == "input_json_delta" and current_tool_call is not None:
                        json_frag = delta.partial_json
                        current_tool_call.arguments_json += json_frag
                        yield StreamChunk(
                            delta_tool_call=DeltaToolCall(
                                index=current_tool_call.index,
                                id=current_tool_call.id,
                                name=current_tool_call.name,
                                arguments_json=json_frag,
                            )
                        )

                elif event_type == "content_block_stop":
                    current_tool_call = None

                elif event_type == "message_delta":
                    stop_reason = getattr(event.delta, "stop_reason", "") or ""
                    output_tokens = 0
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = getattr(event.usage, "output_tokens", 0)
                    yield StreamChunk(
                        is_final=True,
                        stop_reason=stop_reason,
                        usage=LLMUsage(
                            output_tokens=output_tokens,
                            cost=self.estimate_cost(0, output_tokens),
                        ) if output_tokens else None,
                    )

                elif event_type == "message_start":
                    msg = event.message
                    if msg and msg.usage:
                        input_tokens = msg.usage.input_tokens or 0
                        yield StreamChunk(
                            usage=LLMUsage(input_tokens=input_tokens),
                        )

    async def health_check(self) -> bool:
        try:
            await self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            logger.debug("Anthropic health check failed", exc_info=True)
            return False
