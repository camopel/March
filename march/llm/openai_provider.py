"""OpenAI LLM provider via the official openai Python SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

import openai

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


# Models that use the "reasoning" API style and don't accept temperature
_REASONING_MODELS = {"o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o4-mini"}


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is an OpenAI reasoning model (o1/o3 series)."""
    base = model.split("-")[0] if "-" in model else model
    return base in _REASONING_MODELS or model in _REASONING_MODELS


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API provider.

    Supports GPT-4o, o1, o3 series models with tool use and streaming.
    """

    name: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
        input_price: float = 2.5,
        output_price: float = 10.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.input_price = input_price
        self.output_price = output_price

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAIProvider":
        return cls(
            model=config.get("model", "gpt-4o"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url") or config.get("url"),
            organization=config.get("organization"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120.0),
            input_price=config.get("cost", {}).get("input", 2.5),
            output_price=config.get("cost", {}).get("output", 10.0),
        )

    def _format_messages(
        self, messages: list[dict[str, Any]], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert generic message format to OpenAI format."""
        oai_messages: list[dict[str, Any]] = []

        if system:
            role = "developer" if _is_reasoning_model(self.model) else "system"
            oai_messages.append({"role": role, "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                continue  # Already handled above

            if role == "assistant":
                oai_msg: dict[str, Any] = {"role": "assistant"}
                if content:
                    oai_msg["content"] = content
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tc.get("id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": (
                                    json.dumps(tc["arguments"])
                                    if isinstance(tc.get("arguments"), dict)
                                    else str(tc.get("arguments", "{}"))
                                ),
                            },
                        }
                        for tc in tool_calls
                    ]
                oai_messages.append(oai_msg)

            elif role == "user":
                if isinstance(content, list):
                    # Tool results
                    for item in content:
                        if hasattr(item, "id"):
                            error_val = getattr(item, "error", None)
                            content_val = getattr(item, "content", None)
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item.id,
                                "content": str(error_val or content_val or ""),
                            })
                        elif isinstance(item, dict) and "id" in item:
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["id"],
                                "content": str(
                                    item.get("error") or item.get("content", "")
                                ),
                            })
                        else:
                            oai_messages.append({
                                "role": "user",
                                "content": str(item),
                            })
                else:
                    oai_messages.append({"role": "user", "content": str(content)})
            else:
                oai_messages.append({"role": role, "content": str(content)})

        return oai_messages

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        result = []
        for t in tools:
            if isinstance(t, dict):
                result.append(t)
            else:
                result.append(t.to_openai_schema())
        return result

    def _handle_error(self, error: openai.APIError) -> None:
        """Convert openai errors to typed ProviderError."""
        status = getattr(error, "status_code", None)
        message = str(error)

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(f"OpenAI auth: {message}", provider=self.name)
        elif isinstance(error, openai.RateLimitError):
            retry_after = None
            headers = getattr(error, "headers", {}) or {}
            if "retry-after" in headers:
                try:
                    retry_after = float(headers["retry-after"])
                except (ValueError, TypeError):
                    pass
            raise RateLimitError(
                f"OpenAI rate limit: {message}",
                provider=self.name,
                retry_after=retry_after,
            )
        elif status == 400 and "context_length" in message.lower():
            raise ContextLengthError(
                f"OpenAI context length: {message}", provider=self.name
            )
        else:
            retryable = isinstance(
                error, (openai.APIConnectionError, openai.InternalServerError)
            ) or (status is not None and status >= 500)
            raise ProviderError(
                f"OpenAI error: {message}",
                provider=self.name,
                retryable=retryable,
            )

    def _parse_response(
        self, response: Any, duration_ms: float
    ) -> LLMResponse:
        """Parse OpenAI ChatCompletion response into LLMResponse."""
        choice = response.choices[0] if response.choices else None
        content = ""
        tool_calls: list[ToolCall] = []
        stop_reason = ""

        if choice:
            stop_reason = choice.finish_reason or ""
            if choice.message:
                content = choice.message.content or ""
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        args_str = tc.function.arguments or "{}"
                        try:
                            arguments = json.loads(args_str)
                        except json.JSONDecodeError:
                            arguments = {"raw": args_str}
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            args=arguments,
                        ))

        usage_data = response.usage
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=self.estimate_cost(input_tokens, output_tokens),
            ),
            duration_ms=duration_ms,
            stop_reason=stop_reason,
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
        oai_messages = self._format_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
        }

        is_reasoning = _is_reasoning_model(self.model)

        if not is_reasoning:
            kwargs["temperature"] = (
                temperature if temperature is not None else self.temperature
            )

        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens or self.max_tokens
        else:
            kwargs["max_tokens"] = max_tokens or self.max_tokens

        if tools and not is_reasoning:
            kwargs["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with _Timer() as timer:
                    response = await self._client.chat.completions.create(**kwargs)
                return self._parse_response(response, timer.elapsed_ms)
            except openai.APIError as e:
                try:
                    self._handle_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "OpenAI retryable error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"OpenAI unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "OpenAI: max retries exceeded", provider=self.name
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        oai_messages = self._format_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        is_reasoning = _is_reasoning_model(self.model)

        if not is_reasoning:
            kwargs["temperature"] = (
                temperature if temperature is not None else self.temperature
            )

        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens or self.max_tokens
        else:
            kwargs["max_tokens"] = max_tokens or self.max_tokens

        if tools and not is_reasoning:
            kwargs["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        stream = None
        for attempt in range(_MAX_RETRIES):
            try:
                stream = await self._client.chat.completions.create(**kwargs)
                break
            except openai.APIError as e:
                try:
                    self._handle_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "OpenAI stream retryable error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"OpenAI stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        if stream is None:
            raise last_error or ProviderError(
                "OpenAI stream: max retries exceeded", provider=self.name
            )

        # Track active tool calls for accumulation
        active_tool_calls: dict[int, DeltaToolCall] = {}

        async for chunk in stream:
            # Usage chunk (final)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                yield StreamChunk(
                    is_final=True,
                    usage=LLMUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=self.estimate_cost(input_tokens, output_tokens),
                    ),
                )
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta and delta.content:
                yield StreamChunk(delta_text=delta.content)

            # Tool calls
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in active_tool_calls:
                        active_tool_calls[idx] = DeltaToolCall(
                            index=idx,
                            id=tc_delta.id or "",
                            name=(tc_delta.function.name if tc_delta.function else ""),
                            arguments_json="",
                        )
                    dtc = active_tool_calls[idx]
                    if tc_delta.id:
                        dtc.id = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        dtc.name = tc_delta.function.name
                    args_frag = (
                        tc_delta.function.arguments if tc_delta.function else ""
                    ) or ""
                    dtc.arguments_json += args_frag

                    yield StreamChunk(
                        delta_tool_call=DeltaToolCall(
                            index=idx,
                            id=dtc.id,
                            name=dtc.name,
                            arguments_json=args_frag,
                        )
                    )

            # Finish reason
            if choice.finish_reason:
                yield StreamChunk(
                    is_final=True,
                    stop_reason=choice.finish_reason,
                )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            logger.debug("OpenAI health check failed", exc_info=True)
            return False
