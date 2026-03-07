"""LiteLLM proxy provider — OpenAI-compatible catch-all adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

import httpx

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


class LiteLLMProvider(LLMProvider):
    """LiteLLM proxy provider using OpenAI-compatible API.

    Connects to a LiteLLM proxy server that can route to any backend
    (OpenAI, Anthropic, Bedrock, etc.) via a unified OpenAI-compatible interface.
    """

    name: str = "litellm"

    def __init__(
        self,
        model: str = "gpt-4o",
        url: str = "http://localhost:4000",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 600.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.input_price = input_price
        self.output_price = output_price
        self._api_key = api_key

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LiteLLMProvider":
        return cls(
            model=config.get("model", "gpt-4o"),
            url=config.get("url", "http://localhost:4000"),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 600.0),
            input_price=config.get("cost", {}).get("input", 0.0),
            output_price=config.get("cost", {}).get("output", 0.0),
        )

    def _format_messages(
        self, messages: list[dict[str, Any]], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert generic messages to OpenAI-compatible format for LiteLLM."""
        oai_messages: list[dict[str, Any]] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                continue

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
                    # Check if this is multimodal content (images + text)
                    has_media = any(
                        isinstance(item, dict) and item.get("type") in ("image", "image_url")
                        for item in content
                    )
                    if has_media:
                        # Convert to OpenAI multimodal format
                        oai_content: list[dict[str, Any]] = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "image":
                                    # Anthropic format → OpenAI format
                                    source = item.get("source", {})
                                    media_type = source.get("media_type", "image/jpeg")
                                    data = source.get("data", "")
                                    oai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{data}",
                                        },
                                    })
                                elif item.get("type") == "image_url":
                                    # Already OpenAI format
                                    oai_content.append(item)
                                elif item.get("type") == "text":
                                    oai_content.append(item)
                                else:
                                    oai_content.append({
                                        "type": "text",
                                        "text": str(item),
                                    })
                            else:
                                oai_content.append({
                                    "type": "text",
                                    "text": str(item),
                                })
                        oai_messages.append({
                            "role": "user",
                            "content": oai_content,
                        })
                    else:
                        # Non-media list content (tool results, etc.)
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

    def _handle_error(self, status_code: int, body: str) -> None:
        """Convert HTTP errors to typed ProviderError."""
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(
                f"LiteLLM auth error ({status_code}): {body[:200]}",
                provider=self.name,
            )
        elif status_code == 429:
            raise RateLimitError(
                f"LiteLLM rate limit: {body[:200]}",
                provider=self.name,
            )
        elif status_code == 400 and "context" in body.lower():
            raise ContextLengthError(
                f"LiteLLM context length: {body[:200]}",
                provider=self.name,
            )
        else:
            retryable = status_code >= 500
            raise ProviderError(
                f"LiteLLM HTTP {status_code}: {body[:200]}",
                provider=self.name,
                retryable=retryable,
            )

    def _parse_response(
        self, data: dict[str, Any], duration_ms: float
    ) -> LLMResponse:
        """Parse OpenAI-compatible response."""
        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        stop_reason = choice.get("finish_reason", "")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                arguments = {"raw": args_str}
            tool_calls.append(ToolCall(
                id=tc.get("id", str(uuid.uuid4())),
                name=func.get("name", ""),
                args=arguments,
            ))

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with _Timer() as timer:
                    response = await self._client.post(
                        "/v1/chat/completions",
                        json=payload,
                    )
                if response.status_code != 200:
                    self._handle_error(response.status_code, response.text)
                data = response.json()
                return self._parse_response(data, timer.elapsed_ms)
            except (RateLimitError, ProviderError) as pe:
                if pe.retryable and attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    if isinstance(pe, RateLimitError) and pe.retry_after:
                        delay = pe.retry_after
                    logger.warning(
                        "LiteLLM retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, pe,
                    )
                    await asyncio.sleep(delay)
                    last_error = pe
                    continue
                raise
            except (AuthenticationError, ContextLengthError):
                raise
            except httpx.ConnectError as e:
                error = ProviderError(
                    f"LiteLLM connection failed at {self.url}: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "LiteLLM connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except Exception as e:
                raise ProviderError(
                    f"LiteLLM unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "LiteLLM: max retries exceeded", provider=self.name
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        self._handle_error(response.status_code, body.decode())

                    active_tool_calls: dict[int, DeltaToolCall] = {}

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Usage in final chunk
                        usage_data = data.get("usage")
                        if usage_data and not data.get("choices"):
                            input_tokens = usage_data.get("prompt_tokens", 0)
                            output_tokens = usage_data.get("completion_tokens", 0)
                            yield StreamChunk(
                                is_final=True,
                                usage=LLMUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cost=self.estimate_cost(input_tokens, output_tokens),
                                ),
                            )
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})

                        # Text
                        text = delta.get("content")
                        if text:
                            yield StreamChunk(delta_text=text)

                        # Tool calls
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in active_tool_calls:
                                active_tool_calls[idx] = DeltaToolCall(
                                    index=idx,
                                    id=tc_delta.get("id", ""),
                                    name="",
                                    arguments_json="",
                                )
                            dtc = active_tool_calls[idx]
                            if tc_delta.get("id"):
                                dtc.id = tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                dtc.name = func["name"]
                            args_frag = func.get("arguments", "")
                            dtc.arguments_json += args_frag

                            yield StreamChunk(
                                delta_tool_call=DeltaToolCall(
                                    index=idx,
                                    id=dtc.id,
                                    name=dtc.name,
                                    arguments_json=args_frag,
                                )
                            )

                        # Finish
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            yield StreamChunk(
                                is_final=True,
                                stop_reason=finish_reason,
                            )

                    return  # Successfully completed

            except (RateLimitError, ProviderError) as pe:
                if pe.retryable and attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    if isinstance(pe, RateLimitError) and pe.retry_after:
                        delay = pe.retry_after
                    logger.warning(
                        "LiteLLM stream retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, pe,
                    )
                    await asyncio.sleep(delay)
                    last_error = pe
                    continue
                raise
            except (AuthenticationError, ContextLengthError):
                raise
            except httpx.ConnectError as e:
                error = ProviderError(
                    f"LiteLLM stream connection failed at {self.url}: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "LiteLLM stream connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except Exception as e:
                raise ProviderError(
                    f"LiteLLM stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "LiteLLM stream: max retries exceeded", provider=self.name
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/health")
            if response.status_code == 200:
                return True
            # Fallback: try models endpoint
            response = await self._client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            logger.debug("LiteLLM health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        await self._client.aclose()
