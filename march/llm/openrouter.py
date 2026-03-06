"""OpenRouter provider — OpenAI-compatible multi-provider gateway."""

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
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _backoff_delay(attempt: int, base: float = _BASE_DELAY, cap: float = _MAX_DELAY) -> float:
    import random
    delay = min(base * (2 ** attempt), cap)
    return delay * (0.5 + random.random() * 0.5)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using OpenAI-compatible API.

    Connects to OpenRouter, which provides access to 200+ models from
    multiple providers (OpenAI, Anthropic, Google, Meta, etc.) with
    unified billing and API.
    """

    name: str = "openrouter"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
        app_name: str = "march",
        app_url: str = "",
    ) -> None:
        self.model = model
        self.url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.input_price = input_price
        self.output_price = output_price

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "HTTP-Referer": app_url or "https://github.com/march",
            "X-Title": app_name,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenRouterProvider":
        return cls(
            model=config.get("model", "anthropic/claude-sonnet-4-20250514"),
            api_key=config.get("api_key"),
            base_url=config.get("url", _OPENROUTER_BASE_URL),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120.0),
            input_price=config.get("cost", {}).get("input", 0.0),
            output_price=config.get("cost", {}).get("output", 0.0),
            app_name=config.get("app_name", "march"),
            app_url=config.get("app_url", ""),
        )

    def _format_messages(
        self, messages: list[dict[str, Any]], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert generic messages to OpenAI-compatible format."""
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
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(
                f"OpenRouter auth error ({status_code}): {body[:200]}",
                provider=self.name,
            )
        elif status_code == 429:
            retry_after = None
            # Try to parse retry-after from error body
            try:
                data = json.loads(body)
                error_data = data.get("error", {})
                metadata = error_data.get("metadata", {})
                if "retry_after" in metadata:
                    retry_after = float(metadata["retry_after"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            raise RateLimitError(
                f"OpenRouter rate limit: {body[:200]}",
                provider=self.name,
                retry_after=retry_after,
            )
        elif status_code == 400 and "context" in body.lower():
            raise ContextLengthError(
                f"OpenRouter context length: {body[:200]}",
                provider=self.name,
            )
        else:
            retryable = status_code >= 500 or status_code == 502 or status_code == 503
            raise ProviderError(
                f"OpenRouter HTTP {status_code}: {body[:200]}",
                provider=self.name,
                retryable=retryable,
            )

    def _parse_response(
        self, data: dict[str, Any], duration_ms: float
    ) -> LLMResponse:
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
                        "/chat/completions",
                        json=payload,
                    )
                if response.status_code != 200:
                    self._handle_error(response.status_code, response.text)
                data = response.json()

                # OpenRouter may return an error in the response body
                if "error" in data and not data.get("choices"):
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise ProviderError(
                        f"OpenRouter error: {error_msg}",
                        provider=self.name,
                        retryable=False,
                    )

                return self._parse_response(data, timer.elapsed_ms)
            except (RateLimitError, ProviderError) as pe:
                if pe.retryable and attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    if isinstance(pe, RateLimitError) and pe.retry_after:
                        delay = pe.retry_after
                    logger.warning(
                        "OpenRouter retryable error (attempt %d/%d), retrying in %.1fs: %s",
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
                    f"OpenRouter connection failed: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "OpenRouter connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except Exception as e:
                raise ProviderError(
                    f"OpenRouter unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "OpenRouter: max retries exceeded", provider=self.name
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
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
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

                        # Check for error in stream
                        if "error" in data and not data.get("choices"):
                            error_msg = data["error"].get("message", str(data["error"]))
                            raise ProviderError(
                                f"OpenRouter stream error: {error_msg}",
                                provider=self.name,
                                retryable=False,
                            )

                        # Usage data
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
                            # OpenRouter sometimes includes usage in the last chunk
                            usage = data.get("usage")
                            chunk_usage = None
                            if usage:
                                it = usage.get("prompt_tokens", 0)
                                ot = usage.get("completion_tokens", 0)
                                chunk_usage = LLMUsage(
                                    input_tokens=it,
                                    output_tokens=ot,
                                    cost=self.estimate_cost(it, ot),
                                )
                            yield StreamChunk(
                                is_final=True,
                                stop_reason=finish_reason,
                                usage=chunk_usage,
                            )

                    return

            except (RateLimitError, ProviderError) as pe:
                if pe.retryable and attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    if isinstance(pe, RateLimitError) and pe.retry_after:
                        delay = pe.retry_after
                    logger.warning(
                        "OpenRouter stream retryable error (attempt %d/%d), "
                        "retrying in %.1fs: %s",
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
                    f"OpenRouter stream connection failed: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "OpenRouter stream connection error (attempt %d/%d), "
                        "retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except Exception as e:
                raise ProviderError(
                    f"OpenRouter stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "OpenRouter stream: max retries exceeded", provider=self.name
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            logger.debug("OpenRouter health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        await self._client.aclose()
