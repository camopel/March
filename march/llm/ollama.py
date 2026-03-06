"""Ollama local model provider via HTTP /api/chat."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

import httpx

from march.llm.base import (
    DeltaToolCall,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderError,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    _Timer,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 15.0


def _backoff_delay(attempt: int, base: float = _BASE_DELAY, cap: float = _MAX_DELAY) -> float:
    import random
    delay = min(base * (2 ** attempt), cap)
    return delay * (0.5 + random.random() * 0.5)


class OllamaProvider(LLMProvider):
    """Ollama local model provider via HTTP /api/chat endpoint.

    Connects to a local Ollama instance. Supports any GGUF model pulled
    into Ollama, with optional tool use (model-dependent).
    """

    name: str = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        url: str = "http://localhost:11434",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 300.0,
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

        self._client = httpx.AsyncClient(
            base_url=self.url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OllamaProvider":
        return cls(
            model=config.get("model", "llama3.1"),
            url=config.get("url", "http://localhost:11434"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 300.0),
            input_price=config.get("cost", {}).get("input", 0.0),
            output_price=config.get("cost", {}).get("output", 0.0),
        )

    def _format_messages(
        self, messages: list[dict[str, Any]], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert generic messages to Ollama /api/chat format."""
        ollama_messages: list[dict[str, Any]] = []

        if system:
            ollama_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                continue

            if role == "assistant":
                o_msg: dict[str, Any] = {"role": "assistant"}
                if content:
                    o_msg["content"] = content
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    o_msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("arguments", {}),
                            }
                        }
                        for tc in tool_calls
                    ]
                ollama_messages.append(o_msg)

            elif role == "user":
                if isinstance(content, list):
                    # Tool results — Ollama expects them as tool role messages
                    for item in content:
                        tool_content = ""
                        if hasattr(item, "id"):
                            error_val = getattr(item, "error", None)
                            content_val = getattr(item, "content", None)
                            tool_content = str(error_val or content_val or "")
                        elif isinstance(item, dict) and "id" in item:
                            tool_content = str(
                                item.get("error") or item.get("content", "")
                            )
                        else:
                            tool_content = str(item)
                        ollama_messages.append({
                            "role": "tool",
                            "content": tool_content,
                        })
                else:
                    ollama_messages.append({
                        "role": "user",
                        "content": str(content),
                    })
            else:
                ollama_messages.append({"role": role, "content": str(content)})

        return ollama_messages

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        result = []
        for t in tools:
            if isinstance(t, dict):
                result.append(t)
            else:
                result.append(t.to_ollama_schema())
        return result

    def _parse_response(
        self, data: dict[str, Any], duration_ms: float
    ) -> LLMResponse:
        """Parse Ollama /api/chat response."""
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls: list[ToolCall] = []

        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=str(uuid.uuid4()),  # Ollama doesn't provide tool call IDs
                name=func.get("name", ""),
                args=func.get("arguments", {}),
            ))

        # Ollama provides token counts in different fields depending on version
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=self.estimate_cost(prompt_tokens, completion_tokens),
            ),
            duration_ms=duration_ms,
            stop_reason="stop" if data.get("done") else "",
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
        ollama_messages = self._format_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with _Timer() as timer:
                    response = await self._client.post(
                        "/api/chat",
                        json=payload,
                    )
                if response.status_code == 404:
                    raise ProviderError(
                        f"Ollama model '{self.model}' not found. Pull it with: ollama pull {self.model}",
                        provider=self.name,
                        retryable=False,
                    )
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data, timer.elapsed_ms)
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code >= 500
                error = ProviderError(
                    f"Ollama HTTP {e.response.status_code}: {e.response.text[:200]}",
                    provider=self.name,
                    retryable=retryable,
                )
                if retryable and attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Ollama retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, error,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except httpx.ConnectError as e:
                error = ProviderError(
                    f"Ollama connection failed at {self.url}: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Ollama connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(
                    f"Ollama unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "Ollama: max retries exceeded", provider=self.name
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        ollama_messages = self._format_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._client.stream(
                    "POST", "/api/chat", json=payload
                ) as response:
                    if response.status_code == 404:
                        raise ProviderError(
                            f"Ollama model '{self.model}' not found",
                            provider=self.name,
                            retryable=False,
                        )
                    response.raise_for_status()

                    tool_index = 0
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = data.get("message", {})
                        content = message.get("content", "")

                        # Text delta
                        if content:
                            yield StreamChunk(delta_text=content)

                        # Tool calls
                        raw_tool_calls = message.get("tool_calls", [])
                        for tc in raw_tool_calls:
                            func = tc.get("function", {})
                            dtc = DeltaToolCall(
                                index=tool_index,
                                id=str(uuid.uuid4()),
                                name=func.get("name", ""),
                                arguments_json=json.dumps(func.get("arguments", {})),
                            )
                            tool_index += 1
                            yield StreamChunk(delta_tool_call=dtc)

                        # Final chunk
                        if data.get("done"):
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            yield StreamChunk(
                                is_final=True,
                                stop_reason="stop",
                                usage=LLMUsage(
                                    input_tokens=prompt_tokens,
                                    output_tokens=completion_tokens,
                                    cost=self.estimate_cost(
                                        prompt_tokens, completion_tokens
                                    ),
                                ),
                            )
                    return  # Successfully completed streaming

            except httpx.ConnectError as e:
                error = ProviderError(
                    f"Ollama stream connection failed at {self.url}: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Ollama stream connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                raise error from e
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(
                    f"Ollama stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "Ollama stream: max retries exceeded", provider=self.name
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            logger.debug("Ollama health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
