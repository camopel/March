"""AWS Bedrock LLM provider via boto3 Converse API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, AsyncIterator

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BotoConfig = None  # type: ignore[assignment,misc]
    ClientError = Exception  # type: ignore[assignment,misc]

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

# Retry settings
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


def _backoff_delay(attempt: int, base: float = _BASE_DELAY, cap: float = _MAX_DELAY) -> float:
    """Exponential backoff with jitter."""
    import random

    delay = min(base * (2 ** attempt), cap)
    return delay * (0.5 + random.random() * 0.5)


def _mime_to_format(mime: str) -> str:
    """Convert MIME type to Bedrock image format string."""
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    return mapping.get(mime, "png")


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider using the Converse / ConverseStream API.

    Supports Claude, Llama, Mistral, and Nova models via the unified
    Bedrock Converse API.
    """

    name: str = "bedrock"

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
        region: str = "us-west-2",
        profile: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
        input_price: float = 3.0,
        output_price: float = 15.0,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                "boto3 package not installed. Run: pip install march[bedrock]"
            )
        self.model = model
        self.region = region
        self.profile = profile
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.input_price = input_price
        self.output_price = output_price

        self._boto_config = BotoConfig(
            region_name=region,
            read_timeout=int(timeout),
            connect_timeout=10,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )

        self._boto_session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self._client = self._boto_session.client("bedrock-runtime", config=self._boto_config)

    def _refresh_client(self) -> None:
        """Recreate the boto3 client to pick up refreshed AWS credentials."""
        self._boto_session = boto3.Session(profile_name=self.profile) if self.profile else boto3.Session()
        self._client = self._boto_session.client("bedrock-runtime", config=self._boto_config)
        logger.info("Refreshed Bedrock client credentials")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BedrockProvider":
        return cls(
            model=config.get("model", "anthropic.claude-sonnet-4-20250514-v1:0"),
            region=config.get("region", "us-west-2"),
            profile=config.get("profile"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120.0),
            input_price=config.get("cost", {}).get("input", 3.0),
            output_price=config.get("cost", {}).get("output", 15.0),
        )

    def _format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert generic message format to Bedrock Converse format."""
        bedrock_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # System messages handled separately in Bedrock

            # Handle tool results — Bedrock expects these as user messages with toolResult blocks
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", msg.get("id", ""))
                content_text = msg.get("content", "") or "(empty)"
                is_error = content_text.startswith("Error:")
                result_block = {
                    "toolResult": {
                        "toolUseId": tool_call_id,
                        "content": [{"text": content_text}],
                        "status": "error" if is_error else "success",
                    }
                }
                # Merge into previous user message if it exists, or create new one
                if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                    bedrock_messages[-1]["content"].append(result_block)
                else:
                    bedrock_messages.append({
                        "role": "user",
                        "content": [result_block],
                    })
                continue

            bedrock_role = "user" if role == "user" else "assistant"
            content_blocks: list[dict[str, Any]] = []

            raw_content = msg.get("content") or ""

            # Handle tool results (user messages containing tool results)
            if role == "user" and isinstance(raw_content, list):
                for item in raw_content:
                    if hasattr(item, "id"):
                        # ToolResult-like object
                        result_content: list[dict[str, Any]] = []
                        error_val = getattr(item, "error", None)
                        content_val = getattr(item, "content", None)
                        if error_val:
                            result_content.append({"text": str(error_val)})
                            status = "error"
                        else:
                            result_content.append({"text": str(content_val or "")})
                            status = "success"
                        content_blocks.append({
                            "toolResult": {
                                "toolUseId": item.id,
                                "content": result_content,
                                "status": status,
                            }
                        })
                    elif isinstance(item, dict) and "id" in item:
                        result_content_d: list[dict[str, Any]] = []
                        if item.get("error"):
                            result_content_d.append({"text": str(item["error"])})
                            status_d = "error"
                        else:
                            result_content_d.append({"text": str(item.get("content", ""))})
                            status_d = "success"
                        content_blocks.append({
                            "toolResult": {
                                "toolUseId": item["id"],
                                "content": result_content_d,
                                "status": status_d,
                            }
                        })
                    elif isinstance(item, dict) and item.get("type") == "image":
                        # Multimodal image block → Bedrock image format
                        source = item.get("source", {})
                        img_data = source.get("data", "")
                        if isinstance(img_data, str):
                            img_data = base64.b64decode(img_data)
                        content_blocks.append({
                            "image": {
                                "format": _mime_to_format(source.get("media_type", "image/png")),
                                "source": {
                                    "bytes": img_data,
                                },
                            }
                        })
                    elif isinstance(item, dict) and "text" in item:
                        content_blocks.append({"text": item["text"]})
                    else:
                        content_blocks.append({"text": str(item)})
            elif isinstance(raw_content, str):
                if raw_content:  # Skip empty text — Bedrock rejects blank text blocks
                    content_blocks.append({"text": raw_content})
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Multimodal image block → Bedrock image format
                        source = item.get("source", {})
                        content_blocks.append({
                            "image": {
                                "format": _mime_to_format(source.get("media_type", "image/png")),
                                "source": {
                                    "bytes": base64.b64decode(source["data"]),
                                },
                            }
                        })
                    elif isinstance(item, dict) and "text" in item:
                        if item["text"]:  # Skip empty
                            content_blocks.append({"text": item["text"]})
                    elif str(item):
                        content_blocks.append({"text": str(item)})

            # Handle assistant messages with tool calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                if not content_blocks:
                    # Bedrock requires at least some content
                    pass
                for tc in tool_calls:
                    # Handle both OpenAI format (function.name/arguments) and flat format (name/args)
                    func = tc.get("function", {})
                    tc_name = func.get("name", "") or tc.get("name", "")
                    tc_id = tc.get("id", str(uuid.uuid4()))
                    args = func.get("arguments", tc.get("arguments", tc.get("args", {})))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    content_blocks.append({
                        "toolUse": {
                            "toolUseId": tc_id,
                            "name": tc_name,
                            "input": args,
                        }
                    })

            if content_blocks:
                # Bedrock requires alternating user/assistant roles
                # Merge consecutive same-role messages
                if bedrock_messages and bedrock_messages[-1]["role"] == bedrock_role:
                    bedrock_messages[-1]["content"].extend(content_blocks)
                else:
                    bedrock_messages.append({
                        "role": bedrock_role,
                        "content": content_blocks,
                    })

        # Final safety pass: remove any empty text blocks that Bedrock would reject
        for msg in bedrock_messages:
            msg["content"] = [
                block for block in msg["content"]
                if not (list(block.keys()) == ["text"] and not block["text"])
            ]
            # If content is now empty (was all blank text), add a placeholder
            if not msg["content"]:
                msg["content"] = [{"text": "."}]

        # Remove any duplicate consecutive same-role messages that slipped through
        cleaned: list[dict[str, Any]] = []
        for msg in bedrock_messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"].extend(msg["content"])
            else:
                cleaned.append(msg)

        # Repair tool_use/tool_result pairing:
        # Every toolUse in an assistant message must have a matching toolResult
        # in the immediately following user message. If missing, add a dummy result.
        repaired: list[dict[str, Any]] = []
        for i, msg in enumerate(cleaned):
            repaired.append(msg)
            if msg["role"] == "assistant":
                tool_use_ids = [
                    block["toolUse"]["toolUseId"]
                    for block in msg["content"]
                    if "toolUse" in block
                ]
                if tool_use_ids:
                    # Check if next message is user with matching toolResults
                    next_msg = cleaned[i + 1] if i + 1 < len(cleaned) else None
                    if next_msg and next_msg["role"] == "user":
                        existing_result_ids = {
                            block["toolResult"]["toolUseId"]
                            for block in next_msg["content"]
                            if "toolResult" in block
                        }
                        missing_ids = [tid for tid in tool_use_ids if tid not in existing_result_ids]
                        if missing_ids:
                            for tid in missing_ids:
                                next_msg["content"].insert(0, {
                                    "toolResult": {
                                        "toolUseId": tid,
                                        "content": [{"text": "[result lost during context compaction]"}],
                                        "status": "success",
                                    }
                                })
                    elif not next_msg or next_msg["role"] == "assistant":
                        # No user message follows — inject a dummy user message with results
                        dummy_content = []
                        for tid in tool_use_ids:
                            dummy_content.append({
                                "toolResult": {
                                    "toolUseId": tid,
                                    "content": [{"text": "[result lost during context compaction]"}],
                                    "status": "success",
                                }
                            })
                        repaired.append({"role": "user", "content": dummy_content})

        # Final: ensure alternating roles after repair
        final: list[dict[str, Any]] = []
        for msg in repaired:
            if final and final[-1]["role"] == msg["role"]:
                final[-1]["content"].extend(msg["content"])
            else:
                final.append(msg)

        return final

    def _format_tools(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        """Convert tool definitions to Bedrock toolConfig format."""
        tool_list = []
        for t in tools:
            if isinstance(t, dict):
                # Convert OpenAI format dict to Bedrock format
                func = t.get("function", t)
                tool_list.append({
                    "toolSpec": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "inputSchema": {"json": func.get("parameters", {"type": "object", "properties": {}})},
                    }
                })
            else:
                tool_list.append(t.to_bedrock_schema())
        return {"tools": tool_list}

    def _parse_response(
        self, response: dict[str, Any], duration_ms: float
    ) -> LLMResponse:
        """Parse Bedrock Converse response into LLMResponse."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu.get("toolUseId", str(uuid.uuid4())),
                    name=tu.get("name", ""),
                    args=tu.get("input", {}),
                ))

        usage_data = response.get("usage", {})
        input_tokens = usage_data.get("inputTokens", 0)
        output_tokens = usage_data.get("outputTokens", 0)

        stop_reason = response.get("stopReason", "")

        usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self.estimate_cost(input_tokens, output_tokens),
        )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            duration_ms=duration_ms,
            stop_reason=stop_reason,
            model=self.model,
            provider=self.name,
        )

    def _handle_client_error(self, error: ClientError) -> None:
        """Convert boto3 ClientError to typed ProviderError."""
        error_code = error.response.get("Error", {}).get("Code", "")
        error_message = error.response.get("Error", {}).get("Message", str(error))

        if error_code in ("ThrottlingException", "TooManyRequestsException"):
            raise RateLimitError(
                f"Bedrock rate limit: {error_message}",
                provider=self.name,
            )
        elif error_code in (
            "AccessDeniedException",
            "UnrecognizedClientException",
            "ExpiredTokenException",
        ):
            # Auto-refresh credentials on token expiry
            if error_code == "ExpiredTokenException":
                self._refresh_client()
            raise AuthenticationError(
                f"Bedrock auth error: {error_message}",
                provider=self.name,
                retryable=(error_code == "ExpiredTokenException"),
            )
        elif error_code == "ValidationException" and "too long" in error_message.lower():
            raise ContextLengthError(
                f"Bedrock context length: {error_message}",
                provider=self.name,
            )
        else:
            retryable = error_code in (
                "InternalServerException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
            )
            raise ProviderError(
                f"Bedrock error ({error_code}): {error_message}",
                provider=self.name,
                retryable=retryable,
            )

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        bedrock_messages = self._format_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }

        if system:
            kwargs["system"] = [{"text": system}]

        if tools:
            kwargs["toolConfig"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with _Timer() as timer:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._client.converse(**kwargs)
                    )
                return self._parse_response(response, timer.elapsed_ms)
            except ClientError as e:
                try:
                    self._handle_client_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "Bedrock retryable error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"Bedrock unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e

        raise last_error or ProviderError(
            "Bedrock: max retries exceeded", provider=self.name
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        bedrock_messages = self._format_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }

        if system:
            kwargs["system"] = [{"text": system}]

        if tools:
            kwargs["toolConfig"] = self._format_tools(tools)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.converse_stream(**kwargs)
                )
                break
            except ClientError as e:
                try:
                    self._handle_client_error(e)
                except (RateLimitError, ProviderError) as pe:
                    if pe.retryable and attempt < _MAX_RETRIES - 1:
                        delay = _backoff_delay(attempt)
                        if isinstance(pe, RateLimitError) and pe.retry_after:
                            delay = pe.retry_after
                        logger.warning(
                            "Bedrock stream retryable error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRIES, delay, pe,
                        )
                        await asyncio.sleep(delay)
                        last_error = pe
                        continue
                    raise
            except Exception as e:
                raise ProviderError(
                    f"Bedrock stream unexpected error: {e}",
                    provider=self.name,
                    retryable=False,
                ) from e
        else:
            raise last_error or ProviderError(
                "Bedrock stream: max retries exceeded", provider=self.name
            )

        stream = response.get("stream", [])
        current_tool_call: DeltaToolCall | None = None
        tool_index = 0

        # Stream events from the blocking Bedrock iterator via an async queue.
        # A background thread reads events and puts them on the queue;
        # the async generator yields chunks as they arrive.
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def _read_events() -> None:
            try:
                for event in stream:
                    queue.put_nowait(event)
            except Exception as e:
                queue.put_nowait({"_error": str(e)})
            finally:
                queue.put_nowait(None)  # Sentinel: stream finished

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _read_events)

        while True:
            event = await queue.get()
            if event is None:
                break
            if "_error" in event:
                raise ProviderError(
                    f"Bedrock stream read error: {event['_error']}",
                    provider=self.name,
                    retryable=False,
                )
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                block_start = start.get("start", {})
                if "toolUse" in block_start:
                    tu = block_start["toolUse"]
                    current_tool_call = DeltaToolCall(
                        index=tool_index,
                        id=tu.get("toolUseId", str(uuid.uuid4())),
                        name=tu.get("name", ""),
                        arguments_json="",
                    )
                    tool_index += 1
                    yield StreamChunk(delta_tool_call=current_tool_call)

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield StreamChunk(delta_text=delta["text"])
                elif "toolUse" in delta and current_tool_call is not None:
                    json_frag = delta["toolUse"].get("input", "")
                    current_tool_call.arguments_json += json_frag
                    yield StreamChunk(
                        delta_tool_call=DeltaToolCall(
                            index=current_tool_call.index,
                            id=current_tool_call.id,
                            name=current_tool_call.name,
                            arguments_json=json_frag,
                        )
                    )

            elif "contentBlockStop" in event:
                current_tool_call = None

            elif "metadata" in event:
                meta = event["metadata"]
                usage_data = meta.get("usage", {})
                input_tokens = usage_data.get("inputTokens", 0)
                output_tokens = usage_data.get("outputTokens", 0)
                yield StreamChunk(
                    is_final=True,
                    usage=LLMUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=self.estimate_cost(input_tokens, output_tokens),
                    ),
                )

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "")
                yield StreamChunk(
                    is_final=True,
                    stop_reason=stop_reason,
                )

    async def health_check(self) -> bool:
        """Check Bedrock availability by listing foundation models."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.converse(
                    modelId=self.model,
                    messages=[{"role": "user", "content": [{"text": "ping"}]}],
                    inferenceConfig={"maxTokens": 1},
                ),
            )
            return True
        except Exception:
            logger.debug("Bedrock health check failed", exc_info=True)
            return False
