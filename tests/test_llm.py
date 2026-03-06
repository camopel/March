"""Tests for the March LLM layer — providers, router, and base types."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    ToolParameter,
    _Timer,
)
from march.llm.router import LLMRouter, NoProviderAvailable, RouterConfig


# ─── Fixtures ────────────────────────────────────────────────────────────────


SAMPLE_TOOLS = [
    ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters=[
            ToolParameter(
                name="path", type="string", description="File path", required=True
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description="Line offset",
                required=False,
                default=0,
            ),
        ],
    ),
    ToolDefinition(
        name="search",
        description="Search the web",
        parameters=[
            ToolParameter(
                name="query", type="string", description="Search query", required=True
            ),
            ToolParameter(
                name="count",
                type="integer",
                description="Number of results",
                required=False,
                enum=["1", "5", "10"],
            ),
        ],
    ),
]

SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello, can you help me?"},
    {"role": "assistant", "content": "Of course! What do you need?"},
    {"role": "user", "content": "Read the file at /tmp/test.txt"},
]


# ─── Test Tool Schema Conversions ────────────────────────────────────────────


class TestToolDefinition:
    def test_to_openai_schema(self) -> None:
        tool = SAMPLE_TOOLS[0]
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert schema["function"]["description"] == "Read a file"
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert props["path"]["type"] == "string"
        assert "required" in schema["function"]["parameters"]
        assert "path" in schema["function"]["parameters"]["required"]

    def test_to_anthropic_schema(self) -> None:
        tool = SAMPLE_TOOLS[0]
        schema = tool.to_anthropic_schema()

        assert schema["name"] == "read_file"
        assert schema["description"] == "Read a file"
        assert "input_schema" in schema
        props = schema["input_schema"]["properties"]
        assert "path" in props
        assert "required" in schema["input_schema"]

    def test_to_bedrock_schema(self) -> None:
        tool = SAMPLE_TOOLS[0]
        schema = tool.to_bedrock_schema()

        assert "toolSpec" in schema
        spec = schema["toolSpec"]
        assert spec["name"] == "read_file"
        assert spec["description"] == "Read a file"
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]
        props = spec["inputSchema"]["json"]["properties"]
        assert "path" in props

    def test_to_ollama_schema_matches_openai(self) -> None:
        tool = SAMPLE_TOOLS[0]
        openai_schema = tool.to_openai_schema()
        ollama_schema = tool.to_ollama_schema()
        assert openai_schema == ollama_schema

    def test_enum_in_schema(self) -> None:
        tool = SAMPLE_TOOLS[1]
        schema = tool.to_openai_schema()
        count_prop = schema["function"]["parameters"]["properties"]["count"]
        assert count_prop["enum"] == ["1", "5", "10"]

    def test_optional_param_not_required(self) -> None:
        tool = SAMPLE_TOOLS[0]
        schema = tool.to_openai_schema()
        required = schema["function"]["parameters"].get("required", [])
        assert "offset" not in required
        assert "path" in required


# ─── Test Data Types ─────────────────────────────────────────────────────────


class TestDataTypes:
    def test_llm_usage_defaults(self) -> None:
        usage = LLMUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost == 0.0

    def test_llm_response_to_message(self) -> None:
        response = LLMResponse(
            content="Hello!",
            tool_calls=[
                ToolCall(id="tc1", name="read_file", args={"path": "/tmp/x"})
            ],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            duration_ms=100.0,
            stop_reason="stop",
            model="test-model",
            provider="test",
        )

        msg = response.to_message()
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["name"] == "read_file"

    def test_llm_response_to_message_no_content(self) -> None:
        response = LLMResponse(
            tool_calls=[
                ToolCall(id="tc1", name="search", args={"query": "test"})
            ],
        )
        msg = response.to_message()
        assert "content" not in msg
        assert len(msg["tool_calls"]) == 1

    def test_stream_chunk_text(self) -> None:
        chunk = StreamChunk(delta_text="Hello")
        assert chunk.delta_text == "Hello"
        assert not chunk.is_final

    def test_stream_chunk_final(self) -> None:
        chunk = StreamChunk(
            is_final=True,
            usage=LLMUsage(input_tokens=100, output_tokens=50, cost=0.01),
            stop_reason="stop",
        )
        assert chunk.is_final
        assert chunk.usage is not None
        assert chunk.usage.input_tokens == 100

    def test_delta_tool_call(self) -> None:
        dtc = DeltaToolCall(index=0, id="tc1", name="read", arguments_json='{"path": "x"}')
        assert dtc.index == 0
        assert dtc.name == "read"

    def test_tool_call_frozen(self) -> None:
        tc = ToolCall(id="1", name="test", args={})
        with pytest.raises(AttributeError):
            tc.name = "other"  # type: ignore[misc]


# ─── Test Timer ──────────────────────────────────────────────────────────────


class TestTimer:
    def test_timer_measures_time(self) -> None:
        with _Timer() as timer:
            time.sleep(0.01)
        assert timer.elapsed_ms >= 5  # Allow some slack


# ─── Test Exceptions ─────────────────────────────────────────────────────────


class TestExceptions:
    def test_provider_error(self) -> None:
        e = ProviderError("test error", provider="test", retryable=True)
        assert str(e) == "test error"
        assert e.provider == "test"
        assert e.retryable is True

    def test_rate_limit_error(self) -> None:
        e = RateLimitError("rate limited", provider="openai", retry_after=5.0)
        assert e.retryable is True
        assert e.retry_after == 5.0

    def test_auth_error_not_retryable(self) -> None:
        e = AuthenticationError("bad key", provider="openai")
        assert e.retryable is False

    def test_context_length_not_retryable(self) -> None:
        e = ContextLengthError("too long", provider="openai")
        assert e.retryable is False


# ─── Mock Provider for Router Tests ─────────────────────────────────────────


class MockProvider(LLMProvider):
    """A mock provider for testing the router."""

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        should_fail: bool = False,
        fail_retryable: bool = True,
        health: bool = True,
        response_content: str = "mock response",
        input_price: float = 1.0,
        output_price: float = 2.0,
    ) -> None:
        self.name = name
        self.model = model
        self.input_price = input_price
        self.output_price = output_price
        self._should_fail = should_fail
        self._fail_retryable = fail_retryable
        self._health = health
        self._response_content = response_content
        self.call_count = 0
        self.stream_call_count = 0
        self.timeout = 30.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MockProvider":
        return cls(name=config.get("name", "mock"))

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        if self._should_fail:
            raise ProviderError(
                f"{self.name} failed",
                provider=self.name,
                retryable=self._fail_retryable,
            )
        return LLMResponse(
            content=self._response_content,
            usage=LLMUsage(input_tokens=100, output_tokens=50, cost=0.01),
            duration_ms=50.0,
            stop_reason="stop",
            model=self.model,
            provider=self.name,
        )

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_call_count += 1
        if self._should_fail:
            raise ProviderError(
                f"{self.name} stream failed",
                provider=self.name,
                retryable=self._fail_retryable,
            )
        words = self._response_content.split()
        for word in words:
            yield StreamChunk(delta_text=word + " ")
        yield StreamChunk(
            is_final=True,
            stop_reason="stop",
            usage=LLMUsage(input_tokens=100, output_tokens=50, cost=0.01),
        )

    async def health_check(self) -> bool:
        return self._health


# ─── Test Provider Interface Compliance ──────────────────────────────────────


class TestProviderInterfaceCompliance:
    """Verify that all provider implementations satisfy the abstract interface."""

    def test_mock_provider_is_llm_provider(self) -> None:
        provider = MockProvider(name="test")
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_mock_converse(self) -> None:
        provider = MockProvider(name="test", response_content="hello world")
        response = await provider.converse(SAMPLE_MESSAGES)
        assert isinstance(response, LLMResponse)
        assert response.content == "hello world"
        assert response.provider == "test"
        assert response.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_mock_converse_stream(self) -> None:
        provider = MockProvider(name="test", response_content="hello world")
        chunks: list[StreamChunk] = []
        async for chunk in provider.converse_stream(SAMPLE_MESSAGES):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].is_final
        text = "".join(c.delta_text for c in chunks if c.delta_text)
        assert "hello" in text
        assert "world" in text

    @pytest.mark.asyncio
    async def test_mock_health_check(self) -> None:
        healthy_provider = MockProvider(name="healthy", health=True)
        unhealthy_provider = MockProvider(name="unhealthy", health=False)
        assert await healthy_provider.health_check() is True
        assert await unhealthy_provider.health_check() is False

    def test_estimate_cost(self) -> None:
        provider = MockProvider(
            name="test", input_price=3.0, output_price=15.0
        )
        cost = provider.estimate_cost(1000, 500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        assert abs(cost - expected) < 1e-10


# ─── Test Router Fallback Logic ──────────────────────────────────────────────


class TestRouterFallback:
    @pytest.mark.asyncio
    async def test_route_selects_first_healthy(self) -> None:
        providers = {
            "primary": MockProvider(name="primary", response_content="primary response"),
            "secondary": MockProvider(name="secondary", response_content="secondary response"),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        provider = await router.route()
        assert provider.name == "primary"

    @pytest.mark.asyncio
    async def test_route_skips_unhealthy(self) -> None:
        providers = {
            "primary": MockProvider(name="primary"),
            "secondary": MockProvider(name="secondary"),
        }
        config = RouterConfig(
            fallback_chain=["primary", "secondary"],
            max_failures_before_down=2,
        )
        router = LLMRouter(config=config, providers=providers)

        # Mark primary as down
        router.record_failure("primary")
        router.record_failure("primary")

        provider = await router.route()
        assert provider.name == "secondary"

    @pytest.mark.asyncio
    async def test_route_explicit_preference(self) -> None:
        providers = {
            "primary": MockProvider(name="primary"),
            "secondary": MockProvider(name="secondary"),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        provider = await router.route(preference="secondary")
        assert provider.name == "secondary"

    @pytest.mark.asyncio
    async def test_route_preference_even_if_down(self) -> None:
        providers = {
            "primary": MockProvider(name="primary"),
            "secondary": MockProvider(name="secondary"),
        }
        config = RouterConfig(
            fallback_chain=["primary", "secondary"],
            max_failures_before_down=1,
        )
        router = LLMRouter(config=config, providers=providers)

        # Mark secondary as down
        router.record_failure("secondary")

        # Explicit preference overrides health
        provider = await router.route(preference="secondary")
        assert provider.name == "secondary"

    @pytest.mark.asyncio
    async def test_route_no_providers_raises(self) -> None:
        config = RouterConfig(fallback_chain=["nonexistent"])
        router = LLMRouter(config=config, providers={})

        with pytest.raises(NoProviderAvailable):
            await router.route()

    @pytest.mark.asyncio
    async def test_converse_fallback_on_failure(self) -> None:
        providers = {
            "primary": MockProvider(
                name="primary", should_fail=True, fail_retryable=True
            ),
            "secondary": MockProvider(
                name="secondary", response_content="fallback response"
            ),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        response = await router.converse(messages=SAMPLE_MESSAGES)
        assert response.content == "fallback response"
        assert response.provider == "secondary"
        assert providers["primary"].call_count == 1
        assert providers["secondary"].call_count == 1

    @pytest.mark.asyncio
    async def test_converse_non_retryable_does_not_fallback(self) -> None:
        providers = {
            "primary": MockProvider(
                name="primary", should_fail=True, fail_retryable=False
            ),
            "secondary": MockProvider(name="secondary"),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        with pytest.raises(ProviderError):
            await router.converse(messages=SAMPLE_MESSAGES)
        # Secondary should NOT have been called
        assert providers["secondary"].call_count == 0

    @pytest.mark.asyncio
    async def test_converse_all_fail_raises(self) -> None:
        providers = {
            "a": MockProvider(name="a", should_fail=True, fail_retryable=True),
            "b": MockProvider(name="b", should_fail=True, fail_retryable=True),
        }
        config = RouterConfig(fallback_chain=["a", "b"])
        router = LLMRouter(config=config, providers=providers)

        with pytest.raises(ProviderError):
            await router.converse(messages=SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_converse_stream_fallback(self) -> None:
        providers = {
            "primary": MockProvider(
                name="primary", should_fail=True, fail_retryable=True
            ),
            "secondary": MockProvider(
                name="secondary", response_content="stream fallback"
            ),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        chunks: list[StreamChunk] = []
        async for chunk in router.converse_stream(messages=SAMPLE_MESSAGES):
            chunks.append(chunk)

        assert len(chunks) > 0
        text = "".join(c.delta_text for c in chunks if c.delta_text)
        assert "stream" in text
        assert "fallback" in text


# ─── Test Health Check Cycling ───────────────────────────────────────────────


class TestHealthCheckCycling:
    def test_record_success_resets_failures(self) -> None:
        config = RouterConfig(
            fallback_chain=["test"], max_failures_before_down=3
        )
        router = LLMRouter(config=config, providers={"test": MockProvider(name="test")})

        router.record_failure("test")
        router.record_failure("test")
        assert router._health["test"].consecutive_failures == 2

        router.record_success("test")
        assert router._health["test"].consecutive_failures == 0
        assert router._health["test"].healthy is True

    def test_record_failures_marks_down(self) -> None:
        config = RouterConfig(
            fallback_chain=["test"], max_failures_before_down=3
        )
        router = LLMRouter(config=config, providers={"test": MockProvider(name="test")})

        router.record_failure("test")
        router.record_failure("test")
        assert router._health["test"].healthy is True

        router.record_failure("test")
        assert router._health["test"].healthy is False

    def test_cooldown_recovery(self) -> None:
        config = RouterConfig(
            fallback_chain=["test"],
            max_failures_before_down=1,
            cooldown_seconds=0.01,  # Very short for testing
        )
        router = LLMRouter(config=config, providers={"test": MockProvider(name="test")})

        router.record_failure("test")
        assert router._health["test"].healthy is False
        assert not router._is_provider_available("test")

        # Wait for cooldown
        time.sleep(0.02)
        assert router._is_provider_available("test")

    def test_get_health_report(self) -> None:
        providers = {
            "a": MockProvider(name="a"),
            "b": MockProvider(name="b"),
        }
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=2)
        router = LLMRouter(config=config, providers=providers)

        router.record_success("a")
        router.record_failure("b")
        router.record_failure("b")

        health = router.get_health()
        assert health["a"]["healthy"] is True
        assert health["a"]["total_successes"] == 1
        assert health["b"]["healthy"] is False
        assert health["b"]["total_failures"] == 2
        assert "cooldown_remaining_seconds" in health["b"]

    def test_reset_health_single(self) -> None:
        config = RouterConfig(
            fallback_chain=["test"], max_failures_before_down=1
        )
        router = LLMRouter(config=config, providers={"test": MockProvider(name="test")})

        router.record_failure("test")
        assert router._health["test"].healthy is False

        router.reset_health("test")
        assert router._health["test"].healthy is True
        assert router._health["test"].consecutive_failures == 0

    def test_reset_health_all(self) -> None:
        providers = {
            "a": MockProvider(name="a"),
            "b": MockProvider(name="b"),
        }
        config = RouterConfig(
            fallback_chain=["a", "b"], max_failures_before_down=1
        )
        router = LLMRouter(config=config, providers=providers)

        router.record_failure("a")
        router.record_failure("b")

        router.reset_health()
        assert router._health["a"].healthy is True
        assert router._health["b"].healthy is True

    @pytest.mark.asyncio
    async def test_check_all_health(self) -> None:
        providers = {
            "healthy": MockProvider(name="healthy", health=True),
            "unhealthy": MockProvider(name="unhealthy", health=False),
        }
        config = RouterConfig(fallback_chain=["healthy", "unhealthy"])
        router = LLMRouter(config=config, providers=providers)

        results = await router.check_all_health()
        assert results["healthy"] is True
        assert results["unhealthy"] is False
        assert router._health["healthy"].healthy is True
        assert router._health["unhealthy"].consecutive_failures >= 1


# ─── Test Cost-Aware Routing ─────────────────────────────────────────────────


class TestCostAwareRouting:
    @pytest.mark.asyncio
    async def test_cost_aware_selects_cheapest(self) -> None:
        providers = {
            "expensive": MockProvider(
                name="expensive", input_price=10.0, output_price=30.0
            ),
            "cheap": MockProvider(
                name="cheap", input_price=0.5, output_price=1.5
            ),
            "medium": MockProvider(
                name="medium", input_price=3.0, output_price=15.0
            ),
        }
        config = RouterConfig(
            fallback_chain=["expensive", "medium", "cheap"],
            cost_aware=True,
        )
        router = LLMRouter(config=config, providers=providers)

        provider = await router.route()
        assert provider.name == "cheap"

    @pytest.mark.asyncio
    async def test_cost_aware_skips_unhealthy(self) -> None:
        providers = {
            "expensive": MockProvider(
                name="expensive", input_price=10.0, output_price=30.0
            ),
            "cheap": MockProvider(
                name="cheap", input_price=0.5, output_price=1.5
            ),
        }
        config = RouterConfig(
            fallback_chain=["expensive", "cheap"],
            cost_aware=True,
            max_failures_before_down=1,
        )
        router = LLMRouter(config=config, providers=providers)

        # Mark cheap as down
        router.record_failure("cheap")

        provider = await router.route()
        assert provider.name == "expensive"


# ─── Test Router from_config ─────────────────────────────────────────────────


class TestRouterFromConfig:
    def test_from_config_with_chain(self) -> None:
        providers = {
            "a": MockProvider(name="a"),
            "b": MockProvider(name="b"),
        }
        config = {
            "fallback_chain": ["a", "b"],
            "max_failures_before_down": 5,
            "cooldown_seconds": 120.0,
        }
        router = LLMRouter.from_config(config, providers)
        assert router.config.fallback_chain == ["a", "b"]
        assert router.config.max_failures_before_down == 5
        assert router.config.cooldown_seconds == 120.0

    def test_from_config_auto_chain(self) -> None:
        providers = {
            "a": MockProvider(name="a"),
            "b": MockProvider(name="b"),
        }
        config = {"default": "b"}
        router = LLMRouter.from_config(config, providers)
        assert router.config.fallback_chain[0] == "b"
        assert "a" in router.config.fallback_chain


# ─── Test Provider from_config ───────────────────────────────────────────────


class TestProviderFromConfig:
    def test_bedrock_from_config(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {
            "model": "anthropic.claude-sonnet-4-20250514-v1:0",
            "region": "us-east-1",
            "max_tokens": 8192,
            "temperature": 0.5,
            "cost": {"input": 3.0, "output": 15.0},
        }
        provider = BedrockProvider.from_config(config)
        assert provider.model == "anthropic.claude-sonnet-4-20250514-v1:0"
        assert provider.region == "us-east-1"
        assert provider.max_tokens == 8192
        assert provider.temperature == 0.5
        assert provider.input_price == 3.0

    def test_openai_from_config(self) -> None:
        from march.llm.openai_provider import OpenAIProvider

        config = {
            "model": "gpt-4o",
            "api_key": "test-key",
            "max_tokens": 4096,
            "cost": {"input": 2.5, "output": 10.0},
        }
        provider = OpenAIProvider.from_config(config)
        assert provider.model == "gpt-4o"
        assert provider.max_tokens == 4096
        assert provider.input_price == 2.5

    def test_anthropic_from_config(self) -> None:
        from march.llm.anthropic_provider import AnthropicProvider

        config = {
            "model": "claude-sonnet-4-20250514",
            "api_key": "test-key",
            "max_tokens": 4096,
            "cost": {"input": 3.0, "output": 15.0},
        }
        provider = AnthropicProvider.from_config(config)
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.max_tokens == 4096

    def test_ollama_from_config(self) -> None:
        from march.llm.ollama import OllamaProvider

        config = {
            "model": "llama3.1",
            "url": "http://localhost:11434",
            "max_tokens": 2048,
            "cost": {"input": 0.0, "output": 0.0},
        }
        provider = OllamaProvider.from_config(config)
        assert provider.model == "llama3.1"
        assert provider.url == "http://localhost:11434"
        assert provider.input_price == 0.0

    def test_litellm_from_config(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {
            "model": "claude-opus-4-6",
            "url": "http://localhost:4000",
            "max_tokens": 128000,
            "cost": {"input": 15.0, "output": 75.0},
        }
        provider = LiteLLMProvider.from_config(config)
        assert provider.model == "claude-opus-4-6"
        assert provider.url == "http://localhost:4000"
        assert provider.input_price == 15.0

    def test_openrouter_from_config(self) -> None:
        from march.llm.openrouter import OpenRouterProvider

        config = {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": "test-key",
            "max_tokens": 4096,
            "cost": {"input": 3.0, "output": 15.0},
            "app_name": "my-app",
        }
        provider = OpenRouterProvider.from_config(config)
        assert provider.model == "anthropic/claude-sonnet-4-20250514"
        assert provider.max_tokens == 4096


# ─── Test Bedrock Message Formatting ─────────────────────────────────────────


class TestBedrockFormatting:
    def test_format_simple_messages(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        bedrock_msgs = provider._format_messages(messages)

        assert len(bedrock_msgs) == 3
        assert bedrock_msgs[0]["role"] == "user"
        assert bedrock_msgs[0]["content"] == [{"text": "Hello"}]
        assert bedrock_msgs[1]["role"] == "assistant"

    def test_format_tool_calls(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "name": "read_file",
                        "arguments": {"path": "/tmp/test.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"id": "tc1", "content": "file contents here"}
                ],
            },
        ]
        bedrock_msgs = provider._format_messages(messages)
        assert len(bedrock_msgs) == 3

        # Check tool use in assistant message
        assistant_blocks = bedrock_msgs[1]["content"]
        tool_use_blocks = [b for b in assistant_blocks if "toolUse" in b]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["toolUse"]["name"] == "read_file"

        # Check tool result in user message
        user_blocks = bedrock_msgs[2]["content"]
        tool_result_blocks = [b for b in user_blocks if "toolResult" in b]
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["toolResult"]["toolUseId"] == "tc1"

    def test_format_tools(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        tool_config = provider._format_tools(SAMPLE_TOOLS)
        assert "tools" in tool_config
        assert len(tool_config["tools"]) == 2


# ─── Test OpenAI Message Formatting ──────────────────────────────────────────


class TestOpenAIFormatting:
    def test_format_with_system(self) -> None:
        from march.llm.openai_provider import OpenAIProvider

        config = {"model": "gpt-4o", "api_key": "test"}
        provider = OpenAIProvider.from_config(config)

        messages = [{"role": "user", "content": "Hello"}]
        oai_msgs = provider._format_messages(messages, system="You are helpful.")

        assert oai_msgs[0]["role"] == "system"
        assert oai_msgs[0]["content"] == "You are helpful."
        assert oai_msgs[1]["role"] == "user"

    def test_format_reasoning_model_uses_developer(self) -> None:
        from march.llm.openai_provider import OpenAIProvider

        config = {"model": "o3", "api_key": "test"}
        provider = OpenAIProvider.from_config(config)

        messages = [{"role": "user", "content": "Hello"}]
        oai_msgs = provider._format_messages(messages, system="You are helpful.")

        assert oai_msgs[0]["role"] == "developer"

    def test_format_tool_results(self) -> None:
        from march.llm.openai_provider import OpenAIProvider

        config = {"model": "gpt-4o", "api_key": "test"}
        provider = OpenAIProvider.from_config(config)

        messages = [
            {
                "role": "user",
                "content": [
                    {"id": "tc1", "content": "result data"}
                ],
            },
        ]
        oai_msgs = provider._format_messages(messages)

        assert oai_msgs[0]["role"] == "tool"
        assert oai_msgs[0]["tool_call_id"] == "tc1"
        assert oai_msgs[0]["content"] == "result data"


# ─── Test Anthropic Message Formatting ───────────────────────────────────────


class TestAnthropicFormatting:
    def test_format_tool_use_and_result(self) -> None:
        from march.llm.anthropic_provider import AnthropicProvider

        config = {"model": "claude-sonnet-4-20250514", "api_key": "test"}
        provider = AnthropicProvider.from_config(config)

        messages = [
            {"role": "user", "content": "Read a file"},
            {
                "role": "assistant",
                "content": "I'll read it.",
                "tool_calls": [
                    {"id": "tc1", "name": "read", "arguments": {"path": "/x"}}
                ],
            },
            {
                "role": "user",
                "content": [{"id": "tc1", "content": "file data"}],
            },
        ]
        anthropic_msgs = provider._format_messages(messages)

        assert len(anthropic_msgs) == 3

        # Check assistant has both text and tool_use blocks
        assistant = anthropic_msgs[1]
        assert assistant["role"] == "assistant"
        block_types = [b["type"] for b in assistant["content"]]
        assert "text" in block_types
        assert "tool_use" in block_types

        # Check user has tool_result
        user = anthropic_msgs[2]
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "tool_result"
        assert user["content"][0]["tool_use_id"] == "tc1"

    def test_format_tool_error_result(self) -> None:
        from march.llm.anthropic_provider import AnthropicProvider

        config = {"model": "claude-sonnet-4-20250514", "api_key": "test"}
        provider = AnthropicProvider.from_config(config)

        messages = [
            {
                "role": "user",
                "content": [{"id": "tc1", "error": "File not found"}],
            },
        ]
        anthropic_msgs = provider._format_messages(messages)

        block = anthropic_msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["is_error"] is True
        assert "File not found" in block["content"]


# ─── Test Ollama Message Formatting ──────────────────────────────────────────


class TestOllamaFormatting:
    def test_format_with_system(self) -> None:
        from march.llm.ollama import OllamaProvider

        config = {"model": "llama3.1", "url": "http://localhost:11434"}
        provider = OllamaProvider.from_config(config)

        messages = [{"role": "user", "content": "Hello"}]
        ollama_msgs = provider._format_messages(messages, system="Be helpful.")

        assert ollama_msgs[0]["role"] == "system"
        assert ollama_msgs[0]["content"] == "Be helpful."
        assert ollama_msgs[1]["role"] == "user"

    def test_format_tool_results_as_tool_role(self) -> None:
        from march.llm.ollama import OllamaProvider

        config = {"model": "llama3.1", "url": "http://localhost:11434"}
        provider = OllamaProvider.from_config(config)

        messages = [
            {
                "role": "user",
                "content": [{"id": "tc1", "content": "result data"}],
            },
        ]
        ollama_msgs = provider._format_messages(messages)

        assert ollama_msgs[0]["role"] == "tool"
        assert ollama_msgs[0]["content"] == "result data"


# ─── Test LiteLLM Message Formatting ─────────────────────────────────────────


class TestLiteLLMFormatting:
    def test_format_basic(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        formatted = provider._format_messages(messages, system="System prompt")

        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"


# ─── Test OpenRouter Message Formatting ──────────────────────────────────────


class TestOpenRouterFormatting:
    def test_format_basic(self) -> None:
        from march.llm.openrouter import OpenRouterProvider

        config = {"model": "test", "api_key": "key"}
        provider = OpenRouterProvider.from_config(config)

        messages = [{"role": "user", "content": "Hello"}]
        formatted = provider._format_messages(messages, system="Be helpful")

        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"


# ─── Test Bedrock Response Parsing ───────────────────────────────────────────


class TestBedrockResponseParsing:
    def test_parse_text_response(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        raw_response = {
            "output": {
                "message": {
                    "content": [{"text": "Hello world!"}]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20},
            "stopReason": "end_turn",
        }

        response = provider._parse_response(raw_response, 150.0)
        assert response.content == "Hello world!"
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 20
        assert response.stop_reason == "end_turn"
        assert response.duration_ms == 150.0

    def test_parse_tool_use_response(self) -> None:
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        raw_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "I'll read that file."},
                        {
                            "toolUse": {
                                "toolUseId": "tc-123",
                                "name": "read_file",
                                "input": {"path": "/tmp/test.txt"},
                            }
                        },
                    ]
                }
            },
            "usage": {"inputTokens": 200, "outputTokens": 50},
            "stopReason": "tool_use",
        }

        response = provider._parse_response(raw_response, 200.0)
        assert "read that file" in response.content
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
        assert response.stop_reason == "tool_use"


# ─── Test Ollama Response Parsing ────────────────────────────────────────────


class TestOllamaResponseParsing:
    def test_parse_text_response(self) -> None:
        from march.llm.ollama import OllamaProvider

        config = {"model": "llama3.1", "url": "http://localhost:11434"}
        provider = OllamaProvider.from_config(config)

        raw_data = {
            "message": {"content": "Hello from Ollama!"},
            "done": True,
            "prompt_eval_count": 50,
            "eval_count": 15,
        }

        response = provider._parse_response(raw_data, 500.0)
        assert response.content == "Hello from Ollama!"
        assert response.usage.input_tokens == 50
        assert response.usage.output_tokens == 15
        assert response.stop_reason == "stop"

    def test_parse_tool_call_response(self) -> None:
        from march.llm.ollama import OllamaProvider

        config = {"model": "llama3.1", "url": "http://localhost:11434"}
        provider = OllamaProvider.from_config(config)

        raw_data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test"},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 80,
            "eval_count": 30,
        }

        response = provider._parse_response(raw_data, 300.0)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "test"}


# ─── Test LiteLLM Response Parsing ───────────────────────────────────────────


class TestLiteLLMResponseParsing:
    def test_parse_response(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        raw_data = {
            "choices": [
                {
                    "message": {
                        "content": "LiteLLM response",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/x"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

        response = provider._parse_response(raw_data, 100.0)
        assert response.content == "LiteLLM response"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.usage.input_tokens == 100


# ─── Test OpenRouter Response Parsing ────────────────────────────────────────


class TestOpenRouterResponseParsing:
    def test_parse_response(self) -> None:
        from march.llm.openrouter import OpenRouterProvider

        config = {"model": "test", "api_key": "key"}
        provider = OpenRouterProvider.from_config(config)

        raw_data = {
            "choices": [
                {
                    "message": {"content": "OpenRouter response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20},
        }

        response = provider._parse_response(raw_data, 200.0)
        assert response.content == "OpenRouter response"
        assert response.stop_reason == "stop"
        assert response.usage.input_tokens == 80


# ─── Test Streaming Chunking ────────────────────────────────────────────────


class TestStreamingChunking:
    @pytest.mark.asyncio
    async def test_stream_text_chunks(self) -> None:
        provider = MockProvider(name="test", response_content="hello beautiful world")
        chunks: list[StreamChunk] = []
        async for chunk in provider.converse_stream(SAMPLE_MESSAGES):
            chunks.append(chunk)

        # Should have text chunks + final
        text_chunks = [c for c in chunks if c.delta_text]
        final_chunks = [c for c in chunks if c.is_final]

        assert len(text_chunks) == 3  # "hello ", "beautiful ", "world "
        assert len(final_chunks) >= 1
        assert final_chunks[-1].usage is not None

    @pytest.mark.asyncio
    async def test_stream_accumulate_text(self) -> None:
        provider = MockProvider(name="test", response_content="one two three")
        text = ""
        async for chunk in provider.converse_stream(SAMPLE_MESSAGES):
            text += chunk.delta_text

        assert "one" in text
        assert "two" in text
        assert "three" in text


# ─── Test Router Converse with Preference ────────────────────────────────────


class TestRouterConversePreference:
    @pytest.mark.asyncio
    async def test_converse_with_preference(self) -> None:
        providers = {
            "primary": MockProvider(name="primary", response_content="primary"),
            "secondary": MockProvider(name="secondary", response_content="secondary"),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        response = await router.converse(
            messages=SAMPLE_MESSAGES, preference="secondary"
        )
        assert response.content == "secondary"
        assert providers["secondary"].call_count == 1
        assert providers["primary"].call_count == 0

    @pytest.mark.asyncio
    async def test_converse_stream_with_preference(self) -> None:
        providers = {
            "primary": MockProvider(name="primary", response_content="primary"),
            "secondary": MockProvider(name="secondary", response_content="secondary words"),
        }
        config = RouterConfig(fallback_chain=["primary", "secondary"])
        router = LLMRouter(config=config, providers=providers)

        chunks: list[StreamChunk] = []
        async for chunk in router.converse_stream(
            messages=SAMPLE_MESSAGES, preference="secondary"
        ):
            chunks.append(chunk)

        text = "".join(c.delta_text for c in chunks if c.delta_text)
        assert "secondary" in text


# ─── Test Error Handling in LiteLLM/OpenRouter ───────────────────────────────


class TestHTTPProviderErrors:
    def test_litellm_auth_error(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        with pytest.raises(AuthenticationError):
            provider._handle_error(401, "Unauthorized")

    def test_litellm_rate_limit(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        with pytest.raises(RateLimitError):
            provider._handle_error(429, "Too many requests")

    def test_litellm_context_length(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        with pytest.raises(ContextLengthError):
            provider._handle_error(400, "context length exceeded")

    def test_litellm_server_error_retryable(self) -> None:
        from march.llm.litellm_provider import LiteLLMProvider

        config = {"model": "test", "url": "http://localhost:4000"}
        provider = LiteLLMProvider.from_config(config)

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_error(500, "Internal server error")
        assert exc_info.value.retryable is True

    def test_openrouter_rate_limit_with_retry_after(self) -> None:
        from march.llm.openrouter import OpenRouterProvider

        config = {"model": "test", "api_key": "key"}
        provider = OpenRouterProvider.from_config(config)

        body = json.dumps({
            "error": {
                "message": "Rate limited",
                "metadata": {"retry_after": 5.0},
            }
        })

        with pytest.raises(RateLimitError) as exc_info:
            provider._handle_error(429, body)
        assert exc_info.value.retry_after == 5.0


# ─── Test Bedrock Error Handling ─────────────────────────────────────────────


class TestBedrockErrors:
    def test_throttling_raises_rate_limit(self) -> None:
        from botocore.exceptions import ClientError
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Too fast"}},
            "Converse",
        )
        with pytest.raises(RateLimitError):
            provider._handle_client_error(error)

    def test_access_denied_raises_auth(self) -> None:
        from botocore.exceptions import ClientError
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "No access"}},
            "Converse",
        )
        with pytest.raises(AuthenticationError):
            provider._handle_client_error(error)

    def test_internal_error_retryable(self) -> None:
        from botocore.exceptions import ClientError
        from march.llm.bedrock import BedrockProvider

        config = {"model": "test", "region": "us-west-2"}
        provider = BedrockProvider.from_config(config)

        error = ClientError(
            {
                "Error": {
                    "Code": "InternalServerException",
                    "Message": "Internal error",
                }
            },
            "Converse",
        )
        with pytest.raises(ProviderError) as exc_info:
            provider._handle_client_error(error)
        assert exc_info.value.retryable is True


# ─── Test OpenAI Error Handling ──────────────────────────────────────────────


class TestOpenAIErrors:
    def test_openai_reasoning_model_detection(self) -> None:
        from march.llm.openai_provider import _is_reasoning_model

        assert _is_reasoning_model("o1") is True
        assert _is_reasoning_model("o1-preview") is True
        assert _is_reasoning_model("o1-mini") is True
        assert _is_reasoning_model("o3") is True
        assert _is_reasoning_model("o3-mini") is True
        assert _is_reasoning_model("o4-mini") is True
        assert _is_reasoning_model("gpt-4o") is False
        assert _is_reasoning_model("gpt-4") is False
        assert _is_reasoning_model("claude-3") is False


# ─── Test Router Health Report Detail ────────────────────────────────────────


class TestRouterHealthDetail:
    def test_health_report_includes_cooldown(self) -> None:
        config = RouterConfig(
            fallback_chain=["test"],
            max_failures_before_down=1,
            cooldown_seconds=60.0,
        )
        router = LLMRouter(
            config=config, providers={"test": MockProvider(name="test")}
        )

        router.record_failure("test")
        health = router.get_health()

        assert health["test"]["healthy"] is False
        assert "cooldown_remaining_seconds" in health["test"]
        assert health["test"]["cooldown_remaining_seconds"] > 0

    def test_health_report_no_cooldown_for_healthy(self) -> None:
        config = RouterConfig(fallback_chain=["test"])
        router = LLMRouter(
            config=config, providers={"test": MockProvider(name="test")}
        )

        router.record_success("test")
        health = router.get_health()

        assert health["test"]["healthy"] is True
        assert "cooldown_remaining_seconds" not in health["test"]


# ─── Integration-style: Multiple Provider Fallback ───────────────────────────


class TestMultiProviderIntegration:
    @pytest.mark.asyncio
    async def test_three_provider_chain(self) -> None:
        """Test realistic 3-provider fallback: primary → secondary → tertiary."""
        providers = {
            "primary": MockProvider(
                name="primary", should_fail=True, fail_retryable=True
            ),
            "secondary": MockProvider(
                name="secondary", should_fail=True, fail_retryable=True
            ),
            "tertiary": MockProvider(
                name="tertiary", response_content="tertiary saved the day"
            ),
        }
        config = RouterConfig(
            fallback_chain=["primary", "secondary", "tertiary"]
        )
        router = LLMRouter(config=config, providers=providers)

        response = await router.converse(messages=SAMPLE_MESSAGES)
        assert response.content == "tertiary saved the day"
        assert providers["primary"].call_count == 1
        assert providers["secondary"].call_count == 1
        assert providers["tertiary"].call_count == 1

    @pytest.mark.asyncio
    async def test_health_tracking_across_calls(self) -> None:
        """Verify health tracking persists across multiple converse calls."""
        fail_provider = MockProvider(
            name="unreliable", should_fail=True, fail_retryable=True
        )
        good_provider = MockProvider(
            name="reliable", response_content="ok"
        )
        providers = {
            "unreliable": fail_provider,
            "reliable": good_provider,
        }
        config = RouterConfig(
            fallback_chain=["unreliable", "reliable"],
            max_failures_before_down=2,
        )
        router = LLMRouter(config=config, providers=providers)

        # First call: unreliable fails, falls through to reliable
        await router.converse(messages=SAMPLE_MESSAGES)
        assert router._health["unreliable"].consecutive_failures == 1

        # Second call: unreliable fails again, now marked as down
        await router.converse(messages=SAMPLE_MESSAGES)
        assert router._health["unreliable"].consecutive_failures == 2
        assert router._health["unreliable"].healthy is False

        # Third call: unreliable is skipped entirely (marked down)
        fail_provider.call_count = 0
        good_provider.call_count = 0
        await router.converse(messages=SAMPLE_MESSAGES)
        assert fail_provider.call_count == 0  # Skipped!
        assert good_provider.call_count == 1
