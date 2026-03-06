"""Extended tests for the LLM router: fallback chains, health tracking, cost-aware routing, streaming."""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from march.llm.base import LLMProvider, LLMResponse, LLMUsage, ProviderError, StreamChunk
from march.llm.base import ToolCall as LLMToolCall
from march.llm.router import LLMRouter, RouterConfig, NoProviderAvailable, ProviderHealth


# ── Mock Providers ──

class HealthyProvider(LLMProvider):
    name = "healthy"
    model = "healthy-model"
    input_price = 1.0
    output_price = 2.0

    async def converse(self, messages, system=None, tools=None, **kw):
        return LLMResponse(content="healthy response", usage=LLMUsage(input_tokens=10, output_tokens=5))

    async def converse_stream(self, messages, system=None, tools=None, **kw):
        yield StreamChunk(delta="healthy ")
        yield StreamChunk(delta="stream", finish_reason="stop")


class FailingProvider(LLMProvider):
    name = "failing"
    model = "failing-model"
    input_price = 0.5
    output_price = 1.0

    def __init__(self, retryable=True):
        self._retryable = retryable

    async def converse(self, messages, system=None, tools=None, **kw):
        raise ProviderError("Provider is down", provider="failing", retryable=self._retryable)

    async def converse_stream(self, messages, system=None, tools=None, **kw):
        raise ProviderError("Provider stream down", provider="failing", retryable=self._retryable)
        yield  # make it a generator


class CheapProvider(LLMProvider):
    name = "cheap"
    model = "cheap-model"
    input_price = 0.1
    output_price = 0.2

    async def converse(self, messages, system=None, tools=None, **kw):
        return LLMResponse(content="cheap response", usage=LLMUsage(input_tokens=10, output_tokens=5))

    async def converse_stream(self, messages, system=None, tools=None, **kw):
        yield StreamChunk(delta="cheap")
        yield StreamChunk(finish_reason="stop")


class EmptyStreamProvider(LLMProvider):
    name = "empty_stream"
    model = "empty-model"

    async def converse(self, messages, **kw):
        return LLMResponse(content="", usage=LLMUsage())

    async def converse_stream(self, messages, **kw):
        # Empty stream — yields nothing
        return
        yield  # make it a generator


class UnexpectedErrorProvider(LLMProvider):
    name = "unexpected"
    model = "err-model"

    async def converse(self, messages, **kw):
        raise ValueError("Unexpected error!")

    async def converse_stream(self, messages, **kw):
        raise ValueError("Unexpected stream error!")
        yield


# ── Router Config ──

class TestRouterConfig:
    def test_from_dict(self):
        data = {
            "fallback_chain": ["a", "b"],
            "max_failures_before_down": 5,
            "cooldown_seconds": 120.0,
            "cost_aware": True,
            "default": "a",
        }
        config = RouterConfig.from_dict(data)
        assert config.fallback_chain == ["a", "b"]
        assert config.max_failures_before_down == 5
        assert config.cooldown_seconds == 120.0
        assert config.cost_aware
        assert config.default_provider == "a"

    def test_from_dict_defaults(self):
        config = RouterConfig.from_dict({})
        assert config.fallback_chain == []
        assert config.max_failures_before_down == 3
        assert config.cooldown_seconds == 60.0
        assert not config.cost_aware

    def test_from_config_auto_chain(self):
        providers = {"alpha": HealthyProvider(), "beta": CheapProvider()}
        router = LLMRouter.from_config({"default": "alpha"}, providers)
        assert router.config.fallback_chain[0] == "alpha"
        assert "beta" in router.config.fallback_chain

    def test_from_config_no_default(self):
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter.from_config({}, providers)
        assert len(router.config.fallback_chain) == 2


# ── Health Tracking ──

class TestRouterHealth:
    def test_record_success(self):
        config = RouterConfig(fallback_chain=["a"])
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        router.record_success("a")
        health = router._health["a"]
        assert health.healthy
        assert health.consecutive_failures == 0
        assert health.total_successes == 1

    def test_record_failure_marks_down(self):
        config = RouterConfig(fallback_chain=["a"], max_failures_before_down=2)
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        router.record_failure("a")
        assert router._health["a"].healthy  # 1 failure, threshold is 2
        router.record_failure("a")
        assert not router._health["a"].healthy  # 2 failures → down

    def test_cooldown_recovery(self):
        config = RouterConfig(fallback_chain=["a"], max_failures_before_down=1, cooldown_seconds=0.01)
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        router.record_failure("a")
        assert not router._health["a"].healthy

        # Wait for cooldown
        time.sleep(0.02)
        assert router._is_provider_available("a")  # Cooldown expired

    def test_nonexistent_provider_not_available(self):
        config = RouterConfig(fallback_chain=["a"])
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        assert not router._is_provider_available("nonexistent")

    def test_record_success_nonexistent(self):
        config = RouterConfig(fallback_chain=["a"])
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        router.record_success("nonexistent")  # Should not crash

    def test_record_failure_nonexistent(self):
        config = RouterConfig(fallback_chain=["a"])
        router = LLMRouter(config=config, providers={"a": HealthyProvider()})
        router.record_failure("nonexistent")  # Should not crash

    def test_get_health(self):
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=1, cooldown_seconds=60)
        providers = {"a": HealthyProvider(), "b": FailingProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("b")
        health = router.get_health()
        assert health["a"]["healthy"]
        assert not health["b"]["healthy"]
        assert "cooldown_remaining_seconds" in health["b"]

    def test_reset_health_single(self):
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=1)
        providers = {"a": HealthyProvider(), "b": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("a")
        router.record_failure("b")
        router.reset_health("a")
        assert router._health["a"].healthy
        assert not router._health["b"].healthy

    def test_reset_health_all(self):
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=1)
        providers = {"a": HealthyProvider(), "b": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("a")
        router.record_failure("b")
        router.reset_health()
        assert router._health["a"].healthy
        assert router._health["b"].healthy


# ── Route Selection ──

class TestRouterRouting:
    async def test_route_first_available(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        provider = await router.route()
        assert provider.name == "healthy"

    async def test_route_fallback_to_second(self):
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=1)
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("a")  # Mark first as down
        provider = await router.route()
        assert provider.name == "cheap"

    async def test_route_preference(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        provider = await router.route(preference="b")
        assert provider.name == "cheap"

    async def test_route_preference_down_honored(self):
        """Even if preferred is down, honor the explicit request."""
        config = RouterConfig(fallback_chain=["a", "b"], max_failures_before_down=1)
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("b")
        provider = await router.route(preference="b")
        assert provider.name == "cheap"

    async def test_route_cost_aware(self):
        config = RouterConfig(
            fallback_chain=["expensive", "cheap"],
            cost_aware=True,
        )
        providers = {"expensive": HealthyProvider(), "cheap": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        provider = await router.route()
        assert provider.name == "cheap"

    async def test_route_all_down_last_resort(self):
        """When all providers are down, route() falls through to any registered provider."""
        config = RouterConfig(fallback_chain=["a"], max_failures_before_down=1)
        providers = {"a": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        router.record_failure("a")
        # Even though 'a' is down in fallback chain, it's still returned as last resort
        provider = await router.route()
        assert provider is not None

    async def test_route_empty_providers(self):
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        with pytest.raises(NoProviderAvailable):
            await router.route()


# ── Converse with Fallback ──

class TestRouterConverse:
    async def test_converse_success(self):
        config = RouterConfig(fallback_chain=["a"])
        providers = {"a": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        resp = await router.converse(messages=[{"role": "user", "content": "hi"}])
        assert resp.content == "healthy response"

    async def test_converse_fallback(self):
        config = RouterConfig(fallback_chain=["fail", "ok"])
        providers = {"fail": FailingProvider(), "ok": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        resp = await router.converse(messages=[{"role": "user", "content": "hi"}])
        assert resp.content == "healthy response"

    async def test_converse_all_fail(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": FailingProvider(), "b": FailingProvider()}
        router = LLMRouter(config=config, providers=providers)
        with pytest.raises(ProviderError):
            await router.converse(messages=[{"role": "user", "content": "hi"}])

    async def test_converse_non_retryable_raises_immediately(self):
        config = RouterConfig(fallback_chain=["fail", "ok"])
        providers = {"fail": FailingProvider(retryable=False), "ok": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        with pytest.raises(ProviderError):
            await router.converse(messages=[{"role": "user", "content": "hi"}])

    async def test_converse_unexpected_error_continues(self):
        config = RouterConfig(fallback_chain=["bad", "good"])
        providers = {"bad": UnexpectedErrorProvider(), "good": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        resp = await router.converse(messages=[{"role": "user", "content": "hi"}])
        assert resp.content == "healthy response"

    async def test_converse_preference(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": HealthyProvider(), "b": CheapProvider()}
        router = LLMRouter(config=config, providers=providers)
        resp = await router.converse(
            messages=[{"role": "user", "content": "hi"}],
            preference="b",
        )
        assert resp.content == "cheap response"

    async def test_converse_no_providers(self):
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        with pytest.raises(NoProviderAvailable):
            await router.converse(messages=[{"role": "user", "content": "hi"}])


# ── Streaming with Fallback ──

class TestRouterConverseStream:
    async def test_stream_success(self):
        config = RouterConfig(fallback_chain=["a"])
        providers = {"a": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        chunks = []
        async for chunk in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].delta == "healthy "

    async def test_stream_fallback(self):
        config = RouterConfig(fallback_chain=["fail", "ok"])
        providers = {"fail": FailingProvider(), "ok": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        chunks = []
        async for chunk in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert any("healthy" in (c.delta or "") for c in chunks)

    async def test_stream_all_fail(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": FailingProvider(), "b": FailingProvider()}
        router = LLMRouter(config=config, providers=providers)
        with pytest.raises(ProviderError):
            async for _ in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    async def test_stream_non_retryable(self):
        config = RouterConfig(fallback_chain=["fail", "ok"])
        providers = {"fail": FailingProvider(retryable=False), "ok": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        with pytest.raises(ProviderError):
            async for _ in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    async def test_stream_empty(self):
        config = RouterConfig(fallback_chain=["empty"])
        providers = {"empty": EmptyStreamProvider()}
        router = LLMRouter(config=config, providers=providers)
        chunks = []
        async for chunk in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert chunks == []

    async def test_stream_no_providers(self):
        config = RouterConfig(fallback_chain=[])
        router = LLMRouter(config=config, providers={})
        with pytest.raises(NoProviderAvailable):
            async for _ in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    async def test_stream_unexpected_error_fallback(self):
        config = RouterConfig(fallback_chain=["bad", "good"])
        providers = {"bad": UnexpectedErrorProvider(), "good": HealthyProvider()}
        router = LLMRouter(config=config, providers=providers)
        chunks = []
        async for chunk in router.converse_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert any("healthy" in (c.delta or "") for c in chunks)


# ── Health Check ──

class TestRouterHealthCheck:
    async def test_check_all_health(self):
        config = RouterConfig(fallback_chain=["a", "b"])
        providers = {"a": HealthyProvider(), "b": FailingProvider()}

        # Add health_check to mock providers
        providers["a"].health_check = AsyncMock(return_value=True)
        providers["b"].health_check = AsyncMock(return_value=False)

        router = LLMRouter(config=config, providers=providers)
        results = await router.check_all_health()
        assert results["a"] is True
        assert results["b"] is False
        # Health state should be updated
        assert router._health["a"].total_successes >= 1
        assert router._health["b"].total_failures >= 1

    async def test_check_all_health_exception(self):
        config = RouterConfig(fallback_chain=["a"])
        providers = {"a": HealthyProvider()}
        providers["a"].health_check = AsyncMock(side_effect=RuntimeError("check failed"))
        router = LLMRouter(config=config, providers=providers)
        results = await router.check_all_health()
        assert results["a"] is False
