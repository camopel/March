"""LLM Router — smart model selection with fallback chains and health tracking."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from march.llm.base import LLMProvider, LLMResponse, ProviderError, StreamChunk, ToolDefinition

logger = logging.getLogger(__name__)


class NoProviderAvailable(ProviderError):
    """Raised when all providers in the fallback chain are unavailable."""

    def __init__(self, message: str = "All providers are down or unconfigured"):
        super().__init__(message, provider="router", retryable=False)


@dataclass
class ProviderHealth:
    """Health tracking state for a single provider."""

    name: str
    healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0


@dataclass
class RouterConfig:
    """Configuration for the LLM router."""

    fallback_chain: list[str] = field(default_factory=list)
    max_failures_before_down: int = 3
    cooldown_seconds: float = 60.0
    cost_aware: bool = False
    default_provider: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterConfig":
        return cls(
            fallback_chain=data.get("fallback_chain", []),
            max_failures_before_down=data.get("max_failures_before_down", 3),
            cooldown_seconds=data.get("cooldown_seconds", 60.0),
            cost_aware=data.get("cost_aware", False),
            default_provider=data.get("default", ""),
        )


class LLMRouter:
    """Smart model selection with fallback chains, health tracking, and cost-aware routing.

    The router maintains health state for each provider and routes requests
    to the best available provider based on the configured fallback chain.

    Features:
    - Fallback chain: try providers in configured order
    - Health checking: mark providers as down after N consecutive failures
    - Auto-recovery: re-enable providers after cooldown period
    - Cost-aware routing: optionally route to cheapest healthy provider
    - Explicit provider selection: bypass routing with a specific provider name
    """

    def __init__(
        self,
        config: RouterConfig,
        providers: dict[str, LLMProvider],
    ) -> None:
        self.config = config
        self.providers = providers
        self._health: dict[str, ProviderHealth] = {
            name: ProviderHealth(name=name) for name in providers
        }
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(
        cls, config: dict[str, Any], providers: dict[str, LLMProvider]
    ) -> "LLMRouter":
        router_config = RouterConfig.from_dict(config)

        # If no fallback chain configured, use default provider first, then all others
        if not router_config.fallback_chain:
            chain = []
            if router_config.default_provider and router_config.default_provider in providers:
                chain.append(router_config.default_provider)
            for name in providers:
                if name not in chain:
                    chain.append(name)
            router_config.fallback_chain = chain

        return cls(config=router_config, providers=providers)

    def _is_provider_available(self, name: str) -> bool:
        """Check if a provider should be considered available.

        A provider is available if:
        - It's marked healthy, OR
        - It's been down long enough that the cooldown has expired (auto-recovery)
        """
        health = self._health.get(name)
        if health is None:
            return False

        if health.healthy:
            return True

        # Check cooldown — allow retry if enough time has passed
        if health.last_failure_time > 0:
            elapsed = time.monotonic() - health.last_failure_time
            if elapsed >= self.config.cooldown_seconds:
                logger.info(
                    "Provider %s cooldown expired (%.0fs), re-enabling for attempt",
                    name, elapsed,
                )
                return True

        return False

    def record_success(self, name: str) -> None:
        """Record a successful call to a provider."""
        health = self._health.get(name)
        if health is None:
            return
        health.healthy = True
        health.consecutive_failures = 0
        health.last_success_time = time.monotonic()
        health.total_successes += 1

    def record_failure(self, name: str) -> None:
        """Record a failed call to a provider.

        After max_failures_before_down consecutive failures, the provider
        is marked as unhealthy and will be skipped until cooldown expires.
        """
        health = self._health.get(name)
        if health is None:
            return
        health.consecutive_failures += 1
        health.last_failure_time = time.monotonic()
        health.total_failures += 1

        if health.consecutive_failures >= self.config.max_failures_before_down:
            if health.healthy:
                logger.warning(
                    "Provider %s marked as DOWN after %d consecutive failures",
                    name, health.consecutive_failures,
                )
            health.healthy = False

    async def route(
        self,
        message: str | None = None,
        context: Any | None = None,
        preference: str | None = None,
    ) -> LLMProvider:
        """Select the best available provider.

        Args:
            message: Current user message (for future context-based routing).
            context: Current context (for future context-based routing).
            preference: Explicit provider name to use (bypasses fallback chain).

        Returns:
            The selected LLMProvider.

        Raises:
            NoProviderAvailable: If no providers are available.
        """
        # Explicit preference — try it directly
        if preference and preference in self.providers:
            provider = self.providers[preference]
            if self._is_provider_available(preference):
                return provider
            # Even if marked down, honor explicit preference (user knows best)
            logger.warning(
                "Preferred provider %s is marked down, using anyway per explicit request",
                preference,
            )
            return provider

        # Cost-aware routing: sort available providers by cost
        if self.config.cost_aware:
            available = [
                (name, self.providers[name])
                for name in self.config.fallback_chain
                if name in self.providers and self._is_provider_available(name)
            ]
            if available:
                # Sort by estimated cost (input + output price per million tokens)
                available.sort(
                    key=lambda x: x[1].input_price + x[1].output_price
                )
                return available[0][1]

        # Standard fallback chain
        for name in self.config.fallback_chain:
            provider = self.providers.get(name)
            if provider and self._is_provider_available(name):
                return provider

        # All down — try the provider with the oldest failure (most likely to have recovered)
        if self._health:
            candidates = sorted(
                self.providers.items(),
                key=lambda x: self._health.get(x[0], ProviderHealth(name=x[0])).last_failure_time,
            )
            if candidates:
                return candidates[0][1]

        raise NoProviderAvailable()

    async def converse(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        preference: str | None = None,
    ) -> LLMResponse:
        """Route a converse call through the fallback chain.

        Tries each available provider in order. On transient failures,
        moves to the next provider. Records successes and failures for
        health tracking.
        """
        errors: list[tuple[str, Exception]] = []

        # Build ordered list of providers to try
        provider_names = list(self.config.fallback_chain)
        if preference:
            # Put preferred provider first
            if preference in provider_names:
                provider_names.remove(preference)
            provider_names.insert(0, preference)

        for name in provider_names:
            provider = self.providers.get(name)
            if provider is None:
                continue
            if not self._is_provider_available(name):
                continue

            try:
                response = await provider.converse(
                    messages=messages,
                    system=system,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.record_success(name)
                return response
            except ProviderError as e:
                self.record_failure(name)
                errors.append((name, e))
                logger.warning(
                    "Provider %s failed (retryable=%s): %s",
                    name, e.retryable, e,
                )
                if not e.retryable:
                    # Non-retryable errors (auth, context length) — don't fallback
                    raise
                # Retryable — continue to next provider
                continue
            except Exception as e:
                self.record_failure(name)
                errors.append((name, e))
                logger.error("Provider %s unexpected error: %s", name, e)
                continue

        # All providers failed
        if errors:
            last_name, last_error = errors[-1]
            raise ProviderError(
                f"All providers failed. Last error from {last_name}: {last_error}",
                provider="router",
                retryable=False,
            )
        raise NoProviderAvailable()

    async def converse_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        preference: str | None = None,
    ):
        """Route a streaming converse call through the fallback chain.

        Unlike non-streaming, once streaming starts from a provider we commit
        to it (can't switch mid-stream). Fallback only happens on connection-level
        failures before streaming begins.
        """
        errors: list[tuple[str, Exception]] = []

        provider_names = list(self.config.fallback_chain)
        if preference:
            if preference in provider_names:
                provider_names.remove(preference)
            provider_names.insert(0, preference)

        for name in provider_names:
            provider = self.providers.get(name)
            if provider is None:
                continue
            if not self._is_provider_available(name):
                continue

            try:
                stream = provider.converse_stream(
                    messages=messages,
                    system=system,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Yield the first chunk to verify the stream is working
                first_chunk = None
                async for chunk in stream:
                    if first_chunk is None:
                        first_chunk = chunk
                        self.record_success(name)
                        yield chunk
                    else:
                        yield chunk
                if first_chunk is None:
                    # Empty stream — still counts as success
                    self.record_success(name)
                return
            except ProviderError as e:
                self.record_failure(name)
                errors.append((name, e))
                logger.warning(
                    "Provider %s stream failed (retryable=%s): %s",
                    name, e.retryable, e,
                )
                if not e.retryable:
                    raise
                continue
            except Exception as e:
                self.record_failure(name)
                errors.append((name, e))
                logger.error("Provider %s stream unexpected error: %s", name, e)
                continue

        if errors:
            last_name, last_error = errors[-1]
            raise ProviderError(
                f"All providers failed streaming. Last error from {last_name}: {last_error}",
                provider="router",
                retryable=False,
            )
        raise NoProviderAvailable()

    def get_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all providers."""
        result: dict[str, dict[str, Any]] = {}
        for name, health in self._health.items():
            available = self._is_provider_available(name)
            result[name] = {
                "healthy": health.healthy,
                "available": available,
                "consecutive_failures": health.consecutive_failures,
                "total_failures": health.total_failures,
                "total_successes": health.total_successes,
            }
            if health.last_failure_time > 0 and not health.healthy:
                elapsed = time.monotonic() - health.last_failure_time
                remaining = max(0, self.config.cooldown_seconds - elapsed)
                result[name]["cooldown_remaining_seconds"] = round(remaining, 1)
        return result

    def reset_health(self, name: str | None = None) -> None:
        """Reset health tracking for a provider or all providers."""
        if name:
            if name in self._health:
                self._health[name] = ProviderHealth(name=name)
        else:
            for n in self._health:
                self._health[n] = ProviderHealth(name=n)

    async def check_all_health(self) -> dict[str, bool]:
        """Run health checks on all providers and update state."""
        results: dict[str, bool] = {}

        async def _check(provider_name: str, provider: LLMProvider) -> tuple[str, bool]:
            try:
                healthy = await provider.health_check()
                return provider_name, healthy
            except Exception:
                return provider_name, False

        tasks = [_check(name, p) for name, p in self.providers.items()]
        check_results = await asyncio.gather(*tasks)

        for provider_name, healthy in check_results:
            results[provider_name] = healthy
            if healthy:
                self.record_success(provider_name)
            else:
                self.record_failure(provider_name)

        return results
