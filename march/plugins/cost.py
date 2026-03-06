"""CostPlugin — Track token usage and cost per session/day.

Monitors LLM calls, tracks cumulative costs, and alerts when approaching
budget limits.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from march.logging import get_logger
from march.plugins._base import Plugin

if TYPE_CHECKING:
    from march.core.context import Context
    from march.llm.base import LLMResponse

logger = get_logger("march.plugins.cost")


@dataclass
class CostRecord:
    """A single cost entry for an LLM call."""

    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_ms: float


class CostPlugin(Plugin):
    """Track token usage and cost per session/day with budget limits.

    Attributes:
        budget_per_session: Max cost (USD) per session. 0 = unlimited.
        budget_per_day: Max cost (USD) per day. 0 = unlimited.
        alert_threshold: Fraction (0-1) at which to alert (default 0.8 = 80%).
    """

    name = "cost"
    version = "0.1.0"
    priority = 90  # Runs late in the pipeline

    def __init__(
        self,
        budget_per_session: float = 5.00,
        budget_per_day: float = 20.00,
        alert_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.budget_per_session = budget_per_session
        self.budget_per_day = budget_per_day
        self.alert_threshold = alert_threshold

        # Tracking state
        self._session_records: list[CostRecord] = []
        self._daily_records: dict[str, list[CostRecord]] = defaultdict(list)

    async def on_start(self, app: Any) -> None:
        """Load config from app.config.plugins.cost if available."""
        cfg = getattr(getattr(app, "config", None), "plugins", None)
        if cfg:
            cost_cfg = getattr(cfg, "cost", None)
            if cost_cfg:
                self.budget_per_session = getattr(cost_cfg, "budget_per_session", self.budget_per_session)
                self.budget_per_day = getattr(cost_cfg, "budget_per_day", self.budget_per_day)
                self.alert_threshold = getattr(cost_cfg, "alert_threshold", self.alert_threshold)

    async def after_llm(self, context: "Context", response: "LLMResponse") -> "LLMResponse":
        """Track token usage and cost after each LLM call."""
        record = CostRecord(
            timestamp=time.time(),
            provider=getattr(response, "provider", "unknown"),
            model=getattr(response, "model", "unknown"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=response.usage.cost,
            duration_ms=response.duration_ms,
        )

        self._session_records.append(record)
        today = time.strftime("%Y-%m-%d")
        self._daily_records[today].append(record)

        # Prune daily records older than 7 days
        if len(self._daily_records) > 7:
            sorted_days = sorted(self._daily_records.keys())
            for old_day in sorted_days[:-7]:
                del self._daily_records[old_day]

        logger.debug(
            "cost.track provider=%s model=%s in=%d out=%d cost=$%.6f session_total=$%.4f",
            record.provider,
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.cost,
            self.session_cost,
        )

        return response

    async def on_response(self, response: Any) -> Any:
        """Add cost warning footer if over threshold."""
        if not isinstance(response, str):
            return response

        warnings: list[str] = []

        # Check session budget
        if self.budget_per_session > 0:
            ratio = self.session_cost / self.budget_per_session
            if ratio >= 1.0:
                warnings.append(
                    f"⚠️ Session budget exceeded: ${self.session_cost:.4f} / ${self.budget_per_session:.2f}"
                )
            elif ratio >= self.alert_threshold:
                warnings.append(
                    f"💰 Session cost: ${self.session_cost:.4f} / ${self.budget_per_session:.2f} ({ratio:.0%})"
                )

        # Check daily budget
        if self.budget_per_day > 0:
            daily = self.daily_cost
            ratio = daily / self.budget_per_day
            if ratio >= 1.0:
                warnings.append(
                    f"⚠️ Daily budget exceeded: ${daily:.4f} / ${self.budget_per_day:.2f}"
                )
            elif ratio >= self.alert_threshold:
                warnings.append(
                    f"💰 Daily cost: ${daily:.4f} / ${self.budget_per_day:.2f} ({ratio:.0%})"
                )

        if warnings:
            footer = "\n\n---\n" + "\n".join(warnings)
            return response + footer

        return response

    @property
    def session_cost(self) -> float:
        """Total cost for the current session."""
        return sum(r.cost for r in self._session_records)

    @property
    def session_tokens(self) -> int:
        """Total tokens for the current session."""
        return sum(r.input_tokens + r.output_tokens for r in self._session_records)

    @property
    def daily_cost(self) -> float:
        """Total cost for today."""
        today = time.strftime("%Y-%m-%d")
        return sum(r.cost for r in self._daily_records.get(today, []))

    @property
    def daily_tokens(self) -> int:
        """Total tokens for today."""
        today = time.strftime("%Y-%m-%d")
        return sum(
            r.input_tokens + r.output_tokens for r in self._daily_records.get(today, [])
        )

    @property
    def session_records(self) -> list[CostRecord]:
        """Get all session cost records."""
        return list(self._session_records)

    def summary(self) -> dict[str, Any]:
        """Get a cost summary dict."""
        return {
            "session_cost": self.session_cost,
            "session_tokens": self.session_tokens,
            "daily_cost": self.daily_cost,
            "daily_tokens": self.daily_tokens,
            "session_calls": len(self._session_records),
            "budget_per_session": self.budget_per_session,
            "budget_per_day": self.budget_per_day,
        }

    def reset_session(self) -> None:
        """Reset session cost tracking."""
        self._session_records.clear()
