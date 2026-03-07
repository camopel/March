"""SER Adapter — Semantic Execution Runtime interface (stub).

This is a stub for future SER integration. All methods return None/False
to indicate SER is not yet implemented.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from march.core.context import Context
    from march.core.message import Message


class SERAdapter:
    """Interface for Semantic Execution Runtime.

    Reduces token volume via local models + structured reasoning.
    Currently a stub — all methods pass through.
    """

    enabled: bool = False

    async def should_compress(self, context: "Context") -> bool:
        """True if context exceeds token threshold and should be compressed.

        Returns:
            Always False in stub implementation.
        """
        return False

    async def compress_context(self, context: "Context") -> "Context":
        """Compress context using SER's stack-based reasoning.

        Returns:
            The context unchanged in stub implementation.
        """
        return context

    async def route_to_local(self, message: str, context: "Context") -> str | None:
        """Route simple queries to a local model.

        Returns:
            None (SER not implemented yet — always use cloud).
        """
        return None

    async def extract_for_memory(self, conversation: list["Message"]) -> dict[str, Any]:
        """Extract entities/decisions from conversation for structured memory.

        Returns:
            Empty dict in stub implementation.
        """
        return {}
