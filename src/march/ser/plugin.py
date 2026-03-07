"""SER Plugin — Hooks SER into the before_llm pipeline (stub).

Currently passes through — SER is not yet implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from march.logging import get_logger
from march.plugins._base import Plugin
from march.ser.adapter import SERAdapter

if TYPE_CHECKING:
    from march.core.context import Context

logger = get_logger("march.plugins.ser")


class SERPlugin(Plugin):
    """SER integration plugin.

    Hooks into before_llm to route to local models or compress context.
    Currently a passthrough stub — SER is not yet implemented.
    """

    name = "ser"
    version = "0.1.0"
    priority = 10  # Runs early — before most plugins

    def __init__(self, adapter: SERAdapter | None = None) -> None:
        super().__init__()
        self.ser = adapter or SERAdapter()

    async def before_llm(
        self, context: "Context", message: str
    ) -> tuple["Context", str] | tuple["Context", str, str]:
        """Try local routing, then compress if needed.

        Currently passes through since SER is not implemented.
        """
        if not self.ser.enabled:
            return context, message

        # Try local model first
        local_response = await self.ser.route_to_local(message, context)
        if local_response:
            logger.info("ser.local_route message=%s", message[:100])
            return context, message, local_response

        # Compress context if needed
        if await self.ser.should_compress(context):
            context = await self.ser.compress_context(context)
            logger.info("ser.compressed")

        return context, message
