"""Push-based completion announcer for March sub-agents.

When a child finishes, reads the child output and delivers the result
to the parent session using one of three strategies (in priority order):
  1. Steer: Inject into parent's current active turn
  2. Queue: Deliver after parent's current turn ends
  3. Direct: Start a new agent turn if parent is idle

This is push-based — the parent never polls for sub-agent completion.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from march.agents.registry import RunRecord, RunOutcome
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.session import SessionStore

logger = get_logger("march.announce")


class SubagentAnnouncer:
    """Announces sub-agent completion to parent sessions.

    Delivery strategies (tried in order):
    1. Steer — inject result into parent's active turn
    2. Queue — deliver when parent's current turn ends
    3. Direct — start a new agent turn with the result

    The announcer is connected to session infrastructure via callbacks
    to avoid circular dependencies.
    """

    def __init__(
        self,
        read_child_output: Callable[[str], Awaitable[str]] | None = None,
        try_steer: Callable[[str, str], Awaitable[bool]] | None = None,
        try_queue: Callable[[str, str], Awaitable[bool]] | None = None,
        send_direct: Callable[[str, str, str], Awaitable[None]] | None = None,
        delete_session: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._read_child_output = read_child_output
        self._try_steer = try_steer
        self._try_queue = try_queue
        self._send_direct = send_direct
        self._delete_session = delete_session
        self._pending_queue: dict[str, list[str]] = {}  # requester_key → queued messages

    async def announce_completion(self, record: RunRecord, outcome: RunOutcome) -> bool:
        """Push result back to parent session.

        Args:
            record: The run record of the completed sub-agent.
            outcome: The outcome of the run.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        # Read child's final output
        output = ""
        if self._read_child_output:
            try:
                output = await self._read_child_output(record.child_key)
            except Exception as e:
                logger.warning("announce: failed to read child output: %s", e)
                output = f"(failed to read output: {e})"

        # Build completion message
        status_header = {
            "ok": f"✅ Subagent `{record.child_key}` finished",
            "error": f"❌ Subagent `{record.child_key}` failed",
            "timeout": f"⏱️ Subagent `{record.child_key}` timed out",
            "cancelled": f"🚫 Subagent `{record.child_key}` was cancelled",
        }.get(outcome.status, f"Subagent `{record.child_key}` ended ({outcome.status})")

        parts = [status_header]
        if outcome.error:
            parts.append(f"\n**Error:** {outcome.error}")
        if output:
            parts.append(f"\n{output}")
        elif outcome.output:
            parts.append(f"\n{outcome.output}")

        message = "\n".join(parts)

        # Try delivery strategies in order
        delivered = False

        # 1. Try steer (inject into active turn)
        if self._try_steer:
            try:
                delivered = await self._try_steer(record.requester_key, message)
                if delivered:
                    logger.info("announce: delivered via steer to %s", record.requester_key)
            except Exception as e:
                logger.warning("announce: steer failed: %s", e)

        # 2. Try queue (deliver after current turn)
        if not delivered and self._try_queue:
            try:
                delivered = await self._try_queue(record.requester_key, message)
                if delivered:
                    logger.info("announce: queued for %s", record.requester_key)
            except Exception as e:
                logger.warning("announce: queue failed: %s", e)

        # 3. Direct delivery (new agent turn)
        if not delivered and self._send_direct:
            try:
                await self._send_direct(
                    record.requester_key, record.requester_origin, message
                )
                delivered = True
                logger.info("announce: direct delivery to %s", record.requester_key)
            except Exception as e:
                logger.warning("announce: direct delivery failed: %s", e)

        # 4. Fallback: store in pending queue
        if not delivered:
            if record.requester_key not in self._pending_queue:
                self._pending_queue[record.requester_key] = []
            self._pending_queue[record.requester_key].append(message)
            logger.warning(
                "announce: all delivery methods failed for %s, queued locally",
                record.requester_key,
            )

        # Sub-agent sessions are NOT deleted on completion.
        # They persist until the parent session does /reset,
        # so the parent can reference sub-agent context/history.
        # Cleanup is handled by AgentManager.reset_children().

        return delivered

    def get_pending(self, requester_key: str) -> list[str]:
        """Get and clear pending announcements for a requester.

        Called when a requester starts a new turn to deliver any
        queued announcements that couldn't be delivered earlier.
        """
        return self._pending_queue.pop(requester_key, [])

    @property
    def pending_count(self) -> int:
        """Total number of pending announcements across all requesters."""
        return sum(len(msgs) for msgs in self._pending_queue.values())
