"""Send messages across channels (route through channel layer)."""

from __future__ import annotations

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.message_tool")


@tool(name="message", description="Send a message to a channel or user.")
async def message_tool(
    action: str = "send",
    target: str = "",
    message: str = "",
    channel: str = "",
    reply_to: str = "",
) -> str:
    """Send messages across channels.

    Args:
        action: Action: send, read, list.
        target: Target channel ID or user.
        message: Message text to send.
        channel: Channel type (matrix, terminal, etc).
        reply_to: Message ID to reply to.
    """
    if action == "send":
        if not message:
            return "Error: message is required for send action"
        if not target:
            return "Error: target is required for send action"

        # In the full framework, this routes through the channel layer.
        # For now, log the intent and return success.
        logger.info(f"Message send: target={target}, channel={channel}, len={len(message)}")
        return (
            f"Message queued for delivery.\n"
            f"Target: {target}\n"
            f"Channel: {channel or 'default'}\n"
            f"Length: {len(message)} chars\n\n"
            f"Note: Full delivery requires channel layer integration."
        )

    elif action == "read":
        return (
            "Message read requires channel layer integration. "
            "Use the channel's native read capabilities."
        )

    elif action == "list":
        return (
            "Message list requires channel layer integration. "
            "Available channels depend on configuration."
        )

    else:
        return f"Error: Unknown action '{action}'. Use: send, read, list"
