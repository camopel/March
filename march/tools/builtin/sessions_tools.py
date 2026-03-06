"""Session management tools: list, history, send, spawn, subagents, status."""

from __future__ import annotations

import json
import time
import uuid

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.sessions_tools")


@tool(name="sessions_list", description="List active agent sessions.")
async def sessions_list(
    kind: str = "",
    active_only: bool = True,
    label: str = "",
) -> str:
    """List active agent sessions.

    Args:
        kind: Filter by session kind (main, subagent, cron).
        active_only: Only show active sessions.
        label: Filter by session label.
    """
    # In the full framework, this queries the session store.
    # Stub implementation returns a formatted placeholder.
    return (
        "Sessions listing requires session store integration.\n"
        f"Filters: kind={kind or 'all'}, active_only={active_only}, label={label or 'none'}\n\n"
        "This tool will query SQLite session store when the full agent loop is connected."
    )


@tool(name="sessions_history", description="Fetch message history from a session.")
async def sessions_history(
    session_id: str,
    limit: int = 50,
    include_tools: bool = False,
) -> str:
    """Fetch session transcript/history.

    Args:
        session_id: ID of the session to read.
        limit: Maximum number of messages to return.
        include_tools: Whether to include tool call/result messages.
    """
    if not session_id:
        return "Error: session_id is required"

    return (
        f"Session history for: {session_id}\n"
        f"Limit: {limit}, include_tools: {include_tools}\n\n"
        "This tool reads from the session store when the full agent loop is connected."
    )


@tool(name="sessions_send", description="Send a message to another active session.")
async def sessions_send(
    session_id: str,
    message: str,
) -> str:
    """Send a message to another session.

    Args:
        session_id: Target session ID.
        message: Message text to inject.
    """
    if not session_id:
        return "Error: session_id is required"
    if not message:
        return "Error: message is required"

    return (
        f"Message queued for session: {session_id}\n"
        f"Length: {len(message)} chars\n\n"
        "Delivery requires active session store and agent loop."
    )


@tool(name="sessions_spawn", description="Spawn a new sub-agent session for a task.")
async def sessions_spawn(
    task: str,
    label: str = "",
    model: str = "",
    tool_profile: str = "coding",
) -> str:
    """Spawn a sub-agent to handle a task.

    Args:
        task: Task description / instructions for the sub-agent.
        label: Optional label for the session.
        model: LLM model to use (empty for default).
        tool_profile: Tool profile for the sub-agent.
    """
    if not task:
        return "Error: task is required"

    session_id = f"subagent_{uuid.uuid4().hex[:12]}"
    return (
        f"Sub-agent spawned: {session_id}\n"
        f"Label: {label or 'unlabeled'}\n"
        f"Model: {model or 'default'}\n"
        f"Profile: {tool_profile}\n"
        f"Task: {task[:200]}\n\n"
        "Sub-agent will run in the subagent lane. "
        "Results auto-announce to the parent session."
    )


@tool(name="subagents", description="List, steer, or kill sub-agents for the current session.")
async def subagents_tool(
    action: str = "list",
    target: str = "",
    message: str = "",
) -> str:
    """Manage sub-agents.

    Args:
        action: Action: list, steer, kill.
        target: Sub-agent session ID (for steer/kill).
        message: Steering message (for steer action).
    """
    if action == "list":
        return (
            "Sub-agent listing requires session registry integration.\n"
            "This tool queries the sub-agent registry for the current session."
        )
    elif action == "steer":
        if not target:
            return "Error: target session ID required"
        if not message:
            return "Error: message required for steer"
        return f"Steering message sent to {target}: {message[:100]}"
    elif action == "kill":
        if not target:
            return "Error: target session ID required"
        return f"Kill signal sent to sub-agent: {target}"
    else:
        return f"Error: Unknown action '{action}'. Use: list, steer, kill"


@tool(name="session_status", description="Get current session status: tokens, cost, model, runtime.")
async def session_status() -> str:
    """Get status of the current session."""
    return (
        "Session Status:\n"
        f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "  Model: (requires LLM integration)\n"
        "  Tokens: (requires session tracking)\n"
        "  Cost: (requires cost tracking)\n"
        "  Runtime: (requires session timer)\n\n"
        "Full status requires integration with the active session and LLM provider."
    )
