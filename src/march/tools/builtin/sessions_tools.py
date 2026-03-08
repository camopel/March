"""Session management tools: list, history, send, spawn, subagents, status."""

from __future__ import annotations

import time
from typing import Any

from march.logging import get_logger
from march.tools.base import tool
from march.tools.context import current_session_id

logger = get_logger("march.tools.sessions_tools")

# Module-level reference set by MarchApp during initialization
_agent_manager: Any = None


def set_agent_manager(mgr: Any) -> None:
    """Called by MarchApp to wire up the agent manager."""
    global _agent_manager
    _agent_manager = mgr


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
    if _agent_manager is None:
        return (
            "No active sessions (agent manager not initialized).\n"
            f"Filters: kind={kind or 'all'}, active_only={active_only}, label={label or 'none'}"
        )

    statuses = await _agent_manager.list()
    if kind:
        statuses = [s for s in statuses if kind in s.child_key]
    if active_only:
        statuses = [s for s in statuses if s.status == "running"]

    if not statuses:
        return "No matching sessions."

    lines = []
    for s in statuses:
        lines.append(f"- {s.child_key} ({s.status}, {s.duration_seconds:.0f}s) {s.task[:80]}")
    return "\n".join(lines)


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
    if not session_id or not message:
        return "Error: session_id and message required"
    if _agent_manager is None:
        return (
            f"Message queued for session: {session_id}\n"
            f"Length: {len(message)} chars\n\n"
            "Delivery requires active agent manager."
        )

    ok = await _agent_manager.send(session_id, message)
    return f"Message {'delivered' if ok else 'failed'}: {session_id}"


@tool(name="sessions_spawn", description="Spawn a new agent session for a task.")
async def sessions_spawn(
    task: str,
    label: str = "",
    model: str = "",
    tool_profile: str = "coding",
    execution: str = "mt",
) -> str:
    """Spawn an agent to handle a task.

    Args:
        task: Task description / instructions for the agent.
        label: Optional label for the session.
        model: LLM model to use (empty for default).
        tool_profile: Tool profile for the agent.
        execution: Execution mode — "mt" (asyncio, default) or "mp" (isolated process).
    """
    if not task:
        return "Error: task is required"
    if execution not in ("mt", "mp"):
        return f"Error: execution must be 'mt' or 'mp', got '{execution}'"
    if _agent_manager is None:
        import uuid as _uuid
        session_id = f"agent_{_uuid.uuid4().hex[:12]}"
        return (
            f"Agent spawned (queued): {session_id}\n"
            f"Label: {label or 'unlabeled'}\n"
            f"Execution: {execution}\n"
            f"Task: {task[:200]}\n\n"
            "Agent manager not yet initialized. Agent will run when ready."
        )

    # Import here to avoid circular imports
    from march.agents.manager import SpawnParams, SpawnContext

    parent_session = current_session_id.get("")
    result = await _agent_manager.spawn(
        SpawnParams(task=task, label=label, model=model, execution=execution),
        SpawnContext(requester_session=parent_session),
    )

    if result.status != "accepted":
        return f"Error: {result.error}"

    exec_display = "mtAgent" if execution == "mt" else "mpAgent"
    return (
        f"{exec_display} spawned: {result.child_key}\n"
        f"Run ID: {result.run_id}\n"
        f"Label: {label or 'unlabeled'}\n"
        f"Execution: {execution}\n"
        f"Task: {task[:200]}\n\n"
        "Auto-announces on completion, do not poll."
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
    if _agent_manager is None:
        if action == "list":
            return "No active or recent agents (agent manager not initialized)."
        return "Error: agent manager not initialized"

    if action == "list":
        statuses = await _agent_manager.list()
        if not statuses:
            return "No active or recent agents."
        lines = []
        for s in statuses:
            exec_tag = "mt" if s.execution == "mt" else "mp"
            hb_info = ""
            if s.heartbeat and s.status == "running":
                summary = s.heartbeat.get("summary", "")
                if summary:
                    hb_info = f" [{summary}]"
            lines.append(f"- [{exec_tag}] {s.child_key} ({s.status}) task={s.task[:80]}{hb_info}")
        return "\n".join(lines)
    elif action == "steer":
        if not target or not message:
            return "Error: target and message required for steer"
        ok = await _agent_manager.send(target, message)
        return f"Steer {'sent' if ok else 'failed'}: {target}"
    elif action == "kill":
        if not target:
            return "Error: target required for kill"
        ok = await _agent_manager.kill(target)
        return f"Kill {'sent' if ok else 'failed'}: {target}"
    else:
        return f"Error: Unknown action '{action}'"


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
