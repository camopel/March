"""Manage background exec sessions: list, poll, log, write stdin, kill."""

from __future__ import annotations

import signal
import time

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.process_tool")


@tool(name="process", description="Manage background exec sessions: list, poll, log, write stdin, kill.")
async def process_tool(
    action: str,
    session_id: str = "",
    data: str = "",
    offset: int = 0,
    limit: int = 200,
) -> str:
    """Manage running background exec sessions.

    Args:
        action: Action to perform: list, poll, log, write, kill.
        session_id: Session ID (required for poll, log, write, kill).
        data: Data to write to stdin (for 'write' action).
        offset: Log offset in characters (for 'log' action).
        limit: Maximum log lines (for 'log' action).
    """
    from march.tools.builtin.exec_tool import get_sessions

    sessions = get_sessions()

    if action == "list":
        if not sessions:
            return "No background sessions."
        lines = []
        for sid, s in sessions.items():
            status = "finished" if s.finished else "running"
            elapsed = time.monotonic() - s.start_time
            lines.append(
                f"  {sid}: [{status}] pid={s.pid} "
                f"elapsed={elapsed:.0f}s cmd={s.command[:60]}"
            )
        return f"Background sessions ({len(sessions)}):\n" + "\n".join(lines)

    if not session_id:
        return "Error: session_id required for this action"

    session = sessions.get(session_id)
    if not session:
        return f"Error: Session not found: {session_id}"

    if action == "poll":
        status = "finished" if session.finished else "running"
        rc = session.process.returncode
        elapsed = time.monotonic() - session.start_time
        out = session.get_output()
        # Show last 2000 chars
        tail = out[-2000:] if len(out) > 2000 else out
        return (
            f"Session {session_id}: {status}\n"
            f"PID: {session.pid} | Exit: {rc} | Elapsed: {elapsed:.0f}s\n"
            f"--- output (last 2000 chars) ---\n{tail}"
        )

    elif action == "log":
        out = session.get_output()
        lines = out.splitlines()
        total = len(lines)
        selected = lines[offset : offset + limit]
        return (
            f"Session {session_id} log (lines {offset+1}-{offset+len(selected)} of {total}):\n"
            + "\n".join(selected)
        )

    elif action == "write":
        if session.finished:
            return "Error: Session has finished, cannot write to stdin."
        if session.process.stdin is None:
            return "Error: Session stdin not available."
        try:
            session.process.stdin.write(data.encode())
            await session.process.stdin.drain()
            return f"Wrote {len(data)} bytes to session {session_id}"
        except Exception as e:
            return f"Error writing to stdin: {e}"

    elif action == "kill":
        if session.finished:
            # Clean up
            del sessions[session_id]
            return f"Session {session_id} already finished, removed."
        try:
            session.process.send_signal(signal.SIGTERM)
            return f"Sent SIGTERM to session {session_id} (pid={session.pid})"
        except Exception as e:
            return f"Error killing session: {e}"

    else:
        return f"Error: Unknown action '{action}'. Use: list, poll, log, write, kill"
