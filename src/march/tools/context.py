"""Tool execution context — provides session info to tools via contextvars.

Tools that need the current session ID (e.g. session_memory) can read it from
here instead of requiring the LLM to pass it as a parameter.
"""

from __future__ import annotations

from contextvars import ContextVar

# Current session ID — set by the agent loop before executing tools
current_session_id: ContextVar[str] = ContextVar("current_session_id", default="")
