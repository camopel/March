"""Plugin lifecycle hooks for the March agent framework.

Defines all hook points in the agent pipeline where plugins can intercept,
modify, or block execution.
"""

from __future__ import annotations

from enum import Enum


class Hook(str, Enum):
    """All lifecycle hooks available in the March plugin pipeline.

    Hooks are fired at specific points in the agent loop. Plugins
    implement hook methods to intercept and modify behavior.
    """

    # ── Application lifecycle ──
    ON_START = "on_start"
    ON_SHUTDOWN = "on_shutdown"

    # ── Session lifecycle ──
    ON_SESSION_CONNECT = "on_session_connect"
    ON_SESSION_RESET = "on_session_reset"

    # ── LLM pipeline ──
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    ON_LLM_ERROR = "on_llm_error"
    ON_STREAM_CHUNK = "on_stream_chunk"

    # ── Tool pipeline ──
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_TOOL_ERROR = "on_tool_error"

    # ── Response ──
    ON_RESPONSE = "on_response"

    # ── Error handling ──
    ON_ERROR = "on_error"
