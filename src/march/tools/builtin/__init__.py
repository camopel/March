"""Built-in tools for the March agent framework.

Provides register_all_builtin_tools() to register every built-in tool
with a ToolRegistry instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from march.tools.registry import ToolRegistry


def register_all_builtin_tools(registry: "ToolRegistry") -> None:
    """Register all built-in tools with the given registry.

    This is the single entry point for loading all built-in tools.
    Each tool module defines an async function decorated with @tool.
    """
    from march.tools.builtin.file_read import file_read
    from march.tools.builtin.file_write import file_write
    from march.tools.builtin.file_edit import file_edit
    from march.tools.builtin.apply_patch import apply_patch
    from march.tools.builtin.glob_tool import glob_tool
    from march.tools.builtin.diff_tool import diff_tool
    from march.tools.builtin.exec_tool import exec_tool
    from march.tools.builtin.process_tool import process_tool
    from march.tools.builtin.web_search import web_search
    from march.tools.builtin.web_fetch import web_fetch
    from march.tools.builtin.browser_tool import browser_tool
    from march.tools.builtin.pdf_tool import pdf_tool
    from march.tools.builtin.voice_to_text import voice_to_text
    from march.tools.builtin.tts_tool import tts_tool
    from march.tools.builtin.screenshot_tool import screenshot_tool
    from march.tools.builtin.clipboard_tool import clipboard_tool
    from march.tools.builtin.translate_tool import translate_tool
    from march.tools.builtin.message_tool import message_tool
    from march.tools.builtin.sessions_tools import (
        sessions_list,
        sessions_history,
        sessions_send,
        sessions_spawn,
        subagents_tool,
        session_status,
    )

    tools = [
        # Files (fs)
        file_read,
        file_write,
        file_edit,
        apply_patch,
        glob_tool,
        diff_tool,
        # Runtime
        exec_tool,
        process_tool,
        # Web
        web_search,
        web_fetch,
        browser_tool,
        # Media
        pdf_tool,
        voice_to_text,
        tts_tool,
        # System
        screenshot_tool,
        clipboard_tool,
        translate_tool,
        # Messaging
        message_tool,
        # Sessions
        sessions_list,
        sessions_history,
        sessions_send,
        sessions_spawn,
        subagents_tool,
        session_status,
    ]

    for tool_fn in tools:
        registry.register_function(tool_fn)
