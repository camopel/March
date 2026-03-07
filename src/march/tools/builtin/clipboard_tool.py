"""Read clipboard content using pyperclip or platform commands."""

from __future__ import annotations

import asyncio
import shutil

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.clipboard_tool")


@tool(name="clipboard", description="Read the current clipboard content.")
async def clipboard_tool() -> str:
    """Read the current clipboard content."""
    # Try pyperclip first
    try:
        import pyperclip
        content = pyperclip.paste()
        if content:
            return f"Clipboard content ({len(content)} chars):\n{content}"
        return "Clipboard is empty."
    except ImportError:
        pass
    except Exception:
        pass

    # Platform fallback
    cmd = None
    if shutil.which("pbpaste"):
        cmd = "pbpaste"
    elif shutil.which("xclip"):
        cmd = "xclip -selection clipboard -o"
    elif shutil.which("xsel"):
        cmd = "xsel --clipboard --output"
    elif shutil.which("wl-paste"):
        cmd = "wl-paste"

    if cmd is None:
        return "Error: No clipboard tool found (need pyperclip, pbpaste, xclip, xsel, or wl-paste)"

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Error reading clipboard: {stderr.decode()}"
        content = stdout.decode()
        if content:
            return f"Clipboard content ({len(content)} chars):\n{content}"
        return "Clipboard is empty."
    except Exception as e:
        return f"Error reading clipboard: {e}"
