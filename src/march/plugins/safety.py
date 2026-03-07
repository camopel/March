"""SafetyPlugin — Block dangerous commands and require confirmation.

Runs at priority 1 (first in the pipeline) to catch dangerous operations
before they reach tool execution.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from march.logging import get_logger
from march.plugins._base import Plugin

if TYPE_CHECKING:
    from march.core.message import ToolCall

logger = get_logger("march.plugins.safety")

# Default patterns considered dangerous in exec/shell commands
DEFAULT_BLOCKLIST: list[str] = [
    r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$",  # rm -rf /
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\b",  # rm -rf /...
    r"\bmkfs\b",  # format filesystem
    r"\bdd\s+.*of=/dev/",  # dd to device
    r"\b:(){ :\|:& };:",  # fork bomb
    r"\bformat\s+[cCdD]:",  # Windows format
    r"\bchmod\s+-R\s+777\s+/\s*$",  # chmod 777 /
    r"\bchown\s+-R\s+.*\s+/\s*$",  # chown -R ... /
    r">\s*/dev/sd[a-z]",  # write to raw disk
    r"\bshutdown\b",  # shutdown
    r"\breboot\b",  # reboot
    r"\binit\s+0\b",  # init 0
    r"\bhalt\b",  # halt
    r"\bsystemctl\s+(poweroff|halt)\b",  # systemctl poweroff/halt
]


class SafetyPlugin(Plugin):
    """Block dangerous commands in exec.

    Attributes:
        blocklist: List of regex patterns for dangerous commands.
    """

    name = "safety"
    version = "0.1.0"
    priority = 1  # Runs first

    def __init__(
        self,
        blocklist: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.blocklist = [re.compile(p, re.IGNORECASE) for p in (blocklist or DEFAULT_BLOCKLIST)]
        self._security_events: list[dict[str, Any]] = []

    async def on_start(self, app: Any) -> None:
        """Load config from app if available."""
        pass

    async def before_tool(self, tool_call: "ToolCall") -> "ToolCall | None":
        """Block dangerous exec commands.

        Returns None to block, or the (possibly modified) tool_call to proceed.
        """
        # Check exec-like tools for dangerous commands
        if tool_call.name in ("exec", "shell", "bash", "process"):
            command = tool_call.args.get("command", "")
            if not command:
                command = tool_call.args.get("cmd", "")

            if self._is_dangerous(command):
                event = {
                    "type": "blocked_dangerous_command",
                    "tool": tool_call.name,
                    "command": command[:500],
                }
                self._security_events.append(event)
                logger.warning(
                    "safety.blocked tool=%s command=%s",
                    tool_call.name,
                    command[:200],
                )
                return None

        return tool_call

    async def on_error(self, error: Exception) -> None:
        """Log security-related errors."""
        event = {
            "type": "error",
            "error": str(error)[:500],
        }
        self._security_events.append(event)
        logger.error("safety.error %s", error)

    def _is_dangerous(self, command: str) -> bool:
        """Check if a command matches any dangerous pattern.

        Strips common prefixes (sudo, env, nohup, etc.) before matching.
        """
        if not command:
            return False
        # Strip common prefixes that don't change the danger of the underlying command
        stripped = re.sub(
            r"^(sudo\s+|env\s+\S+=\S+\s+|nohup\s+|nice\s+(-n\s+\d+\s+)?)+",
            "",
            command.strip(),
            flags=re.IGNORECASE,
        )
        for pattern in self.blocklist:
            if pattern.search(command) or pattern.search(stripped):
                return True
        return False

    @property
    def security_events(self) -> list[dict[str, Any]]:
        """Get the list of recorded security events."""
        return list(self._security_events)

    def clear_events(self) -> None:
        """Clear the security event log."""
        self._security_events.clear()
