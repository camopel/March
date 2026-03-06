"""Skill base class for the March agent framework.

Defines the Skill data class that represents a loaded skill package
with its metadata, tools, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class Skill:
    """A loaded skill with all its metadata and components.

    Attributes:
        name: Unique skill name (from SKILL.md).
        version: Skill version string.
        description: Human-readable description.
        triggers: Keywords/phrases that activate this skill.
        author: Skill author name.
        tools: List of tool functions defined by this skill.
        config: Skill-specific configuration dict.
        dependencies_python: Python package dependencies.
        dependencies_mcp: MCP server dependencies.
        path: Filesystem path to the skill directory.
    """

    name: str
    version: str = "0.0.0"
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    author: str = ""
    tools: list[Callable[..., Coroutine[Any, Any, str]]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    dependencies_python: list[str] = field(default_factory=list)
    dependencies_mcp: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def tool_names(self) -> list[str]:
        """Get names of all tools provided by this skill."""
        names = []
        for fn in self.tools:
            meta = getattr(fn, "_tool_meta", None)
            if meta:
                names.append(meta.name)
            else:
                names.append(fn.__name__)
        return names

    def matches_trigger(self, text: str) -> bool:
        """Check if the text matches any of this skill's trigger keywords."""
        if not self.triggers:
            return False
        text_lower = text.lower()
        return any(trigger.lower() in text_lower for trigger in self.triggers)

    def to_dict(self) -> dict[str, Any]:
        """Serialize skill metadata to a dict."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "triggers": self.triggers,
            "author": self.author,
            "tools": self.tool_names,
            "config": self.config,
            "dependencies_python": self.dependencies_python,
            "dependencies_mcp": self.dependencies_mcp,
            "path": self.path,
        }
