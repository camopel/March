"""Context builder for the March agent framework.

Assembles the system prompt from all context pieces: system rules, agent profile,
long-term memory, daily memory, and session context.
Token-aware assembly that truncates if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Rough token estimation: ~4 chars per token for English text
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    # Truncate at a reasonable boundary
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    return truncated + "\n\n[...truncated due to context length...]"


@dataclass
class Context:
    """Assembled context for an agent turn.

    Contains all the pieces that make up the system prompt and contextual
    information the agent needs to respond.

    Attributes:
        system_rules: Content from SYSTEM.md — persona, voice, behavior rules.
        agent_profile: Content from AGENT.md — specialization, role behavior.
        tool_inventory: Content from TOOLS.md — lean tool inventory.
        long_term_memory: Content from MEMORY.md — curated long-term memory.
        session_context: Session metadata (channel, source, etc.).
        extra_context: Additional context injected by plugins.
    """

    system_rules: str = ""
    agent_profile: str = ""
    tool_inventory: str = ""
    long_term_memory: str = ""
    session_context: dict[str, Any] = field(default_factory=dict)
    extra_context: list[str] = field(default_factory=list)

    def add(self, context: str) -> None:
        """Add extra context (used by plugins to inject additional information)."""
        if context.strip():
            self.extra_context.append(context.strip())

    def build_system_prompt(self, max_tokens: int = 0) -> str:
        """Assemble the full system prompt from all context pieces.

        Sections are included in priority order. If max_tokens is set,
        lower-priority sections are truncated first.

        Priority (highest = preserved first when truncating):
        1. System rules (persona, behavior) — IMMUTABLE, never compressed
        2. Agent profile (specialization) — IMMUTABLE, never compressed
        3. Tool inventory (available tools) — IMMUTABLE, never compressed
        4. Long-term memory (curated facts) — IMMUTABLE, never compressed
        5. Extra context (plugin-injected) — expendable, can be truncated
        6. Session context (metadata) — expendable, can be truncated
        """
        sections: list[tuple[str, str, int]] = []

        if self.system_rules:
            sections.append(("System Rules", self.system_rules, 1))
        if self.agent_profile:
            sections.append(("Agent Profile", self.agent_profile, 2))
        if self.tool_inventory:
            sections.append(("Available Tools", self.tool_inventory, 3))
        if self.long_term_memory:
            sections.append(("Long-Term Memory", self.long_term_memory, 4))
        for i, extra in enumerate(self.extra_context):
            sections.append((f"Context {i + 1}", extra, 5))
        if self.session_context:
            ctx_lines = [f"- **{k}**: {v}" for k, v in self.session_context.items()]
            sections.append(("Session Context", "\n".join(ctx_lines), 6))

        if not sections:
            return ""

        if max_tokens > 0:
            return self._build_with_budget(sections, max_tokens)

        return self._build_all(sections)

    def _build_all(self, sections: list[tuple[str, str, int]]) -> str:
        """Build system prompt without token budget — include everything."""
        parts: list[str] = []
        for title, content, _ in sections:
            parts.append(f"## {title}\n\n{content}")
        return "\n\n---\n\n".join(parts)

    def _build_with_budget(
        self, sections: list[tuple[str, str, int]], max_tokens: int
    ) -> str:
        """Build system prompt within a token budget.

        Priority 1-3 (System Rules, Agent Profile, Long-Term Memory) are
        NEVER truncated or compressed — they are always included in full.
        Only priority 4+ (Extra Context, Session Context) can be truncated
        or omitted when the budget runs out.
        """
        # Immutable threshold: priorities 1-4 are never compressed
        IMMUTABLE_PRIORITY = 4

        budget_remaining = max_tokens
        included: list[str] = []

        # Sort by priority (lower number = higher priority)
        sorted_sections = sorted(sections, key=lambda s: s[2])

        for title, content, priority in sorted_sections:
            section_text = f"## {title}\n\n{content}"
            section_tokens = estimate_tokens(section_text)

            if priority <= IMMUTABLE_PRIORITY:
                # Always include in full — never truncate
                included.append(section_text)
                budget_remaining -= section_tokens
            elif section_tokens <= budget_remaining:
                included.append(section_text)
                budget_remaining -= section_tokens
            elif budget_remaining > 100:
                # Truncate this expendable section to fit remaining budget
                truncated = truncate_to_tokens(content, budget_remaining - 20)
                included.append(f"## {title}\n\n{truncated}")
                budget_remaining = 0
            # else: skip this expendable section entirely

        return "\n\n---\n\n".join(included)

    @property
    def system_prompt(self) -> str:
        """Convenience property — build the system prompt with no token limit."""
        return self.build_system_prompt()

    @property
    def estimated_tokens(self) -> int:
        """Estimate total tokens in the system prompt."""
        return estimate_tokens(self.build_system_prompt())
