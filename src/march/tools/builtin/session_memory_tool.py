"""Session memory tool — save facts, plans, and attachment content to session memory.

The agent calls this tool to persist important information from the conversation
to the session memory directory (~/.march/memory/{session_id}/). This information
is folded into the compaction summary when context is compressed, ensuring
nothing important is lost.

Entries are timestamped so the LLM can identify the latest decisions when
facts evolve or get superseded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool
from march.tools.context import current_session_id

logger = get_logger("march.tools.session_memory")


def _resolve_session_id(session_id: str) -> str:
    """Resolve the actual session ID.

    The LLM often doesn't know the real session UUID and may pass
    placeholders like '$SESSION_ID' or empty strings. Fall back to
    the contextvars-based current_session_id set by the agent loop.
    """
    # Treat common placeholders as empty
    if session_id and session_id not in ("$SESSION_ID", "${SESSION_ID}", ""):
        return session_id
    return current_session_id.get("")


@tool(
    name="session_memory",
    description=(
        "Save facts, plans, or attachment content to session memory. "
        "Use when the user provides important information, documents, or attachments. "
        "Entries are timestamped — if a fact evolves, save the update (don't worry about duplicates, "
        "the latest timestamp wins). Type: 'facts' or 'plan' (both append-only)."
    ),
)
async def session_memory_tool(
    session_id: str,
    type: str,
    content: str,
) -> str:
    """Save content to session memory.

    Args:
        session_id: The current session ID.
        type: Type of memory: 'facts' or 'plan'.
        content: The content to save (markdown text). Will be auto-timestamped.

    Examples:
        type="facts", content="- User's project uses Python 3.12\\n- Deploy target is AWS ECS"
        type="plan", content="1. Refactor DB layer\\n2. Add API tests\\n3. Deploy to staging"
        type="facts", content="## From requirements.pdf\\n- Must support 10k concurrent users"
        type="facts", content="- [UPDATE] Deploy target changed from ECS to Lambda"
    """
    # Resolve session_id: use context var if LLM passed a placeholder
    resolved_id = _resolve_session_id(session_id)
    if not resolved_id:
        return "Error: session_id could not be resolved"

    if type not in ("facts", "plan"):
        return f"Error: type must be 'facts' or 'plan', got '{type}'"

    if not content.strip():
        return "Error: empty content"

    memory_dir = Path.home() / ".march" / "memory" / resolved_id
    memory_dir.mkdir(parents=True, exist_ok=True)

    filename = "facts.md" if type == "facts" else "plan.md"
    filepath = memory_dir / filename

    # Auto-timestamp the entry
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    timestamped = f"[{ts}]\n{content.strip()}"

    # Always append
    existing = ""
    if filepath.exists():
        existing = filepath.read_text(encoding="utf-8").strip()

    with open(filepath, "w", encoding="utf-8") as f:
        if existing:
            f.write(existing + "\n\n")
        f.write(timestamped + "\n")

    logger.info("session_memory saved: %s/%s (%d chars)", resolved_id[:8], filename, len(content))
    return f"Saved to {type}: {len(content.strip())} chars"
