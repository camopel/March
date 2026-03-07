"""Session memory tool — save facts, plans, and attachment content to session memory.

The agent calls this tool to persist important information from the conversation
to the session memory directory (~/.march/memory/{session_id}/). This information
is folded into the compaction summary when context is compressed, ensuring
nothing important is lost.
"""

from __future__ import annotations

from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.session_memory")


@tool(
    name="session_memory",
    description=(
        "Save facts, plans, or attachment content to session memory. "
        "Use this when the user provides important information, documents, "
        "or attachments that should be preserved across context compaction. "
        "Type: 'facts' (append) or 'plan' (append). "
        "Content is plain text or markdown."
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
        content: The content to save (markdown text).

    Examples:
        type="facts", content="- User's project uses Python 3.12\\n- Deploy target is AWS ECS"
        type="plan", content="1. Refactor DB layer\\n2. Add API tests\\n3. Deploy to staging"
        type="facts", content="## Document: requirements.pdf\\n- Must support 10k concurrent users\\n- Budget: $50k/month"
    """
    if not session_id:
        return "Error: session_id required"

    if type not in ("facts", "plan"):
        return f"Error: type must be 'facts' or 'plan', got '{type}'"

    if not content.strip():
        return "Error: empty content"

    memory_dir = Path.home() / ".march" / "memory" / session_id
    memory_dir.mkdir(parents=True, exist_ok=True)

    filename = "facts.md" if type == "facts" else "plan.md"
    filepath = memory_dir / filename

    # Always append
    existing = ""
    if filepath.exists():
        existing = filepath.read_text(encoding="utf-8").strip()

    with open(filepath, "w", encoding="utf-8") as f:
        if existing:
            f.write(existing + "\n\n")
        f.write(content.strip() + "\n")

    logger.info("session_memory saved: %s/%s (%d chars)", session_id[:8], filename, len(content))
    return f"Saved to {type}: {len(content.strip())} chars"
