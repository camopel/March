"""Context compaction — summarize old messages when context grows too large.

Inspired by OpenClaw's compaction strategy:
1. Split old messages into chunks
2. Summarize each chunk with the LLM
3. Replace old messages with a summary message
4. Preserve recent messages intact
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keep at least this many recent messages untouched
MIN_RECENT_KEEP = 10

# When context exceeds this fraction of the window, trigger compaction
COMPACTION_THRESHOLD = 0.90

# Reserve this fraction of the context window for the summary + new messages
SUMMARY_BUDGET_RATIO = 0.15

# Budget for session memory within the compaction summary
# Facts: max 20% of context window, Plans: max 5%
FACTS_BUDGET_RATIO = 0.20
PLAN_BUDGET_RATIO = 0.05

# Safety margin for token estimation inaccuracy
SAFETY_MARGIN = 1.2

SUMMARIZE_PROMPT = """You are a conversation summarizer. Compress chat history into a summary that is at most 30% the length of the original.

**Keep (exact, never paraphrase):**
- Decisions made and their rationale
- Action items and assignments
- File paths, URLs, UUIDs, variable names, commands
- Important code snippets and error resolutions
- Anything the user explicitly asked to "remember" or "note"
- Technical conclusions and working solutions

**Drop aggressively:**
- Greetings, thanks, filler, social niceties
- Failed debugging attempts that led nowhere (mention only: "X was tried and failed")
- Redundant explanations and repeated information
- Verbose tool/command output (keep only the relevant result)
- Back-and-forth clarification once the answer is established

**Format rules:**
- Write a concise narrative, not a transcript — no "User said / Assistant said"
- Use bullet points for lists of facts, decisions, or items
- Group related topics under short **bold headers**
- Preserve all identifiers verbatim in backticks
- The summary MUST be at most 30% the size of the input. Be ruthless about brevity.

Summarize the following conversation. Be ruthless about brevity. Every token costs money."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(str(text)) // 4)


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single message including tool calls and images."""
    content = msg.get("content", "")
    if isinstance(content, list):
        # Multimodal content — estimate each block
        tokens = 4
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    # Images cost ~1600 tokens for Bedrock (fixed overhead)
                    tokens += 1600
                elif "text" in block:
                    tokens += estimate_tokens(block["text"])
                else:
                    tokens += estimate_tokens(str(block))
            else:
                tokens += estimate_tokens(str(block))
    else:
        tokens = estimate_tokens(str(content)) + 4
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", tc)
        tokens += estimate_tokens(str(func.get("arguments", func.get("args", "")))) + 10
    return tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def needs_compaction(
    messages: list[dict[str, Any]],
    context_window: int,
    system_tokens: int = 0,
) -> bool:
    """Check if messages need compaction."""
    available = int(context_window * COMPACTION_THRESHOLD) - system_tokens
    total = estimate_messages_tokens(messages)
    return total > available and len(messages) > MIN_RECENT_KEEP


def split_for_compaction(
    messages: list[dict[str, Any]],
    context_window: int,
    system_tokens: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into (old_to_summarize, recent_to_keep).

    Keeps enough recent messages to fit within the context window
    with room for the summary.
    """
    available = int(context_window * 0.8) - system_tokens
    summary_budget = int(context_window * SUMMARY_BUDGET_RATIO)
    keep_budget = available - summary_budget

    # Walk backwards from the end, keeping messages until we hit the budget
    kept: list[dict[str, Any]] = []
    kept_tokens = 0

    for msg in reversed(messages):
        msg_tokens = int(estimate_message_tokens(msg) * SAFETY_MARGIN)
        if kept_tokens + msg_tokens > keep_budget and len(kept) >= MIN_RECENT_KEEP:
            break
        kept.insert(0, msg)
        kept_tokens += msg_tokens

    # Everything before the kept messages gets summarized
    split_idx = len(messages) - len(kept)
    old = messages[:split_idx]
    recent = messages[split_idx:]

    return old, recent


def build_summary_prompt(
    messages: list[dict[str, Any]],
    previous_summary: str | None = None,
) -> str:
    """Build the prompt for summarizing a chunk of messages."""
    parts = [SUMMARIZE_PROMPT]

    if previous_summary:
        parts.append(f"\n\nPrevious context summary:\n{previous_summary}")

    parts.append("\n\nConversation to summarize:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal — extract text parts only
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
            content = " ".join(text_parts) if text_parts else "[non-text content]"
        # Truncate very long messages for the summary
        if len(str(content)) > 2000:
            content = str(content)[:2000] + "... [truncated]"
        parts.append(f"\n[{role}]: {content}")

        # Include tool call names (but not full args)
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", tc)
            name = func.get("name", "unknown")
            parts.append(f"  → tool: {name}")

    return "\n".join(parts)


EXTRACT_PROMPT = """You are a memory curator. Extract key information from this conversation that should be preserved across context compaction.

Output TWO sections in markdown:

## Facts
- Concrete facts, decisions, conclusions, technical details
- Names, IDs, paths, URLs, config values mentioned
- What was built, fixed, or changed
- If a fact was updated/changed during the conversation, note the latest version with [UPDATE]
- Only include things that are STILL RELEVANT (skip resolved issues)

## Plan
- Active tasks, next steps, TODOs that are NOT yet done
- Ongoing goals or intentions
- Skip anything already completed

Rules:
- Be concise — bullet points only, no prose
- If nothing fits a section, write "None" under it
- Max 50 lines total
- Preserve identifiers verbatim in backticks
- For conflicting/evolved facts, keep only the latest decision"""


def _load_session_memory(session_id: str) -> tuple[str, str]:
    """Load facts and plans from session memory separately.

    Files are NOT cleared — they accumulate across compactions so every
    compaction sees the full facts and plans. This prevents important
    details from being lost across multiple compressions.

    Returns:
        (facts_content, plan_content) — raw text from each file.
    """
    from pathlib import Path

    memory_dir = Path.home() / ".march" / "memory" / session_id
    if not memory_dir.is_dir():
        return "", ""

    facts = ""
    plan = ""

    facts_path = memory_dir / "facts.md"
    plan_path = memory_dir / "plan.md"

    if facts_path.exists():
        facts = facts_path.read_text(encoding="utf-8", errors="replace").strip()
    if plan_path.exists():
        plan = plan_path.read_text(encoding="utf-8", errors="replace").strip()

    # Also pick up any other .md/.txt files as facts
    for path in sorted(memory_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in ("facts.md", "plan.md"):
            continue
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                rel = path.relative_to(memory_dir)
                facts += f"\n\n[{rel}]\n{content}"
        except Exception:
            continue

    return facts.strip(), plan.strip()


async def _compress_facts(facts: str, max_tokens: int, summarize_fn, facts_path: str = "") -> str:
    """Create a compressed index of facts that fits within token budget.

    Does NOT modify the original facts file. Instead, creates a concise
    summary with a reference to the full file so the LLM can read details
    on demand.
    """
    prompt = (
        "Create a concise INDEX of these facts. Rules:\n"
        "- One bullet per fact (merge duplicates, keep latest by timestamp)\n"
        "- Each bullet: brief summary of the fact\n"
        "- Preserve all identifiers, paths, URLs verbatim in backticks\n"
        "- Drop [UPDATE] prefixes — just keep the final value\n"
        f"- Target: under {max_tokens} tokens (~{max_tokens * 4} chars)\n\n"
        f"Facts to index:\n{facts}"
    )
    try:
        compressed = await summarize_fn(prompt)
        index = compressed.strip() if compressed else facts
    except Exception:
        # If compression fails, truncate to fit
        max_chars = max_tokens * 4
        if len(facts) > max_chars:
            index = facts[-max_chars:] + "\n... (older facts truncated)"
        else:
            index = facts

    # Add reference to full file
    if facts_path:
        index += f"\n\n_Full details: `{facts_path}` (use read tool to access)_"

    return index


def delete_session_memory(session_id: str) -> bool:
    """Delete all session memory files for a session (used by /reset).

    Returns True if directory existed and was deleted.
    """
    import shutil
    from pathlib import Path

    memory_dir = Path.home() / ".march" / "memory" / session_id
    if memory_dir.is_dir():
        shutil.rmtree(memory_dir, ignore_errors=True)
        return True
    return False


async def extract_session_memory(
    messages: list[dict[str, Any]],
    session_id: str,
    summarize_fn,
    memory_dir: str | None = None,
) -> None:
    """Extract facts and plans from messages and save to session memory dir.

    Called automatically before compaction so important information
    survives context compression.

    Args:
        messages: Messages about to be compacted (the old ones being removed).
        session_id: Current session ID.
        summarize_fn: Async function(prompt: str) -> str that calls the LLM.
        memory_dir: Override memory directory (default: ~/.march/memory/{session_id}/).
    """
    from pathlib import Path

    if not messages:
        return

    target_dir = Path(memory_dir) if memory_dir else (
        Path.home() / ".march" / "memory" / session_id
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # Build extraction prompt
    parts = [EXTRACT_PROMPT, "\n\nConversation:"]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
            content = " ".join(text_parts) if text_parts else "[non-text content]"
        if len(str(content)) > 1500:
            content = str(content)[:1500] + "..."
        parts.append(f"\n[{role}]: {content}")

    prompt = "\n".join(parts)

    try:
        extracted = await summarize_fn(prompt)
    except Exception as e:
        logger.error("Session memory extraction failed: %s", e)
        return

    if not extracted or len(extracted.strip()) < 10:
        return

    # Parse into facts and plan sections
    facts_content = ""
    plan_content = ""

    lines = extracted.strip().split("\n")
    current_section = None
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("## fact"):
            current_section = "facts"
            continue
        elif stripped.startswith("## plan"):
            current_section = "plan"
            continue

        if current_section == "facts":
            facts_content += line + "\n"
        elif current_section == "plan":
            plan_content += line + "\n"

    # Save/append to session memory files
    facts_content = facts_content.strip()
    plan_content = plan_content.strip()

    if facts_content and facts_content.lower() != "none":
        facts_path = target_dir / "facts.md"
        existing = facts_path.read_text() if facts_path.exists() else ""
        with open(facts_path, "a") as f:
            if existing:
                f.write("\n\n")
            f.write(facts_content)
        logger.info("Saved session facts: %s (%d chars)", facts_path, len(facts_content))

    if plan_content and plan_content.lower() != "none":
        # Plan is overwritten (not appended) — always reflects latest state
        plan_path = target_dir / "plan.md"
        plan_path.write_text(plan_content)
        logger.info("Saved session plan: %s (%d chars)", plan_path, len(plan_content))


async def compact_messages(
    messages: list[dict[str, Any]],
    context_window: int,
    system_tokens: int,
    summarize_fn,
    previous_summary: str | None = None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Compact messages by summarizing old ones.

    Args:
        messages: Full message history.
        context_window: Model's context window in tokens.
        system_tokens: Tokens used by system prompt.
        summarize_fn: Async function(prompt: str) -> str that calls the LLM.
        previous_summary: Previous compaction summary to include.
        session_id: Session ID for extracting facts/plans before compaction.

    Returns:
        (new_messages, summary_text) — new_messages starts with a summary
        message followed by recent messages.
    """
    if not needs_compaction(messages, context_window, system_tokens):
        return messages, previous_summary or ""

    old, recent = split_for_compaction(messages, context_window, system_tokens)

    if not old:
        return messages, previous_summary or ""

    logger.info(
        "Compacting: %d old messages → summary, keeping %d recent (est. %d → %d tokens)",
        len(old), len(recent),
        estimate_messages_tokens(messages),
        estimate_messages_tokens(recent),
    )

    # Extract facts and plans BEFORE compaction so nothing important is lost
    if session_id:
        try:
            await extract_session_memory(old, session_id, summarize_fn)
        except Exception as e:
            logger.error("Session memory extraction failed (non-fatal): %s", e)

    # Build and run the summarization
    prompt = build_summary_prompt(old, previous_summary)
    try:
        summary = await summarize_fn(prompt)
    except Exception as e:
        logger.error("Compaction summarization failed: %s — falling back to truncation", e)
        # Fallback: just drop old messages without summary
        return recent, previous_summary or ""

    # Fold session memory (facts + plans) into the summary with budget limits
    # Facts: max 20% of context window, Plans: max 5% (plans never compressed)
    if session_id:
        facts, plan = _load_session_memory(session_id)

        if facts or plan:
            facts_budget = int(context_window * FACTS_BUDGET_RATIO)
            plan_budget = int(context_window * PLAN_BUDGET_RATIO)

            # Compress facts if they exceed budget — creates an index, keeps original file
            facts_tokens = estimate_tokens(facts) if facts else 0
            if facts and facts_tokens > facts_budget:
                logger.info(
                    "Session facts exceed budget (%d > %d tokens), creating index",
                    facts_tokens, facts_budget,
                )
                from pathlib import Path
                facts_path = str(Path.home() / ".march" / "memory" / session_id / "facts.md")
                facts = await _compress_facts(facts, facts_budget, summarize_fn, facts_path)

            # Truncate plan if it exceeds budget (no compression — just keep latest)
            if plan and estimate_tokens(plan) > plan_budget:
                max_chars = plan_budget * 4
                plan = plan[-max_chars:]

            memory_parts = []
            if facts:
                memory_parts.append(f"**Facts:**\n{facts}")
            if plan:
                memory_parts.append(f"**Plan:**\n{plan}")

            if memory_parts:
                summary = summary + "\n\n---\n\n**Preserved Session Memory:**\n" + "\n\n".join(memory_parts)

    # Build the compacted message list
    summary_msg = {
        "role": "user",
        "content": (
            f"[Context Summary — {len(old)} earlier messages were compacted]\n\n"
            f"{summary}"
        ),
    }

    compacted = [summary_msg] + recent
    logger.info(
        "Compaction complete: %d messages → %d (summary + %d recent), ~%d tokens",
        len(messages), len(compacted), len(recent),
        estimate_messages_tokens(compacted),
    )

    return compacted, summary
