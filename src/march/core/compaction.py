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


async def compact_messages(
    messages: list[dict[str, Any]],
    context_window: int,
    system_tokens: int,
    summarize_fn,
    previous_summary: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Compact messages by summarizing old ones.

    Args:
        messages: Full message history.
        context_window: Model's context window in tokens.
        system_tokens: Tokens used by system prompt.
        summarize_fn: Async function(prompt: str) -> str that calls the LLM.
        previous_summary: Previous compaction summary to include.

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

    # Build and run the summarization
    prompt = build_summary_prompt(old, previous_summary)
    try:
        summary = await summarize_fn(prompt)
    except Exception as e:
        logger.error("Compaction summarization failed: %s — falling back to truncation", e)
        # Fallback: just drop old messages without summary
        return recent, previous_summary or ""

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
