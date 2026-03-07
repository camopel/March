"""Translation using the LLM itself (no external API)."""

from __future__ import annotations

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.translate_tool")


@tool(name="translate", description="Translate text between languages using the LLM.")
async def translate_tool(
    text: str,
    target_language: str,
    source_language: str = "auto",
) -> str:
    """Translate text between languages.

    This tool prepares a translation prompt for the LLM. In the full agent loop,
    the LLM performs the translation directly. As a standalone tool, it returns
    the formatted request.

    Args:
        text: Text to translate.
        target_language: Target language (e.g. 'English', 'German', 'Japanese').
        source_language: Source language (default 'auto' for auto-detection).
    """
    if not text.strip():
        return "Error: Empty text to translate"

    if not target_language.strip():
        return "Error: target_language is required"

    # In the full agent framework, this would route through the LLM.
    # For now, we return a structured translation request that the agent loop
    # can process or that can be sent to any LLM.
    source_note = f" from {source_language}" if source_language != "auto" else ""
    return (
        f"[Translation request{source_note} → {target_language}]\n"
        f"Text ({len(text)} chars): {text}\n\n"
        f"Note: Translation is performed by the LLM in the agent loop. "
        f"This tool formats the request for the model."
    )
