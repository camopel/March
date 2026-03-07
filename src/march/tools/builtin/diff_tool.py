"""Read-only unified diff between two files using difflib."""

from __future__ import annotations

import difflib
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.diff_tool")


@tool(name="diff", description="Show unified diff between two files or two strings.")
async def diff_tool(
    file_a: str = "",
    file_b: str = "",
    text_a: str = "",
    text_b: str = "",
    context_lines: int = 3,
) -> str:
    """Show unified diff between two files or two text strings.

    Args:
        file_a: Path to the first file.
        file_b: Path to the second file.
        text_a: First text (alternative to file_a).
        text_b: Second text (alternative to file_b).
        context_lines: Number of context lines around changes.
    """
    label_a = file_a or "text_a"
    label_b = file_b or "text_b"

    # Read content
    if file_a:
        p_a = Path(file_a).expanduser().resolve()
        if not p_a.is_file():
            return f"Error: File not found: {file_a}"
        lines_a = p_a.read_text(errors="replace").splitlines(keepends=True)
    else:
        lines_a = text_a.splitlines(keepends=True)

    if file_b:
        p_b = Path(file_b).expanduser().resolve()
        if not p_b.is_file():
            return f"Error: File not found: {file_b}"
        lines_b = p_b.read_text(errors="replace").splitlines(keepends=True)
    else:
        lines_b = text_b.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=label_a,
            tofile=label_b,
            n=context_lines,
        )
    )

    if not diff:
        return "No differences found."

    return "".join(diff)
