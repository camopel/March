"""Precise text replacement — find exact old_string and replace with new_string."""

from __future__ import annotations

from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.file_edit")


@tool(name="edit", description="Edit a file by replacing exact text. The old_string must match exactly (including whitespace).")
async def file_edit(
    path: str,
    old_string: str,
    new_string: str,
) -> str:
    """Edit a file with precise text replacement.

    Args:
        path: Path to the file to edit.
        old_string: Exact text to find and replace (must match exactly).
        new_string: New text to replace the old text with.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = p.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

    if not old_string:
        return "Error: old_string must not be empty"

    count = content.count(old_string)
    if count == 0:
        # Show a snippet to help debug
        lines = content.splitlines()
        preview = "\n".join(lines[:20])
        return (
            f"Error: old_string not found in {p.name}. "
            f"File has {len(lines)} lines. First 20 lines:\n{preview}"
        )
    if count > 1:
        return (
            f"Error: old_string found {count} times in {p.name}. "
            f"Please provide a more specific match that appears exactly once."
        )

    new_content = content.replace(old_string, new_string, 1)
    try:
        p.write_text(new_content)
    except Exception as e:
        return f"Error writing file: {e}"

    # Report what changed
    old_lines = old_string.count("\n") + 1
    new_lines = new_string.count("\n") + 1
    return f"Edited {p.name}: replaced {old_lines} lines with {new_lines} lines"
