"""Apply multi-hunk unified diff patches to files."""

from __future__ import annotations

import re
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.apply_patch")


def _parse_hunks(patch: str) -> list[dict]:
    """Parse unified diff into hunks with line ranges and content."""
    hunks = []
    current_hunk = None
    lines = patch.splitlines(keepends=True)

    for line in lines:
        # Skip file headers
        if line.startswith("---") or line.startswith("+++"):
            continue
        # Hunk header
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or 1),
                "old_lines": [],
                "new_lines": [],
            }
            continue
        if current_hunk is None:
            continue
        if line.startswith("-"):
            current_hunk["old_lines"].append(line[1:])
        elif line.startswith("+"):
            current_hunk["new_lines"].append(line[1:])
        elif line.startswith(" "):
            current_hunk["old_lines"].append(line[1:])
            current_hunk["new_lines"].append(line[1:])
        else:
            # Context line without prefix
            current_hunk["old_lines"].append(line)
            current_hunk["new_lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)
    return hunks


@tool(name="apply_patch", description="Apply a unified diff patch to a file. Supports multi-hunk patches.")
async def apply_patch(
    path: str,
    patch: str,
) -> str:
    """Apply a unified diff patch to a file.

    Args:
        path: Path to the file to patch.
        patch: The unified diff patch content.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File not found: {path}"

    try:
        content = p.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

    hunks = _parse_hunks(patch)
    if not hunks:
        return "Error: No hunks found in patch"

    file_lines = content.splitlines(keepends=True)
    # Ensure all lines end with newline for matching
    if file_lines and not file_lines[-1].endswith("\n"):
        file_lines[-1] += "\n"

    # Apply hunks in reverse order to preserve line numbers
    offset = 0
    applied = 0
    for hunk in hunks:
        start = hunk["old_start"] - 1 + offset
        old_lines = hunk["old_lines"]
        new_lines = hunk["new_lines"]

        # Normalize line endings for comparison
        def norm(lines):
            return [l.rstrip("\n") for l in lines]

        actual = norm(file_lines[start : start + len(old_lines)])
        expected = norm(old_lines)

        if actual != expected:
            # Try fuzzy match: search nearby
            found = False
            for delta in range(-3, 4):
                s = start + delta
                if s < 0 or s + len(old_lines) > len(file_lines):
                    continue
                if norm(file_lines[s : s + len(old_lines)]) == expected:
                    start = s
                    found = True
                    break
            if not found:
                return (
                    f"Error: Hunk at line {hunk['old_start']} does not match. "
                    f"Expected:\n{''.join(old_lines[:5])}\n"
                    f"Got:\n{''.join(file_lines[start:start+min(5, len(old_lines))])}"
                )

        # Replace old lines with new lines
        file_lines[start : start + len(old_lines)] = new_lines
        offset += len(new_lines) - len(old_lines)
        applied += 1

    try:
        p.write_text("".join(file_lines))
    except Exception as e:
        return f"Error writing patched file: {e}"

    return f"Applied {applied} hunk(s) to {p.name}"
