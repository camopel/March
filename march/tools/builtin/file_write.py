"""Write content to a file, auto-creating parent directories."""

from __future__ import annotations

from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.file_write")


@tool(name="write", description="Create or overwrite a file. Automatically creates parent directories.")
async def file_write(
    path: str,
    content: str,
) -> str:
    """Write content to a file.

    Args:
        path: Path to the file to write (relative or absolute).
        content: Content to write to the file.
    """
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Wrote {len(content)} bytes ({lines} lines) to {p}"
    except Exception as e:
        return f"Error writing file: {e}"
