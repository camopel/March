"""Read file contents — text files with offset/limit, image passthrough."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.file_read")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_MAX_SIZE = 50 * 1024  # 50 KB text cap
_MAX_LINES = 2000


@tool(name="read", description="Read the contents of a file. Supports text files and images.")
async def file_read(
    path: str,
    offset: int = 0,
    limit: int = 0,
) -> str:
    """Read file contents. Supports text files and images (jpg, png, gif, webp).

    Args:
        path: Path to the file to read (relative or absolute).
        offset: Line number to start reading from (1-indexed, 0 means start).
        limit: Maximum number of lines to read (0 means all).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"

    # Image files → return base64 data URI
    if p.suffix.lower() in _IMAGE_EXTENSIONS:
        try:
            data = p.read_bytes()
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            b64 = base64.b64encode(data).decode()
            return f"[Image: {p.name} ({len(data)} bytes)]\ndata:{mime};base64,{b64}"
        except Exception as e:
            return f"Error reading image: {e}"

    # Text files
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = text.splitlines(keepends=True)
    total = len(lines)

    # Apply offset (1-indexed)
    start = max(0, offset - 1) if offset > 0 else 0
    end = start + limit if limit > 0 else total
    end = min(end, total)
    selected = lines[start:end]

    # Enforce limits
    if len(selected) > _MAX_LINES:
        selected = selected[:_MAX_LINES]
        truncated = True
    else:
        truncated = False

    result = "".join(selected)
    if len(result) > _MAX_SIZE:
        result = result[:_MAX_SIZE]
        truncated = True

    header = f"[{p.name}] Lines {start + 1}-{start + len(selected)} of {total}"
    if truncated:
        header += " (truncated)"
    return f"{header}\n{result}"
