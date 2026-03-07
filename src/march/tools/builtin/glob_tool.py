"""File discovery using pathlib glob/rglob with tree output format."""

from __future__ import annotations

from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.glob_tool")

_DEFAULT_EXCLUDES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".egg-info", ".tox",
}

_MAX_RESULTS = 500


@tool(name="glob", description="Find files matching a glob pattern. Returns a tree-style listing.")
async def glob_tool(
    pattern: str,
    path: str = ".",
    include_hidden: bool = False,
) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. '**/*.py', '*.md', 'src/**/*.ts').
        path: Root directory to search from.
        include_hidden: Whether to include hidden files and directories.
    """
    root = Path(path).expanduser().resolve()

    # Handle absolute patterns: split into directory + glob part
    if pattern.startswith("/") or pattern.startswith("~"):
        abs_pattern = Path(pattern).expanduser()
        root = abs_pattern.parent
        pattern = abs_pattern.name

    if not root.exists():
        return f"Error: Directory not found: {path}"
    if not root.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        matches = sorted(root.glob(pattern))
    except Exception as e:
        return f"Error in glob: {e}"

    # Filter
    results = []
    for m in matches:
        rel = m.relative_to(root)
        parts = rel.parts

        # Skip excluded directories
        if any(p in _DEFAULT_EXCLUDES for p in parts):
            continue
        if not include_hidden and any(p.startswith(".") for p in parts):
            continue

        results.append(rel)
        if len(results) >= _MAX_RESULTS:
            break

    if not results:
        return f"No files matching '{pattern}' in {root}"

    # Build tree output
    lines = [f"{root.name}/"]
    for i, rel in enumerate(results):
        is_last = i == len(results) - 1
        prefix = "└── " if is_last else "├── "
        lines.append(f"{prefix}{rel}")

    total = len(results)
    suffix = f"\n({total} matches)" if total > 1 else ""
    truncated = " (truncated)" if total >= _MAX_RESULTS else ""
    return "\n".join(lines) + suffix + truncated
