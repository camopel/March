"""GitHub tool — wraps the `gh` CLI for issues, PRs, repos, and API queries."""

from __future__ import annotations

import asyncio
import shutil

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.github")

_MAX_OUTPUT = 8000  # Truncate long outputs


async def _run_gh(*args: str, timeout: int = 30) -> str:
    """Run a gh CLI command and return stdout."""
    gh_path = shutil.which("gh")
    if not gh_path:
        return "Error: `gh` CLI not found. Install: https://cli.github.com/"

    try:
        proc = await asyncio.create_subprocess_exec(
            gh_path, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return f"Error: gh command timed out after {timeout}s"
    except Exception as e:
        return f"Error running gh: {e}"

    output = stdout.decode("utf-8", errors="replace").strip()
    err = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        return f"Error (exit {proc.returncode}): {err or output}"

    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + f"\n... (truncated, {len(output)} chars total)"

    return output or "(no output)"


@tool(
    name="github",
    description=(
        "Interact with GitHub via the `gh` CLI. "
        "Supports issues, PRs, repos, releases, actions, and API queries. "
        "Pass the full gh subcommand as a string (e.g. 'issue list --repo owner/repo')."
    ),
)
async def github_tool(
    command: str,
    repo: str = "",
) -> str:
    """Run a GitHub CLI command.

    Args:
        command: The gh subcommand to run (e.g. 'issue list', 'pr view 42', 'api /repos/owner/repo').
                 Do NOT include 'gh' prefix — just the subcommand.
        repo: Optional owner/repo to scope the command to. Added as --repo flag.

    Examples:
        command="issue list --state open --limit 10", repo="owner/repo"
        command="pr view 42"
        command="pr list --search 'is:open review:required'"
        command="release list"
        command="api /repos/owner/repo/actions/runs --jq '.workflow_runs[:5] | .[].name'"
        command="repo view owner/repo"
    """
    if not command.strip():
        return "Error: Empty command. Provide a gh subcommand like 'issue list' or 'pr view 42'."

    # Build args
    parts = command.strip().split()

    # Add --repo if provided and not already in command
    if repo and "--repo" not in command and "-R" not in command:
        parts.extend(["--repo", repo])

    return await _run_gh(*parts)
