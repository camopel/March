"""GitContextPlugin — Auto-detect git repos and inject context.

Detects git repositories in the working directory and injects branch name,
git status, and optionally staged diff into the LLM context.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, TYPE_CHECKING

from march.logging import get_logger
from march.plugins._base import Plugin

if TYPE_CHECKING:
    from march.core.context import Context

logger = get_logger("march.plugins.git_context")


class GitContextPlugin(Plugin):
    """Auto-detect git repos and inject branch/status into LLM context.

    Attributes:
        auto_detect: Whether to auto-detect git repos.
        inject_branch: Whether to inject current branch name.
        inject_status: Whether to inject git status output.
        inject_diff: Whether to inject staged diff (disabled by default — can be large).
        working_dir: Working directory to check for git repos (default: cwd).
    """

    name = "git_context"
    version = "0.1.0"
    priority = 80  # Runs before logger/cost but after safety/rate_limiter

    def __init__(
        self,
        auto_detect: bool = True,
        inject_branch: bool = True,
        inject_status: bool = True,
        inject_diff: bool = False,
        working_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.auto_detect = auto_detect
        self.inject_branch = inject_branch
        self.inject_status = inject_status
        self.inject_diff = inject_diff
        self.working_dir = working_dir

    async def on_start(self, app: Any) -> None:
        """Load config from app.config.plugins.git_context if available."""
        cfg = getattr(getattr(app, "config", None), "plugins", None)
        if cfg:
            gc_cfg = getattr(cfg, "git_context", None)
            if gc_cfg:
                self.auto_detect = getattr(gc_cfg, "auto_detect", self.auto_detect)
                self.inject_branch = getattr(gc_cfg, "inject_branch", self.inject_branch)
                self.inject_status = getattr(gc_cfg, "inject_status", self.inject_status)
                self.inject_diff = getattr(gc_cfg, "inject_diff", self.inject_diff)

    async def before_llm(
        self, context: "Context", message: str
    ) -> tuple["Context", str]:
        """Inject git context into the LLM context if in a git repo."""
        if not self.auto_detect:
            return context, message

        cwd = self.working_dir or os.getcwd()
        if not await self._is_git_repo(cwd):
            return context, message

        parts: list[str] = ["**Git Context:**"]

        if self.inject_branch:
            branch = await self._get_branch(cwd)
            if branch:
                parts.append(f"- Branch: `{branch}`")

        if self.inject_status:
            status = await self._get_status(cwd)
            if status:
                # Truncate long status output
                if len(status) > 1000:
                    status = status[:1000] + "\n... (truncated)"
                parts.append(f"- Status:\n```\n{status}\n```")

        if self.inject_diff:
            diff = await self._get_staged_diff(cwd)
            if diff:
                if len(diff) > 3000:
                    diff = diff[:3000] + "\n... (truncated)"
                parts.append(f"- Staged diff:\n```diff\n{diff}\n```")

        if len(parts) > 1:
            context.add("\n".join(parts))
            logger.debug("git_context.injected parts=%d", len(parts) - 1)

        return context, message

    async def _is_git_repo(self, cwd: str) -> bool:
        """Check if the directory is inside a git repository."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--is-inside-work-tree",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return proc.returncode == 0 and b"true" in stdout
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return False

    async def _get_branch(self, cwd: str) -> str:
        """Get the current git branch name."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "--show-current",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip() if proc.returncode == 0 else ""
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return ""

    async def _get_status(self, cwd: str) -> str:
        """Get a short git status."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--short",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip() if proc.returncode == 0 else ""
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return ""

    async def _get_staged_diff(self, cwd: str) -> str:
        """Get the staged diff."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--cached",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip() if proc.returncode == 0 else ""
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return ""
