"""File memory for the March memory system (Tier 1).

Layered file resolution:
  1. User config dir (~/.march/) — mutable, environment-specific
  2. Workspace (cwd) — project-level overrides
  3. Package templates (march/templates/) — immutable defaults

MEMORY.md is mutable-only: always from config_dir, never from templates.
SYSTEM.md, AGENT.md, TOOLS.md: config_dir → workspace → templates.

File watching uses polling. Auto-reindex on change via callback.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date, timedelta
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("march.memory.file_memory")

# Polling interval for file change detection (seconds)
POLL_INTERVAL = 5.0

# Default config directory
DEFAULT_CONFIG_DIR = Path.home() / ".march"


class FileMemory:
    """Tier 1: Human-readable file-based memory.

    Loads markdown files with layered resolution and watches for changes.

    Resolution order for SYSTEM.md, AGENT.md, TOOLS.md:
      1. config_dir (~/.march/SYSTEM.md)  — user override
      2. workspace (./SYSTEM.md)          — project override
      3. march/templates/SYSTEM.md        — package default

    MEMORY.md and daily files are mutable-only:
      1. config_dir (~/.march/MEMORY.md)  — the only source
    """

    def __init__(
        self,
        workspace: Path,
        config_dir: Path | None = None,
        system_rules_path: str = "SYSTEM.md",
        agent_profile_path: str = "AGENT.md",
        tool_rules_path: str = "TOOLS.md",
        memory_path: str = "MEMORY.md",
        daily_dir: str = "memory",
    ):
        self.workspace = workspace
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.system_rules_path = system_rules_path
        self.agent_profile_path = agent_profile_path
        self.tool_rules_path = tool_rules_path
        self.memory_path = memory_path
        self.daily_dir = daily_dir

        # Ensure config_dir exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # File content cache
        self._cache: dict[str, str] = {}
        # File hashes for change detection
        self._hashes: dict[str, str] = {}
        # Resolved paths cache: rel_path → absolute path that was used
        self._resolved_paths: dict[str, Path | None] = {}
        # Callback for reindex on change
        self._on_change: Callable[[str, str], Coroutine[Any, Any, None]] | None = None
        # Watching task
        self._watch_task: asyncio.Task | None = None
        self._running = False

    def set_on_change(
        self, callback: Callable[[str, str], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for file changes: callback(file_path, new_content)."""
        self._on_change = callback

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _resolve_layered(self, rel_path: str) -> Path | None:
        """Resolve a file using the layered lookup: config_dir → workspace → templates.

        Returns the Path of the first file found, or None.
        """
        # Layer 1: User config dir (~/.march/)
        config_path = self.config_dir / rel_path
        if config_path.exists() and config_path.is_file():
            return config_path

        # Layer 2: Workspace (cwd)
        workspace_path = self.workspace / rel_path
        if workspace_path.exists() and workspace_path.is_file():
            return workspace_path

        # Layer 3: Package templates
        try:
            templates = pkg_files("march.templates")
            template_ref = templates / rel_path
            # importlib.resources may return a Traversable, check if it exists
            if hasattr(template_ref, "is_file") and template_ref.is_file():
                # For package resources, we need the actual path
                # Use as_posix or read directly
                return Path(str(template_ref))
        except (TypeError, FileNotFoundError, ModuleNotFoundError):
            pass

        return None

    def _resolve_mutable(self, rel_path: str) -> Path:
        """Resolve a mutable-only file (MEMORY.md, daily files).

        Always resolves to config_dir. Creates parent dirs if needed.
        """
        return self.config_dir / rel_path

    def _read_file_at(self, path: Path | None) -> str:
        """Read a file at an absolute path, return empty string if not found."""
        if path is None:
            return ""
        try:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Error reading %s: %s", path, e)
        return ""

    def _load_layered(self, rel_path: str) -> str:
        """Load a file using layered resolution, cache it, track its hash."""
        resolved = self._resolve_layered(rel_path)
        self._resolved_paths[rel_path] = resolved
        content = self._read_file_at(resolved)
        self._cache[rel_path] = content
        self._hashes[rel_path] = self._hash_content(content)
        if resolved:
            logger.debug("Loaded %s from %s", rel_path, resolved)
        return content

    def _load_mutable(self, rel_path: str) -> str:
        """Load a mutable-only file from config_dir, cache it."""
        path = self._resolve_mutable(rel_path)
        self._resolved_paths[rel_path] = path
        content = self._read_file_at(path)
        self._cache[rel_path] = content
        self._hashes[rel_path] = self._hash_content(content)
        return content

    # ── Public API ──

    def load_system_rules(self) -> str:
        """Load SYSTEM.md: config_dir → workspace → templates."""
        return self._load_layered(self.system_rules_path)

    def load_agent_profile(self) -> str:
        """Load AGENT.md: config_dir → workspace → templates."""
        return self._load_layered(self.agent_profile_path)

    def load_tool_rules(self) -> str:
        """Load TOOLS.md: config_dir → workspace → templates."""
        return self._load_layered(self.tool_rules_path)

    def load_long_term(self) -> str:
        """Load MEMORY.md: config_dir only (mutable)."""
        return self._load_mutable(self.memory_path)

    def load_today(self) -> str:
        """Load today's daily memory file from config_dir."""
        today_path = f"{self.daily_dir}/{date.today().isoformat()}.md"
        return self._load_mutable(today_path)

    def load_yesterday(self) -> str:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        yesterday_path = f"{self.daily_dir}/{yesterday}.md"
        return self._load_mutable(yesterday_path)

    def save_memory(self, content: str) -> None:
        """Append content to MEMORY.md in config_dir."""
        path = self._resolve_mutable(self.memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_file_at(path)
        with open(path, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n" + content + "\n")
        # Refresh cache
        self._load_mutable(self.memory_path)

    def save_daily(self, content: str) -> None:
        """Append content to today's daily file in config_dir."""
        today_path = f"{self.daily_dir}/{date.today().isoformat()}.md"
        full_path = self._resolve_mutable(today_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        # Refresh cache
        self._load_mutable(today_path)

    def get_all_daily_files(self) -> list[tuple[str, str]]:
        """Return list of (relative_path, content) for all daily files."""
        daily_dir = self.config_dir / self.daily_dir
        if not daily_dir.is_dir():
            return []
        files = sorted(daily_dir.glob("*.md"))
        result: list[tuple[str, str]] = []
        for f in files:
            rel = f"{self.daily_dir}/{f.name}"
            content = self._load_mutable(rel)
            if content.strip():
                result.append((rel, content))
        return result

    def get_all_watched_files(self) -> list[tuple[str, str]]:
        """Return (path, content) for all tracked files (for indexing)."""
        layered = [
            self.system_rules_path,
            self.agent_profile_path,
            self.tool_rules_path,
        ]
        mutable = [self.memory_path]

        result: list[tuple[str, str]] = []
        for p in layered:
            content = self._load_layered(p)
            if content.strip():
                result.append((p, content))
        for p in mutable:
            content = self._load_mutable(p)
            if content.strip():
                result.append((p, content))
        # Daily files
        result.extend(self.get_all_daily_files())
        return result

    def get_tracked_sources(self) -> set[str]:
        """Return the set of all file sources currently being tracked."""
        sources: set[str] = set()
        for p in [self.system_rules_path, self.agent_profile_path,
                   self.tool_rules_path]:
            if self._resolve_layered(p) is not None:
                sources.add(p)
        if self._resolve_mutable(self.memory_path).exists():
            sources.add(self.memory_path)
        daily_dir = self.config_dir / self.daily_dir
        if daily_dir.is_dir():
            for f in daily_dir.glob("*.md"):
                sources.add(f"{self.daily_dir}/{f.name}")
        return sources

    def get_resolved_path(self, rel_path: str) -> Path | None:
        """Return the absolute path that was resolved for a given rel_path.

        Useful for debugging which layer a file came from.
        """
        return self._resolved_paths.get(rel_path)

    # ── File watching ──

    async def start_watching(self) -> None:
        """Start file change watching."""
        if self._running:
            return
        self._running = True
        self._watch_task = asyncio.create_task(self._poll_loop())
        logger.info("File watching started (poll interval: %.1fs)", POLL_INTERVAL)

    async def stop_watching(self) -> None:
        """Stop file change watching."""
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

    async def _poll_loop(self) -> None:
        """Poll files for changes."""
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._check_changes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in file watch poll: %s", e)

    async def _check_changes(self) -> None:
        """Check all tracked files for content changes.

        Re-resolves layered files each time so that adding a file to
        config_dir will override the template on next poll.
        """
        # Layered files: re-resolve each time
        layered_files = [
            self.system_rules_path,
            self.agent_profile_path,
            self.tool_rules_path,
        ]
        for rel_path in layered_files:
            resolved = self._resolve_layered(rel_path)
            content = self._read_file_at(resolved)
            new_hash = self._hash_content(content)
            old_hash = self._hashes.get(rel_path)

            if old_hash is not None and new_hash != old_hash:
                logger.info("File changed: %s (from %s)", rel_path, resolved)
                self._cache[rel_path] = content
                self._hashes[rel_path] = new_hash
                self._resolved_paths[rel_path] = resolved
                if self._on_change:
                    try:
                        await self._on_change(rel_path, content)
                    except Exception as e:
                        logger.error("Error in on_change callback for %s: %s", rel_path, e)
            elif old_hash is None:
                self._cache[rel_path] = content
                self._hashes[rel_path] = new_hash
                self._resolved_paths[rel_path] = resolved

        # Mutable files: config_dir only
        mutable_files = [self.memory_path]
        daily_dir = self.config_dir / self.daily_dir
        if daily_dir.is_dir():
            for f in daily_dir.glob("*.md"):
                mutable_files.append(f"{self.daily_dir}/{f.name}")

        for rel_path in mutable_files:
            path = self._resolve_mutable(rel_path)
            content = self._read_file_at(path)
            new_hash = self._hash_content(content)
            old_hash = self._hashes.get(rel_path)

            if old_hash is not None and new_hash != old_hash:
                logger.info("File changed: %s", rel_path)
                self._cache[rel_path] = content
                self._hashes[rel_path] = new_hash
                if self._on_change:
                    try:
                        await self._on_change(rel_path, content)
                    except Exception as e:
                        logger.error("Error in on_change callback for %s: %s", rel_path, e)
            elif old_hash is None:
                self._cache[rel_path] = content
                self._hashes[rel_path] = new_hash

    def check_needs_reindex(self) -> list[str]:
        """Check which files have changed since last load (for startup reindex)."""
        changed: list[str] = []
        for rel_path in list(self._hashes.keys()):
            resolved = self._resolved_paths.get(rel_path)
            if resolved is None:
                resolved = self._resolve_layered(rel_path)
            content = self._read_file_at(resolved)
            new_hash = self._hash_content(content)
            if new_hash != self._hashes.get(rel_path):
                changed.append(rel_path)
        return changed
