"""SkillLoader — Parse SKILL.md manifests and load skill tools.

Handles discovery, parsing, and registration of skills from the filesystem.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml

from march.logging import get_logger
from march.tools.base import ToolMeta
from march.tools.registry import ToolRegistry
from march.tools.skills.base import Skill

logger = get_logger("march.tools.skills")


class SkillLoadError(Exception):
    """Raised when a skill fails to load."""

    pass


class SkillLoader:
    """Load skills from directories, parse SKILL.md manifests, and register tools.

    Attributes:
        skills: Dict of loaded skills, keyed by name.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    @property
    def skills(self) -> dict[str, Skill]:
        """Get all loaded skills."""
        return dict(self._skills)

    def load(self, skill_dir: Path, registry: ToolRegistry | None = None) -> Skill:
        """Load a single skill from a directory.

        Args:
            skill_dir: Path to the skill directory.
            registry: Optional tool registry to register tools into.

        Returns:
            The loaded Skill object.

        Raises:
            SkillLoadError: If the skill fails to load.
        """
        skill_dir = Path(skill_dir)
        if not skill_dir.is_dir():
            raise SkillLoadError(f"Skill directory not found: {skill_dir}")

        # Parse SKILL.md manifest
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise SkillLoadError(f"Missing SKILL.md in {skill_dir}")

        manifest = self._parse_skill_md(skill_md)
        name = manifest.get("name", skill_dir.name)

        # Load config.yaml if present
        config = self._load_config(skill_dir)

        # Load tools from tools.py if present
        tools = self._load_tools(skill_dir)

        # Build skill object
        skill = Skill(
            name=name,
            version=manifest.get("version", "0.0.0"),
            description=manifest.get("description", ""),
            triggers=manifest.get("triggers", []),
            author=manifest.get("author", ""),
            tools=tools,
            config=config,
            dependencies_python=manifest.get("dependencies_python", []),
            dependencies_mcp=manifest.get("dependencies_mcp", []),
            path=str(skill_dir),
        )

        # Register tools in the main registry
        if registry:
            for fn in tools:
                registry.register_function(fn, source=f"skill:{name}")

        self._skills[name] = skill
        logger.info(
            "skill.loaded name=%s version=%s tools=%d",
            name,
            skill.version,
            len(tools),
        )
        return skill

    def load_directory(
        self, skills_dir: Path, registry: ToolRegistry | None = None
    ) -> list[Skill]:
        """Load all skills from a directory of skill subdirectories.

        Args:
            skills_dir: Path to the skills parent directory.
            registry: Optional tool registry.

        Returns:
            List of successfully loaded skills.
        """
        if not skills_dir.is_dir():
            return []

        loaded: list[Skill] = []
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                try:
                    skill = self.load(child, registry=registry)
                    loaded.append(skill)
                except SkillLoadError as e:
                    logger.error("skill.load_error path=%s error=%s", child, e)

        return loaded

    def get(self, name: str) -> Skill | None:
        """Get a loaded skill by name."""
        return self._skills.get(name)

    def _parse_skill_md(self, path: Path) -> dict[str, Any]:
        """Parse a SKILL.md manifest file.

        Extracts metadata from bold-labeled lines like:
            **Name**: my-skill
            **Version**: 1.0.0
            **Triggers**: ["keyword1", "keyword2"]
        """
        content = path.read_text(encoding="utf-8")
        manifest: dict[str, Any] = {}

        # Extract bold-labeled metadata
        patterns = {
            "name": r"\*\*Name\*\*\s*:\s*(.+)",
            "version": r"\*\*Version\*\*\s*:\s*(.+)",
            "description": r"\*\*Description\*\*\s*:\s*(.+)",
            "author": r"\*\*Author\*\*\s*:\s*(.+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                manifest[key] = match.group(1).strip()

        # Extract triggers (JSON array or comma-separated)
        triggers_match = re.search(
            r"\*\*Triggers?\*\*\s*:\s*(.+)", content, re.IGNORECASE
        )
        if triggers_match:
            triggers_str = triggers_match.group(1).strip()
            manifest["triggers"] = self._parse_list(triggers_str)

        # Extract dependencies
        python_deps: list[str] = []
        mcp_deps: list[str] = []
        in_deps = False
        dep_section = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("## dependencies"):
                in_deps = True
                continue
            if in_deps and stripped.startswith("##"):
                in_deps = False
                continue
            if in_deps:
                if stripped.lower().startswith("- python:"):
                    python_deps.append(stripped.split(":", 1)[1].strip())
                elif stripped.lower().startswith("- mcp:"):
                    mcp_deps.append(stripped.split(":", 1)[1].strip())

        if python_deps:
            manifest["dependencies_python"] = python_deps
        if mcp_deps:
            manifest["dependencies_mcp"] = mcp_deps

        return manifest

    def _parse_list(self, text: str) -> list[str]:
        """Parse a list from a string (JSON array or comma-separated)."""
        text = text.strip()
        # Try JSON array
        if text.startswith("["):
            import json

            try:
                result = json.loads(text)
                if isinstance(result, list):
                    return [str(x) for x in result]
            except json.JSONDecodeError:
                pass

        # Comma-separated
        items = [x.strip().strip('"').strip("'") for x in text.split(",")]
        return [x for x in items if x]

    def _load_config(self, skill_dir: Path) -> dict[str, Any]:
        """Load config.yaml from a skill directory if present."""
        config_path = skill_dir / "config.yaml"
        if not config_path.exists():
            config_path = skill_dir / "config.yml"
        if not config_path.exists():
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("skill.config_error path=%s error=%s", config_path, e)
            return {}

    def _load_tools(
        self, skill_dir: Path
    ) -> list[Callable[..., Coroutine[Any, Any, str]]]:
        """Load tool functions from tools.py in a skill directory."""
        tools_file = skill_dir / "tools.py"
        if not tools_file.exists():
            return []

        module_name = f"march_skill_{skill_dir.name}_tools"
        try:
            spec = importlib.util.spec_from_file_location(module_name, tools_file)
            if spec is None or spec.loader is None:
                return []

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            tools: list[Callable[..., Coroutine[Any, Any, str]]] = []
            for _, obj in inspect.getmembers(module):
                if callable(obj) and hasattr(obj, "_tool_meta"):
                    tools.append(obj)

            return tools
        except Exception as e:
            logger.error("skill.tools_error path=%s error=%s", tools_file, e)
            return []

    def validate_dependencies(self, skill: Skill) -> list[str]:
        """Validate that a skill's dependencies are available.

        Returns a list of missing dependency descriptions.
        """
        missing: list[str] = []

        for dep in skill.dependencies_python:
            # Extract package name from requirement string
            pkg = re.split(r"[>=<!\s]", dep)[0].strip()
            try:
                importlib.import_module(pkg.replace("-", "_"))
            except ImportError:
                missing.append(f"Python: {dep}")

        # MCP dependencies can't be validated here easily
        # Just note them as potential issues
        for dep in skill.dependencies_mcp:
            missing.append(f"MCP (unverified): {dep}")

        return missing
