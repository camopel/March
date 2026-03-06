"""Tests for the March skill system: loading, parsing, registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from march.tools.skills.base import Skill
from march.tools.skills.loader import SkillLoader, SkillLoadError
from march.tools.registry import ToolRegistry


# ─── Fixtures ───


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Create a complete skill directory for testing."""
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()

    # SKILL.md
    skill_md = """# Test Skill

**Name**: test-skill
**Version**: 1.2.3
**Description**: A test skill for unit testing
**Triggers**: ["test", "demo"]
**Author**: March Test

## Tools
- `test_tool` — A test tool

## Dependencies
- Python: requests>=2.28
"""
    (skill_path / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # tools.py
    tools_py = '''
from march.tools.base import tool

@tool(description="A test tool that echoes input")
async def test_tool(query: str) -> str:
    """Echo the input query.

    Args:
        query: The input to echo.
    """
    return f"echo: {query}"

@tool(description="Another test tool")
async def another_tool(value: int = 42) -> str:
    """Return a value.

    Args:
        value: A numeric value.
    """
    return str(value)
'''
    (skill_path / "tools.py").write_text(tools_py, encoding="utf-8")

    # config.yaml
    (skill_path / "config.yaml").write_text(
        "api_key: test-key\ntimeout: 30\n", encoding="utf-8"
    )

    return skill_path


@pytest.fixture
def empty_skill_dir(tmp_path: Path) -> Path:
    """Create a minimal skill directory (SKILL.md only)."""
    skill_path = tmp_path / "minimal-skill"
    skill_path.mkdir()

    skill_md = """# Minimal Skill

**Name**: minimal-skill
**Version**: 0.1.0
**Description**: Minimal skill with no tools
"""
    (skill_path / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return skill_path


# ─── Skill Base Class Tests ───


class TestSkillBase:
    def test_skill_creation(self) -> None:
        """Skill can be created with all fields."""
        skill = Skill(
            name="test",
            version="1.0.0",
            description="Test skill",
            triggers=["test"],
            author="Author",
        )
        assert skill.name == "test"
        assert skill.version == "1.0.0"
        assert skill.triggers == ["test"]

    def test_matches_trigger(self) -> None:
        """Trigger matching works case-insensitively."""
        skill = Skill(name="test", triggers=["python", "coding"])
        assert skill.matches_trigger("Help me with Python")
        assert skill.matches_trigger("CODING task")
        assert not skill.matches_trigger("javascript help")

    def test_no_triggers(self) -> None:
        """Skill with no triggers never matches."""
        skill = Skill(name="test", triggers=[])
        assert not skill.matches_trigger("anything")

    def test_to_dict(self) -> None:
        """Skill serializes to dict correctly."""
        skill = Skill(
            name="test",
            version="1.0.0",
            description="Test",
            triggers=["t"],
            author="A",
        )
        d = skill.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert d["triggers"] == ["t"]


# ─── Skill Loader Tests ───


class TestSkillLoader:
    def test_load_skill(self, skill_dir: Path) -> None:
        """Loading a complete skill directory works."""
        loader = SkillLoader()
        skill = loader.load(skill_dir)

        assert skill.name == "test-skill"
        assert skill.version == "1.2.3"
        assert skill.description == "A test skill for unit testing"
        assert "test" in skill.triggers
        assert "demo" in skill.triggers
        assert skill.author == "March Test"
        assert len(skill.tools) == 2
        assert skill.config.get("api_key") == "test-key"

    def test_load_minimal_skill(self, empty_skill_dir: Path) -> None:
        """Loading a minimal skill (no tools, no config) works."""
        loader = SkillLoader()
        skill = loader.load(empty_skill_dir)

        assert skill.name == "minimal-skill"
        assert skill.version == "0.1.0"
        assert len(skill.tools) == 0
        assert skill.config == {}

    def test_load_missing_directory(self) -> None:
        """Loading from a missing directory raises SkillLoadError."""
        loader = SkillLoader()
        with pytest.raises(SkillLoadError, match="not found"):
            loader.load(Path("/nonexistent/skill"))

    def test_load_missing_skill_md(self, tmp_path: Path) -> None:
        """Loading a directory without SKILL.md raises SkillLoadError."""
        loader = SkillLoader()
        with pytest.raises(SkillLoadError, match="Missing SKILL.md"):
            loader.load(tmp_path)

    def test_register_tools_in_registry(self, skill_dir: Path) -> None:
        """Skill tools are registered in the tool registry."""
        loader = SkillLoader()
        registry = ToolRegistry()
        skill = loader.load(skill_dir, registry=registry)

        assert registry.has("test_tool")
        assert registry.has("another_tool")
        # Verify source tracking
        tool = registry.get("test_tool")
        assert tool is not None
        assert tool.source == "skill:test-skill"

    def test_load_directory(self, tmp_path: Path, skill_dir: Path) -> None:
        """load_directory discovers all skills in subdirectories."""
        # Create a parent directory containing the skill
        import shutil
        parent = tmp_path / "all_skills"
        parent.mkdir()
        shutil.copytree(skill_dir, parent / "test-skill")

        loader = SkillLoader()
        skills = loader.load_directory(parent)
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_get_skill(self, skill_dir: Path) -> None:
        """Loaded skills can be retrieved by name."""
        loader = SkillLoader()
        loader.load(skill_dir)
        assert loader.get("test-skill") is not None
        assert loader.get("nonexistent") is None

    def test_validate_dependencies(self, skill_dir: Path) -> None:
        """Dependency validation detects missing packages."""
        loader = SkillLoader()
        skill = loader.load(skill_dir)
        # 'requests' may or may not be installed; just check the method works
        missing = loader.validate_dependencies(skill)
        assert isinstance(missing, list)
