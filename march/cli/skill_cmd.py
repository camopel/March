"""march skill — Skill management commands."""

from __future__ import annotations

import click


@click.group()
def skill() -> None:
    """Manage skills."""
    pass


@skill.command("list")
def skill_list() -> None:
    """List installed skills."""
    from pathlib import Path

    skills_dirs = [
        Path.cwd() / "skills",
        Path.home() / ".march" / "skills",
    ]

    found = False
    for skills_dir in skills_dirs:
        if not skills_dir.is_dir():
            continue
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                # Parse basic info
                content = skill_md.read_text(encoding="utf-8")
                import re

                name = child.name
                version = "?"
                desc = ""
                name_match = re.search(r"\*\*Name\*\*\s*:\s*(.+)", content)
                if name_match:
                    name = name_match.group(1).strip()
                ver_match = re.search(r"\*\*Version\*\*\s*:\s*(.+)", content)
                if ver_match:
                    version = ver_match.group(1).strip()
                desc_match = re.search(r"\*\*Description\*\*\s*:\s*(.+)", content)
                if desc_match:
                    desc = desc_match.group(1).strip()

                click.echo(f"  {name} v{version} — {desc}")
                found = True

    if not found:
        click.echo("No skills installed.")
        click.echo("  Install with: march skill install <path|url>")


@skill.command("install")
@click.argument("source")
def skill_install(source: str) -> None:
    """Install a skill from a path or URL."""
    from pathlib import Path
    import shutil

    source_path = Path(source)
    if source_path.is_dir():
        # Install from local directory
        target = Path.cwd() / "skills" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            click.echo(f"Skill already exists at {target}. Remove first to reinstall.")
            raise SystemExit(1)
        shutil.copytree(source_path, target)
        click.echo(f"✅ Installed skill: {source_path.name} → {target}")
    else:
        click.echo(f"URL-based skill install not yet supported: {source}")
        raise SystemExit(1)


@skill.command("create")
@click.argument("name")
def skill_create(name: str) -> None:
    """Scaffold a new skill directory."""
    from pathlib import Path

    skill_dir = Path.cwd() / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # SKILL.md
    skill_md = f"""# {name}

**Name**: {name}
**Version**: 0.1.0
**Description**: A custom March skill
**Triggers**: ["{name}"]
**Author**: March user

## Tools
- `{name}_tool` — Description of what it does

## Dependencies
- Python: (none)

## Config
- (none)
"""
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # tools.py
    tools_py = f'''"""Tools for the {name} skill."""

from march.tools.base import tool


@tool(description="Example tool for {name}")
async def {name}_tool(query: str) -> str:
    """Example tool function.

    Args:
        query: The input query.
    """
    return f"Result for: {{query}}"
'''
    (skill_dir / "tools.py").write_text(tools_py, encoding="utf-8")

    # config.yaml
    (skill_dir / "config.yaml").write_text(f"# Config for {name}\n", encoding="utf-8")

    # README.md
    (skill_dir / "README.md").write_text(
        f"# {name}\n\nA custom March skill.\n", encoding="utf-8"
    )

    click.echo(f"✅ Created skill scaffold: {skill_dir}")
    click.echo(f"  Files: SKILL.md, tools.py, config.yaml, README.md")


@skill.command("info")
@click.argument("name")
def skill_info(name: str) -> None:
    """Show detailed skill information."""
    from pathlib import Path

    from march.tools.skills.loader import SkillLoader

    # Search for the skill
    skill_dirs = [
        Path.cwd() / "skills" / name,
        Path.home() / ".march" / "skills" / name,
    ]

    for skill_dir in skill_dirs:
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            loader = SkillLoader()
            try:
                s = loader.load(skill_dir)
                click.echo(f"Name:        {s.name}")
                click.echo(f"Version:     {s.version}")
                click.echo(f"Description: {s.description}")
                click.echo(f"Author:      {s.author}")
                click.echo(f"Triggers:    {', '.join(s.triggers)}")
                click.echo(f"Tools:       {', '.join(s.tool_names)}")
                click.echo(f"Path:        {s.path}")

                # Check dependencies
                missing = loader.validate_dependencies(s)
                if missing:
                    click.echo(f"⚠️  Missing deps: {', '.join(missing)}")
                return
            except Exception as e:
                click.echo(f"Error loading skill: {e}", err=True)
                raise SystemExit(1)

    click.echo(f"Skill not found: {name}", err=True)
    raise SystemExit(1)
