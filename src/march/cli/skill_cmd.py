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
    import re

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
                content = skill_md.read_text(encoding="utf-8")
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
        click.echo("  Install with: march skill install <path>")


@skill.command("install")
@click.argument("source")
def skill_install(source: str) -> None:
    """Install a skill from a local path."""
    import shutil
    from pathlib import Path

    source_path = Path(source).resolve()
    if not source_path.is_dir():
        click.echo(f"Not a directory: {source}", err=True)
        raise SystemExit(1)

    if not (source_path / "SKILL.md").exists():
        click.echo(f"No SKILL.md found in {source}", err=True)
        raise SystemExit(1)

    target = Path.home() / ".march" / "skills" / source_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        click.echo(f"Skill already exists: {target}")
        click.echo("  Remove it first to reinstall.")
        raise SystemExit(1)

    shutil.copytree(source_path, target)
    click.echo(f"✅ Installed: {source_path.name} → {target}")


@skill.command("show")
@click.argument("name")
def skill_show(name: str) -> None:
    """Show detailed skill information."""
    from pathlib import Path

    skill_dirs = [
        Path.cwd() / "skills" / name,
        Path.home() / ".march" / "skills" / name,
    ]

    for skill_dir in skill_dirs:
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            _show_skill(skill_dir, name)
            return

    click.echo(f"Skill not found: {name}", err=True)
    raise SystemExit(1)


def _show_skill(skill_dir, name: str) -> None:
    """Display skill details."""
    import re
    from pathlib import Path

    skill_md = skill_dir / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")

    # Parse metadata
    fields = {}
    for key in ("Name", "Version", "Description", "Author", "Triggers"):
        match = re.search(rf"\*\*{key}\*\*\s*:\s*(.+)", content)
        if match:
            fields[key] = match.group(1).strip()

    click.echo(f"Skill: {fields.get('Name', name)}")
    click.echo("═" * 50)
    click.echo(f"  Version:     {fields.get('Version', '?')}")
    click.echo(f"  Description: {fields.get('Description', '—')}")
    click.echo(f"  Author:      {fields.get('Author', '—')}")
    click.echo(f"  Triggers:    {fields.get('Triggers', '—')}")
    click.echo(f"  Path:        {skill_dir}")

    # List files
    click.echo("\n  Files:")
    for f in sorted(skill_dir.rglob("*")):
        if f.is_file() and "__pycache__" not in str(f):
            rel = f.relative_to(skill_dir)
            click.echo(f"    {rel}")

    # Show tools if tools.py exists
    tools_py = skill_dir / "tools.py"
    if tools_py.exists():
        tool_content = tools_py.read_text(encoding="utf-8")
        tool_names = re.findall(r"@tool.*\ndef\s+(\w+)", tool_content)
        if tool_names:
            click.echo(f"\n  Tools: {', '.join(tool_names)}")
