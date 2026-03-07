"""march init — Initialize March environment."""

from __future__ import annotations

import click


@click.command("init")
def init() -> None:
    """Initialize March: copy templates to ~/.march/ and create project dirs.

    Copies from package templates:
    - config.yaml, MEMORY.md, SYSTEM.md, AGENT.md, TOOLS.md

    Creates in current directory:
    - plugins/, skills/
    """
    from importlib.resources import files as pkg_files
    from pathlib import Path

    config_dir = Path.home() / ".march"
    config_dir.mkdir(parents=True, exist_ok=True)
    templates_pkg = pkg_files("march.templates")

    click.echo("Initializing March...\n")

    created = 0
    skipped = 0

    # Copy all templates to ~/.march/
    for name in ("config.yaml", "MEMORY.md", "SYSTEM.md", "AGENT.md", "TOOLS.md"):
        dest = config_dir / name
        if dest.exists():
            click.echo(f"  ⏭️  Exists  ~/.march/{name}")
            skipped += 1
            continue
        try:
            content = (templates_pkg / name).read_text(encoding="utf-8")
            dest.write_text(content, encoding="utf-8")
            click.echo(f"  ✅ Created ~/.march/{name}")
            created += 1
        except Exception as e:
            click.echo(f"  ⚠️  Failed  {name}: {e}")

    # Create directories in cwd
    for dirname in ("plugins", "skills"):
        Path(dirname).mkdir(exist_ok=True)
        click.echo(f"  📁 {dirname}/")

    click.echo(f"\n✅ March initialized. ({created} created, {skipped} existing)")
    click.echo("")
    click.echo("  Next steps:")
    click.echo("    1. Edit ~/.march/config.yaml to set your LLM provider")
    click.echo("    2. Customize ~/.march/SYSTEM.md for your agent's persona")
    click.echo("    3. Run 'march chat' to start chatting")
    click.echo("")
