"""march memory — Memory management commands."""

from __future__ import annotations

import click


@click.group()
def memory() -> None:
    """Manage agent memory."""
    pass


@memory.command("show")
def memory_show() -> None:
    """Show memory statistics."""
    from pathlib import Path

    march_dir = Path.home() / ".march"
    memory_md = march_dir / "MEMORY.md"

    click.echo("Memory Statistics:")
    click.echo(f"  MEMORY.md:  {'exists' if memory_md.exists() else 'not found'}")
    if memory_md.exists():
        size = memory_md.stat().st_size
        click.echo(f"  Size:       {size:,} bytes")


@memory.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear MEMORY.md?")
def memory_clear() -> None:
    """Clear MEMORY.md contents."""
    from pathlib import Path

    memory_md = Path.home() / ".march" / "MEMORY.md"
    if memory_md.exists():
        memory_md.write_text(
            "# Memory\n\n"
            "Long-term curated memory. Updated by the agent over time.\n",
            encoding="utf-8",
        )
        click.echo("✅ MEMORY.md cleared.")
    else:
        click.echo("MEMORY.md not found. Run 'march init' first.")
