"""march config — Configuration commands."""

from __future__ import annotations

import click


@click.group()
def config() -> None:
    """Manage March configuration."""
    pass


@config.command("show")
def config_show() -> None:
    """Show the config file path."""
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    if config_path.exists():
        click.echo(str(config_path))
    else:
        click.echo("Config not found. Run 'march start' to initialize.")
