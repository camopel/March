"""march config — Configuration management commands."""

from __future__ import annotations

import click


@click.group()
def config() -> None:
    """Manage March configuration."""
    pass


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    try:
        from march.config.loader import load_config

        cfg = load_config(use_cache=False)
        click.echo(cfg.model_dump_json(indent=2))
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        raise SystemExit(1)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value (dot-notation key).

    Example: march config set llm.default litellm
    """
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    if not config_path.exists():
        click.echo(f"Config file not found: {config_path}", err=True)
        raise SystemExit(1)

    # Load existing config
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Navigate dot-notation path and set value
    keys = key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]

    # Try to parse value as YAML (handles numbers, booleans, etc.)
    try:
        parsed = yaml.safe_load(value)
    except Exception:
        parsed = value

    current[keys[-1]] = parsed

    # Write back
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    click.echo(f"Set {key} = {parsed}")


@config.command("edit")
def config_edit() -> None:
    """Open config in $EDITOR."""
    import os
    from pathlib import Path

    editor = os.environ.get("EDITOR", "vi")
    config_path = Path.home() / ".march" / "config.yaml"
    if not config_path.exists():
        from march.config.loader import ensure_config_exists

        ensure_config_exists()
    click.edit(filename=str(config_path), editor=editor)


@config.command("validate")
def config_validate() -> None:
    """Validate current configuration."""
    try:
        from march.config.loader import load_config

        load_config(use_cache=False)
        click.echo("✅ Configuration is valid.")
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        raise SystemExit(1)
