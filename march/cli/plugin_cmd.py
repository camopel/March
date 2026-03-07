"""march plugin — Plugin management commands."""

from __future__ import annotations

import click


@click.group()
def plugin() -> None:
    """Manage plugins."""
    pass


def _load_raw_config() -> dict:
    """Load config.yaml as raw dict (no env var expansion)."""
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _save_raw_config(data: dict) -> None:
    """Save raw dict back to config.yaml."""
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


@plugin.command("list")
def plugin_list() -> None:
    """List plugins with enabled/disabled status."""
    data = _load_raw_config()
    enabled = set(data.get("plugins", {}).get("enabled", []))

    builtins = ["safety", "cost", "logger", "rate_limiter", "git_context"]
    click.echo("Built-in Plugins:")
    for name in builtins:
        status = "✅ enabled" if name in enabled else "⬚ disabled"
        click.echo(f"  {name:20s} {status}")

    # Show any custom plugins in enabled list
    custom = [p for p in enabled if p not in builtins]
    if custom:
        click.echo("\nCustom Plugins:")
        for name in sorted(custom):
            click.echo(f"  {name:20s} ✅ enabled")


@plugin.command("enable")
@click.argument("name")
def plugin_enable(name: str) -> None:
    """Enable a plugin by name."""
    data = _load_raw_config()
    plugins = data.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])

    if name in enabled:
        click.echo(f"Plugin '{name}' is already enabled.")
        return

    enabled.append(name)
    _save_raw_config(data)
    click.echo(f"✅ Enabled plugin: {name}")


@plugin.command("disable")
@click.argument("name")
def plugin_disable(name: str) -> None:
    """Disable a plugin by name."""
    data = _load_raw_config()
    plugins = data.get("plugins", {})
    enabled = plugins.get("enabled", [])

    if name not in enabled:
        click.echo(f"Plugin '{name}' is not enabled.")
        return

    enabled.remove(name)
    plugins["enabled"] = enabled
    _save_raw_config(data)
    click.echo(f"⬚ Disabled plugin: {name}")
