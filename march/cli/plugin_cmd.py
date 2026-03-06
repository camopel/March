"""march plugin — Plugin management commands."""

from __future__ import annotations

import click


@click.group()
def plugin() -> None:
    """Manage plugins."""
    pass


@plugin.command("list")
def plugin_list() -> None:
    """List plugins with enabled/disabled status."""
    try:
        from march.config.loader import load_config

        config = load_config(use_cache=False)
        enabled = set(config.plugins.enabled)

        # List builtin plugins
        builtins = ["safety", "cost", "logger", "rate_limiter", "git_context"]
        click.echo("Built-in Plugins:")
        for name in builtins:
            status = "✅ enabled" if name in enabled else "⬚ disabled"
            click.echo(f"  {name:20s} {status}")

        # List plugins from directory
        from pathlib import Path

        plugin_dir = Path.cwd() / config.plugins.directory
        if plugin_dir.is_dir():
            custom = []
            for f in sorted(plugin_dir.glob("*.py")):
                if not f.name.startswith("_"):
                    custom.append(f.stem)

            if custom:
                click.echo("\nCustom Plugins:")
                for name in custom:
                    status = "✅ enabled" if name in enabled else "⬚ disabled"
                    click.echo(f"  {name:20s} {status}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@plugin.command("enable")
@click.argument("name")
def plugin_enable(name: str) -> None:
    """Enable a plugin by name."""
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    if not config_path.exists():
        click.echo("Config file not found. Run 'march init' first.", err=True)
        raise SystemExit(1)

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    plugins = data.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])

    if name in enabled:
        click.echo(f"Plugin '{name}' is already enabled.")
        return

    enabled.append(name)

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    click.echo(f"✅ Enabled plugin: {name}")


@plugin.command("disable")
@click.argument("name")
def plugin_disable(name: str) -> None:
    """Disable a plugin by name."""
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    if not config_path.exists():
        click.echo("Config file not found. Run 'march init' first.", err=True)
        raise SystemExit(1)

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    plugins = data.get("plugins", {})
    enabled = plugins.get("enabled", [])

    if name not in enabled:
        click.echo(f"Plugin '{name}' is not enabled.")
        return

    enabled.remove(name)
    plugins["enabled"] = enabled

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    click.echo(f"⬚ Disabled plugin: {name}")


@plugin.command("create")
@click.argument("name")
def plugin_create(name: str) -> None:
    """Scaffold a new plugin."""
    from pathlib import Path

    plugin_dir = Path.cwd() / "plugins"
    plugin_dir.mkdir(exist_ok=True)

    plugin_file = plugin_dir / f"{name}.py"
    if plugin_file.exists():
        click.echo(f"Plugin file already exists: {plugin_file}")
        raise SystemExit(1)

    plugin_code = f'''"""Custom plugin: {name}."""

from march.plugins._base import Plugin


class {name.title().replace("_", "").replace("-", "")}Plugin(Plugin):
    """Custom plugin — {name}."""

    name = "{name}"
    version = "0.1.0"
    priority = 100

    async def before_llm(self, context, message):
        """Called before the LLM is invoked."""
        return context, message

    async def after_llm(self, context, response):
        """Called after the LLM responds."""
        return response

    async def before_tool(self, tool_call):
        """Called before a tool is executed. Return None to block."""
        return tool_call
'''
    plugin_file.write_text(plugin_code, encoding="utf-8")
    click.echo(f"✅ Created plugin: {plugin_file}")
