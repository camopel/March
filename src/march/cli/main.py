"""March CLI — Click-based command-line interface.

Entry point: `march` command, registered via pyproject.toml [project.scripts].
Subcommands are organized into focused modules and registered here.
"""

from __future__ import annotations

import click

from march import __version__

# Import subcommand modules
from march.cli.chat import chat
from march.cli.config_cmd import config
from march.cli.agent_cmd import agent
from march.cli.skill_cmd import skill
from march.cli.plugin_cmd import plugin
from march.cli.log_cmd import log_group
from march.cli.start import start, stop, restart, enable, disable


class MarchGroup(click.Group):
    """Custom group that shows curated help instead of Click's default."""

    def format_help(self, ctx, formatter):
        _print_help()


@click.group(cls=MarchGroup, invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx: click.Context) -> None:
    """March — A framework-first agent runtime."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ─── Register subcommands ───

cli.add_command(chat)
cli.add_command(config)
cli.add_command(agent)
cli.add_command(skill)
cli.add_command(plugin)
cli.add_command(log_group, name="log")
cli.add_command(start)
cli.add_command(stop)
cli.add_command(restart)
cli.add_command(enable)
cli.add_command(disable)


# ─── Help & Status ───


def _print_help() -> None:
    """Print the curated help text."""
    click.echo(f"""march {__version__} — A framework-first agent runtime.

USAGE
  march <command> [options]

LIFECYCLE
  march start                  Init (if needed) + start agent + dashboard
  march start --channel matrix Start with Matrix channel
  march start --all            Start all enabled channels
  march start --headless       WS proxy channel only (no interactive channels)
  march stop                   Stop March and all services
  march restart                Stop + start
  march enable                 Install as systemd service (auto-start on boot)
  march disable                Remove systemd service

INTERACTIVE
  march chat                   Interactive terminal session
  march chat "prompt"          One-shot mode

CONFIGURATION
  march config show            Show config file path
  march status                 Health, version, model, plugins, skills

AGENTS
  march agent list             List active sub-agents
  march agent show             Show agent details (db, logs, memory, files)

SKILLS & PLUGINS
  march skill list             List installed skills
  march skill install PATH     Install a skill from path
  march skill show NAME        Show skill details
  march plugin list            List active plugins
  march plugin enable NAME     Enable a plugin
  march plugin disable NAME    Disable a plugin

LOGS
  march log                    Follow log stream (default)
  march log -n 100             Last 100 lines + follow
  march log --no-follow        Print and exit

Run 'march <command> -h' for detailed help.""")


@cli.command()
def status() -> None:
    """Health, version, model, plugins, skills, channels."""
    import yaml
    from pathlib import Path

    config_path = Path.home() / ".march" / "config.yaml"
    click.echo(f"March v{__version__}")
    click.echo("═" * 40)

    # Health + config validation
    config_errors = []
    try:
        from march.config.loader import load_config
        cfg = load_config(use_cache=False)
        click.echo("  Status:    ✅ healthy")
    except Exception as e:
        cfg = None
        config_errors.append(str(e))
        click.echo("  Status:    ❌ unhealthy")

    # Raw config for fields that don't need env expansion
    raw = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    elif not config_errors:
        config_errors.append("config.yaml not found")

    # Model
    llm = raw.get("llm", {})
    click.echo(f"  Model:     {llm.get('default', '?')}")

    # Channels
    channels = raw.get("channels", {})
    enabled_ch = [ch for ch, v in channels.items() if isinstance(v, dict) and v.get("enabled")]
    click.echo(f"  Channels:  {', '.join(enabled_ch) if enabled_ch else 'none'}")

    # Plugins
    plugins = raw.get("plugins", {}).get("enabled", [])
    click.echo(f"  Plugins:   {', '.join(plugins) if plugins else 'none'}")

    # Skills
    skills_dirs = [Path.cwd() / "skills", Path.home() / ".march" / "skills"]
    skill_names = []
    for sd in skills_dirs:
        if sd.is_dir():
            for child in sd.iterdir():
                if child.is_dir() and (child / "SKILL.md").exists():
                    skill_names.append(child.name)
    click.echo(f"  Skills:    {', '.join(skill_names) if skill_names else 'none'}")

    # Process check
    from march.cli.start import _find_march_pids
    pids = _find_march_pids()
    if pids:
        labels = []
        dashboard_url = None
        for _, cmd in pids:
            if "dashboard" in cmd:
                labels.append("dashboard")
                # Extract port from cmdline if present
                import re
                port_match = re.search(r"--port\s+(\d+)", cmd)
                port = port_match.group(1) if port_match else "8200"
                dashboard_url = f"http://localhost:{port}"
            else:
                labels.append("agent")
        click.echo(f"  Running:   {', '.join(labels)} ({len(pids)} processes)")
        if dashboard_url:
            click.echo(f"  Dashboard: {dashboard_url}")
    else:
        click.echo("  Running:   not running")

    # Config errors
    if config_errors:
        click.echo("")
        click.echo("  Errors:")
        for err in config_errors:
            click.echo(f"    ❌ {err}")


if __name__ == "__main__":
    cli()
