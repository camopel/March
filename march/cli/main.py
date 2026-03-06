"""March CLI — Click-based command-line interface.

Entry point: `march` command, registered via pyproject.toml [project.scripts].
Subcommands are organized into focused modules and registered here.
"""

from __future__ import annotations

import click
from pathlib import Path

from march import __version__

# Import subcommand modules
from march.cli.chat import chat
from march.cli.serve import serve
from march.cli.config_cmd import config
from march.cli.agent_cmd import agent
from march.cli.skill_cmd import skill
from march.cli.plugin_cmd import plugin
from march.cli.memory_cmd import memory
from march.cli.log_cmd import log_group
from march.cli.init_cmd import init, init_templates


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="march")
@click.option("--status", is_flag=True, help="Quick health check (exit 0=healthy, 1=unhealthy).")
@click.pass_context
def cli(ctx: click.Context, status: bool) -> None:
    """March — A framework-first agent runtime.

    Like FastAPI for web → March for agents.
    """
    if status:
        _quick_status()
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _quick_status() -> None:
    """Quick health check — exit 0 if healthy, exit 1 with details if not."""
    try:
        from march.config.loader import load_config

        config = load_config(use_cache=False)
        click.echo(f"march {__version__} — healthy")
        click.echo(f"  default LLM: {config.llm.default}")
        click.echo(
            f"  channels: terminal={config.channels.terminal.enabled}, "
            f"matrix={config.channels.matrix.enabled}"
        )
        click.echo(f"  plugins: {', '.join(config.plugins.enabled)}")
        raise SystemExit(0)
    except Exception as e:
        click.echo(f"march {__version__} — unhealthy: {e}", err=True)
        raise SystemExit(1)


# ─── Register subcommands ───

cli.add_command(chat)
cli.add_command(serve)
cli.add_command(config)
cli.add_command(agent)
cli.add_command(skill)
cli.add_command(plugin)
cli.add_command(memory)
cli.add_command(log_group, name="log")
cli.add_command(init)
cli.add_command(init_templates, name="init-templates")


# ─── Additional top-level commands ───


@cli.command()
def acp() -> None:
    """ACP mode — launched by IDEs (IntelliJ, Zed, VS Code)."""
    click.echo("march acp — not yet implemented (Phase 3)")


@cli.command()
def status() -> None:
    """Full status: health, model, tools, plugins, cost, memory."""
    _quick_status()


@cli.command("dashboard")
@click.option("--port", default=8200, help="Dashboard port.")
@click.option("--no-open", is_flag=True, help="Don't open browser.")
def dashboard(port: int, no_open: bool) -> None:
    """Open the dashboard in a browser."""
    from march.dashboard.server import DashboardServer

    server = DashboardServer(port=port)
    server.start(open_browser=not no_open)
    click.echo(f"Dashboard running at {server.url}")
    click.echo("Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        click.echo("\nDashboard stopped.")


@cli.command("version")
def version() -> None:
    """Show version information."""
    click.echo(f"march {__version__}")


@cli.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def logs(follow: bool, lines: int) -> None:
    """Tail the March log file."""
    import subprocess

    log_path = Path.home() / ".march" / "logs" / "march.log"
    # Also check session.log (the default config name)
    if not log_path.exists():
        log_path = Path.home() / ".march" / "logs" / "session.log"
    if not log_path.exists():
        # Try to find any .log file in the logs directory
        log_dir = Path.home() / ".march" / "logs"
        if log_dir.is_dir():
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                log_path = log_files[0]
            else:
                click.echo(f"No log files found in {log_dir}")
                return
        else:
            click.echo(f"Log directory not found: {log_dir}")
            return

    cmd = ["tail"]
    if follow:
        cmd.append("-f")
    cmd.extend(["-n", str(lines), str(log_path)])
    subprocess.run(cmd)


if __name__ == "__main__":
    cli()
