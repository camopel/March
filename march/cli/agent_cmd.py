"""march agent — Sub-agent management commands."""

from __future__ import annotations

import click


@click.group()
def agent() -> None:
    """Manage sub-agents."""
    pass


@agent.command("list")
def agent_list() -> None:
    """List active sub-agents with their status."""
    click.echo("ID                      Task                    Status    Duration")
    click.echo("─" * 72)
    # TODO: Connect to actual agent manager when implemented
    click.echo("No active sub-agents.")


@agent.command("kill")
@click.argument("agent_id")
def agent_kill(agent_id: str) -> None:
    """Kill a sub-agent by ID."""
    # TODO: Connect to actual agent manager
    click.echo(f"Killing sub-agent: {agent_id}")
    click.echo(f"Sub-agent {agent_id} terminated.")


@agent.command("logs")
@click.argument("agent_id")
@click.option("--tail", default=50, help="Number of lines to show.")
def agent_logs(agent_id: str, tail: int) -> None:
    """Show sub-agent logs."""
    from pathlib import Path

    log_path = Path.home() / ".march" / "logs" / f"{agent_id}.log"
    if not log_path.exists():
        click.echo(f"No logs found for agent: {agent_id}", err=True)
        raise SystemExit(1)

    lines = log_path.read_text().strip().split("\n")
    for line in lines[-tail:]:
        click.echo(line)


@agent.command("send")
@click.argument("agent_id")
@click.argument("message")
def agent_send(agent_id: str, message: str) -> None:
    """Send a steering message to a running sub-agent."""
    # TODO: Connect to actual agent manager
    click.echo(f"Sending to {agent_id}: {message}")
    click.echo("Message sent.")
