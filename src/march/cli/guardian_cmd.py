"""march guardian — Guardian process management."""

from __future__ import annotations

import click


@click.group("guardian")
def guardian() -> None:
    """Guardian process — monitors PIDs and protects restarts."""
    pass


@guardian.command("start")
def guardian_start() -> None:
    """Start the guardian as a background process."""
    import asyncio
    import os
    import sys
    from pathlib import Path

    from march.agents.guardian import run_guardian

    pid = os.fork()
    if pid > 0:
        click.echo(f"Guardian started (PID {pid})")
        return

    # Child: daemonize
    os.setsid()
    sys.stdin.close()

    log_dir = Path.home() / ".march" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "guardian.log", "a")
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

    asyncio.run(run_guardian())


@guardian.command("stop")
def guardian_stop() -> None:
    """Stop the running guardian process."""
    import os
    import signal
    import subprocess

    result = subprocess.run(
        ["pgrep", "-f", "march.*guardian"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo("No guardian process found.")
        return

    my_pid = os.getpid()
    for pid_str in result.stdout.strip().split("\n"):
        try:
            pid = int(pid_str)
            if pid == my_pid:
                continue
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Stopped guardian (PID {pid})")
        except (ProcessLookupError, ValueError):
            pass


@guardian.command("status")
def guardian_status() -> None:
    """Show guardian status and watched entries."""
    import json
    import os
    from pathlib import Path

    registry = Path.home() / ".march" / "guardian" / "watched.json"
    if not registry.exists():
        click.echo("Guardian: no registry found")
        return

    try:
        data = json.loads(registry.read_text())
    except (json.JSONDecodeError, OSError):
        click.echo("Guardian: registry corrupted")
        return

    if not data:
        click.echo("Guardian: running, 0 watched entries")
        return

    click.echo(f"Guardian: {len(data)} watched entries\n")
    for eid, entry in data.items():
        pid = entry.get("pid", 0)
        cmd = entry.get("command", "")
        alive = "?"
        if pid:
            try:
                os.kill(pid, 0)
                alive = "alive"
            except ProcessLookupError:
                alive = "dead"
            except PermissionError:
                alive = "alive"
        click.echo(f"  {eid}: PID {pid} ({alive}) — {cmd}")
