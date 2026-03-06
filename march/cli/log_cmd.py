"""march log — Log viewing and filtering commands."""

from __future__ import annotations

import click


@click.group(name="log", invoke_without_command=True)
@click.option("--level", help="Filter by level (DEBUG/INFO/WARNING/ERROR).")
@click.option("--event", help="Filter by event type (e.g. llm.call).")
@click.option("--lines", "-n", default=50, help="Number of lines.")
@click.pass_context
def log_group(ctx: click.Context, level: str | None, event: str | None, lines: int) -> None:
    """View and filter logs.

    Without subcommand: tail live logs.
    """
    if ctx.invoked_subcommand is not None:
        return

    # Default: tail logs
    from pathlib import Path
    import json

    log_dir = Path.home() / ".march" / "logs"
    if not log_dir.is_dir():
        click.echo("No log directory found.")
        return

    # Find the most recent log file
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        click.echo("No log files found.")
        return

    log_file = log_files[0]
    all_lines = log_file.read_text().strip().split("\n")

    # Filter
    filtered = []
    for line in all_lines:
        if level and level.upper() not in line.upper():
            continue
        if event and event not in line:
            continue
        filtered.append(line)

    for line in filtered[-lines:]:
        click.echo(line)


@log_group.command("tail")
@click.option("--level", help="Filter by level.")
@click.option("--event", help="Filter by event type.")
@click.option("--lines", "-n", default=50, help="Number of lines.")
def log_tail(level: str | None, event: str | None, lines: int) -> None:
    """Tail live logs."""
    from pathlib import Path

    log_dir = Path.home() / ".march" / "logs"
    if not log_dir.is_dir():
        click.echo("No log directory found.")
        return

    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        click.echo("No log files found.")
        return

    all_lines = log_files[0].read_text().strip().split("\n")
    filtered = []
    for line in all_lines:
        if level and level.upper() not in line.upper():
            continue
        if event and event not in line:
            continue
        filtered.append(line)

    for line in filtered[-lines:]:
        click.echo(line)


@log_group.command("cost")
@click.option("--today", is_flag=True, help="Show today's cost only.")
def log_cost(today: bool) -> None:
    """Show cost summary from logs."""
    from pathlib import Path
    import json
    import re

    log_dir = Path.home() / ".march" / "logs"
    if not log_dir.is_dir():
        click.echo("No log directory found.")
        return

    total_cost = 0.0
    total_tokens = 0
    calls = 0

    for log_file in log_dir.glob("*.log"):
        for line in log_file.read_text().strip().split("\n"):
            if "cost=$" in line:
                cost_match = re.search(r"cost=\$([0-9.]+)", line)
                in_match = re.search(r"in=(\d+)", line)
                out_match = re.search(r"out=(\d+)", line)
                if cost_match:
                    total_cost += float(cost_match.group(1))
                    calls += 1
                if in_match:
                    total_tokens += int(in_match.group(1))
                if out_match:
                    total_tokens += int(out_match.group(1))

    click.echo("Cost Summary:")
    click.echo(f"  Total cost:   ${total_cost:.4f}")
    click.echo(f"  Total tokens: {total_tokens:,}")
    click.echo(f"  LLM calls:    {calls}")


@log_group.command("audit")
@click.option("--event", help="Filter by event type.")
@click.option("--limit", default=100, help="Max results.")
def log_audit(event: str | None, limit: int) -> None:
    """Show security audit trail."""
    from pathlib import Path

    audit_db = Path.home() / ".march" / "audit.db"
    if not audit_db.exists():
        click.echo("No audit database found.")
        return

    try:
        import sqlite3

        conn = sqlite3.connect(str(audit_db))
        cursor = conn.cursor()

        query = "SELECT timestamp, event_type, details FROM audit_trail ORDER BY timestamp DESC"
        params: list[str] = []
        if event:
            query += " WHERE event_type = ?"
            params.append(event)
        query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            click.echo("No audit entries found.")
            return

        for ts, event_type, details in rows:
            click.echo(f"  [{ts}] {event_type}: {details}")

    except Exception as e:
        click.echo(f"Error reading audit trail: {e}", err=True)
