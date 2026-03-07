"""march agent — Agent inspection commands."""

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


@agent.command("show")
def agent_show() -> None:
    """Show agent details: config, database, logs, memory, sub-agents."""
    from pathlib import Path

    config_dir = Path.home() / ".march"
    log_dir = config_dir / "logs"
    db_path = config_dir / "march.db"
    memory_path = config_dir / "MEMORY.md"

    click.echo("March Agent")
    click.echo("═" * 50)

    # ── Config ──
    click.echo("\n📋 Configuration")
    config_path = config_dir / "config.yaml"
    if config_path.exists():
        size = config_path.stat().st_size
        click.echo(f"  config:   {config_path} ({_human_size(size)})")
    else:
        click.echo("  config:   not found")

    # ── Database ──
    click.echo("\n💾 Database")
    if db_path.exists():
        size = db_path.stat().st_size
        click.echo(f"  path:     {db_path} ({_human_size(size)})")
        _show_db_stats(db_path)
    else:
        # Check for other db files
        dbs = list(config_dir.glob("*.db"))
        if dbs:
            for db in dbs:
                click.echo(f"  path:     {db} ({_human_size(db.stat().st_size)})")
        else:
            click.echo("  no database found")

    # ── Memory ──
    click.echo("\n🧠 Memory")
    if memory_path.exists():
        lines = memory_path.read_text().strip().split("\n")
        click.echo(f"  MEMORY.md: {len(lines)} lines ({_human_size(memory_path.stat().st_size)})")
    else:
        click.echo("  MEMORY.md: not found")

    memory_dir = config_dir / "memory"
    if memory_dir.exists():
        daily_files = sorted(memory_dir.glob("*.md"))
        if daily_files:
            click.echo(f"  daily:     {len(daily_files)} files ({daily_files[0].stem} → {daily_files[-1].stem})")
        else:
            click.echo("  daily:     no files")

    # ── Logs ──
    click.echo("\n📝 Logs")
    if log_dir.exists():
        log_files = sorted(log_dir.iterdir())
        if log_files:
            for lf in log_files[-5:]:  # Show last 5
                click.echo(f"  {lf.name:30s} {_human_size(lf.stat().st_size)}")
            if len(log_files) > 5:
                click.echo(f"  ... and {len(log_files) - 5} more")
        else:
            click.echo("  no log files")
    else:
        click.echo("  log directory not found")

    # ── Markdown files ──
    click.echo("\n📄 Files")
    for name in ("SYSTEM.md", "AGENT.md", "TOOLS.md"):
        fp = config_dir / name
        if fp.exists():
            click.echo(f"  {name:20s} {_human_size(fp.stat().st_size)}")

    # ── Sub-agents ──
    click.echo("\n🤖 Sub-agents")
    # TODO: Connect to actual agent manager
    click.echo("  No active sub-agents.")

    click.echo("")


def _human_size(size: int) -> str:
    """Format bytes as human-readable."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _show_db_stats(db_path) -> None:
    """Show basic SQLite table stats."""
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        if tables:
            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                    click.echo(f"  table:    {table} ({count:,} rows)")
                except Exception:
                    click.echo(f"  table:    {table} (error reading)")
        else:
            click.echo("  no tables")
        conn.close()
    except Exception as e:
        click.echo(f"  error: {e}")
