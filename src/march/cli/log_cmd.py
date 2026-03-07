"""march log — Follow the log stream."""

from __future__ import annotations

import click


@click.command(name="log")
@click.option("--lines", "-n", default=50, help="Number of lines to show before following.")
@click.option("--no-follow", is_flag=True, help="Print last N lines and exit (don't follow).")
@click.option("--level", help="Filter by level (DEBUG/INFO/WARNING/ERROR).")
def log_group(lines: int, no_follow: bool, level: str | None) -> None:
    """Follow the March log stream.

    By default, shows the last 50 lines then follows new output.
    Use -n to change how many lines to show, --no-follow to just print and exit.

    \b
    Examples:
        march log                # follow live
        march log -n 100         # last 100 lines + follow
        march log --no-follow    # print last 50 and exit
    """
    import subprocess
    from pathlib import Path

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

    if level:
        # Filter mode: read, filter, optionally follow with grep
        all_lines = log_file.read_text().strip().split("\n")
        filtered = [l for l in all_lines if level.upper() in l.upper()]
        for line in filtered[-lines:]:
            click.echo(line)
        if not no_follow:
            cmd = ["tail", "-f", "-n", "0", str(log_file)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            try:
                for line in proc.stdout:
                    if level.upper() in line.upper():
                        click.echo(line, nl=False)
            except KeyboardInterrupt:
                proc.terminate()
    else:
        # Simple tail
        cmd = ["tail"]
        if not no_follow:
            cmd.append("-f")
        cmd.extend(["-n", str(lines), str(log_file)])
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
