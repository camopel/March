"""march start/stop/restart/enable/disable — Lifecycle management."""

from __future__ import annotations

import click


@click.command("start")
@click.option("--port", default=8100, help="WebSocket server port.")
@click.option("--all", "all_channels", is_flag=True, help="Start all enabled channels.")
@click.option("--channel", multiple=True, help="Enable specific channel(s).")
@click.option("--headless", is_flag=True, help="Plugins only, no channels.")
@click.option("--no-guardian", is_flag=True, help="Don't start guardian.")
@click.option("--no-dashboard", is_flag=True, help="Don't start dashboard.")
@click.option("--dashboard-port", default=8200, help="Dashboard port.")
def start(port: int, all_channels: bool, channel: tuple[str, ...], headless: bool,
          no_guardian: bool, no_dashboard: bool, dashboard_port: int) -> None:
    """Initialize (if needed) and start March.

    On first run, copies default templates to ~/.march/.
    Then starts the agent, guardian, and dashboard.

    \b
    Examples:
        march start                    # terminal + guardian + dashboard
        march start --channel matrix   # matrix channel
        march start --all              # all enabled channels
        march start --headless         # ws_proxy channel only
    """
    import asyncio
    import os
    import signal
    from pathlib import Path

    # ── Auto-init if needed ──
    config_dir = Path.home() / ".march"
    config_dir.mkdir(parents=True, exist_ok=True)
    _ensure_templates(config_dir)

    # ── Check if already running ──
    if _find_march_pids():
        click.echo("March is already running. Use 'march restart' or 'march stop' first.")
        return

    # ── Start app ──
    from march.app import MarchApp

    config_path = config_dir / "config.yaml"
    app = MarchApp(config=config_path)

    # Start services
    guardian_pid = _start_subprocess("guardian", "start") if not no_guardian else None
    dashboard_pid = _start_subprocess("dashboard", "--port", str(dashboard_port), "--no-open") if not no_dashboard else None

    child_pids = [p for p in (guardian_pid, dashboard_pid) if p]

    def _cleanup(signum=None, frame=None):
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if signum:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    try:
        if headless:
            click.echo("Starting March (headless)")
            _print_services(guardian_pid, dashboard_pid, dashboard_port)
            asyncio.run(app._run_headless())
            return

        if all_channels:
            channels = []
            if app.config.channels.terminal.enabled:
                channels.append("terminal")
            if app.config.channels.matrix.enabled:
                channels.append("matrix")
            if not channels:
                channels = ["terminal"]
        elif channel:
            channels = list(channel)
        else:
            channels = ["terminal"]

        click.echo(f"Starting March — channels: {', '.join(channels)}")
        _print_services(guardian_pid, dashboard_pid, dashboard_port)
        app.run(channels=channels)
    finally:
        _cleanup()


@click.command("stop")
def stop() -> None:
    """Stop March and all its services (guardian, dashboard)."""
    import os
    import signal

    pids = _find_march_pids()
    if not pids:
        click.echo("March is not running.")
        return

    stopped = []
    for pid, cmdline in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            label = "guardian" if "guardian" in cmdline else "dashboard" if "dashboard" in cmdline else "agent"
            stopped.append(f"{label} (PID {pid})")
        except ProcessLookupError:
            pass

    if stopped:
        click.echo(f"Stopped: {', '.join(stopped)}")
    else:
        click.echo("No March processes found.")


@click.command("restart")
@click.pass_context
def restart(ctx: click.Context) -> None:
    """Restart March (stop + start)."""
    import time

    ctx.invoke(stop)
    time.sleep(1)
    click.echo("")
    ctx.invoke(start)


@click.command("enable")
@click.option("--channel", multiple=True, help="Channel(s) to start with.")
@click.option("--all", "all_channels", is_flag=True, help="Start all enabled channels.")
@click.option("--headless", is_flag=True, help="Plugins only.")
def enable(channel: tuple[str, ...], all_channels: bool, headless: bool) -> None:
    """Install March as a systemd user service (auto-start on boot).

    Creates ~/.config/systemd/user/march.service and enables it.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / "march.service"

    march_bin = Path(sys.executable).parent / "march"
    exec_args = [str(march_bin), "start"]
    if all_channels:
        exec_args.append("--all")
    elif channel:
        for ch in channel:
            exec_args.extend(["--channel", ch])
    if headless:
        exec_args.append("--headless")

    # Collect env vars to pass through (PATH + MARCH_* + common ones)
    env_lines = [f"Environment=PATH={os.environ.get('PATH', '/usr/bin')}"]
    for key, val in os.environ.items():
        if key.startswith("MARCH_") or key in ("AWS_PROFILE", "AWS_DEFAULT_REGION", "HOME"):
            env_lines.append(f"Environment={key}={val}")

    service_content = f"""\
[Unit]
Description=March Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={" ".join(exec_args)}
Restart=always
RestartSec=5
{chr(10).join(env_lines)}
WorkingDirectory={Path.home()}

[Install]
WantedBy=default.target
"""

    service_path.write_text(service_content)
    click.echo(f"  ✅ Created {service_path}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "march.service"], check=False)
    subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")], check=False)

    click.echo("  ✅ Enabled march.service (auto-start on boot)")
    click.echo("")
    click.echo("  Commands:")
    click.echo("    systemctl --user start march    # start now")
    click.echo("    systemctl --user status march   # check status")
    click.echo("    journalctl --user -u march -f   # follow logs")
    click.echo("    march disable                   # remove service")


@click.command("disable")
def disable() -> None:
    """Remove March systemd service."""
    import subprocess
    from pathlib import Path

    service_path = Path.home() / ".config" / "systemd" / "user" / "march.service"
    if not service_path.exists():
        click.echo("March service not installed.")
        return

    subprocess.run(["systemctl", "--user", "stop", "march.service"], check=False)
    subprocess.run(["systemctl", "--user", "disable", "march.service"], check=False)
    service_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    click.echo("  ✅ March service removed.")


# ─── Helpers ──────────────────────────────────────────────────────────


def _find_march_pids() -> list[tuple[int, str]]:
    """Find running march server processes by scanning /proc. Returns [(pid, cmdline)]."""
    import os
    from pathlib import Path

    my_pid = os.getpid()
    # Commands that are management CLIs, not server processes
    mgmt_commands = ("march stop", "march restart", "march enable", "march disable",
                     "march help", "march config", "march plugin", "march skill",
                     "march agent", "march version", "march status", "march log")
    results = []

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_text().replace("\x00", " ").strip()
            # Must contain march and be a server-like process
            if "march" not in cmdline:
                continue
            if "march start" not in cmdline and "march.cli.main start" not in cmdline \
               and "guardian" not in cmdline and "dashboard" not in cmdline:
                continue
            # Exclude management CLI commands
            if any(mc in cmdline for mc in mgmt_commands):
                continue
            results.append((pid, cmdline))
        except (PermissionError, FileNotFoundError, OSError):
            continue

    return results


def _ensure_templates(config_dir) -> None:
    """Copy missing templates to ~/.march/."""
    from importlib.resources import files as pkg_files
    from pathlib import Path

    templates_pkg = pkg_files("march.templates")
    any_created = False

    for name in ("config.yaml", "MEMORY.md", "SYSTEM.md", "AGENT.md", "TOOLS.md"):
        dest = config_dir / name
        if dest.exists():
            continue
        try:
            content = (templates_pkg / name).read_text(encoding="utf-8")
            dest.write_text(content, encoding="utf-8")
            click.echo(f"  ✅ Created ~/.march/{name}")
            any_created = True
        except Exception as e:
            click.echo(f"  ⚠️  Failed  {name}: {e}")

    if any_created:
        click.echo("")


def _print_services(guardian_pid, dashboard_pid, dashboard_port) -> None:
    if guardian_pid:
        click.echo(f"  ✅ Guardian (PID {guardian_pid})")
    if dashboard_pid:
        click.echo(f"  ✅ Dashboard: http://localhost:{dashboard_port}")


def _start_subprocess(*args) -> int | None:
    """Start a march subcommand as a background process."""
    import subprocess
    import sys
    from datetime import date
    from pathlib import Path

    log_dir = Path.home() / ".march" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = args[0] if args else "subprocess"

    # Use categorised subdirectory with date-based log file
    sub_log_dir = log_dir / log_name
    sub_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = sub_log_dir / f"{date.today().isoformat()}.log"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "march.cli.main", *args],
            stdout=open(log_file_path, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.pid
    except Exception:
        return None
