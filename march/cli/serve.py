"""march serve — Start the March agent server."""

from __future__ import annotations

import click


@click.command()
@click.option("--port", default=8100, help="WebSocket server port.")
@click.option("--all", "all_channels", is_flag=True, help="Start all enabled channels.")
@click.option("--channel", multiple=True, help="Enable specific channel(s).")
@click.option("--headless", is_flag=True, help="Start with plugins only, no channels (for plugin-based servers).")
def serve(port: int, all_channels: bool, channel: tuple[str, ...], headless: bool) -> None:
    """Start the March agent server.

    By default starts the terminal channel. Use --all to start all enabled
    channels, --channel to specify individual channels, or --headless to
    run with plugins only (e.g. ws_proxy plugin provides its own server).
    """
    from pathlib import Path
    from march.app import MarchApp

    config_path = Path.home() / ".march" / "config.yaml"
    app = MarchApp(config=config_path if config_path.exists() else None)

    if headless:
        click.echo("Starting March in headless mode (plugins only)")
        import asyncio
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

    click.echo(f"Starting March server with channels: {', '.join(channels)}")
    app.run(channels=channels)
