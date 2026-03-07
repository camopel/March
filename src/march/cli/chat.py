"""march chat — Interactive terminal chat and one-shot mode."""

from __future__ import annotations

import click


@click.command()
@click.argument("message", required=False)
@click.option("--new", "new_session", is_flag=True, help="Start a new session.")
@click.option("--model", help="Override the default model.")
def chat(message: str | None, new_session: bool, model: str | None) -> None:
    """Interactive terminal chat, or one-shot mode with a message argument.

    Without arguments: starts an interactive chat session.
    With a message: one-shot mode — sends the message and prints the response.
    """
    import asyncio

    from march.app import MarchApp
    from march.core.session import Session

    from pathlib import Path; _cfg = Path.home() / ".march" / "config.yaml"; app = MarchApp(config=_cfg if _cfg.exists() else None)

    if message:
        # One-shot mode
        async def _one_shot() -> None:
            await app.initialize()
            try:
                session = Session(
                    source_type="terminal",
                    metadata={"mode": "one-shot"},
                )
                if model:
                    session.metadata["model_override"] = model
                response = await app.agent.run(message, session)
                click.echo(response.content)
            finally:
                await app.shutdown()

        asyncio.run(_one_shot())
    else:
        # Interactive mode
        app.run(channels=["terminal"])
