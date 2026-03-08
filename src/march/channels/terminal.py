"""Terminal channel for the March agent framework.

Pure I/O adapter: reads stdin, writes stdout, delegates all agent logic
to the Orchestrator.  Never touches Agent or SessionStore directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from march.channels.base import Channel
from march.core.orchestrator import (
    Cancelled,
    Error,
    FinalResponse,
    Orchestrator,
    OrchestratorEvent,
    TextDelta,
    ToolProgress,
)
from march.core.session import deterministic_session_id
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent
    from march.core.session import SessionStore

logger = get_logger("march.terminal")

# Deterministic session id for the interactive terminal.
_TERMINAL_SESSION_ID = deterministic_session_id("terminal", "interactive")


class TerminalChannel(Channel):
    """Interactive terminal channel using Rich.

    Acts as a pure I/O adapter — all agent interaction flows through the
    Orchestrator.  The channel is responsible only for:

    - Reading user input (readline via Rich)
    - Converting user text → ``Orchestrator.handle_message()``
    - Rendering ``OrchestratorEvent`` objects to the terminal
    - Handling slash commands (/stop, /reset, /quit, etc.)

    Features:
    - Formatted markdown output via Rich
    - Streaming response display (TextDelta → incremental render)
    - Tool execution indicators (ToolProgress → status lines)
    - Ctrl+C / /stop handling via cancel_event
    """

    name: str = "terminal"

    def __init__(
        self,
        streaming: bool = True,
        theme: str = "dark",
    ):
        self.streaming = streaming
        self.theme = theme
        self.console = Console()
        self._orchestrator: Orchestrator | None = None
        self._running = False
        self._cancel_event: asyncio.Event = asyncio.Event()

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the interactive terminal loop.

        Creates an Orchestrator from the provided agent + session_store,
        then enters the read-eval-print loop.
        """
        session_store: SessionStore | None = kwargs.get("session_store")
        if session_store is None:
            # Fallback: create an in-memory session store so the orchestrator
            # can still function (useful for quick testing).
            from march.core.session import SessionStore
            session_store = SessionStore(":memory:")
            await session_store.initialize()

        self._orchestrator = Orchestrator(agent=agent, session_store=session_store)
        self._running = True

        # Display welcome banner
        self._show_banner()

        while self._running:
            try:
                # Get user input
                user_input = await self._get_input()
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle special commands
                if await self._handle_command(user_input):
                    continue

                # Process message through the Orchestrator
                self.console.print()
                await self._process_message(user_input)
                self.console.print()

            except KeyboardInterrupt:
                if self._cancel_event.is_set():
                    # Double Ctrl+C — quit
                    self.console.print("\n[dim]Goodbye![/dim]")
                    break
                self._cancel_event.set()
                self.console.print("\n[yellow]⏹ Stopping...[/yellow] [dim](Ctrl+C again to quit)[/dim]")
                # cancel_event is cleared at the start of the next
                # _process_message() call, so no timer is needed.
            except EOFError:
                break

        self._running = False

    async def stop(self) -> None:
        """Stop the terminal channel."""
        self._running = False

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a complete response to the terminal."""
        self.console.print()
        md = Markdown(content)
        self.console.print(
            Panel(md, title="[bold blue]March[/bold blue]", border_style="blue", expand=True)
        )

    async def send_stream(
        self, chunks: AsyncIterator, **kwargs: Any
    ) -> None:
        """Legacy send_stream — kept for Channel ABC compliance.

        The refactored terminal processes OrchestratorEvents directly in
        ``_process_message`` rather than consuming raw StreamChunks.
        """
        # Drain the iterator so callers don't hang.
        async for _ in chunks:
            pass

    # ── Input ────────────────────────────────────────────────────────────

    async def _get_input(self) -> str | None:
        """Get user input asynchronously."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: self.console.input("[bold green]You:[/bold green] "),
            )
        except EOFError:
            return None
        except KeyboardInterrupt:
            raise

    # ── Message processing (Orchestrator) ────────────────────────────────

    async def _process_message(self, user_input: str) -> None:
        """Send user_input through the Orchestrator and render events."""
        assert self._orchestrator is not None

        # Clear cancel_event at the start of every new message to avoid
        # stale cancellation from a previous Ctrl+C.
        self._cancel_event.clear()

        collected = ""

        with Live(
            Text("Thinking...", style="dim italic"),
            console=self.console,
            refresh_per_second=15,
            transient=True,
        ) as live:
            async for event in self._orchestrator.handle_message(
                session_id=_TERMINAL_SESSION_ID,
                content=user_input,
                source="terminal",
                cancel_event=self._cancel_event,
            ):
                if isinstance(event, TextDelta):
                    collected += event.delta
                    live.update(Markdown(collected))

                elif isinstance(event, ToolProgress):
                    self._render_tool_progress(event)

                elif isinstance(event, FinalResponse):
                    # Stop the Live context before final render
                    break

                elif isinstance(event, Cancelled):
                    live.update(Text(""))
                    self.console.print("[yellow]⏹ Stopped[/yellow]")
                    if event.partial_content:
                        # Show whatever was collected before cancellation
                        self.console.print(
                            Panel(
                                Markdown(event.partial_content),
                                title="[bold blue]March[/bold blue] [dim](partial)[/dim]",
                                border_style="yellow",
                                expand=True,
                            )
                        )
                    return

                elif isinstance(event, Error):
                    live.update(Text(""))
                    self.console.print(f"[bold red]Error:[/bold red] {event.message}")
                    return

        # Final render of the complete response
        display_content = collected
        final_event: FinalResponse | None = None

        # Check if the last event was a FinalResponse (we broke out of the loop)
        if isinstance(event, FinalResponse):  # type: ignore[possibly-undefined]
            final_event = event
            # Use the FinalResponse content if we didn't collect any streaming text
            if not display_content and final_event.content:
                display_content = final_event.content

        if display_content:
            self.console.print(
                Panel(
                    Markdown(display_content),
                    title="[bold blue]March[/bold blue]",
                    border_style="blue",
                    expand=True,
                )
            )

        # Show usage info
        if final_event and final_event.total_tokens > 0:
            self.console.print(
                f"[dim]tokens: {final_event.total_tokens} | "
                f"cost: ${final_event.total_cost:.4f} | "
                f"tools: {final_event.tool_calls_made}[/dim]"
            )

    def _render_tool_progress(self, event: ToolProgress) -> None:
        """Render a tool progress event as a status line."""
        if event.status == "complete":
            icon = "[green]✓[/green]"
            detail = f"[dim]({event.duration_ms:.0f}ms)[/dim]"
        elif event.status == "error":
            icon = "[red]✗[/red]"
            detail = f"[red]{event.summary}[/red]" if event.summary else ""
        elif event.status == "started":
            icon = "[dim]⚙[/dim]"
            detail = ""
        else:
            icon = "[dim]·[/dim]"
            detail = event.summary or ""

        self.console.print(f"  {icon} {event.name} {detail}")

    # ── Banner & commands ────────────────────────────────────────────────

    def _show_banner(self) -> None:
        """Display the welcome banner."""
        self.console.print()
        self.console.print(
            Panel(
                "[bold]march[/bold] — Agent Framework\n"
                "[dim]Type your message. /help for commands. Ctrl+C twice to quit.[/dim]",
                border_style="blue",
                expand=False,
            )
        )
        self.console.print()

    async def _handle_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if the input was a command."""
        if not text.startswith("/"):
            return False

        cmd = text.lower().split()[0]

        if cmd == "/help":
            self.console.print(
                Panel(
                    "[bold]/help[/bold]  — Show this help\n"
                    "[bold]/stop[/bold]  — Cancel current generation\n"
                    "[bold]/reset[/bold] — Clear session history\n"
                    "[bold]/history[/bold] — Show conversation history\n"
                    "[bold]/status[/bold] — Show session status\n"
                    "[bold]/quit[/bold]  — Exit",
                    title="Commands",
                    border_style="dim",
                    expand=False,
                )
            )
            return True

        if cmd in ("/quit", "/exit", "/q"):
            self._running = False
            self.console.print("[dim]Goodbye![/dim]")
            return True

        if cmd == "/stop":
            self._cancel_event.set()
            self.console.print("[yellow]⏹ Stopped[/yellow]")
            return True

        if cmd == "/reset":
            assert self._orchestrator is not None
            try:
                await self._orchestrator.reset_session(_TERMINAL_SESSION_ID)
                self.console.print("[dim]Session reset.[/dim]")
            except Exception as exc:
                self.console.print(f"[red]Reset failed: {exc}[/red]")
            return True

        if cmd == "/history":
            assert self._orchestrator is not None
            session = self._orchestrator.get_cached_session(_TERMINAL_SESSION_ID)
            if session is None or not session.history:
                self.console.print("[dim]No history yet.[/dim]")
            else:
                for msg in session.history:
                    role = msg.role.value if hasattr(msg.role, "value") else msg.role
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                    preview = (content_str[:100] + "...") if len(content_str) > 100 else content_str
                    self.console.print(f"[bold]{role}[/bold]: {preview}")
            return True

        if cmd == "/status":
            assert self._orchestrator is not None
            session = self._orchestrator.get_cached_session(_TERMINAL_SESSION_ID)
            msg_count = len(session.history) if session else 0
            tool_count = self._orchestrator.agent.tools.tool_count if self._orchestrator.agent.tools else 0
            self.console.print(
                Panel(
                    f"Session: {_TERMINAL_SESSION_ID}\n"
                    f"Messages: {msg_count}\n"
                    f"Streaming: {self.streaming}\n"
                    f"Tools: {tool_count}",
                    title="Status",
                    border_style="dim",
                    expand=False,
                )
            )
            return True

        # /rmb — delegate to agent (not a terminal-level command)
        if cmd == "/rmb":
            return False

        self.console.print(f"[dim]Unknown command: {cmd}. Type /help for available commands.[/dim]")
        return True
