"""Terminal channel for the March agent framework.

Interactive terminal interface using Rich for formatted output.
Reads user input, sends to the agent, and displays streaming responses
with tool execution indicators.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, AsyncIterator, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from march.channels.base import Channel
from march.core.agent import Agent, AgentResponse
from march.core.session import Session
from march.llm.base import StreamChunk
from march.logging import get_logger

logger = get_logger("march.terminal")


class TerminalChannel(Channel):
    """Interactive terminal channel using Rich.

    Features:
    - Formatted markdown output
    - Streaming response display
    - Tool execution indicators
    - Ctrl+C handling (cancel current, double to quit)
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
        self._agent: Agent | None = None
        self._session: Session | None = None
        self._running = False
        self._cancel_event: asyncio.Event = asyncio.Event()

    async def start(self, agent: Agent, **kwargs: Any) -> None:
        """Start the interactive terminal loop.

        Reads user input, sends to agent, displays response.
        Handles Ctrl+C gracefully.
        """
        self._agent = agent
        self._session = kwargs.get("session") or Session(
            source_type="terminal",
            source_id="terminal-interactive",
        )
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
                if self._handle_command(user_input):
                    continue

                # Process message
                self.console.print()
                if self.streaming:
                    await self._process_streaming(user_input)
                else:
                    await self._process_blocking(user_input)
                self.console.print()

            except KeyboardInterrupt:
                if self._cancel_event.is_set():
                    # Double Ctrl+C — quit
                    self.console.print("\n[dim]Goodbye![/dim]")
                    break
                self._cancel_event.set()
                self.console.print("\n[dim]Press Ctrl+C again to quit[/dim]")
                # Reset after a short delay
                asyncio.get_event_loop().call_later(
                    2.0, self._cancel_event.clear
                )
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
        self, chunks: AsyncIterator[StreamChunk], **kwargs: Any
    ) -> None:
        """Send a streaming response to the terminal."""
        collected = ""
        with Live(
            Text("", style="dim"),
            console=self.console,
            refresh_per_second=15,
            transient=True,
        ) as live:
            async for chunk in chunks:
                if isinstance(chunk, AgentResponse):
                    # Final response metadata — don't display in stream
                    break

                if chunk.tool_call_delta:
                    name = chunk.tool_call_delta.get("name", "")
                    status = chunk.tool_call_delta.get("status", "")
                    duration = chunk.tool_call_delta.get("duration_ms", 0)
                    self.console.print(
                        f"  [dim]⚙ {name}[/dim] {status} [dim]({duration:.0f}ms)[/dim]"
                    )

                if chunk.delta:
                    collected += chunk.delta
                    live.update(Markdown(collected))

        # Final render
        if collected:
            self.console.print(
                Panel(
                    Markdown(collected),
                    title="[bold blue]March[/bold blue]",
                    border_style="blue",
                    expand=True,
                )
            )

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

    async def _process_streaming(self, user_input: str) -> None:
        """Process a message with streaming output."""
        assert self._agent is not None
        assert self._session is not None

        collected = ""
        tool_notifications: list[str] = []

        with Live(
            Text("Thinking...", style="dim italic"),
            console=self.console,
            refresh_per_second=15,
            transient=True,
        ) as live:
            final_response: AgentResponse | None = None

            async for item in self._agent.run_stream(user_input, self._session):
                if isinstance(item, AgentResponse):
                    final_response = item
                    break

                chunk: StreamChunk = item

                if chunk.tool_call_delta:
                    name = chunk.tool_call_delta.get("name", "")
                    status = chunk.tool_call_delta.get("status", "")
                    duration = chunk.tool_call_delta.get("duration_ms", 0)
                    tool_notifications.append(
                        f"  ⚙ {name} {status} ({duration:.0f}ms)"
                    )
                    # Show tool notification above the streaming text
                    for tn in tool_notifications:
                        self.console.print(f"[dim]{tn}[/dim]")
                    tool_notifications.clear()

                if chunk.delta:
                    collected += chunk.delta
                    live.update(Markdown(collected))

        # Final render
        if collected:
            self.console.print(
                Panel(
                    Markdown(collected),
                    title="[bold blue]March[/bold blue]",
                    border_style="blue",
                    expand=True,
                )
            )

        # Show usage info if available
        if final_response and final_response.total_tokens > 0:
            self.console.print(
                f"[dim]tokens: {final_response.total_tokens} | "
                f"cost: ${final_response.total_cost:.4f} | "
                f"time: {final_response.duration_ms:.0f}ms | "
                f"tools: {final_response.tool_calls_made}[/dim]"
            )

    async def _process_blocking(self, user_input: str) -> None:
        """Process a message without streaming."""
        assert self._agent is not None
        assert self._session is not None

        with self.console.status("[bold blue]Thinking...[/bold blue]"):
            response = await self._agent.run(user_input, self._session)

        self.console.print(
            Panel(
                Markdown(response.content),
                title="[bold blue]March[/bold blue]",
                border_style="blue",
                expand=True,
            )
        )

        if response.total_tokens > 0:
            self.console.print(
                f"[dim]tokens: {response.total_tokens} | "
                f"cost: ${response.total_cost:.4f} | "
                f"time: {response.duration_ms:.0f}ms | "
                f"tools: {response.tool_calls_made}[/dim]"
            )

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

    def _handle_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if the input was a command."""
        if not text.startswith("/"):
            return False

        cmd = text.lower().split()[0]

        if cmd == "/help":
            self.console.print(
                Panel(
                    "[bold]/help[/bold]  — Show this help\n"
                    "[bold]/reset[/bold] — Clear session history\n"
                    "[bold]/history[/bold] — Show conversation history\n"
                    "[bold]/status[/bold] — Show agent status\n"
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

        if cmd == "/reset":
            # Delegate to agent for full memory cleanup
            return False

        if cmd == "/rmb":
            # Delegate to agent for global memory storage
            return False

        if cmd == "/history":
            assert self._session is not None
            if not self._session.history:
                self.console.print("[dim]No history yet.[/dim]")
            else:
                for msg in self._session.history:
                    role = msg.role.value if hasattr(msg.role, "value") else msg.role
                    preview = (msg.content[:100] + "...") if len(msg.content) > 100 else msg.content
                    self.console.print(f"[bold]{role}[/bold]: {preview}")
            return True

        if cmd == "/status":
            self.console.print(
                Panel(
                    f"Session: {self._session.id if self._session else 'none'}\n"
                    f"Messages: {len(self._session.history) if self._session else 0}\n"
                    f"Streaming: {self.streaming}\n"
                    f"Tools: {self._agent.tools.tool_count if self._agent else 0}",
                    title="Status",
                    border_style="dim",
                    expand=False,
                )
            )
            return True

        self.console.print(f"[dim]Unknown command: {cmd}. Type /help for available commands.[/dim]")
        return True
