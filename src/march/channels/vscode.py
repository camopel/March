"""VS Code extension bridge channel for March.

Connects to the ws_proxy plugin's WebSocket server to provide VS Code
integration. Session identity is based on the workspace path. This channel
acts as a WebSocket client that bridges VS Code extension messages to the
March agent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, TYPE_CHECKING

import websockets
from websockets.asyncio.client import ClientConnection

from march.channels.base import Channel
from march.core.session import Session
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.llm.base import StreamChunk

logger = get_logger("march.vscode")


class VSCodeChannel(Channel):
    """VS Code extension bridge.

    Connects to the ws_proxy plugin's WebSocket server as a client.
    Each VS Code workspace gets its own session. The extension communicates
    with March through the shared WebSocket protocol.
    """

    name: str = "vscode"

    def __init__(
        self,
        ws_url: str = "ws://localhost:8100",
        workspace_path: str = "",
    ) -> None:
        self.ws_url = ws_url
        self.workspace_path = workspace_path
        self._agent: Agent | None = None
        self._session: Session | None = None
        self._ws: ClientConnection | None = None
        self._running = False
        self._message_handlers: dict[str, Any] = {}

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the VS Code channel — connect to ws_proxy WebSocket."""
        self._agent = agent
        self._running = True

        workspace = kwargs.get("workspace_path", self.workspace_path)
        self._session = Session(
            source_type="vscode",
            source_id=workspace or "vscode-default",
        )

        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    logger.info("vscode: connected to %s (workspace: %s)", self.ws_url, workspace)

                    # Register with the server
                    await self._send_ws({
                        "type": "session.create",
                        "data": {
                            "source": "vscode",
                            "workspace_path": workspace,
                        },
                    })

                    # Listen for messages
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw)
                            await self._handle_server_message(data)
                        except json.JSONDecodeError:
                            logger.warning("vscode: invalid JSON from server")
                        except Exception as e:
                            logger.error("vscode: error handling message: %s", e)

            except websockets.exceptions.ConnectionClosed:
                if self._running:
                    logger.warning("vscode: connection lost, reconnecting in 5s...")
                    await asyncio.sleep(5)
            except ConnectionRefusedError:
                if self._running:
                    logger.warning("vscode: server not available, retrying in 5s...")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

        self._ws = None

    async def stop(self) -> None:
        """Stop the VS Code channel."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a complete response through the WebSocket."""
        if self._ws:
            await self._send_ws({
                "type": "stream.end",
                "data": {"text": content},
            })

    async def send_stream(
        self, chunks: AsyncIterator["StreamChunk"], **kwargs: Any
    ) -> None:
        """Send streaming response through the WebSocket."""
        if not self._ws:
            return

        await self._send_ws({"type": "stream.start", "data": {}})

        collected = ""
        async for chunk in chunks:
            if hasattr(chunk, "delta") and chunk.delta:
                collected += chunk.delta
                await self._send_ws({
                    "type": "stream.delta",
                    "data": {"text": chunk.delta},
                })

        await self._send_ws({
            "type": "stream.end",
            "data": {"text": collected},
        })

    # ── Server Message Handling ──────────────────────────────────────────

    async def _handle_server_message(self, data: dict[str, Any]) -> None:
        """Handle messages from the HomeHub WebSocket server."""
        msg_type = data.get("type", "")

        if msg_type == "stream.delta":
            # Forward streaming delta to the extension
            pass  # The extension reads these directly from the WS

        elif msg_type == "stream.end":
            # Response complete
            pass

        elif msg_type == "error":
            error_msg = data.get("data", {}).get("message", "Unknown error")
            logger.error("vscode: server error: %s", error_msg)

        elif msg_type == "status":
            # Status update from server
            logger.debug("vscode: status update: %s", data.get("data"))

    # ── VS Code Extension API ────────────────────────────────────────────

    async def send_message(self, text: str) -> None:
        """Send a user message to the agent (called by VS Code extension)."""
        if self._ws:
            await self._send_ws({
                "type": "message",
                "data": {"text": text},
            })

    async def send_edit_context(
        self,
        file_path: str,
        selection: str = "",
        language: str = "",
    ) -> None:
        """Send editor context along with a message."""
        if self._ws:
            await self._send_ws({
                "type": "message",
                "data": {
                    "text": f"Working on: {file_path}",
                    "context": {
                        "file": file_path,
                        "selection": selection,
                        "language": language,
                    },
                },
            })

    async def request_status(self) -> None:
        """Request agent status."""
        if self._ws:
            await self._send_ws({"type": "status.get", "data": {}})

    # ── Internal Helpers ─────────────────────────────────────────────────

    async def _send_ws(self, data: dict[str, Any]) -> None:
        """Send a JSON message over WebSocket."""
        if self._ws:
            try:
                await self._ws.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("vscode: can't send, connection closed")

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is currently connected."""
        return self._ws is not None and self._ws.open
