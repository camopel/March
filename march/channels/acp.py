"""ACP (Agent Client Protocol) channel for March.

JSON-RPC over stdio for IDE integration (IntelliJ, Zed, VS Code).
Implements the open Agent Client Protocol spec:
  - initialize: Capability negotiation
  - agent/message: User message with editor context
  - agent/stream: Streaming response deltas
  - agent/edit: Apply file edits via IDE
  - agent/terminal: Run commands via IDE terminal
  - shutdown: Clean disconnect
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, AsyncIterator, TYPE_CHECKING

from march.channels.base import Channel
from march.core.session import Session
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.llm.base import StreamChunk

logger = get_logger("march.acp")

# ACP JSON-RPC version
JSONRPC_VERSION = "2.0"

# ACP Protocol version
ACP_PROTOCOL_VERSION = "0.1"


class ACPChannel(Channel):
    """ACP (Agent Client Protocol) channel via stdio JSON-RPC.

    Reads JSON-RPC messages from stdin, processes them, writes responses
    to stdout. Designed for IDE integration where the IDE spawns
    `march acp` and communicates over stdin/stdout.
    """

    name: str = "acp"

    def __init__(self) -> None:
        self._agent: Agent | None = None
        self._session: Session | None = None
        self._running = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._capabilities: dict[str, Any] = {}
        self._initialized = False

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the ACP channel, reading from stdin."""
        self._agent = agent
        self._running = True

        # Set up async stdin/stdout
        loop = asyncio.get_running_loop()
        self._reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        transport, _ = await loop.connect_write_pipe(
            lambda: asyncio.Protocol(), sys.stdout
        )
        self._writer = asyncio.StreamWriter(
            transport, protocol, self._reader, loop
        )

        logger.info("acp channel started")

        # Main message loop
        while self._running:
            try:
                message = await self._read_message()
                if message is None:
                    break
                await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("acp error: %s", e)

    async def stop(self) -> None:
        """Stop the ACP channel."""
        self._running = False

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a complete response."""
        request_id = kwargs.get("request_id")
        if request_id is not None:
            await self._send_response(request_id, {"content": content})

    async def send_stream(
        self, chunks: AsyncIterator["StreamChunk"], **kwargs: Any
    ) -> None:
        """Send streaming response via JSON-RPC notifications."""
        request_id = kwargs.get("request_id")
        collected = ""

        async for chunk in chunks:
            if hasattr(chunk, "delta") and chunk.delta:
                collected += chunk.delta
                await self._send_notification("agent/stream", {
                    "delta": chunk.delta,
                })

        # Final complete message
        if request_id is not None:
            await self._send_response(request_id, {"content": collected})

    # ── Message I/O ──────────────────────────────────────────────────────

    async def _read_message(self) -> dict[str, Any] | None:
        """Read a JSON-RPC message from stdin.

        ACP uses Content-Length headers (like LSP):
        Content-Length: <len>\r\n
        \r\n
        <JSON payload>
        """
        assert self._reader is not None

        # Read headers
        content_length = 0
        while True:
            line = await self._reader.readline()
            if not line:
                return None
            line_str = line.decode("utf-8").strip()
            if not line_str:
                break  # Empty line = end of headers
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":")[1].strip())

        if content_length == 0:
            return None

        # Read body
        body = await self._reader.readexactly(content_length)
        return json.loads(body.decode("utf-8"))

    async def _write_message(self, data: dict[str, Any]) -> None:
        """Write a JSON-RPC message to stdout with Content-Length header."""
        assert self._writer is not None
        body = json.dumps(data)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self._writer.write(header.encode("utf-8"))
        self._writer.write(body.encode("utf-8"))
        await self._writer.drain()

    async def _send_response(self, request_id: Any, result: Any) -> None:
        """Send a JSON-RPC response."""
        await self._write_message({
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "result": result,
        })

    async def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send a JSON-RPC error response."""
        await self._write_message({
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        })

    async def _send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no id = no response expected)."""
        msg: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        await self._write_message(msg)

    # ── Message Handlers ─────────────────────────────────────────────────

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Route a JSON-RPC message to the appropriate handler."""
        method = message.get("method", "")
        params = message.get("params", {})
        request_id = message.get("id")

        handlers = {
            "initialize": self._handle_initialize,
            "agent/message": self._handle_agent_message,
            "agent/stream": self._handle_agent_stream,
            "agent/edit": self._handle_agent_edit,
            "agent/terminal": self._handle_agent_terminal,
            "shutdown": self._handle_shutdown,
        }

        handler = handlers.get(method)
        if handler:
            await handler(request_id, params)
        elif request_id is not None:
            await self._send_error(request_id, -32601, f"Method not found: {method}")

    async def _handle_initialize(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle initialize request — capability negotiation."""
        client_caps = params.get("capabilities", {})
        self._capabilities = client_caps

        # Create session based on workspace path if provided
        workspace = params.get("workspacePath", "")
        source_id = workspace or "acp-default"
        self._session = Session(
            source_type="acp",
            source_id=source_id,
        )
        self._initialized = True

        # Respond with server capabilities
        await self._send_response(request_id, {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "capabilities": {
                "streaming": True,
                "tools": True,
                "edits": True,
                "terminal": True,
            },
            "serverInfo": {
                "name": "march",
                "version": "0.1.0",
            },
        })

    async def _handle_agent_message(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle agent/message — non-streaming response."""
        assert self._agent is not None

        if not self._initialized:
            await self._send_error(request_id, -32002, "Not initialized")
            return

        text = params.get("message", "").strip()
        if not text:
            await self._send_error(request_id, -32602, "Empty message")
            return

        if not self._session:
            self._session = Session(source_type="acp", source_id="acp-default")

        # Inject editor context if provided
        context_parts = []
        if params.get("activeFile"):
            context_parts.append(f"Active file: {params['activeFile']}")
        if params.get("selection"):
            context_parts.append(f"Selection:\n```\n{params['selection']}\n```")
        if context_parts:
            text = "\n".join(context_parts) + "\n\n" + text

        try:
            response = await self._agent.run(text, self._session)
            await self._send_response(request_id, {
                "content": response.content,
                "totalTokens": response.total_tokens,
                "totalCost": response.total_cost,
                "toolCallsMade": response.tool_calls_made,
            })
        except Exception as e:
            await self._send_error(request_id, -32000, str(e))

    async def _handle_agent_stream(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle agent/stream — streaming response via notifications."""
        assert self._agent is not None

        if not self._initialized:
            await self._send_error(request_id, -32002, "Not initialized")
            return

        text = params.get("message", "").strip()
        if not text:
            await self._send_error(request_id, -32602, "Empty message")
            return

        if not self._session:
            self._session = Session(source_type="acp", source_id="acp-default")

        collected = ""
        final_response = None

        try:
            from march.core.agent import AgentResponse
            async for item in self._agent.run_stream(text, self._session):
                if isinstance(item, AgentResponse):
                    final_response = item
                    break

                if hasattr(item, "delta") and item.delta:
                    collected += item.delta
                    await self._send_notification("agent/stream", {
                        "delta": item.delta,
                    })

                if hasattr(item, "tool_call_delta") and item.tool_call_delta:
                    await self._send_notification("agent/tool", {
                        "tool": item.tool_call_delta,
                    })

            result: dict[str, Any] = {"content": collected}
            if final_response:
                result.update({
                    "totalTokens": final_response.total_tokens,
                    "totalCost": final_response.total_cost,
                    "toolCallsMade": final_response.tool_calls_made,
                })
            await self._send_response(request_id, result)

        except Exception as e:
            await self._send_error(request_id, -32000, str(e))

    async def _handle_agent_edit(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle agent/edit — apply file edits via IDE.

        The agent suggests edits; the IDE applies them through its API.
        """
        # Acknowledge — actual edit application happens on IDE side
        await self._send_response(request_id, {
            "status": "acknowledged",
            "message": "Edit requests are handled by the IDE",
        })

    async def _handle_agent_terminal(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle agent/terminal — run commands via IDE terminal.

        The agent requests command execution; the IDE runs it in its terminal.
        """
        await self._send_response(request_id, {
            "status": "acknowledged",
            "message": "Terminal requests are handled by the IDE",
        })

    async def _handle_shutdown(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle shutdown — clean disconnect."""
        self._running = False
        await self._send_response(request_id, {"status": "shutdown"})
