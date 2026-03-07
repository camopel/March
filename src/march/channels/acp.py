"""ACP (Agent Client Protocol) channel for March.

Implements the official ACP spec (https://agentclientprotocol.com).
JSON-RPC 2.0 over stdio with newline-delimited framing.

Flow:
  1. Client → Agent: initialize (version + capabilities)
  2. Client → Agent: session/new
  3. Client → Agent: session/prompt
  4. Agent → Client: session/update notifications (streaming)
  5. Agent → Client: session/prompt response (stopReason)
  6. Client → Agent: session/cancel (optional)
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any, AsyncIterator, TYPE_CHECKING

from march.channels.base import Channel
from march.core.session import Session
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.llm.base import StreamChunk

logger = get_logger("march.acp")

# ACP protocol version (integer, per spec)
ACP_PROTOCOL_VERSION = 1
JSONRPC = "2.0"


class ACPChannel(Channel):
    """ACP channel — JSON-RPC 2.0 over stdio, newline-delimited.

    Designed for IDE integration (IntelliJ, Zed, VS Code).
    The IDE spawns `march start --channel acp` and communicates via stdin/stdout.
    """

    name: str = "acp"

    def __init__(self) -> None:
        self._agent: Agent | None = None
        self._sessions: dict[str, Session] = {}
        self._running = False
        self._initialized = False
        self._client_capabilities: dict[str, Any] = {}
        self._current_task: asyncio.Task | None = None

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the ACP channel, reading newline-delimited JSON from stdin."""
        self._agent = agent
        self._running = True

        # ACP uses stdout for JSON-RPC — redirect all logging to stderr
        _redirect_logging_to_stderr()

        logger.info("acp channel started")

        # Read lines from stdin
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_running_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    break  # EOF
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError as e:
                    logger.error("invalid JSON: %s", e)
                    continue

                await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("acp error: %s", e)

    async def stop(self) -> None:
        """Stop the ACP channel."""
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a complete response (used by agent internals)."""
        session_id = kwargs.get("session_id")
        if session_id:
            await self._send_update(session_id, {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": content},
            })

    async def send_stream(
        self, chunks: AsyncIterator["StreamChunk"], **kwargs: Any
    ) -> None:
        """Send streaming response via session/update notifications."""
        session_id = kwargs.get("session_id", "")
        async for chunk in chunks:
            if hasattr(chunk, "delta") and chunk.delta:
                await self._send_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": chunk.delta},
                })

    # ── I/O ──────────────────────────────────────────────────────────────

    def _write(self, data: dict[str, Any]) -> None:
        """Write a newline-delimited JSON message to stdout."""
        line = json.dumps(data, separators=(",", ":")) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    def _send_response(self, request_id: Any, result: Any) -> None:
        """Send a JSON-RPC response."""
        self._write({"jsonrpc": JSONRPC, "id": request_id, "result": result})

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send a JSON-RPC error."""
        self._write({
            "jsonrpc": JSONRPC,
            "id": request_id,
            "error": {"code": code, "message": message},
        })

    async def _send_update(self, session_id: str, update: dict[str, Any]) -> None:
        """Send a session/update notification."""
        self._write({
            "jsonrpc": JSONRPC,
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update},
        })

    async def _request_permission(
        self, session_id: str, tool_name: str, args: dict[str, Any]
    ) -> str:
        """Request tool permission from the client. Returns outcome."""
        request_id = str(uuid.uuid4())
        self._write({
            "jsonrpc": JSONRPC,
            "id": request_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": session_id,
                "permissions": [{
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                    "toolName": tool_name,
                    "args": args,
                    "options": [
                        {"id": "allow-once", "label": "Allow Once"},
                        {"id": "allow-always", "label": "Allow Always"},
                        {"id": "reject-once", "label": "Reject"},
                    ],
                }],
            },
        })
        # In a real implementation, we'd wait for the client's response.
        # For now, auto-approve (the agent's safety plugin handles blocking).
        return "allow-once"

    # ── Message Routing ──────────────────────────────────────────────────

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Route a JSON-RPC message."""
        method = message.get("method", "")
        params = message.get("params", {})
        request_id = message.get("id")

        # Notifications (no id) — handle inline
        if method == "session/cancel":
            self._handle_cancel(params)
            return

        # Methods (have id) — need response
        handlers = {
            "initialize": self._handle_initialize,
            "session/new": self._handle_session_new,
            "session/load": self._handle_session_load,
            "session/prompt": self._handle_session_prompt,
            "session/set_mode": self._handle_set_mode,
            "shutdown": self._handle_shutdown,
        }

        handler = handlers.get(method)
        if handler:
            try:
                await handler(request_id, params)
            except Exception as e:
                logger.error("handler error for %s: %s", method, e)
                if request_id is not None:
                    self._send_error(request_id, -32000, str(e))
        elif request_id is not None:
            self._send_error(request_id, -32601, f"Method not found: {method}")

    # ── Handlers ─────────────────────────────────────────────────────────

    async def _handle_initialize(self, request_id: Any, params: dict[str, Any]) -> None:
        """Initialize — negotiate version and capabilities."""
        self._client_capabilities = params.get("clientCapabilities", {})
        client_info = params.get("clientInfo", {})
        requested_version = params.get("protocolVersion", 1)

        logger.info(
            "acp initialize: client=%s version=%s",
            client_info.get("name", "unknown"),
            requested_version,
        )

        # Respond with our version (match if we support it)
        version = min(requested_version, ACP_PROTOCOL_VERSION)

        self._initialized = True
        self._send_response(request_id, {
            "protocolVersion": version,
            "agentCapabilities": {
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": True,
                },
            },
            "agentInfo": {
                "name": "march",
                "title": "March Agent",
                "version": "0.1.0",
            },
            "authMethods": [],
        })

    async def _handle_session_new(self, request_id: Any, params: dict[str, Any]) -> None:
        """Create a new session."""
        if not self._initialized:
            self._send_error(request_id, -32002, "Not initialized")
            return

        session_id = str(uuid.uuid4())
        workspace = params.get("workspacePath", "")

        self._sessions[session_id] = Session(
            source_type="acp",
            source_id=session_id,
        )

        logger.info("acp session created: %s workspace=%s", session_id, workspace)
        self._send_response(request_id, {"sessionId": session_id})

    async def _handle_session_load(self, request_id: Any, params: dict[str, Any]) -> None:
        """Load an existing session (not supported yet)."""
        self._send_error(request_id, -32601, "session/load not supported")

    async def _handle_session_prompt(self, request_id: Any, params: dict[str, Any]) -> None:
        """Handle a user prompt — run the agent and stream updates."""
        assert self._agent is not None

        session_id = params.get("sessionId", "")
        session = self._sessions.get(session_id)
        if not session:
            self._send_error(request_id, -32602, f"Unknown session: {session_id}")
            return

        # Extract text from content blocks
        content_blocks = params.get("content", [])
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "resource_link":
                    text_parts.append(f"[Resource: {block.get('uri', '')}]")
                elif block.get("type") == "resource":
                    # Embedded context
                    text_parts.append(block.get("text", block.get("data", "")))
            elif isinstance(block, str):
                text_parts.append(block)

        text = "\n".join(text_parts).strip()
        if not text:
            self._send_error(request_id, -32602, "Empty prompt")
            return

        logger.info("acp prompt: session=%s len=%d", session_id[:8], len(text))

        # Run agent
        try:
            response = await self._agent.run(text, session)

            # Send the full response as a message chunk
            if response.content:
                await self._send_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": response.content},
                })

            # Respond with stop reason
            self._send_response(request_id, {
                "stopReason": "endTurn",
                "_meta": {
                    "totalTokens": response.total_tokens,
                    "totalCost": response.total_cost,
                    "toolCallsMade": response.tool_calls_made,
                },
            })

        except asyncio.CancelledError:
            self._send_response(request_id, {"stopReason": "cancelled"})
        except Exception as e:
            logger.error("acp prompt error: %s", e)
            self._send_error(request_id, -32000, str(e))

    async def _handle_set_mode(self, request_id: Any, params: dict[str, Any]) -> None:
        """Set session mode (agent/plan/ask)."""
        mode = params.get("mode", "agent")
        logger.info("acp mode set: %s", mode)
        self._send_response(request_id, {"mode": mode})

    def _handle_cancel(self, params: dict[str, Any]) -> None:
        """Cancel ongoing operation (notification — no response)."""
        session_id = params.get("sessionId", "")
        logger.info("acp cancel: session=%s", session_id[:8] if session_id else "?")
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def _handle_shutdown(self, request_id: Any, params: dict[str, Any]) -> None:
        """Shutdown — clean disconnect."""
        logger.info("acp shutdown")
        self._running = False
        self._send_response(request_id, {})


def _redirect_logging_to_stderr() -> None:
    """Ensure ALL output goes to stderr (ACP uses stdout exclusively for JSON-RPC)."""
    import logging

    # Redirect all logging handlers from stdout to stderr
    for name in list(logging.Logger.manager.loggerDict) + ['']:
        lgr = logging.getLogger(name)
        for handler in getattr(lgr, 'handlers', []):
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is sys.stdout:
                    handler.stream = sys.stderr

    # Reconfigure structlog to use stderr
    try:
        import structlog
        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
    except Exception:
        pass
