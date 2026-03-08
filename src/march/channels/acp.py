"""ACP (Agent Client Protocol) channel for March.

Implements the official ACP spec (https://agentclientprotocol.com).
JSON-RPC 2.0 over stdio with newline-delimited framing.

This channel is a **pure I/O adapter**: it translates between the ACP wire
protocol and the Orchestrator layer.  It does NOT touch the Agent or
SessionStore directly.

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
from march.core.orchestrator import (
    Cancelled,
    Error,
    FinalResponse,
    Orchestrator,
    TextDelta,
    ToolProgress,
)
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent
    from march.llm.base import StreamChunk

logger = get_logger("march.acp")

# ACP protocol version (integer, per spec)
ACP_PROTOCOL_VERSION = 1
JSONRPC = "2.0"


class ACPChannel(Channel):
    """ACP channel — JSON-RPC 2.0 over stdio, newline-delimited.

    Designed for IDE integration (IntelliJ, Zed, VS Code).
    The IDE spawns `march start --channel acp` and communicates via stdin/stdout.

    This channel is a pure I/O adapter.  All session management, agent
    execution, and message persistence are delegated to the Orchestrator.
    """

    name: str = "acp"

    def __init__(self) -> None:
        self._orchestrator: Orchestrator | None = None
        self._running = False
        self._initialized = False
        self._client_capabilities: dict[str, Any] = {}
        # Per-session cancel events — keyed by session ID
        self._cancel_events: dict[str, asyncio.Event] = {}
        # Per-session prompt tasks — for cancellation via asyncio.Task.cancel()
        self._prompt_tasks: dict[str, asyncio.Task] = {}

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the ACP channel, reading newline-delimited JSON from stdin.

        The ``orchestrator`` keyword argument is **required**.  The channel
        does not interact with the Agent or SessionStore directly.
        """
        orchestrator = kwargs.get("orchestrator")
        if orchestrator is None:
            raise ValueError(
                "ACPChannel requires an 'orchestrator' keyword argument"
            )
        self._orchestrator = orchestrator
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
        # Cancel all in-flight prompt tasks
        for task in self._prompt_tasks.values():
            if not task.done():
                task.cancel()
        self._prompt_tasks.clear()
        self._cancel_events.clear()

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
            "session/destroy": self._handle_session_destroy,
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

    async def _handle_initialize(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
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

    async def _handle_session_new(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
        """Create a new session via the Orchestrator."""
        assert self._orchestrator is not None

        if not self._initialized:
            self._send_error(request_id, -32002, "Not initialized")
            return

        workspace = params.get("workspacePath", "")

        # Build a deterministic session ID from the workspace path so
        # reconnecting to the same workspace resumes the same session.
        from march.core.session import deterministic_session_id

        session_id = deterministic_session_id("acp", workspace or str(uuid.uuid4()))

        # Warm the Orchestrator's session cache by issuing a no-op peek.
        # The Orchestrator will lazily create the session on the first
        # handle_message() call, so we just need to track the ID here.
        logger.info("acp session created: %s workspace=%s", session_id, workspace)
        self._send_response(request_id, {"sessionId": session_id})

    async def _handle_session_destroy(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
        """Destroy a session — reset via Orchestrator."""
        assert self._orchestrator is not None

        session_id = params.get("sessionId", "")
        if not session_id:
            self._send_error(request_id, -32602, "Missing sessionId")
            return

        # Cancel any in-flight prompt for this session
        self._signal_cancel(session_id)

        try:
            await self._orchestrator.reset_session(session_id)
        except Exception as e:
            logger.error("session destroy failed: %s", e)
            self._send_error(request_id, -32000, str(e))
            return

        # Clean up local cancel state
        self._cancel_events.pop(session_id, None)
        self._prompt_tasks.pop(session_id, None)

        logger.info("acp session destroyed: %s", session_id[:8])
        self._send_response(request_id, {})

    async def _handle_session_prompt(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
        """Handle a user prompt — delegate to Orchestrator and stream events."""
        assert self._orchestrator is not None

        session_id = params.get("sessionId", "")
        if not session_id:
            self._send_error(request_id, -32602, "Missing sessionId")
            return

        # Extract text from ACP content blocks
        content_blocks = params.get("content", [])
        text_parts: list[str] = []
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

        # Create a fresh cancel event for this turn
        cancel_event = asyncio.Event()
        self._cancel_events[session_id] = cancel_event

        # Run the prompt in a task so session/cancel can interrupt it
        task = asyncio.create_task(
            self._run_prompt(request_id, session_id, text, cancel_event)
        )
        self._prompt_tasks[session_id] = task

    async def _run_prompt(
        self,
        request_id: Any,
        session_id: str,
        text: str,
        cancel_event: asyncio.Event,
    ) -> None:
        """Stream OrchestratorEvents and translate them to ACP wire messages."""
        assert self._orchestrator is not None

        try:
            async for event in self._orchestrator.handle_message(
                session_id=session_id,
                content=text,
                source="acp",
                cancel_event=cancel_event,
            ):
                if isinstance(event, TextDelta):
                    await self._send_update(session_id, {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": event.delta},
                    })

                elif isinstance(event, ToolProgress):
                    await self._send_update(session_id, {
                        "sessionUpdate": "tool_call",
                        "toolName": event.name,
                        "status": event.status,
                        "summary": event.summary,
                        "durationMs": event.duration_ms,
                    })

                elif isinstance(event, FinalResponse):
                    self._send_response(request_id, {
                        "stopReason": "endTurn",
                        "_meta": {
                            "totalTokens": event.total_tokens,
                            "totalCost": event.total_cost,
                            "toolCallsMade": event.tool_calls_made,
                        },
                    })
                    return

                elif isinstance(event, Cancelled):
                    self._send_response(request_id, {
                        "stopReason": "cancelled",
                    })
                    return

                elif isinstance(event, Error):
                    self._send_error(request_id, -32000, event.message)
                    return

        except asyncio.CancelledError:
            self._send_response(request_id, {"stopReason": "cancelled"})
        except Exception as e:
            logger.error("acp prompt error: %s", e, exc_info=True)
            self._send_error(request_id, -32000, str(e))
        finally:
            # Clean up task tracking
            self._prompt_tasks.pop(session_id, None)
            self._cancel_events.pop(session_id, None)

    async def _handle_set_mode(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
        """Set session mode (agent/plan/ask)."""
        mode = params.get("mode", "agent")
        logger.info("acp mode set: %s", mode)
        self._send_response(request_id, {"mode": mode})

    def _handle_cancel(self, params: dict[str, Any]) -> None:
        """Cancel ongoing operation (notification — no response)."""
        session_id = params.get("sessionId", "")
        logger.info(
            "acp cancel: session=%s", session_id[:8] if session_id else "?"
        )
        self._signal_cancel(session_id)

    def _signal_cancel(self, session_id: str) -> None:
        """Signal cancellation for a session's in-flight prompt."""
        # Set the cooperative cancel event (Orchestrator checks this)
        cancel_event = self._cancel_events.get(session_id)
        if cancel_event is not None:
            cancel_event.set()

        # Also cancel the asyncio task as a fallback
        task = self._prompt_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()

    async def _handle_shutdown(
        self, request_id: Any, params: dict[str, Any]
    ) -> None:
        """Shutdown — clean disconnect."""
        logger.info("acp shutdown")
        self._running = False
        self._send_response(request_id, {})


def _redirect_logging_to_stderr() -> None:
    """Ensure ALL output goes to stderr (ACP uses stdout exclusively for JSON-RPC)."""
    import logging

    # Redirect all stdlib logging handlers from stdout to stderr
    for name in list(logging.Logger.manager.loggerDict) + [""]:
        lgr = logging.getLogger(name)
        for handler in getattr(lgr, "handlers", []):
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is sys.stdout:
                    handler.stream = sys.stderr

    # If structlog is using stdlib integration (LoggerFactory), the above
    # handler redirect is sufficient. If it's using PrintLogger/WriteLogger
    # directly, reconfigure to use stderr.
    try:
        import structlog

        cfg = structlog.get_config()
        factory = cfg.get("logger_factory")
        # Only reconfigure if NOT using stdlib (stdlib is already redirected above)
        if factory and not isinstance(factory, structlog.stdlib.LoggerFactory):
            structlog.configure(
                logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
            )
    except Exception:
        pass
