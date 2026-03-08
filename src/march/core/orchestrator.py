"""Orchestrator — unified agent control layer between channels and the agent.

The Orchestrator owns:
  - Session cache (in-memory dict[str, Session])
  - Message persistence (save user msg on receive, assistant msg on completion)
  - Cancel support (checks cancel_event between LLM calls and tool executions)
  - Clean history (only user + final assistant reply in session.history)
  - Translation of StreamChunks into OrchestratorEvents for channels

Channels call ``orchestrator.handle_message()`` and iterate over
``OrchestratorEvent`` objects — they never touch the Agent or SessionStore
directly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

from march.core.agent import Agent, AgentResponse, _extract_text
from march.core.message import Message
from march.core.session import Session, SessionStore
from march.core.turn_log import TurnLogger
from march.llm.base import StreamChunk
from march.logging import get_logger

logger = get_logger("march.orchestrator", subsystem="orchestrator")


# ─── OrchestratorEvent types ────────────────────────────────────────────────


@dataclass(frozen=True)
class TextDelta:
    """Streaming text fragment from the LLM."""

    delta: str


@dataclass(frozen=True)
class ToolProgress:
    """Tool execution progress update."""

    name: str
    status: str  # "started" | "complete" | "error"
    summary: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class FinalResponse:
    """Turn complete — the final assistant reply with metadata."""

    content: str
    tool_calls_made: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    turn_summary: str = ""


@dataclass(frozen=True)
class Error:
    """An error occurred during the turn."""

    message: str


@dataclass(frozen=True)
class Cancelled:
    """The turn was cancelled by the user."""

    partial_content: str = ""


# Union of all event types that handle_message can yield
OrchestratorEvent = Union[TextDelta, ToolProgress, FinalResponse, Error, Cancelled]


# ─── Orchestrator ────────────────────────────────────────────────────────────


class Orchestrator:
    """Unified agent control layer that sits between channels and the Agent.

    Channels create one Orchestrator at startup and call ``handle_message``
    for every inbound user message.  The orchestrator:

    1. Manages an in-memory session cache (cold-start loads from DB).
    2. Persists user messages on receive and assistant messages on completion.
    3. Delegates LLM + tool execution to ``Agent.run_stream()``.
    4. Translates ``StreamChunk`` / ``AgentResponse`` into typed
       ``OrchestratorEvent`` objects.
    5. Supports cooperative cancellation via ``asyncio.Event``.
    6. Keeps session history clean — only user + final assistant reply are
       stored; tool-call intermediates live in ephemeral turn context only.

    Args:
        agent: The core Agent instance (handles LLM calls, tool execution,
               context building, compaction, etc.).
        session_store: SQLite-backed session persistence layer.
    """

    def __init__(self, agent: Agent, session_store: SessionStore) -> None:
        self.agent = agent
        self.session_store = session_store
        # In-memory session cache — avoids DB round-trips on every turn.
        self._sessions: dict[str, Session] = {}
        # Structured JSONL turn logger for debugging.
        self._turn_log = TurnLogger()

    # ── Public API ───────────────────────────────────────────────────────

    async def handle_message(
        self,
        session_id: str,
        content: str | list,
        source: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[OrchestratorEvent]:
        """Process one user message and yield events as the agent works.

        This is the **single entry-point** for all channels.

        Args:
            session_id: Deterministic session identifier (from the channel).
            content: User message — plain text or multimodal content blocks.
            source: Human-readable source label (e.g. ``"matrix"``, ``"ws"``).
            cancel_event: Optional event that, when set, signals the turn
                should be cancelled at the next safe checkpoint.

        Yields:
            ``OrchestratorEvent`` instances in order:
            - Zero or more ``TextDelta`` (streaming LLM text)
            - Zero or more ``ToolProgress`` (tool execution updates)
            - Exactly one terminal event: ``FinalResponse``, ``Error``,
              or ``Cancelled``.
        """
        cancel = cancel_event or asyncio.Event()
        turn_id = uuid.uuid4().hex
        turn_start_time = time.monotonic()

        # 1. Resolve session (cache hit or cold-start from DB)
        try:
            session = await self._get_or_create_session(session_id, source)
        except Exception as exc:
            logger.error("session load failed", session_id=session_id, error=str(exc))
            self._turn_log.turn_error(turn_id=turn_id, session_id=session_id, error=f"session load failed: {exc}")
            yield Error(message=f"Failed to load session: {exc}")
            return

        # Log turn start
        user_text = _extract_text(content) if not isinstance(content, str) else content
        self._turn_log.turn_start(turn_id=turn_id, session_id=session_id, user_msg=user_text, source=source)

        # 2. Persist the user message to DB
        try:
            user_msg = Message.user(content)
            await self.session_store.add_message(session_id, user_msg)
        except Exception as exc:
            logger.warning(
                "failed to persist user message (non-fatal)",
                session_id=session_id,
                error=str(exc),
            )

        # 2b. Deliver any pending sub-agent announcements
        agent_manager = getattr(self.agent, "agent_manager", None)
        if agent_manager is not None:
            announcer = getattr(agent_manager, "announcer", None)
            if announcer is not None and hasattr(announcer, "get_pending"):
                try:
                    pending = announcer.get_pending(session_id)
                    if pending:
                        for announcement in pending:
                            session.add_system_message(announcement)
                except Exception as exc:
                    logger.warning(
                        "failed to deliver pending announcements (non-fatal)",
                        session_id=session_id,
                        error=str(exc),
                    )

        # 3. Check for early cancellation
        if cancel.is_set():
            self._turn_log.turn_cancelled(turn_id=turn_id, session_id=session_id, partial_content_length=0)
            yield Cancelled(partial_content="")
            return

        # 4. Delegate to Agent.run_stream() and translate events
        partial_content = ""
        _llm_call_count = 0
        try:
            async for item in self.agent.run_stream(content, session):
                # ── Cancel checkpoint ────────────────────────────────
                if cancel.is_set():
                    self._turn_log.turn_cancelled(
                        turn_id=turn_id, session_id=session_id,
                        partial_content_length=len(partial_content),
                    )
                    yield Cancelled(partial_content=partial_content)
                    return

                if isinstance(item, StreamChunk):
                    # Streaming text delta
                    if item.delta:
                        partial_content += item.delta
                        yield TextDelta(delta=item.delta)

                    # LLM usage data (emitted on final chunk of each LLM call)
                    if item.usage and (item.usage.input_tokens or item.usage.output_tokens):
                        _llm_call_count += 1
                        self._turn_log.llm_call(
                            turn_id=turn_id,
                            session_id=session_id,
                            provider="",
                            model="",
                            input_tokens=item.usage.input_tokens,
                            output_tokens=item.usage.output_tokens,
                            cost=item.usage.cost,
                            duration_ms=0.0,
                        )

                    # Tool progress embedded in StreamChunk
                    if item.tool_progress:
                        tp = item.tool_progress
                        self._turn_log.tool_call(
                            turn_id=turn_id,
                            session_id=session_id,
                            name=tp.get("name", ""),
                            args={},
                            duration_ms=tp.get("duration_ms", 0.0),
                            status=tp.get("status", ""),
                            summary=tp.get("summary", ""),
                        )
                        yield ToolProgress(
                            name=tp.get("name", ""),
                            status=tp.get("status", ""),
                            summary=tp.get("summary", ""),
                            duration_ms=tp.get("duration_ms", 0.0),
                        )

                elif isinstance(item, AgentResponse):
                    # Turn complete — Agent already called session.add_exchange()
                    # so history is up to date.  Persist the assistant message.
                    try:
                        assistant_msg = Message.assistant(item.content)
                        await self.session_store.add_message(session_id, assistant_msg)
                    except Exception as exc:
                        logger.warning(
                            "failed to persist assistant message (non-fatal)",
                            session_id=session_id,
                            error=str(exc),
                        )

                    turn_duration = (time.monotonic() - turn_start_time) * 1000
                    self._turn_log.turn_complete(
                        turn_id=turn_id,
                        session_id=session_id,
                        tool_calls=item.tool_calls_made,
                        total_tokens=item.total_tokens,
                        total_cost=item.total_cost,
                        duration_ms=turn_duration,
                        final_reply_length=len(item.content),
                    )

                    yield FinalResponse(
                        content=item.content,
                        tool_calls_made=item.tool_calls_made,
                        total_tokens=item.total_tokens,
                        total_cost=item.total_cost,
                        turn_summary=item.turn_summary,
                    )
                    return

        except asyncio.CancelledError:
            self._turn_log.turn_cancelled(
                turn_id=turn_id, session_id=session_id,
                partial_content_length=len(partial_content),
            )
            yield Cancelled(partial_content=partial_content)
            return
        except Exception as exc:
            logger.error(
                "agent run_stream failed",
                session_id=session_id,
                error=str(exc),
                exc_info=True,
            )
            self._turn_log.turn_error(turn_id=turn_id, session_id=session_id, error=str(exc))
            yield Error(message=str(exc))
            return

        # If we exit the loop without yielding a terminal event (shouldn't
        # happen in normal flow), emit an error so the channel isn't left
        # hanging.
        self._turn_log.turn_error(
            turn_id=turn_id, session_id=session_id,
            error="Agent stream ended without a final response.",
        )
        yield Error(message="Agent stream ended without a final response.")

    async def reset_session(self, session_id: str) -> dict[str, Any]:
        """Reset a session — clear in-memory cache, DB, and agent memory.

        Performs a full reset:
          1. Evict from in-memory session cache
          2. Clear messages from the session store (DB)
          3. Reset agent memory for this session (facts, embeddings, etc.)
          4. Delete session memory files on disk (facts.md, plan.md, etc.)

        Args:
            session_id: The session to reset.

        Returns:
            Dict with cleanup details (e.g. memory entries cleared).
        """
        result: dict[str, Any] = {}

        # 1. Clear from cache
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.clear()

        # 2. Clear from DB
        try:
            await self.session_store.clear_session(session_id)
        except Exception as exc:
            logger.error("session reset DB clear failed", session_id=session_id, error=str(exc))
            raise

        # 3. Reset agent memory (MemoryStore: facts, embeddings, etc.)
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            try:
                mem_result = await self.agent.memory.reset_session(session_id)
                result["memory"] = mem_result
            except Exception as exc:
                logger.warning(
                    "session reset memory clear failed (non-fatal)",
                    session_id=session_id,
                    error=str(exc),
                )

        # 4. Delete session memory files on disk
        try:
            from march.core.compaction import delete_session_memory
            if delete_session_memory(session_id):
                result["session_memory_deleted"] = True
        except Exception as exc:
            logger.warning(
                "session reset memory file delete failed (non-fatal)",
                session_id=session_id,
                error=str(exc),
            )

        # 5. Reset sub-agent data: kill active children, delete their sessions/memory
        agent_manager = getattr(self.agent, "agent_manager", None)
        if agent_manager is not None:
            try:
                children_cleaned = await agent_manager.reset_children(session_id)
                result["children_cleaned"] = children_cleaned
            except Exception as exc:
                logger.warning(
                    "session reset children cleanup failed (non-fatal)",
                    session_id=session_id,
                    error=str(exc),
                )

        return result

    def get_cached_session(self, session_id: str) -> Session | None:
        """Return the cached session if present, else ``None``.

        This is a read-only peek — channels can use it to check session
        state without triggering a DB load.
        """
        return self._sessions.get(session_id)

    def evict_session(self, session_id: str) -> None:
        """Remove a session from the in-memory cache.

        The session is not deleted from the DB — it will be reloaded on the
        next ``handle_message`` call.
        """
        self._sessions.pop(session_id, None)

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _get_or_create_session(
        self,
        session_id: str,
        source: str,
    ) -> Session:
        """Resolve a session from cache or DB (cold-start).

        On cold start the session is loaded from the DB and its message
        history is populated.  The session is then cached for subsequent
        turns.
        """
        # Fast path: cache hit
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Cold start: try loading from DB
        session = await self.session_store.get_session(session_id)
        if session is not None:
            # Load message history into the session object
            messages = await self.session_store.get_messages(session_id)
            session.history = messages
            self._sessions[session_id] = session
            logger.info(
                "session loaded from DB (cold start)",
                session_id=session_id,
                message_count=len(messages),
            )
            return session

        # Brand-new session — create in DB and cache
        session = await self.session_store.create_session(
            source_type=source,
            source_id=session_id,
            name="",
            session_id=session_id,
        )
        self._sessions[session_id] = session
        logger.info("new session created", session_id=session_id, source=source)
        return session
