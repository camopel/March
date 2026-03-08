"""Core agent loop for the March agent framework.

This is the heart of March: receive a message, build context, call the LLM,
execute tools, and return a response. Supports both streaming and non-streaming modes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from march.core.context import Context, estimate_tokens
from march.core.message import ToolCall, ToolResult
from march.core.attachments import (
    strip_attachments_from_messages,
)
from march.core.session import Session
from march.logging import get_logger
from march.logging.logger import MarchLogger, MetricsLogger
from march.llm.base import LLMProvider, LLMResponse, ProviderError, StreamChunk
from march.llm.router import LLMRouter
from march.memory.store import MemoryStore
from march.plugins._manager import PluginManager
from march.tools.context import current_session_id
from march.tools.registry import ToolRegistry, ToolNotFound

logger = get_logger("march.agent", subsystem="agent")


MAX_LLM_RETRIES = 3  # Maximum retries on transient LLM errors
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff delays in seconds
DEFAULT_CONTEXT_WINDOW = 200000  # Default context window in tokens
MIN_MESSAGES_KEEP = 4  # Minimum number of recent messages to keep during truncation


def _extract_text(user_message: str | list) -> str:
    """Extract the text portion from a user message (str or multimodal list)."""
    if isinstance(user_message, str):
        return user_message
    # Multimodal: find the first text block
    for block in user_message:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


@dataclass
class AgentResponse:
    """The final response from the agent.

    Attributes:
        content: The text response to show the user.
        tool_calls_made: Number of tool calls executed during this turn.
        total_tokens: Total tokens used (input + output) across all LLM calls.
        total_cost: Total cost in USD across all LLM calls.
        duration_ms: Total wall-clock time for this agent turn.
        turn_summary: Short LLM-generated summary of the turn (≤200 words).
    """

    content: str = ""
    tool_calls_made: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
    turn_summary: str = ""


class Agent:
    """The core agent loop.

    Orchestrates: context building → plugin hooks → LLM calls → tool execution → response.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry,
        plugin_manager: PluginManager,
        memory_store: MemoryStore,
        config: Any = None,
    ):
        self.llm = llm_router
        self.tools = tool_registry
        self.plugins = plugin_manager
        self.memory = memory_store
        self.config = config
        self.agent_manager: Any = None  # Set by MarchApp after initialization
        self.session_store: Any = None  # Set by MarchApp for auto-persistence
        self._mlogger = MarchLogger()
        self._metrics = MetricsLogger.get()
        self._steering_queues: dict[str, asyncio.Queue] = {}  # session_id → Queue

    # ── Steering API ─────────────────────────────────────────────────

    def get_steering_queue(self, session_id: str) -> asyncio.Queue:
        """Get or create a steering queue for a session."""
        if session_id not in self._steering_queues:
            self._steering_queues[session_id] = asyncio.Queue()
        return self._steering_queues[session_id]

    def steer(self, session_id: str, message: str) -> bool:
        """Queue a steering message for a running session.

        Returns True if the session has an active queue (is running).
        """
        queue = self._steering_queues.get(session_id)
        if queue is None:
            return False
        queue.put_nowait(message)
        return True

    def _drain_steering(self, session_id: str) -> list[str]:
        """Drain all pending steering messages for a session."""
        queue = self._steering_queues.get(session_id)
        if queue is None:
            return []
        messages = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def handle_command(self, user_message: str, session: Session) -> AgentResponse | None:
        """Handle slash commands (/rmb, /reset). Returns AgentResponse if handled, None otherwise."""
        stripped = user_message.strip()

        if stripped.lower().startswith("/rmb"):
            instruction = stripped[4:].strip()
            if not instruction:
                return AgentResponse(content="Usage: /rmb <what to remember>")
            return await self._handle_rmb(instruction, session)

        if stripped.lower() == "/reset":
            return await self._handle_reset(session)

        return None

    async def _handle_rmb(self, instruction: str, session: Session) -> AgentResponse:
        """Handle /rmb command: LLM extracts key info, appends to MEMORY.md."""
        # 1. Gather recent conversation context
        recent_messages = session.get_messages_for_llm()[-20:]
        context_lines: list[str] = []
        for m in recent_messages:
            content = m.get("content", "")
            if content and isinstance(content, str):
                role = m.get("role", "unknown")
                context_lines.append(f"{role}: {content}")
        context_text = "\n".join(context_lines)

        # 2. Build extraction prompt
        extraction_prompt = (
            "You are a memory curator. The user wants to save something to long-term memory.\n\n"
            f"Their instruction: {instruction}\n\n"
            f"Recent conversation context:\n{context_text[:4000]}\n\n"
            "Extract the key information and write a concise memory entry. Rules:\n"
            "- Write 1-5 bullet points, no fluff\n"
            "- Preserve exact identifiers (paths, URLs, names, commands) in backticks\n"
            "- Include a short header (## or ###) describing the topic\n"
            "- If the instruction is self-contained, just distill it directly\n"
            "- If it references conversation context, extract the relevant facts\n\n"
            "Output ONLY the memory entry (markdown), nothing else."
        )

        # 3. Call LLM to extract memory entry
        try:
            provider = await self.llm.route()
            llm_response = await provider.converse(
                messages=[{"role": "user", "content": extraction_prompt}],
                system="You are a concise memory curator. Output only markdown.",
                tools=None,
            )
            memory_entry = llm_response.content.strip()
        except Exception as e:
            logger.error("rmb extraction failed", error=str(e))
            # Fallback: store the instruction directly
            memory_entry = f"### {instruction}"

        # 4. Append to MEMORY.md
        await self.memory.append_memory(memory_entry)

        return AgentResponse(content=f"✓ Remembered: {instruction}")

    async def _handle_reset(self, session: Session) -> AgentResponse:
        """Handle /reset: clear session data, keep global memory."""
        result = await self.memory.reset_session(session.id)
        session.clear()

        # Delete session memory files (facts.md, plan.md, etc.)
        from march.core.compaction import delete_session_memory
        deleted_memory = delete_session_memory(session.id)

        msg = f"✓ Session reset. Removed {result.get('sqlite_entries', 0)} database entries."
        if deleted_memory:
            msg += " Session memory cleared."
        return AgentResponse(content=msg)

    async def run(self, user_message: str | list, session: Session) -> AgentResponse:
        """The main agent loop. Called once per user message."""
        try:
            return await self._run_inner(user_message, session)
        except Exception as e:
            logger.error("unexpected error in agent loop", error=str(e), exc_info=True)
            return AgentResponse(
                content=f"An unexpected error occurred: {e}",
                duration_ms=0.0,
            )

    async def _run_inner(self, user_message: str | list, session: Session) -> AgentResponse:
        """Inner implementation of run() — separated for clean error handling."""
        # Check for slash commands first (only for text messages)
        if isinstance(user_message, str):
            cmd_response = await self.handle_command(user_message, session)
            if cmd_response is not None:
                return cmd_response

        # Create steering queue for this session's turn
        self._steering_queues[session.id] = asyncio.Queue()

        start_time = time.monotonic()
        total_tokens = 0
        total_cost = 0.0
        tool_calls_made = 0

        # Set session ID in context var so tools can access it
        current_session_id.set(session.id)

        # Bind session_id to structured logger for this turn
        mlog = self._mlogger.bind(session_id=session.id)

        # Log turn start
        text_preview = _extract_text(user_message)
        mlog.turn_start(session_id=session.id, message_length=len(text_preview))

        # 1. Build context
        context = await self._build_context(session)

        # 2. Plugin: before_llm (can modify context, message, or short-circuit)
        context, user_message, short_circuit = await self.plugins.dispatch_before_llm(
            context, user_message
        )
        if short_circuit is not None:
            # A plugin returned a direct response — skip the LLM
            return await self._finalize(
                content=short_circuit,
                user_message=user_message,
                session=session,
                tool_calls_made=0,
                total_tokens=0,
                total_cost=0.0,
                start_time=start_time,
            )

        # 3. Select LLM provider
        text_for_routing = _extract_text(user_message)
        try:
            provider = await self.llm.route(text_for_routing, context)
        except (RuntimeError, ProviderError) as e:
            await self.plugins.dispatch_on_llm_error(e)
            mlog.llm_error(provider="none", error=str(e), will_retry=False,
                           model="none")
            return AgentResponse(
                content=f"Error: No LLM provider available. {e}",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # 4. Build messages list: session messages + new user message
        messages = session.get_messages_for_llm()
        messages.append({"role": "user", "content": user_message})

        # Strip attachment data (images, etc.) from history messages.
        # Only the current request (last message) keeps full attachment data.
        messages = strip_attachments_from_messages(messages, skip_last=True)

        # 5. Compact or truncate messages if they exceed the context window
        from march.core.compaction import needs_compaction, split_for_compaction, \
            build_summary_prompt, MIN_RECENT_KEEP
        context_window = self._get_context_window()
        system_tokens = context.estimated_tokens

        # Get compaction config if available
        _compaction_threshold = None
        _summary_budget_ratio = None
        _dedup_max_ratio = None
        if self.config and hasattr(self.config, 'memory') and hasattr(self.config.memory, 'compaction'):
            _compaction_threshold = self.config.memory.compaction.threshold
            _summary_budget_ratio = self.config.memory.compaction.summary_budget_ratio
            _dedup_max_ratio = getattr(self.config.memory.compaction, 'dedup_max_ratio', None)

        if needs_compaction(messages, context_window, system_tokens, threshold=_compaction_threshold):
            old, recent = split_for_compaction(messages, context_window, system_tokens,
                                                summary_budget_ratio=_summary_budget_ratio)
            if old:
                async def _summarize_sync(prompt: str) -> str:
                    r = await provider.converse(
                        messages=[{"role": "user", "content": prompt}],
                        system="You are a helpful assistant.",
                        max_tokens=500,
                    )
                    return r.content.strip() if r and r.content else ""

                try:
                    new_rolling = await self._two_step_compaction(
                        session, messages, _summarize_sync,
                        context_window, _dedup_max_ratio,
                    )
                    session.compact(new_rolling)
                    logger.info(
                        "session compacted (non-streaming)",
                        session_id=session.id,
                        old_messages=len(old),
                    )
                    messages = session.get_messages_for_llm()
                    messages.append({"role": "user", "content": user_message})
                except Exception as e:
                    logger.warning("compaction failed, falling back to truncation",
                                   session_id=session.id, error=str(e))
                    messages = self._truncate_messages(messages, context)
            else:
                messages = self._truncate_messages(messages, context)
        else:
            messages = self._truncate_messages(messages, context)

        # 6. Agent loop (LLM → tools → LLM → ... until no more tool calls)
        system_prompt = context.build_system_prompt()

        tool_definitions = self.tools.definitions() if self.tools.tool_count > 0 else None

        # Unlimited tool loop — runs until LLM stops calling tools
        while True:
            # Call LLM with retry on transient errors
            llm_response = await self._call_llm_with_retry(
                provider, messages, system_prompt, tool_definitions, mlog=mlog
            )
            if llm_response is None:
                return AgentResponse(
                    content="Error: LLM call failed after retries.",
                    duration_ms=(time.monotonic() - start_time) * 1000,
                )

            # Track usage
            total_tokens += llm_response.usage.input_tokens + llm_response.usage.output_tokens
            total_cost += llm_response.usage.cost
            mlog.llm_call(
                provider=provider.name,
                model=provider.model,
                input_tokens=llm_response.usage.input_tokens,
                output_tokens=llm_response.usage.output_tokens,
                cost=llm_response.usage.cost,
                duration_ms=llm_response.duration_ms,
            )
            self._metrics.llm_call(
                session_id=session.id,
                provider=provider.name,
                model=provider.model,
                input_tokens=llm_response.usage.input_tokens,
                output_tokens=llm_response.usage.output_tokens,
                cost_usd=llm_response.usage.cost,
                duration_ms=llm_response.duration_ms,
            )

            # Plugin: after_llm
            llm_response = await self.plugins.dispatch_after_llm(context, llm_response)

            # If no tool calls, we're done
            if not llm_response.tool_calls:
                return await self._finalize(
                    content=llm_response.content,
                    user_message=user_message,
                    session=session,
                    tool_calls_made=tool_calls_made,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    start_time=start_time,
                )

            # Execute tool calls
            tool_results: list[ToolResult] = []
            steering_messages: list[str] = []
            for i, llm_tool_call in enumerate(llm_response.tool_calls):
                core_tool_call = ToolCall(
                    id=llm_tool_call.id,
                    name=llm_tool_call.name,
                    args=llm_tool_call.args,
                )

                modified_call = await self.plugins.dispatch_before_tool(core_tool_call)
                if modified_call is None:
                    tool_results.append(
                        ToolResult(
                            id=core_tool_call.id,
                            error=f"Tool '{core_tool_call.name}' was blocked by a plugin.",
                        )
                    )
                    continue

                core_tool_call = modified_call

                try:
                    _tool_t0 = time.monotonic()
                    result = await self.tools.execute(core_tool_call)
                    _tool_dur = (time.monotonic() - _tool_t0) * 1000
                    result = await self.plugins.dispatch_after_tool(core_tool_call, result)
                    mlog.tool_call(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        result_summary=result.summary[:200],
                        duration_ms=_tool_dur,
                    )
                    self._metrics.tool_call(
                        session_id=session.id,
                        tool=core_tool_call.name,
                        duration_ms=_tool_dur,
                    )
                except ToolNotFound:
                    mlog.tool_error(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        error=f"Unknown tool: {core_tool_call.name}",
                    )
                    result = ToolResult(
                        id=core_tool_call.id,
                        error=f"Unknown tool: {core_tool_call.name}",
                    )
                except Exception as e:
                    await self.plugins.dispatch_on_tool_error(core_tool_call, e)
                    mlog.tool_error(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        error=str(e),
                    )
                    result = ToolResult(
                        id=core_tool_call.id,
                        error=str(e),
                    )

                tool_results.append(result)
                tool_calls_made += 1

                # ── Steering checkpoint ──
                steering_messages = self._drain_steering(session.id)
                if steering_messages:
                    # Skip remaining tools
                    for remaining_tc in llm_response.tool_calls[i + 1:]:
                        skip_result = ToolResult(
                            id=remaining_tc.id,
                            error="Skipped due to queued user message.",
                        )
                        tool_results.append(skip_result)
                        tool_calls_made += 1
                    break  # Exit tool loop

            # Add assistant message (with tool calls) + tool results to messages
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": llm_response.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.args},
                    }
                    for tc in llm_response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tr in tool_results:
                content = tr.content if not tr.is_error else f"Error: {tr.error}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.id,
                        "content": content,
                    }
                )

            # Tool call intermediate messages stay in `messages` (for current
            # turn's LLM context) but are NOT added to session.messages.
            # Only the final assistant reply is saved via _finalize() →
            # session.add_exchange(). This keeps session messages clean:
            # [user, assistant, user, assistant, ...]

            # ── Inject steering messages as user messages ──
            if not steering_messages:
                steering_messages = self._drain_steering(session.id)
            if steering_messages:
                for steer_msg in steering_messages:
                    messages.append({"role": "user", "content": steer_msg})
                continue  # Skip truncation, go straight to next LLM call

            # Truncate messages if context is growing too large
            messages = self._truncate_messages(messages, context)

    async def _call_llm_with_retry(
        self,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_definitions: list[dict[str, Any]] | None,
        mlog: MarchLogger | None = None,
    ) -> LLMResponse | None:
        """Call the LLM with retry on transient errors."""
        last_error: Exception | None = None
        _mlog = mlog or self._mlogger
        for attempt in range(MAX_LLM_RETRIES):
            try:
                return await provider.converse(
                    messages=messages,
                    system=system_prompt,
                    tools=tool_definitions,
                )
            except ProviderError as e:
                last_error = e
                if e.retryable and attempt < MAX_LLM_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                    _mlog.llm_error(
                        provider=provider.name, error=str(e), will_retry=True,
                        attempt=attempt + 1, max_retries=MAX_LLM_RETRIES,
                        model=getattr(provider, 'model', ''),
                    )
                    await asyncio.sleep(delay)
                else:
                    await self.plugins.dispatch_on_llm_error(e)
                    _mlog.llm_error(
                        provider=provider.name, error=str(e), will_retry=False,
                        attempt=attempt + 1, max_retries=MAX_LLM_RETRIES,
                        model=getattr(provider, 'model', ''),
                    )
                    return None
            except Exception as e:
                last_error = e
                await self.plugins.dispatch_on_llm_error(e)
                _mlog.llm_error(
                    provider=provider.name, error=str(e), will_retry=False,
                    attempt=attempt + 1, max_retries=MAX_LLM_RETRIES,
                    model=getattr(provider, 'model', ''),
                )
                return None

        if last_error:
            await self.plugins.dispatch_on_llm_error(last_error)
        return None

    def _get_context_window(self) -> int:
        """Get the context window size from config."""
        context_window = DEFAULT_CONTEXT_WINDOW
        if self.config:
            default_provider = getattr(self.config.llm, 'default', '')
            providers = getattr(self.config.llm, 'providers', {})
            if default_provider and default_provider in providers:
                context_window = providers[default_provider].context_window
            elif hasattr(self.config.llm, 'providers') and providers:
                first_provider = next(iter(providers.values()), None)
                if first_provider:
                    context_window = first_provider.context_window
        return context_window

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
        context: Context,
    ) -> list[dict[str, Any]]:
        """Truncate oldest messages if they exceed the model's context window."""
        # Determine context window from config
        context_window = DEFAULT_CONTEXT_WINDOW
        if self.config:
            default_provider = getattr(self.config.llm, 'default', '')
            providers = getattr(self.config.llm, 'providers', {})
            if default_provider and default_provider in providers:
                context_window = providers[default_provider].context_window
            elif hasattr(self.config.llm, 'providers') and providers:
                first_provider = next(iter(providers.values()), None)
                if first_provider:
                    context_window = first_provider.context_window

        # Estimate current token usage
        system_tokens = context.estimated_tokens
        # Reserve 20% for output and overhead
        available_tokens = int(context_window * 0.8) - system_tokens

        if available_tokens <= 0:
            if len(messages) > MIN_MESSAGES_KEEP:
                return messages[-MIN_MESSAGES_KEEP:]
            return messages

        # Estimate message tokens
        total_msg_tokens = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total_msg_tokens += estimate_tokens(str(content)) + 4
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", tc)
                total_msg_tokens += estimate_tokens(str(func.get("arguments", func.get("args", "")))) + 10

        if total_msg_tokens <= available_tokens:
            return messages

        while len(messages) > MIN_MESSAGES_KEEP and total_msg_tokens > available_tokens:
            removed = messages.pop(0)
            removed_content = removed.get("content", "") or ""
            total_msg_tokens -= estimate_tokens(str(removed_content)) + 4

        return messages

    # ── Two-step compaction ──────────────────────────────────────────

    async def _two_step_compaction(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        summarize_fn,
        context_window: int,
        dedup_max_ratio: float | None = None,
    ) -> str:
        """Two-step compaction: summarize → dedup with .md files.

        Step 1: Summarize rolling_summary + all messages into a conversation summary.
        Step 2: Dedup the summary against all .md files + session memory to produce
                a self-contained rolling context.

        Returns:
            The new rolling summary (deduped).
        """
        from march.core.compaction import (
            SUMMARIZE_PROMPT, DEDUP_ROLLING_CONTEXT_PROMPT,
            _load_session_memory, estimate_tokens as est_tokens,
        )

        _dedup_max_ratio = dedup_max_ratio or 0.30

        # Step 1: Summarize entire rolling context
        full_context = ""
        if session.rolling_summary:
            full_context = session.rolling_summary + "\n\n"
        full_context += "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
            for m in messages
            if isinstance(m.get('content', ''), str)
        )

        summary_prompt = SUMMARIZE_PROMPT + "\n\n" + full_context
        summary = await summarize_fn(summary_prompt)

        if not summary or not summary.strip():
            # Fallback: use rolling_summary as-is
            return session.rolling_summary or ""

        # Step 2: Dedup with all .md files + session memory
        system_rules = await self.memory.load_system_rules()
        agent_profile = await self.memory.load_agent_profile()
        tool_inventory = await self.memory.load_tool_rules()
        long_term = await self.memory.load_long_term()

        session_mem = _load_session_memory(session.id)

        dedup_input = f"""## System Rules
{system_rules}

## Agent Profile
{agent_profile}

## Tools
{tool_inventory}

## Long-Term Memory
{long_term}

## Session Facts
{session_mem.get('facts', '')}

## Session Plan
{session_mem.get('plan', '')}

## Conversation Summary
{summary}"""

        # Dedup target: min(dedup_max_ratio * context_window, current_input_tokens)
        current_tokens = est_tokens(dedup_input)
        target_tokens = min(int(context_window * _dedup_max_ratio), current_tokens)
        target_chars = target_tokens * 4

        dedup_prompt = DEDUP_ROLLING_CONTEXT_PROMPT.format(
            target_tokens=target_tokens,
            target_chars=target_chars,
            dedup_input=dedup_input,
        )

        dedup_result = await summarize_fn(dedup_prompt)

        if not dedup_result or not dedup_result.strip():
            # Fallback: use the summary without dedup
            return summary

        return dedup_result.strip()

    async def run_stream(
        self, user_message: str | list, session: Session
    ) -> AsyncIterator[StreamChunk | AgentResponse]:
        """Streaming version of the agent loop."""
        # Check for slash commands first (only for text messages)
        if isinstance(user_message, str):
            cmd_response = await self.handle_command(user_message, session)
            if cmd_response is not None:
                yield StreamChunk(delta=cmd_response.content, finish_reason="stop")
                yield cmd_response
                return

        start_time = time.monotonic()
        total_tokens = 0
        total_cost = 0.0
        tool_calls_made = 0

        # Create steering queue for this session's turn
        self._steering_queues[session.id] = asyncio.Queue()

        # Set session ID in context var so tools can access it
        current_session_id.set(session.id)

        # Bind session_id to structured logger for this streaming turn
        mlog = self._mlogger.bind(session_id=session.id)

        # 1. Build context
        context = await self._build_context(session)

        # 2. Plugin: before_llm
        context, user_message, short_circuit = await self.plugins.dispatch_before_llm(
            context, user_message
        )
        if short_circuit is not None:
            yield StreamChunk(delta=short_circuit, finish_reason="stop")
            yield await self._finalize(
                content=short_circuit,
                user_message=user_message,
                session=session,
                tool_calls_made=0,
                total_tokens=0,
                total_cost=0.0,
                start_time=start_time,
            )
            return

        # 3. Select LLM provider
        text_for_routing = _extract_text(user_message)
        try:
            provider = await self.llm.route(text_for_routing, context)
        except (RuntimeError, ProviderError) as e:
            await self.plugins.dispatch_on_llm_error(e)
            error_msg = f"Error: No LLM provider available. {e}"
            yield StreamChunk(delta=error_msg, finish_reason="error")
            return

        # 4. Build messages
        messages = session.get_messages_for_llm()
        messages.append({"role": "user", "content": user_message})

        # Strip attachment data from history messages.
        messages = strip_attachments_from_messages(messages, skip_last=True)

        # Context management: compact if too large
        from march.core.compaction import needs_compaction, split_for_compaction, \
            build_summary_prompt, MIN_RECENT_KEEP
        context_window = self._get_context_window()
        system_tokens = context.estimated_tokens

        # Get compaction config if available
        _compaction_threshold = None
        _summary_budget_ratio = None
        _dedup_max_ratio = None
        if self.config and hasattr(self.config, 'memory') and hasattr(self.config.memory, 'compaction'):
            _compaction_threshold = self.config.memory.compaction.threshold
            _summary_budget_ratio = self.config.memory.compaction.summary_budget_ratio
            _dedup_max_ratio = getattr(self.config.memory.compaction, 'dedup_max_ratio', None)

        if needs_compaction(messages, context_window, system_tokens, threshold=_compaction_threshold):
            # Summarize old messages and permanently compact the session
            async def _summarize(prompt: str) -> str:
                result = ""
                async for chunk in provider.converse_stream(
                    messages=[{"role": "user", "content": prompt}],
                    system="You are a helpful assistant that summarizes conversations concisely.",
                    tools=None,
                ):
                    if hasattr(chunk, 'delta') and chunk.delta:
                        result += chunk.delta
                return result

            old, recent = split_for_compaction(messages, context_window, system_tokens,
                                                summary_budget_ratio=_summary_budget_ratio)
            if old:
                try:
                    new_rolling = await self._two_step_compaction(
                        session, messages, _summarize,
                        context_window, _dedup_max_ratio,
                    )
                    session.compact(new_rolling)
                    get_logger("march.compaction", subsystem="compaction").info(
                        "session compacted (streaming)",
                        session_id=session.id,
                        old_messages=len(old),
                    )
                    # Rebuild messages from the now-compacted session
                    messages = session.get_messages_for_llm()
                    messages.append({"role": "user", "content": user_message})
                except Exception as e:
                    get_logger("march.compaction", subsystem="compaction").error(
                        "session compaction failed",
                        session_id=session.id, error=str(e),
                        action="falling back to truncation",
                    )
                    messages = self._truncate_messages(messages, context)
        else:
            # No compaction needed — just truncate if still too long
            messages = self._truncate_messages(messages, context)

        system_prompt = context.build_system_prompt()

        tool_definitions = self.tools.definitions() if self.tools.tool_count > 0 else None

        # Unlimited tool loop — runs until LLM stops calling tools
        while True:
            # Stream LLM response
            collected_content = ""
            collected_tool_calls: list[dict[str, Any]] = []
            try:
                async for chunk in provider.converse_stream(
                    messages=messages,
                    system=system_prompt,
                    tools=tool_definitions,
                ):
                    chunk = await self.plugins.dispatch_on_stream_chunk(chunk)

                    if chunk.delta:
                        collected_content += chunk.delta
                        yield chunk

                    if chunk.tool_call_delta:
                        self._merge_tool_call_delta(
                            collected_tool_calls, chunk.tool_call_delta
                        )

                    if chunk.delta_tool_call:
                        dtc = chunk.delta_tool_call
                        self._merge_tool_call_delta(
                            collected_tool_calls,
                            {
                                "index": dtc.index,
                                "id": dtc.id or "",
                                "name": dtc.name or "",
                                "arguments": dtc.arguments_json or "",
                            },
                        )

                    if chunk.usage:
                        total_tokens += (
                            chunk.usage.input_tokens + chunk.usage.output_tokens
                        )
                        total_cost += chunk.usage.cost

            except Exception as e:
                await self.plugins.dispatch_on_llm_error(e)
                mlog.llm_stream_error(
                    provider=provider.name,
                    model=getattr(provider, 'model', ''),
                    error=str(e),
                    collected_length=len(collected_content),
                )
                logger.error("stream interrupted", subsystem="stream",
                             session_id=session.id, error=str(e), exc_info=True)
                yield StreamChunk(delta=f"\nError: {e}", finish_reason="error")
                return

            llm_tool_calls = self._parse_collected_tool_calls(collected_tool_calls)

            # If no tool calls, we're done
            if not llm_tool_calls:
                yield await self._finalize(
                    content=collected_content,
                    user_message=user_message,
                    session=session,
                    tool_calls_made=tool_calls_made,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    start_time=start_time,
                )
                return

            # Execute tool calls
            tool_results: list[ToolResult] = []
            steering_messages: list[str] = []
            for i, tc in enumerate(llm_tool_calls):
                core_tool_call = ToolCall(id=tc.id, name=tc.name, args=tc.args)

                modified_call = await self.plugins.dispatch_before_tool(core_tool_call)
                if modified_call is None:
                    tool_results.append(
                        ToolResult(
                            id=core_tool_call.id,
                            error=f"Tool '{core_tool_call.name}' was blocked by a plugin.",
                        )
                    )
                    continue

                core_tool_call = modified_call
                try:
                    _tool_t0 = time.monotonic()
                    result = await self.tools.execute(core_tool_call)
                    _tool_dur = (time.monotonic() - _tool_t0) * 1000
                    result = await self.plugins.dispatch_after_tool(core_tool_call, result)
                    mlog.tool_call(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        result_summary=result.summary[:200],
                        duration_ms=_tool_dur,
                    )
                    self._metrics.tool_call(
                        session_id=session.id,
                        tool=core_tool_call.name,
                        duration_ms=_tool_dur,
                    )
                except ToolNotFound:
                    mlog.tool_error(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        error=f"Unknown tool: {core_tool_call.name}",
                    )
                    result = ToolResult(
                        id=core_tool_call.id,
                        error=f"Unknown tool: {core_tool_call.name}",
                    )
                except Exception as e:
                    await self.plugins.dispatch_on_tool_error(core_tool_call, e)
                    mlog.tool_error(
                        tool=core_tool_call.name,
                        args=core_tool_call.args,
                        error=str(e),
                    )
                    result = ToolResult(id=core_tool_call.id, error=str(e))

                tool_results.append(result)
                tool_calls_made += 1

                status = "✓" if not result.is_error else "✗"
                yield StreamChunk(
                    delta="",
                    tool_call_delta={
                        "name": core_tool_call.name,
                        "status": status,
                        "duration_ms": result.duration_ms,
                    },
                )

                # Yield tool progress for real-time frontend updates
                yield StreamChunk(
                    delta="",
                    tool_progress={
                        "name": core_tool_call.name,
                        "status": "complete" if not result.is_error else "error",
                        "summary": result.summary[:200] if hasattr(result, 'summary') else "",
                        "duration_ms": _tool_dur,
                    },
                )

                # ── Steering checkpoint ──
                steering_messages = self._drain_steering(session.id)
                if steering_messages:
                    for remaining_tc in llm_tool_calls[i + 1:]:
                        skip_result = ToolResult(
                            id=remaining_tc.id,
                            error="Skipped due to queued user message.",
                        )
                        tool_results.append(skip_result)
                        tool_calls_made += 1
                        yield StreamChunk(
                            delta="",
                            tool_progress={
                                "name": remaining_tc.name,
                                "status": "skipped",
                                "summary": "Skipped due to steering",
                                "duration_ms": 0,
                            },
                        )
                    break  # Exit tool loop

            # Add to messages for next LLM call
            assistant_msg = {
                "role": "assistant",
                "content": collected_content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.args},
                    }
                    for tc in llm_tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tr in tool_results:
                content = tr.content if not tr.is_error else f"Error: {tr.error}"
                messages.append(
                    {"role": "tool", "tool_call_id": tr.id, "content": content}
                )

            # Tool call intermediate messages stay in `messages` only —
            # not saved to session.messages. See run() for explanation.

            # ── Inject steering messages as user messages ──
            if not steering_messages:
                steering_messages = self._drain_steering(session.id)
            if steering_messages:
                for steer_msg in steering_messages:
                    messages.append({"role": "user", "content": steer_msg})
                continue  # Skip truncation, go straight to next LLM call

            messages = self._truncate_messages(messages, context)

    def _merge_tool_call_delta(
        self,
        collected: list[dict[str, Any]],
        delta: dict[str, Any],
    ) -> None:
        """Merge a streaming tool call delta into the collection."""
        index = delta.get("index", 0)
        while len(collected) <= index:
            collected.append({"id": "", "name": "", "arguments": ""})

        entry = collected[index]
        if delta.get("id") and not entry["id"]:
            entry["id"] = delta["id"]
        if delta.get("name") and not entry["name"]:
            entry["name"] = delta["name"]
        if delta.get("arguments"):
            entry["arguments"] += delta["arguments"]

    def _parse_collected_tool_calls(
        self, collected: list[dict[str, Any]]
    ) -> list[ToolCall]:
        """Parse collected tool call fragments into ToolCall objects."""
        import json

        result: list[ToolCall] = []
        for entry in collected:
            if not entry.get("name"):
                continue
            try:
                args = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                args = {"raw": entry["arguments"]}
            result.append(
                ToolCall(
                    id=entry.get("id", f"call_{id(entry)}"),
                    name=entry["name"],
                    args=args,
                )
            )
        return result

    async def _build_context(self, session: Session) -> Context:
        """Assemble system prompt from config + memory files.

        For the FIRST turn of a new session (rolling_summary is empty),
        reads .md files to bootstrap. For subsequent turns, uses the
        rolling_summary which already contains the deduped content.
        """
        if session.rolling_summary:
            # Use cached rolling summary — no .md file reads
            ctx = dict(session.metadata)
            return Context(
                system_rules=session.rolling_summary,
                session_context=ctx,
            )
        else:
            # First turn — bootstrap from .md files
            system_rules = await self.memory.load_system_rules()
            agent_profile = await self.memory.load_agent_profile()
            tool_inventory = await self.memory.load_tool_rules()
            long_term = await self.memory.load_long_term()

            ctx = dict(session.metadata)

            return Context(
                system_rules=system_rules,
                agent_profile=agent_profile,
                tool_inventory=tool_inventory,
                long_term_memory=long_term,
                session_context=ctx,
            )

    async def _generate_turn_summary(
        self,
        user_message: str,
        assistant_content: str,
        tool_calls_made: int,
    ) -> str:
        """Generate a short summary (≤200 words) of the agent turn."""
        if tool_calls_made == 0 and len(assistant_content) < 500:
            return ""

        user_text = user_message if isinstance(user_message, str) else "[multimodal input]"
        user_preview = user_text[:500]
        assistant_preview = assistant_content[:2000]

        summary_prompt = (
            "Write a single short paragraph summarizing this agent turn. "
            "No bullets, no lists, no headers — just one concise paragraph. "
            "Maximum 200 words but shorter is better. "
            "Cover what was asked, what was done, and the result.\n\n"
            f"User: {user_preview}\n\n"
            f"Assistant ({tool_calls_made} tool calls): {assistant_preview}"
        )

        try:
            result = await self.llm.converse(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300,
            )
            summary = result.content.strip() if result and result.content else ""
            logger.debug("turn summary generated", length=len(summary))
            return summary
        except Exception as e:
            logger.warning("turn summary generation failed", error=str(e))
            return ""

    async def _finalize(
        self,
        content: str,
        user_message: str,
        session: Session,
        tool_calls_made: int,
        total_tokens: int,
        total_cost: float,
        start_time: float,
    ) -> AgentResponse:
        """Post-process response and save to session."""
        duration_ms = (time.monotonic() - start_time) * 1000

        # Log turn completion via MarchLogger
        mlog = self._mlogger.bind(session_id=session.id)
        mlog.turn_complete(
            session_id=session.id,
            tool_calls=tool_calls_made,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_ms=duration_ms,
        )

        # Emit turn.complete metric
        self._metrics.turn_complete(
            session_id=session.id,
            tool_calls=tool_calls_made,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_ms=duration_ms,
        )

        # Plugin: on_response
        content = await self.plugins.dispatch_on_response(content)
        if not isinstance(content, str):
            content = str(content)

        # Generate turn summary
        turn_summary = await self._generate_turn_summary(
            user_message, content, tool_calls_made
        )

        # Save exchange to session
        session.add_exchange(user_message, content)

        # Clean up steering queue for this session's turn
        self._steering_queues.pop(session.id, None)

        return AgentResponse(
            content=content,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            total_cost=total_cost,
            duration_ms=duration_ms,
            turn_summary=turn_summary,
        )
