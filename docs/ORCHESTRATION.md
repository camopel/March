# Agent Orchestration Refactor — Design Document

**Status:** Draft  
**Date:** 2026-03-07  
**Scope:** Replace inline tool execution in the agent loop with a sub-agent orchestration model.

---

## 1. Problem Statement

Currently, `Agent.run()` and `Agent.run_stream()` in `agent.py` execute tool calls **inline** within a `while True` loop. The LLM is called, tool calls are extracted, tools are executed sequentially, results are appended to the message list, and the LLM is called again — all within a single async coroutine.

This design has several limitations:

1. **No cancellation** — Once the agent loop starts, the only way to stop it is to close the WebSocket. The `cancel_event` in `_WSConn` exists but is never checked inside the agent loop itself.
2. **No progress visibility** — Tool execution happens inside the agent; the frontend only sees stream deltas and tool status chips. There's no structured way to report intermediate reasoning or partial results.
3. **Blocking turns** — A long-running tool (e.g., a 30-second code execution) blocks the entire agent turn. The user can queue messages but can't redirect the agent mid-turn.
4. **History pollution risk** — Although the current code correctly keeps tool intermediates out of `session.history`, the boundary is implicit and easy to break during future changes.
5. **No parallelism** — Multiple independent tool calls from a single LLM response are executed sequentially.

---

## 2. Architecture Overview

The new architecture introduces a **two-tier execution model**: a **main agent** (the "orchestrator") that handles LLM reasoning, and **sub-agents** that handle tool execution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (WebSocket)                         │
│                                                                     │
│  User message ──►  stream.delta / tool.progress / stream.end  ◄──  │
│  stop command ──►  stream.cancelled                           ◄──  │
└─────────────┬───────────────────────────────────────────────────────┘
              │ WS messages (JSON)
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WSProxyPlugin (_WSConn)                          │
│                                                                     │
│  _ws_handle_message() ──► _run_agent() ──► Orchestrator             │
│  stop/cancel ──────────► cancel_event.set()                         │
└─────────────┬───────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator                                  │
│                   (new: orchestrator.py)                              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LOOP:                                                        │   │
│  │  1. Call LLM with [user msg + context + sub-agent results]   │   │
│  │  2. LLM returns text? → yield final reply, break             │   │
│  │  3. LLM returns tool_calls? → spawn SubAgent                 │   │
│  │  4. Await SubAgent results (with cancel check)               │   │
│  │  5. Yield progress updates to frontend                       │   │
│  │  6. Feed results back to LLM → goto 1                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Ephemeral turn state:                                              │
│    - tool_messages[]    (assistant+tool msgs, never persisted)      │
│    - sub_agent_results  (structured results from sub-agents)        │
│    - turn_token_usage   (accumulated across all LLM calls)          │
└─────────────┬───────────────────────────────────────────────────────┘
              │ spawn / cancel
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SubAgent                                     │
│                    (new: sub_agent.py)                                │
│                                                                     │
│  Receives: list[ToolCall]                                           │
│  Executes: tools via ToolRegistry.execute()                         │
│  Returns:  list[ToolResult]                                         │
│  Emits:    progress events (tool started, tool finished, error)     │
│                                                                     │
│  Lifecycle: spawn → execute → return → cleanup                      │
│  Cancellable via asyncio.Event                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `Orchestrator` | `core/orchestrator.py` (new) | Drives the LLM↔tool loop, manages sub-agents, handles cancellation |
| `SubAgent` | `core/sub_agent.py` (new) | Executes a batch of tool calls, reports progress |
| `Agent` | `core/agent.py` (modified) | Simplified to delegate to `Orchestrator`; retains `_build_context`, `_finalize` |
| `_WSConn` | `plugins/ws_proxy.py` (modified) | Passes cancel events to orchestrator, relays progress |
| `Session` | `core/session.py` (unchanged) | Stores only user + final assistant messages |

---

## 3. Data Flow — Typical Multi-Tool Turn

### Sequence: User asks "What's the weather in Tokyo and translate it to Japanese"

```
Frontend                WSProxy              Orchestrator           SubAgent
   │                      │                      │                     │
   │── message ──────────►│                      │                     │
   │                      │── save user msg ────►│(DB)                 │
   │                      │── run_agent() ──────►│                     │
   │                      │                      │                     │
   │                      │                      │── LLM call #1 ────►│(LLM)
   │                      │                      │◄── tool_calls: ────│
   │                      │                      │    [weather, translate]
   │                      │                      │                     │
   │                      │                      │── spawn SubAgent ──►│
   │◄─ tool.progress ─────│◄── progress ─────────│◄── tool.started ───│
   │   {weather,started}  │                      │    (weather)        │
   │                      │                      │                     │
   │                      │                      │                     │── execute weather
   │                      │                      │                     │── execute translate
   │                      │                      │                     │
   │◄─ tool.progress ─────│◄── progress ─────────│◄── tool.done ──────│
   │   {weather,done}     │                      │    (weather result) │
   │◄─ tool.progress ─────│◄── progress ─────────│◄── tool.done ──────│
   │   {translate,done}   │                      │    (translate result)│
   │                      │                      │                     │
   │                      │                      │◄── SubAgent returns │
   │                      │                      │    list[ToolResult] │
   │                      │                      │                     │
   │                      │                      │── LLM call #2 ────►│(LLM)
   │                      │                      │◄── final text ─────│
   │                      │                      │                     │
   │◄─ stream.delta ──────│◄── yield chunks ─────│                     │
   │◄─ stream.end ────────│◄── yield AgentResp ──│                     │
   │                      │── save assistant ────►│(DB)                 │
   │                      │── session.add_exchange()                   │
```

### What the LLM sees at each call

**LLM Call #1 (initial):**
```
system: [context from _build_context]
messages: [
  ...session history (user + assistant pairs only)...
  {role: "user", content: "What's the weather in Tokyo and translate it to Japanese"}
]
tools: [weather_tool, translate_tool, ...]
```

**LLM Call #2 (after sub-agent returns):**
```
system: [same context]
messages: [
  ...session history...
  {role: "user", content: "What's the weather in Tokyo..."},
  {role: "assistant", content: null, tool_calls: [{weather}, {translate}]},
  {role: "tool", tool_call_id: "...", content: "Tokyo: 15°C, partly cloudy"},
  {role: "tool", tool_call_id: "...", content: "東京：15°C、曇り時々晴れ"},
]
tools: [weather_tool, translate_tool, ...]
```

**LLM Call #2 response (final):**
```
{content: "The weather in Tokyo is 15°C and partly cloudy. In Japanese: 東京：15°C、曇り時々晴れ"}
```

### What gets persisted

| Storage | Content |
|---------|---------|
| `session.history` | `[Message(user, "What's the weather..."), Message(assistant, "The weather in Tokyo...")]` |
| SQLite `messages` | Same two rows: user message + final assistant reply |
| **NOT stored** | The two intermediate assistant+tool messages from the tool loop |

---

## 4. Stop / Cancel Flow

### Current State

`_WSConn` has a `cancel_event: asyncio.Event` and `STOP_COMMANDS = {"stop", "停止", "/stop"}`. When a stop command arrives during `busy=True`, the event is set and `pending` is cleared. However, the agent loop in `agent.py` **never checks this event**, so cancellation doesn't actually interrupt tool execution.

### New Flow

```
Frontend              WSProxy                Orchestrator              SubAgent
   │                    │                        │                        │
   │── "stop" ─────────►│                        │                        │
   │                    │── cancel_event.set() ──►│                        │
   │                    │── pending.clear()       │                        │
   │                    │                         │── sub_agent.cancel() ──►│
   │                    │                         │                        │── (raises Cancelled
   │                    │                         │                        │    or returns partial)
   │                    │                         │                        │
   │                    │                         │◄── CancelledError ─────│
   │                    │                         │                        │
   │                    │                         │── cleanup ephemeral ───│
   │                    │                         │   state (tool_messages │
   │                    │                         │   turn_usage, etc.)    │
   │                    │                         │                        │
   │◄─ stream.cancelled │◄── yield cancelled ─────│                        │
   │                    │                         │                        │
   │                    │── conn.busy = False     │                        │
```

### Cancellation Contract

1. **Orchestrator** checks `cancel_event.is_set()` at every iteration boundary:
   - Before each LLM call
   - Before spawning a sub-agent
   - After receiving sub-agent results (before feeding back to LLM)

2. **SubAgent** receives a reference to the cancel event. Between tool executions (and during `await` points within tool execution), it checks the event. If set:
   - Currently-running tool is allowed to complete (tools are atomic — no partial execution)
   - Remaining queued tools are skipped
   - SubAgent returns partial results with a `cancelled=True` flag

3. **Session history** is NOT modified on cancel — the user message was already added, but since no final assistant reply was produced, `add_exchange()` is never called. The user message remains as the last entry. On the next user message, it will appear as two consecutive user messages in history, which is acceptable.

4. **Ephemeral state** (the `messages` list with tool intermediates) is discarded entirely.

### Edge Case: Cancel During LLM Streaming

If the LLM is mid-stream when cancel arrives:
- The orchestrator wraps the `converse_stream()` call in a task
- On cancel, the task is cancelled via `task.cancel()`
- Any partial text collected is discarded (not sent to frontend)
- The stream buffer is marked as cancelled

---

## 5. Session History Management

### Principle: Clean History

Session history stores **only** the conversational turns visible to the user:

```
session.history = [
    Message(user, "Hello"),
    Message(assistant, "Hi! How can I help?"),
    Message(user, "What's the weather in Tokyo and translate it to Japanese"),
    Message(assistant, "The weather in Tokyo is 15°C..."),
]
```

No tool calls, no tool results, no intermediate assistant messages.

### What's Ephemeral (Turn-Scoped)

The orchestrator maintains a **turn context** that exists only for the duration of a single user→assistant exchange:

```python
@dataclass
class TurnContext:
    """Ephemeral state for a single orchestration turn."""
    
    # The working message list sent to the LLM (includes tool intermediates)
    messages: list[dict[str, Any]]
    
    # Accumulated sub-agent results for the current turn
    sub_agent_results: list[SubAgentResult]
    
    # Token/cost tracking across all LLM calls in this turn
    total_tokens: int = 0
    total_cost: float = 0.0
    tool_calls_made: int = 0
    
    # Reference to the cancel event
    cancel_event: asyncio.Event | None = None
```

This is created at the start of each turn and garbage-collected when the turn completes (or is cancelled).

### Compaction Interaction

The existing compaction system (`core/compaction.py`) continues to work unchanged. It operates on `session.history`, which only contains user+assistant pairs. Since tool intermediates are never stored in history, compaction never sees them — which is exactly correct.

The compaction trigger point remains the same: at the start of `run_stream()` / `run()`, before the orchestration loop begins.

### Database Persistence

The `SessionStore` and `ChatDB` adapter persist messages at the same points as today:

| Event | Persisted By | What's Stored |
|-------|-------------|---------------|
| User sends message | `_ws_handle_message()` → `save_message()` | User message text |
| Agent finishes turn | `_stream_response()` → `finalize_draft()` or `save_message()` | Final assistant reply |
| Stream in progress | `save_draft()` (periodic) | Partial assistant text (overwritten on finalize) |

Tool calls and results are **never** written to the `messages` table.

---

## 6. Sub-Agent Lifecycle

### SubAgent Class

```python
@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    tool_results: list[ToolResult]
    cancelled: bool = False
    error: str | None = None
    duration_ms: float = 0.0


class SubAgent:
    """Executes a batch of tool calls with progress reporting and cancellation."""
    
    def __init__(
        self,
        tool_calls: list[ToolCall],
        tool_registry: ToolRegistry,
        plugin_manager: PluginManager,
        cancel_event: asyncio.Event,
        config: SubAgentConfig,
    ):
        self.tool_calls = tool_calls
        self.tools = tool_registry
        self.plugins = plugin_manager
        self.cancel_event = cancel_event
        self.config = config
        self._progress_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
    
    async def execute(self) -> SubAgentResult:
        """Execute all tool calls. Returns results (possibly partial on cancel)."""
        ...
    
    def progress_iter(self) -> AsyncIterator[ProgressEvent]:
        """Async iterator over progress events."""
        ...
```

### Lifecycle Phases

```
  SPAWN                    EXECUTE                    RETURN          CLEANUP
    │                        │                          │                │
    │  SubAgent created      │  For each ToolCall:      │  Returns       │  SubAgent
    │  with tool_calls,      │    check cancel_event    │  SubAgentResult│  dereferenced,
    │  cancel_event,         │    dispatch before_tool   │  with all      │  GC'd
    │  config                │    execute tool           │  ToolResults   │
    │                        │    dispatch after_tool    │                │
    │  Progress queue        │    emit progress event   │  If cancelled: │
    │  created               │    check cancel_event    │  partial results│
    │                        │                          │  cancelled=True│
    │                        │  If parallel mode:       │                │
    │                        │    asyncio.gather()      │  If error:     │
    │                        │    with cancel support   │  error string  │
    │                        │                          │  + partial     │
```

### Parallel vs Sequential Execution

The sub-agent supports two execution modes, controlled by config:

1. **Sequential** (default, safe): Tools execute one at a time. Cancel checks happen between each tool.

2. **Parallel**: Independent tool calls from a single LLM response execute concurrently via `asyncio.gather()`. Cancel triggers `task.cancel()` on all pending tasks. Currently-executing tools complete; unstarted ones are skipped.

The orchestrator determines which mode to use based on:
- Config setting (`orchestration.parallel_tools: bool`)
- Whether any tool calls have dependencies (future: LLM can annotate dependencies)

### Progress Events

```python
@dataclass
class ProgressEvent:
    """Emitted by SubAgent during execution."""
    type: str          # "tool.started" | "tool.done" | "tool.error" | "cancelled"
    tool_name: str
    tool_call_id: str
    result: ToolResult | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
```

The orchestrator consumes these events and forwards them to the frontend via the WebSocket as `tool.progress` messages.

---

## 7. Streaming / Progress Protocol (WebSocket Message Types)

### Existing Messages (unchanged)

| Type | Direction | Description |
|------|-----------|-------------|
| `message` | client→server | User sends a text message |
| `attachment` | client→server | User sends a file |
| `voice` | client→server | User sends voice audio |
| `stream.start` | server→client | Agent begins responding |
| `stream.delta` | server→client | Text chunk from LLM |
| `stream.end` | server→client | Agent finished responding |
| `stream.cancelled` | server→client | Response was cancelled |
| `error` | server→client | Error occurred |
| `message.queued` | server→client | Message queued while busy |

### New Messages

| Type | Direction | Payload | Description |
|------|-----------|---------|-------------|
| `tool.progress` | server→client | `{tool_name, tool_call_id, status, duration_ms?}` | Tool execution progress |
| `orchestration.step` | server→client | `{step: int, action: "llm_call"\|"tool_exec", detail: str}` | High-level orchestration step (optional, for debug UI) |
| `orchestration.thinking` | server→client | `{content: str}` | LLM's intermediate reasoning text before tool calls (if model emits text alongside tool_calls) |

### `tool.progress` Status Values

| Status | Meaning |
|--------|---------|
| `started` | Tool execution has begun |
| `done` | Tool completed successfully |
| `error` | Tool failed (includes error message) |
| `skipped` | Tool was skipped due to cancellation |

### Message Flow Example

```json
← {"type": "stream.start", "chunk_id": 0}
← {"type": "orchestration.thinking", "content": "I'll check the weather and translate it.", "chunk_id": 1}
← {"type": "tool.progress", "tool_name": "weather", "tool_call_id": "call_abc", "status": "started", "chunk_id": 2}
← {"type": "tool.progress", "tool_name": "translate", "tool_call_id": "call_def", "status": "started", "chunk_id": 3}
← {"type": "tool.progress", "tool_name": "weather", "tool_call_id": "call_abc", "status": "done", "duration_ms": 1200, "chunk_id": 4}
← {"type": "tool.progress", "tool_name": "translate", "tool_call_id": "call_def", "status": "done", "duration_ms": 800, "chunk_id": 5}
← {"type": "stream.delta", "content": "The weather in Tokyo is ", "chunk_id": 6}
← {"type": "stream.delta", "content": "15°C and partly cloudy...", "chunk_id": 7}
← {"type": "stream.end", "usage": {...}, "chunk_id": 8}
```

### Backward Compatibility

The existing `tool.start` message type (emitted in current `run_stream`) is replaced by `tool.progress`. The frontend should handle both during the migration period. The `stream.delta`, `stream.start`, and `stream.end` messages are unchanged.

---

## 8. Orchestrator Implementation Detail

### Core Loop (Pseudocode)

```python
class Orchestrator:
    async def run_stream(
        self,
        user_message: str | list,
        session: Session,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[StreamChunk | ProgressEvent | AgentResponse]:
        
        # 1. Build context, handle compaction (same as current agent.py)
        context = await self.agent._build_context(session)
        messages = session.get_messages_for_llm()
        messages.append({"role": "user", "content": user_message})
        messages = strip_attachments_from_messages(messages, skip_last=True)
        # ... compaction logic (unchanged) ...
        
        system_prompt = context.build_system_prompt()
        tool_definitions = self.agent.tools.definitions()
        
        turn = TurnContext(messages=messages, cancel_event=cancel_event)
        
        # 2. Orchestration loop
        while True:
            # ── Cancel check ──
            if cancel_event.is_set():
                yield ProgressEvent(type="cancelled", ...)
                return
            
            # ── LLM call (streaming) ──
            collected_content = ""
            collected_tool_calls = []
            
            async for chunk in provider.converse_stream(
                messages=turn.messages,
                system=system_prompt,
                tools=tool_definitions,
            ):
                if cancel_event.is_set():
                    break  # Abort mid-stream
                
                if chunk.delta:
                    collected_content += chunk.delta
                    yield chunk  # Forward text deltas to frontend
                
                # Accumulate tool call deltas...
                if chunk.usage:
                    turn.total_tokens += chunk.usage.input_tokens + chunk.usage.output_tokens
                    turn.total_cost += chunk.usage.cost
            
            if cancel_event.is_set():
                return
            
            llm_tool_calls = parse_tool_calls(collected_tool_calls)
            
            # ── No tools? We're done ──
            if not llm_tool_calls:
                yield await self.agent._finalize(
                    content=collected_content,
                    user_message=user_message,
                    session=session,
                    tool_calls_made=turn.tool_calls_made,
                    total_tokens=turn.total_tokens,
                    total_cost=turn.total_cost,
                    start_time=turn.start_time,
                )
                return
            
            # ── Emit intermediate thinking ──
            if collected_content:
                yield ProgressEvent(type="orchestration.thinking", content=collected_content)
            
            # ── Spawn sub-agent ──
            sub_agent = SubAgent(
                tool_calls=llm_tool_calls,
                tool_registry=self.agent.tools,
                plugin_manager=self.agent.plugins,
                cancel_event=cancel_event,
                config=self.config.sub_agent,
            )
            
            # Execute and stream progress
            result = None
            execute_task = asyncio.create_task(sub_agent.execute())
            
            try:
                # Concurrently consume progress events and wait for completion
                async for event in sub_agent.progress_iter():
                    yield event  # Forward to frontend
                
                result = await execute_task
            except asyncio.CancelledError:
                execute_task.cancel()
                return
            
            turn.tool_calls_made += len(result.tool_results)
            
            if result.cancelled:
                return
            
            # ── Append tool messages to working context ──
            assistant_msg = {
                "role": "assistant",
                "content": collected_content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": tc.args}}
                    for tc in llm_tool_calls
                ],
            }
            turn.messages.append(assistant_msg)
            
            for tr in result.tool_results:
                content = tr.content if not tr.is_error else f"Error: {tr.error}"
                turn.messages.append({
                    "role": "tool",
                    "tool_call_id": tr.id,
                    "content": content,
                })
            
            # Truncate if needed
            turn.messages = self.agent._truncate_messages(turn.messages, context)
            
            # Loop back → LLM sees tool results, decides next action
```

### Integration with Agent

The `Agent` class is simplified. `run()` and `run_stream()` delegate to the orchestrator:

```python
class Agent:
    async def run_stream(self, user_message, session) -> AsyncIterator[StreamChunk | AgentResponse]:
        # Slash commands (unchanged)
        if isinstance(user_message, str):
            cmd = await self.handle_command(user_message, session)
            if cmd:
                yield StreamChunk(delta=cmd.content, finish_reason="stop")
                yield cmd
                return
        
        # Delegate to orchestrator
        orchestrator = Orchestrator(agent=self, config=self.config)
        async for item in orchestrator.run_stream(user_message, session, cancel_event):
            yield item
```

---

## 9. Edge Cases

### 9.1 Sub-Agent Crash (Unhandled Exception in Tool)

**Current behavior:** Tool exceptions are caught in the agent loop and converted to `ToolResult(error=...)`. This continues.

**New behavior:** The sub-agent wraps each tool execution in try/except. If a tool raises an unexpected exception:
1. The error is captured as a `ToolResult` with `is_error=True`
2. A `tool.progress` event with `status: "error"` is emitted
3. Execution continues with the remaining tools (unless `config.fail_fast=True`)
4. The orchestrator receives the error result and feeds it to the LLM, which can decide how to proceed

If the sub-agent itself crashes (e.g., out of memory, asyncio internal error):
1. The `execute_task` raises an exception
2. The orchestrator catches it, logs the error
3. An error `StreamChunk` is yielded to the frontend
4. The turn is abandoned; no assistant message is saved to history
5. The user message remains in history as the last entry

### 9.2 LLM Error Mid-Turn

**Scenario:** LLM call #1 succeeds (returns tool calls), tools execute, LLM call #2 fails.

**Handling:**
1. The existing `_call_llm_with_retry` logic (3 retries with exponential backoff) applies to each LLM call independently
2. If all retries fail on call #2:
   - The orchestrator yields an error response
   - Tool results from the sub-agent are lost (ephemeral)
   - No assistant message is saved
   - The user message remains in history
3. The user can retry by sending the same message (or a new one)

**Partial recovery option (future):** If the LLM produced useful text in call #1 alongside tool calls, and call #2 fails, we could save the partial text as the assistant response. This is deferred — for now, it's all-or-nothing per turn.

### 9.3 Concurrent User Messages

**Current behavior:** Messages arriving while `busy=True` are queued in `conn.pending`. After the current turn completes, `_drain_queue()` combines them and runs a new agent turn.

**New behavior:** Unchanged. The queue mechanism in `_WSConn` works at the WebSocket layer, above the orchestrator. The orchestrator handles one turn at a time.

**Nuance:** If a user sends a message while the orchestrator is between LLM call #1 and sub-agent execution, the message is queued (not injected into the current turn). This is correct — injecting mid-turn would corrupt the tool call context.

### 9.4 Sub-Agent Timeout

If a tool takes longer than `config.tool_timeout_seconds`:
1. The sub-agent cancels the tool's asyncio task
2. A `ToolResult(error="Tool execution timed out after Xs")` is produced
3. Execution continues with remaining tools
4. The LLM receives the timeout error and can decide to retry or work around it

### 9.5 Context Window Overflow During Tool Loop

**Scenario:** Many tool calls produce large results, pushing the working message list beyond the context window.

**Handling:** The orchestrator calls `_truncate_messages()` after each sub-agent return (same as current behavior). If truncation removes the tool call/result messages that the LLM just produced, the LLM may get confused on the next call. To mitigate:
1. Tool results are truncated to `config.max_tool_result_chars` before being added to messages
2. If the working context is still too large after truncation, the orchestrator yields an error and aborts the turn

### 9.6 Infinite Tool Loop

**Scenario:** The LLM keeps requesting tool calls indefinitely.

**Handling:** A configurable `max_tool_rounds` (default: 20) limits the number of orchestration loop iterations. When exceeded:
1. The orchestrator forces a final LLM call with an injected system message: `"You have reached the maximum number of tool call rounds. Please provide your final response now."`
2. If the LLM still returns tool calls, they are ignored and only the text content is used
3. If no text content, a fallback message is returned: `"I wasn't able to complete the task within the allowed number of steps."`

---

## 10. Configuration

New configuration under `config.yaml → orchestration`:

```yaml
orchestration:
  # Maximum number of LLM↔tool rounds per turn
  max_tool_rounds: 20
  
  # Maximum characters per tool result (truncated if exceeded)
  max_tool_result_chars: 50000
  
  # Execute independent tool calls in parallel
  parallel_tools: false
  
  # Maximum concurrent tool executions (when parallel_tools=true)
  max_concurrent_tools: 5
  
  # Per-tool execution timeout (seconds)
  tool_timeout_seconds: 120
  
  # Sub-agent overall timeout (seconds, across all tools in one batch)
  sub_agent_timeout_seconds: 300
  
  # Abort remaining tools on first failure
  fail_fast: false
  
  # Enable orchestration.thinking messages to frontend
  emit_thinking: true
  
  # Enable orchestration.step debug messages
  emit_debug_steps: false
```

### Config Schema Addition

```python
# In config/schema.py

class OrchestrationConfig(BaseModel):
    max_tool_rounds: int = 20
    max_tool_result_chars: int = 50000
    parallel_tools: bool = False
    max_concurrent_tools: int = 5
    tool_timeout_seconds: int = 120
    sub_agent_timeout_seconds: int = 300
    fail_fast: bool = False
    emit_thinking: bool = True
    emit_debug_steps: bool = False

class MarchConfig(BaseModel):
    # ... existing fields ...
    orchestration: OrchestrationConfig = OrchestrationConfig()
```

---

## 11. Migration Plan

### Phase 1: Extract Orchestrator (Non-Breaking)

**Goal:** Move the tool loop out of `Agent` into `Orchestrator` without changing behavior.

1. Create `core/orchestrator.py` with `Orchestrator` class
2. Create `core/sub_agent.py` with `SubAgent` class
3. Move the `while True` loop from `Agent.run_stream()` into `Orchestrator.run_stream()`
4. `Agent.run_stream()` delegates to `Orchestrator.run_stream()`
5. SubAgent initially just wraps the existing sequential tool execution
6. **No WS protocol changes** — orchestrator emits the same `StreamChunk` objects

**Tests:** All existing tests pass. New unit tests for `Orchestrator` and `SubAgent`.

**Rollback:** Delete new files, revert `Agent` delegation. Zero risk.

### Phase 2: Wire Cancellation

**Goal:** Make stop/cancel actually work end-to-end.

1. Pass `cancel_event` from `_WSConn` through `_run_agent()` → `Agent.run_stream()` → `Orchestrator`
2. Add cancel checks at loop boundaries in `Orchestrator`
3. Add cancel checks between tool executions in `SubAgent`
4. Add `stream.cancelled` emission on cancel

**Tests:** Integration test: send message → send "stop" → verify cancellation.

### Phase 3: Progress Protocol

**Goal:** Rich tool progress reporting to the frontend.

1. Add `ProgressEvent` type
2. SubAgent emits progress events via async queue
3. Orchestrator forwards events as `tool.progress` WS messages
4. Frontend updated to render progress (separate frontend PR)
5. Deprecate old `tool.start` message type

### Phase 4: Parallel Tool Execution

**Goal:** Execute independent tools concurrently.

1. Add `parallel_tools` config option
2. SubAgent uses `asyncio.gather()` when enabled
3. Cancel support for parallel execution (cancel all pending tasks)
4. Concurrency limit via `asyncio.Semaphore(max_concurrent_tools)`

### Phase 5: Cleanup

1. Remove legacy inline tool execution code from `Agent`
2. Remove deprecated `tool.start` WS message handling
3. Update documentation

### Timeline Estimate

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1 | 2-3 days | Low (pure refactor, no behavior change) |
| Phase 2 | 1-2 days | Low (additive) |
| Phase 3 | 2-3 days | Medium (WS protocol change, frontend coordination) |
| Phase 4 | 2-3 days | Medium (concurrency bugs) |
| Phase 5 | 1 day | Low (cleanup) |

---

## 12. File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `core/orchestrator.py` | `Orchestrator` class — drives the LLM↔sub-agent loop |
| `core/sub_agent.py` | `SubAgent` class — executes tool batches with progress |
| `core/types.py` | `TurnContext`, `SubAgentResult`, `ProgressEvent`, `SubAgentConfig` |
| `tests/test_orchestrator.py` | Unit tests for orchestrator |
| `tests/test_sub_agent.py` | Unit tests for sub-agent |

### Modified Files

| File | Changes |
|------|---------|
| `core/agent.py` | `run()` and `run_stream()` delegate to `Orchestrator`; tool loop removed. Retains `_build_context`, `_finalize`, `_truncate_messages`, `_call_llm_with_retry`. |
| `plugins/ws_proxy.py` | `_stream_response()` passes `cancel_event` to agent. Handles new `tool.progress` and `orchestration.*` message types. |
| `config/schema.py` | Add `OrchestrationConfig` model and `orchestration` field to `MarchConfig`. |

### Unchanged Files

| File | Why |
|------|-----|
| `core/session.py` | Session history management is already correct (only stores user+assistant). No changes needed. |
| `core/message.py` | Message types are sufficient. `ToolCall`, `ToolResult`, `Message` used as-is. |
| `core/compaction.py` | Operates on session history, which is unaffected. |
| `llm/base.py` | `StreamChunk`, `LLMResponse`, `LLMProvider` unchanged. |
| `tools/registry.py` | `ToolRegistry.execute()` unchanged — sub-agent calls it the same way. |
| `plugins/_manager.py` | Plugin hooks (`before_tool`, `after_tool`, etc.) unchanged — sub-agent dispatches them. |

---

## 13. Open Questions

1. **Should the orchestrator support "planning" mode?** — Where the LLM first outputs a plan (text), then the orchestrator confirms before executing tools. Deferred to a future design.

2. **Should tool results be summarized before feeding back to the LLM?** — For large tool outputs, an intermediate summarization step could save context. This could be a sub-agent config option (`summarize_results: true`). Deferred.

3. **Should the frontend be able to approve/reject tool calls?** — Human-in-the-loop for sensitive tools (e.g., file deletion, API calls). This would add a `tool.approval_required` WS message and block the sub-agent until approval. Deferred.

4. **Multi-model orchestration?** — Using a cheaper/faster model for tool-calling rounds and a more capable model for the final response. The `LLMRouter` already supports multiple providers, but the orchestrator would need per-step model selection. Deferred.
