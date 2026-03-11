# subagents — March Agent System

> Technical reference for March's agent orchestration layer: mtAgent (asyncio tasks) and mpAgent (isolated processes).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Execution Modes](#2-execution-modes)
3. [Architecture](#3-architecture)
4. [Core Components](#4-core-components)
5. [Lifecycle](#5-lifecycle)
6. [Failure Handling](#6-failure-handling)
7. [Heartbeat & Monitoring](#7-heartbeat--monitoring)
8. [Configuration](#8-configuration)
9. [Session IDs & Logging](#9-session-ids--logging)
10. [Nested Agents (Depth > 1)](#10-nested-agents-depth--1)
11. [Comparison with Other Systems](#11-comparison-with-other-systems)

---

## 1. Overview

March lets a main agent break work into subtasks and hand them off to independent sub-agents that run in parallel. The system is built around three principles:

1. **Push, don't poll.** When a child finishes, `AgentAnnouncer` delivers the result to the parent automatically. The parent never busy-waits.
2. **Always get an answer.** `MpRunner.wait_result()` is contractually guaranteed to return a `RunOutcome` no matter what happens to the child — crash, OOM, signal, IPC failure, or timeout.
3. **In-memory only.** `AgentRegistry` keeps all run state in RAM. There is no disk persistence; crash recovery relies on logs.

Two execution modes are available:

| | mtAgent | mpAgent |
|---|---------|---------|
| Runs as | `asyncio.Task` in the main event loop | `multiprocessing.Process` (start method `spawn`) |
| Isolation | None — shares the main process | Full — independent process group via `os.setpgrp()` |
| Best for | I/O-bound work, chat, file ops, API calls | Heavy compute, GPU, simulations, anything that might OOM |

When in doubt, use mpAgent. The overhead is higher, but a crash stays contained.

---

## 2. Execution Modes

### 2.1 mtAgent

An mtAgent is an `asyncio.Task` on the main event loop. It shares memory with the main agent and every other mtAgent.

```
Main process event loop
  ├── Main agent turn
  ├── mtAgent-1 (asyncio.Task)
  ├── mtAgent-2 (asyncio.Task)
  └── ...
```

Tasks enter through the `TaskQueue`'s `mt` lane. When a slot opens, the queue drains the next entry and wraps it in `asyncio.create_task()`.

**Steering** works through a per-child `asyncio.Queue`. The parent calls `AgentManager.send()`, which puts a message on the queue; the child consumes it before its next LLM call.

**Limitations:** No process isolation. No heartbeat monitoring. An OOM or segfault kills the entire main process. The GIL prevents CPU parallelism.

### 2.2 mpAgent

An mpAgent is a fully independent child process created with `multiprocessing.get_context("spawn").Process()`. The child immediately calls `os.setpgrp()` to form its own process group, which lets the parent `os.killpg()` the entire tree in one shot.

Parent and child communicate over a Unix socketpair using msgpack-serialized messages with a 4-byte big-endian length prefix.

```
Main process                         Child process
  │                                    │
  ├── MpRunner                         ├── os.setpgrp()
  │   ├── _recv_loop (IPC recv)        ├── Agent (own LLM, tools, memory)
  │   ├── _monitor (heartbeat watch)   ├── _HeartbeatThread (heartbeat + steer recv)
  │   └── parent_sock ◄──────────────► child_sock
  │       (Unix socketpair, msgpack + 4-byte length prefix)
```

The child loads `config.yaml` from scratch, creates its own LLM providers, tool registry, and memory store. It runs the task via `Agent.run()` with a pure in-memory `Session` — it never touches `SessionStore`. Results travel back over IPC; the parent persists them if needed.

**Steering** goes through IPC: the parent sends a `steer` message, the `_HeartbeatThread` in the child picks it up, and a `_steering_pump` asyncio task injects it into the agent's steering queue.

### 2.3 Side-by-Side

| | mtAgent | mpAgent |
|---|---------|---------|
| Execution | `asyncio.Task` in main process | `multiprocessing.Process` (own process group) |
| Startup cost | Negligible (coroutine creation) | Higher (process spawn + full Agent init) |
| Crash blast radius | Main process dies | Only child process group affected |
| OOM protection | None | Yes — SIGKILL hits child group only |
| Heartbeat | None | Periodic, with RSS/tokens/cost/tool metrics |
| IPC | `asyncio.Queue` (in-memory) | Unix socketpair + msgpack |
| Steering | Queue injection | IPC `steer` → HeartbeatThread → Agent |
| Kill | `asyncio.Task.cancel()` | `os.killpg(SIGTERM)` → grace → `os.killpg(SIGKILL)` |
| SessionStore access | Shared with parent | None (parent persists results) |
| Session key format | `{id}:mtagent:{hex12}` | `{id}:mpagent:{hex12}` |
| Default max concurrency | 8 | 8 |

---

## 3. Architecture

### 3.1 Component Map

```
┌──────────────────────────────────────────────────────────────────┐
│                         AgentManager                             │
│  ┌──────────┐  ┌───────────────┐  ┌───────────────────────────┐  │
│  │ spawn()  │  │ kill()        │  │ send() / steer            │  │
│  │ list()   │  │ kill_recursive│  │ reset_children()          │  │
│  │ logs()   │  │               │  │ get_child_sessions()      │  │
│  └────┬─────┘  └──────┬────────┘  └────────────┬──────────────┘  │
│       │               │                        │                 │
│  ┌────▼───────────────▼────────────────────────▼──────────────┐  │
│  │                     TaskQueue                               │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │  │
│  │  │ main     │ │ mt       │ │ mp       │ │ cron         │   │  │
│  │  │ (auto)   │ │ (max: 8) │ │ (max: 8) │ │ (max: 1)    │   │  │
│  │  └──────────┘ └────┬─────┘ └────┬─────┘ └──────────────┘   │  │
│  └──────────────────────┼──────────┼──────────────────────────┘  │
│                         │          │                             │
│  ┌──────────────────────┼──────────┼──────────────────────────┐  │
│  │               AgentRegistry (in-memory)                     │  │
│  │  RunRecord: run_id, child_key, status, outcome, pid, ...   │  │
│  │  get_subtree() / get_tree_with_depth()                     │  │
│  └──────────────────────┼──────────┼──────────────────────────┘  │
│                         │          │                             │
│  ┌──────────────────────▼──────────▼──────────────────────────┐  │
│  │               AgentAnnouncer                                │  │
│  │  Delivery: steer → queue → direct → pending_queue           │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 mtAgent Data Flow

```
Parent Agent
    │
    │ spawn(execution="mt")
    ▼
AgentManager.spawn()
    │  1. Create RunRecord → Registry.register()
    │  2. Create asyncio.Queue for steering
    │  3. TaskQueue.enqueue("mt", _execute_child)
    ▼
TaskQueue mt lane
    │  drain → asyncio.create_task()
    ▼
_execute_child()
    │  agent_factory(task, model, tools, child_key, parent_key)
    │  → Child Agent.run()
    │  → RunOutcome (ok / error / timeout / cancelled)
    ▼
Registry.complete(run_id, outcome)
    │
    ▼
AgentAnnouncer.announce_completion()
    │  try_steer → try_queue → send_direct → pending_queue
    ▼
Result delivered to parent session
```

### 3.3 mpAgent Data Flow

```
Parent (MpRunner)                     Child (mp_child_main)
    │                                    │
    │  create_socket_pair()              │
    │  Process.start() ──────────────►   │ os.setpgrp()
    │                                    │ load_config()
    │                                    │ init Agent, LLM, Tools, Memory
    │                                    │ create in-memory Session
    │                                    │ create _SpawnProxy
    │                                    │ start _HeartbeatThread
    │                                    │ start _steering_pump
    │                                    │
    │  start _recv_loop                  │ Agent.run(task, session)
    │  start _monitor                    │
    │                                    │
    │  ◄── heartbeat (every 60s) ──────  │ {rss, tokens, cost, tools, ...}
    │  ── steer ──────────────────────►  │ → Agent.steer(session_id, msg)
    │  ── kill ───────────────────────►  │ → kill_requested = True
    │  ◄── progress ───────────────────  │ tool execution updates
    │  ◄── result ─────────────────────  │ Agent.run() finished
    │                                    │
    │  _recv_loop → _resolve_result()    │ HeartbeatThread.stop()
    │  wait_result() returns             │ child exits
    │                                    │
    ▼                                    ▼
Registry.complete() → Announcer → parent session
```

---

## 4. Core Components

### 4.1 AgentManager

**File:** `src/march/agents/manager.py`

The single public entry point. Everything goes through here.

```python
class AgentManager:
    async def spawn(self, params: SpawnParams, ctx: SpawnContext) -> SpawnResult
    async def list(self) -> list[AgentStatus]
    async def kill(self, agent_id: str) -> bool
    async def kill_recursive(self, agent_id: str) -> int
    async def send(self, agent_id: str, message: str) -> bool
    async def logs(self, agent_id: str, tail: int = 50) -> list[str]
    async def reset_children(self, parent_session_key: str) -> int
    def get_child_sessions(self, parent_session_key: str) -> list[RunRecord]
```

**`SpawnParams`** controls what gets spawned:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `task` | str | (required) | Task description |
| `agent_id` | str | `"agent-{uuid[:8]}"` | Identifier; auto-generated if empty |
| `model` | str | `""` | LLM model override (empty = default) |
| `tools` | list[str] \| None | None | Restrict available tools |
| `timeout` | int | 0 | Timeout in seconds (0 = none) |
| `mode` | str | `"run"` | `"run"` (one-shot) or `"session"` (keep alive) |
| `cleanup` | str | `"keep"` | `"keep"` or `"delete"` |
| `label` | str | `""` | Human-readable label |
| `execution` | str | `"mt"` | `"mt"` (asyncio) or `"mp"` (process) |

**`SpawnContext`** carries the caller's identity:

| Field | Type | Purpose |
|-------|------|---------|
| `requester_session` | str | Parent's session key |
| `origin` | str | Parent's channel/source |
| `caller_depth` | int | Current nesting depth |

**Key behaviors:**

- `spawn()` rejects the call if `caller_depth >= max_spawn_depth`.
- When the parent is itself a sub-agent, `spawn()` enforces **execution consistency**: the child must use the same execution mode as its parent. If they differ, the child's mode is silently forced to match.
- `kill()` resolves by either `run_id` or `child_key`. For mtAgents it cancels the asyncio task; for mpAgents it calls `MpRunner.kill()` which sends SIGKILL to the process group.
- `kill_recursive()` collects all descendants via `registry.get_subtree()`, then kills them bottom-up (deepest first) to avoid orphans.
- `reset_children()` is called when the parent session does `/reset`. It recursively kills active children, deletes their session memory from disk, removes their sessions from the store, and clears their registry entries.
- `send()` routes steering messages: `asyncio.Queue` for mtAgents, `MpRunner.send_steer()` (IPC) for mpAgents.

**mpAgent spawn internals (`_execute_child_mp`):**

The manager builds three async closures — `_handle_spawn_request`, `_handle_steer`, `_handle_kill` — and passes them to `MpRunner` as spawn proxy handlers. These closures let the child process delegate spawn/steer/kill operations back to the parent's `AgentManager` over IPC (see [Section 10](#10-nested-agents-depth--1)).

After the child completes, the manager optionally persists the result to `SessionStore` by creating a session record and storing the output as an assistant message.

The `_announce()` helper has special logic for nested agents: if the parent is itself an active mpAgent, it calls `runner.notify_child_completed()` to push a `child_completed` IPC message so the grandchild's result reaches the mpAgent child process directly.

### 4.2 AgentRegistry

**File:** `src/march/agents/registry.py`

Pure in-memory run tracking. No disk persistence.

**`RunRecord`** fields:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | UUID, unique per run |
| `child_key` | str | Child's session key |
| `requester_key` | str | Parent's session key |
| `requester_origin` | str | Parent's channel (for announce routing) |
| `task` | str | Task description |
| `started_at` | float | Unix timestamp |
| `ended_at` | float | Unix timestamp (0.0 = still running) |
| `mode` | str | `"run"` or `"session"` |
| `cleanup` | str | `"delete"` or `"keep"` |
| `outcome` | RunOutcome \| None | Set on completion |
| `cleanup_done` | bool | True once announce has been delivered |
| `execution` | str | `"mt"` or `"mp"` |
| `pid` | int | Process ID (mpAgent only; 0 for mt) |
| `log_path` | str | Log directory |

**`RunOutcome`:** `status` is one of `"ok"`, `"error"`, `"timeout"`, `"cancelled"`. Plus `error`, `output`, and `duration_ms` fields.

**Properties:** `is_active` checks `ended_at == 0.0`. `duration_seconds` computes elapsed time in real time.

**Tree operations:**

- `get_subtree(root_key)` — depth-first recursive traversal following `requester_key` → `child_key` links. Returns all descendants.
- `get_tree_with_depth(root_key)` — same traversal but returns `(depth, RunRecord)` tuples.
- `list_for_requester(key)` — direct children only.

**Cleanup:** `cleanup_old(max_age_seconds)` is a safety net that only removes records where `cleanup_done=True` AND the record has been complete for longer than `max_age_seconds`. Normal cleanup happens through `AgentManager.reset_children()`.

### 4.3 AgentAnnouncer

**File:** `src/march/agents/announce.py`

Delivers completion results to the parent session. Connected to session infrastructure via callbacks (to avoid circular imports).

**Delivery cascade:**

| Priority | Method | When it works |
|----------|--------|---------------|
| 1 | **Steer** | Parent is mid-turn — inject result into the active turn |
| 2 | **Queue** | Parent is mid-turn — deliver after current turn ends |
| 3 | **Direct** | Parent is idle — start a new agent turn with the result |
| 4 | **Pending** | Everything failed — buffer in `_pending_queue` |

Pending messages are drained when the parent starts its next turn via `get_pending(requester_key)`.

**Message format examples:**

```
✅ mtAgent `agent-abc12345:mtagent:def678901234` finished
<output>

❌ mpAgent `agent-xyz:mpagent:abc123456789` failed
**Error:** Child process killed by SIGKILL (likely OOM)

⏱️ mpAgent `agent-xyz:mpagent:abc123456789` timed out
**Error:** No heartbeat for 300s (timeout: 300s)

🚫 mtAgent `agent-abc:mtagent:def678` was cancelled
**Error:** killed by user
```

**Important:** Agent sessions are NOT deleted after completion. They persist until the parent does `/reset`, so the parent can reference child context and history. The `cleanup_done` flag only tracks whether the announcement was delivered.

### 4.4 TaskQueue

**File:** `src/march/agents/task_queue.py`

Lane-based async task queue. Each lane has its own concurrency limit and waiting queue.

| Lane | Purpose | Default Max Concurrent |
|------|---------|------------------------|
| `main` | User sessions | `os.cpu_count()` |
| `mt` | mtAgent runs | 8 |
| `mp` | mpAgent runs | 8 |
| `cron` | Scheduled jobs | 1 |

Internally each lane has a `deque` (waiting) and a `set` (active task IDs). `_drain()` promotes queued tasks to active whenever a slot opens. Lanes are independent — mpAgents never compete with mtAgents for slots.

```python
# Block until complete
result = await task_queue.enqueue("mt", my_coroutine_fn)

# Fire and forget
task_id = task_queue.enqueue_fire_and_forget("mp", my_coroutine_fn)

# Inspect
task_queue.lane_stats("mt")   # {"name":"mt", "max_concurrent":8, "active":3, "queued":2}
task_queue.all_stats()         # All lanes
task_queue.total_active        # Sum across lanes
task_queue.total_queued        # Sum across lanes
```

### 4.5 MpRunner

**File:** `src/march/agents/mp_runner.py`

Manages a single mpAgent child process from spawn to result collection.

**Core guarantee:** `wait_result()` always returns a `RunOutcome`. It never hangs.

**Internal structure:**

```
MpRunner
  ├── _recv_loop (asyncio.Task)     # Reads IPC messages from child
  ├── _monitor (asyncio.Task)       # Watches heartbeat timing + process liveness
  ├── _parent_sock                  # Non-blocking Unix socket, driven by asyncio
  ├── _result_future                # asyncio.Future[RunOutcome]
  ├── _latest_heartbeat             # Most recent heartbeat dict
  ├── _spawn_handler                # Callback: delegate grandchild spawns to AgentManager
  ├── _steer_handler                # Callback: forward steer to grandchild
  └── _kill_handler                 # Callback: forward kill to grandchild
```

**Spawn sequence:**

1. `create_socket_pair()` → two connected AF_UNIX SOCK_STREAM sockets, both `set_inheritable(True)`
2. Set parent socket to non-blocking
3. `multiprocessing.get_context("spawn").Process(target=mp_child_main, args=(...))` — pass child socket fd, config path, task, session ID, log dir, heartbeat interval
4. `process.start()` → record `pid`; set `pgid = pid` (child calls `setpgrp`)
5. Close child socket on parent side
6. Launch `_recv_loop` and `_monitor` as asyncio tasks
7. Return `MpRunHandle(pid, session_id, runner)`

**Kill sequence:**

1. Try sending `{type: "kill"}` via IPC (graceful notification)
2. `os.killpg(pgid, SIGKILL)` (immediate)
3. Resolve `_result_future` with `RunOutcome(status="cancelled")`

**Graceful kill** (used by heartbeat timeout): SIGTERM → wait `kill_grace_seconds` → SIGKILL if still alive → `process.join(timeout=5.0)`.

**`_resolve_result()`** is idempotent. Multiple code paths (recv loop, monitor, kill) may race to set the result; the `_done` flag ensures only the first caller wins.

**`wait_result()`** wraps the future in `asyncio.wait_for()` with a safety timeout of `heartbeat_timeout + kill_grace + 60s`. If even that fires, it force-kills the process and returns a timeout outcome.

**Spawn proxy handling:** When `_recv_loop` receives `MSG_SPAWN_REQUEST`, `MSG_SPAWN_STEER`, or `MSG_SPAWN_KILL`, it dispatches to the corresponding handler callback (provided by `AgentManager._execute_child_mp`). Results flow back via `_send_spawn_result()` or `notify_child_completed()`.

### 4.6 IPC Protocol

**File:** `src/march/agents/ipc.py`

**Wire format:** 4-byte big-endian length prefix + msgpack payload (max 64 MB).

```
┌──────────────┬──────────────────────────┐
│ 4 bytes (BE) │ msgpack payload          │
│ payload len  │ (max 64 MB)             │
└──────────────┴──────────────────────────┘
```

Falls back to JSON if msgpack is not installed (logs a warning).

**Message types:**

| Direction | Type Constant | Purpose |
|-----------|---------------|---------|
| Parent → Child | `MSG_STEER` | Inject steering message |
| Parent → Child | `MSG_KILL` | Request graceful shutdown |
| Parent → Child | `MSG_SPAWN_RESULT` | Response to child's spawn request |
| Parent → Child | `MSG_CHILD_COMPLETED` | Grandchild finished notification |
| Child → Parent | `MSG_HEARTBEAT` | Periodic metrics |
| Child → Parent | `MSG_PROGRESS` | Tool execution progress |
| Child → Parent | `MSG_RESULT` | Final run result |
| Child → Parent | `MSG_LOG` | Log entry |
| Child → Parent | `MSG_SPAWN_REQUEST` | Request parent to spawn grandchild |
| Child → Parent | `MSG_SPAWN_STEER` | Forward steer to grandchild |
| Child → Parent | `MSG_SPAWN_KILL` | Request kill of grandchild |

**Two API surfaces:**

- **Async** (`send_message` / `recv_message`): Used by the parent process (MpRunner) on the asyncio event loop. Uses `loop.sock_sendall()` / `loop.sock_recv()`.
- **Sync** (`send_message_sync` / `recv_message_sync`): Used by the child's `_HeartbeatThread` (a plain `threading.Thread`). Blocking socket I/O with optional timeout.

**Socket inheritance:** Both ends are marked `set_inheritable(True)` so the fd survives `multiprocessing.Process(start_method="spawn")`.

**Key message schemas:**

```python
# Heartbeat (child → parent)
{
    "type": "heartbeat",
    "ts": 1709913600.0,
    "data": {
        "memory_rss_mb": 256.3,
        "elapsed_seconds": 45.2,
        "tokens_used": 12500,
        "total_cost": 0.0384,
        "tool_calls_made": 7,
        "llm_calls_made": 3,
        "summary": "Executing tool: exec",
        "current_tool": "exec",
        "current_tool_detail": "{'command': 'pytest tests/'}",
        "recent_tools": [
            {"name": "read", "status": "done", "ms": 12, "summary": ""},
            {"name": "edit", "status": "done", "ms": 45, "summary": ""}
        ]
    }
}

# Result (child → parent)
{
    "type": "result",
    "status": "ok",
    "output": "Task complete: created 12 test files...",
    "error": "",
    "tokens": 25000,
    "cost": 0.0768
}

# Spawn request (child → parent, for nested agents)
{
    "type": "spawn_request",
    "task": "Run the test suite",
    "agent_id": "",
    "model": "",
    "timeout": 0,
    "request_id": "a1b2c3d4e5f6g7h8"
}

# Spawn result (parent → child)
{
    "type": "spawn_result",
    "request_id": "a1b2c3d4e5f6g7h8",
    "status": "accepted",
    "child_key": "agent-xyz:mpagent:abc123456789",
    "run_id": "...",
    "error": ""
}

# Grandchild completed (parent → child)
{
    "type": "child_completed",
    "child_key": "agent-xyz:mpagent:abc123456789",
    "status": "ok",
    "output": "All tests passed.",
    "error": ""
}
```

---

## 5. Lifecycle

### 5.1 mtAgent

```
1. AgentManager.spawn(execution="mt")
   ├── Validate depth < max_spawn_depth
   ├── Generate run_id (UUID) + child_key
   ├── RunRecord → Registry.register()
   ├── Create steer asyncio.Queue
   └── TaskQueue.enqueue("mt", _execute_child)

2. TaskQueue drains → asyncio.create_task()

3. _execute_child()
   ├── agent_factory() → Child Agent.run(task)
   ├── Catches: CancelledError → "cancelled"
   │            TimeoutError  → "timeout"
   │            Exception     → "error"
   └── Normal return         → "ok"

4. Registry.complete(run_id, outcome)
   Clean up _active_tasks, _steer_queues

5. AgentAnnouncer.announce_completion()
   steer → queue → direct → pending
```

### 5.2 mpAgent

```
1. AgentManager.spawn(execution="mp")
   ├── Validate depth < max_spawn_depth
   ├── Generate run_id + child_key
   ├── RunRecord → Registry.register()
   └── TaskQueue.enqueue("mp", _execute_child_mp)

2. _execute_child_mp()
   ├── Build MpRunner with spawn/steer/kill handler closures
   ├── runner.spawn() → child process starts
   │   └── mp_child_main():
   │       ├── os.setpgrp()
   │       ├── Reconstruct socket from fd
   │       ├── _setup_child_logging()
   │       ├── load_config() → init LLM, tools, memory
   │       ├── Create in-memory Session
   │       ├── Create _SpawnProxy
   │       ├── Start _HeartbeatThread
   │       ├── Register spawn_agent tool
   │       ├── Wrap tools.execute → _tracked_execute
   │       ├── Start _steering_pump task
   │       ├── Agent.run(task, session)
   │       ├── Stop steering pump + heartbeat thread
   │       └── Send result via IPC
   ├── Record PID in RunRecord
   └── runner.wait_result() ← guaranteed to return

3. Parent runs concurrently:
   ├── _recv_loop: heartbeat / progress / log / result / spawn proxy
   └── _monitor: heartbeat timeout + process liveness

4. Result arrives → _resolve_result() → wait_result() returns

5. Cleanup:
   ├── Remove runner from _active_runners
   ├── Registry.complete()
   ├── Optionally persist to SessionStore
   └── AgentAnnouncer.announce_completion()
```

---

## 6. Failure Handling

### 6.1 Failure Matrix

| Scenario | Detection | Resolution | Outcome |
|----------|-----------|------------|---------|
| mtAgent exception | try/except in `_execute_child` | Catch, build outcome | `error` |
| mtAgent cancelled | `CancelledError` | Catch | `cancelled` |
| mtAgent timeout | `TimeoutError` | Catch | `timeout` |
| mpAgent normal result | IPC `result` message | `_recv_loop` resolves future | `ok` or `error` |
| mpAgent OOM (SIGKILL) | `process.is_alive()=False`, `exitcode=-9` | `_monitor` → `_handle_process_exit` | `error` + "killed by SIGKILL (likely OOM)" |
| mpAgent killed by signal | `exitcode < 0` | `_handle_process_exit` | `error` + signal name |
| mpAgent heartbeat timeout | `_monitor` sees elapsed > threshold | SIGTERM → grace → SIGKILL | `timeout` |
| mpAgent IPC disconnect | `ConnectionError` in `_recv_loop` | `_resolve_result` immediately | `error` + "IPC connection lost" |
| mpAgent parent disappears | `BrokenPipeError` in HeartbeatThread send | Child sets `kill_requested`, exits | N/A (parent gone) |
| mpAgent exits cleanly but no result | `exitcode=0`, `_done=False` | Wait 0.5s, then error | `error` + "no result received" |
| All announce methods fail | Steer/queue/direct all fail | Store in `_pending_queue` | Delivered on next parent turn |
| Safety timeout in `wait_result` | `asyncio.wait_for` fires | Force kill process | `timeout` + "safety timeout" |

### 6.2 The Five-Layer Guarantee

`MpRunner.wait_result()` always returns because five independent mechanisms cover every failure mode:

1. **IPC result message** — the happy path. Child sends `{type: "result"}`.
2. **Process exit detection** — `_monitor` polls `process.is_alive()`. If the child is dead but no result arrived, it generates an outcome from the exit code.
3. **Heartbeat timeout** — if the child is alive but silent (hung), `_monitor` kills it after `heartbeat_timeout_seconds`.
4. **IPC disconnect** — `_recv_loop` catches `ConnectionError` and resolves immediately.
5. **Safety timeout** — `wait_result()` wraps everything in `asyncio.wait_for(timeout=heartbeat_timeout + kill_grace + 60)`. Last resort.

`_resolve_result()` is idempotent via the `_done` flag, so multiple layers racing to set the result is safe.

```python
safety_timeout = heartbeat_timeout_seconds + kill_grace_seconds + 60
# Default: 300 + 10 + 60 = 370s
```

---

## 7. Heartbeat & Monitoring

### 7.1 Heartbeat Content

The child's `_HeartbeatThread` (a daemon thread) sends heartbeats at `heartbeat_interval_seconds` (default 60s). Each heartbeat includes:

| Field | Source |
|-------|--------|
| `memory_rss_mb` | `resource.getrusage(RUSAGE_SELF).ru_maxrss` — KB on Linux, bytes on macOS |
| `elapsed_seconds` | `time.time() - start_time` |
| `tokens_used` | Updated by agent loop |
| `total_cost` | Updated by agent loop |
| `tool_calls_made` | Incremented by `_tracked_execute` wrapper |
| `llm_calls_made` | Updated by agent loop |
| `summary` | Current status text (e.g., "Executing tool: exec") |
| `current_tool` | Name of the tool currently running |
| `current_tool_detail` | First 200 chars of tool args |
| `recent_tools` | Last 3 tool calls: name, status, duration_ms, summary |

Tool tracking works by monkey-patching `agent.tools.execute` with `_tracked_execute`, which wraps every tool call to record timing and status.

### 7.2 Heartbeat Thread Internals

The thread alternates between sending a heartbeat and listening for incoming messages in short recv windows (1s timeout per iteration). This lets it pick up `steer`, `kill`, `spawn_result`, and `child_completed` messages between heartbeats.

**Orphan detection:** If `send_message_sync()` raises `BrokenPipeError` or `ConnectionError`, the parent is gone. The thread sets `kill_requested = True` and stops, causing the child process to exit cleanly.

**Steering flow:** Incoming `steer` messages are appended to `_steer_messages` (protected by a lock). The `_steering_pump` asyncio task in the child's event loop calls `drain_steer_messages()` every 0.5s and injects them into the agent via `agent.steer(session_id, msg)`.

### 7.3 Monitor Loop

`_monitor` runs in the parent process as an asyncio task. Check interval: `min(heartbeat_timeout / 4, 15s)`.

Each iteration:

1. Is the process dead? → `_handle_process_exit()` (generate outcome from exit code)
2. Has it been too long since the last heartbeat? → Kill the process group (SIGTERM → grace → SIGKILL) → `RunOutcome(status="timeout")`

### 7.4 Querying Status

```python
# Detailed status for one agent
lines = await manager.logs("agent-abc12345")

# All agents with heartbeat data
statuses = await manager.list()
for s in statuses:
    if s.heartbeat:
        print(f"RSS: {s.heartbeat['memory_rss_mb']}MB, Tokens: {s.heartbeat['tokens_used']}")
```

---

## 8. Configuration

### 8.1 config.yaml

```yaml
agents:
  max_concurrent: 4              # Unused — each lane has its own limit

  mt:
    max_concurrent: 8            # mtAgent lane concurrency

  mp:
    max_concurrent: 8            # mpAgent lane concurrency
    heartbeat_interval_seconds: 60
    heartbeat_timeout_seconds: 300
    kill_grace_seconds: 10
    spawn_method: spawn          # "spawn" or "forkserver"

  subagents:
    max_spawn_depth: 1           # Max nesting depth (default: 1)
```

### 8.2 Parameter Reference

**MtConfig** (`agents.mt`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent` | int | 8 | Max simultaneous mtAgents |

**MpConfig** (`agents.mp`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent` | int | 8 | Max simultaneous mpAgents |
| `heartbeat_interval_seconds` | int | 60 | How often the child sends heartbeats |
| `heartbeat_timeout_seconds` | int | 300 | How long the parent waits before declaring the child dead |
| `kill_grace_seconds` | int | 10 | SIGTERM → SIGKILL grace period |
| `spawn_method` | str | `"spawn"` | `multiprocessing` start method. `"spawn"` is safest (clean re-init). `"forkserver"` is faster but may inherit parent state. |

**SubagentsCommonConfig** (`agents.subagents`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_spawn_depth` | int | 1 | Max nesting depth. `1` = sub-agents only, no grandchildren. Increase to enable nested spawning. |

**AgentManagerConfig** (code-internal):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_spawn_depth` | int | 1 | Mapped from `SubagentsCommonConfig` |
| `reset_after_complete_minutes` | int | 60 | Safety-net TTL for orphaned completed records |
| `announce_timeout_seconds` | int | 60 | Max time to wait for announce delivery |

---

## 9. Session IDs & Logging

### 9.1 Naming

Session keys follow the pattern:

```
{agent_id}:{execution_suffix}:{uuid_hex[:12]}
```

- `agent_id`: Caller-specified or auto-generated as `agent-{uuid[:8]}`
- `execution_suffix`: `mtagent` or `mpagent`
- `uuid_hex[:12]`: 12 random hex characters

Examples:

```
agent-a1b2c3d4:mtagent:e5f6a7b8c9d0
agent-a1b2c3d4:mpagent:e5f6a7b8c9d0
refactor-db:mtagent:abc123def456
```

Run IDs are full UUID v4 strings, used internally only.

### 9.2 Log Structure

```
~/.march/logs/
  ├── {session_id}/
  │   ├── 2026-03-08.log          # Date-partitioned log file
  │   └── heartbeats.jsonl        # mpAgent heartbeat records
  └── ...
```

The child process configures its own file logger via `_setup_child_logging()`. Format:

```
2026-03-08T16:00:00 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — mpAgent child started: pid=12345 pgid=12345 session=agent-xyz:mpagent:abc456
```

The child also captures all `march.*` logger output to the same file, so LLM calls, tool executions, and other framework logs are preserved.

---

## 10. Nested Agents (Depth > 1)

### 10.1 The Default: Depth 1

By default, `max_spawn_depth` is **1**. Sub-agents cannot spawn their own children. This is the safe default — it prevents runaway recursive spawning and keeps the system simple.

### 10.2 Enabling Deeper Nesting

Set `max_spawn_depth` higher in config:

```yaml
agents:
  subagents:
    max_spawn_depth: 3
```

This allows:

```
Root Agent (depth 0)
  ├── Sub-agent A (depth 1)
  │   ├── Grandchild A1 (depth 2)
  │   │   └── Great-grandchild A1a (depth 3)  ← max reached
  │   └── Grandchild A2 (depth 2)
  └── Sub-agent B (depth 1)
```

Each `spawn()` checks `caller_depth >= max_spawn_depth` and rejects if the limit is hit.

### 10.3 How It Works: _SpawnProxy

An mpAgent child process has no access to the parent's `AgentManager`. It runs in a separate process with its own Python interpreter. The `_SpawnProxy` class bridges this gap by proxying spawn operations over the existing IPC channel.

**Child side (`_SpawnProxy` in `mp_child.py`):**

```
Child agent calls spawn_agent tool
  → _SpawnProxy.spawn(task, agent_id, model, timeout)
    → IPC send: {type: "spawn_request", ..., request_id: "abc123"}
    → Block on asyncio.Future keyed by request_id
    ← IPC recv: {type: "spawn_result", request_id: "abc123", child_key: "...", ...}
    → Future resolves → return (child_key, run_id)
```

**Parent side (`MpRunner._recv_loop`):**

```
_recv_loop receives MSG_SPAWN_REQUEST
  → _handle_spawn_request(msg)
    → Calls _spawn_handler callback (closure from AgentManager._execute_child_mp)
      → AgentManager.spawn(params, ctx) with caller_depth + 1
    → Sends MSG_SPAWN_RESULT back to child
```

The child process also gets a `spawn_agent` tool registered in its tool registry, so the child's LLM can naturally decide to spawn grandchildren.

**Waiting for grandchild completion:**

```
Child calls _SpawnProxy.wait_child(child_key)
  → Creates asyncio.Future keyed by child_key
  ← Parent sends MSG_CHILD_COMPLETED when grandchild finishes
  → _HeartbeatThread receives it → calls spawn_proxy.handle_child_completed()
    → Resolves the Future from the event loop thread
  → wait_child() returns (status, output)
```

The `spawn_agent` tool supports `wait=True` (default) to block until the grandchild completes, or `wait=False` to get back the `child_key` immediately for manual management.

### 10.4 Execution Consistency

When a sub-agent spawns a grandchild, the manager enforces that the grandchild uses the **same execution mode** as its parent. If the parent is an mpAgent, the grandchild must also be mp. The code detects this by looking up the parent's `RunRecord` via `registry.get_by_child_key()` and overriding the requested execution mode if it differs (with a warning log).

This prevents mixed mt/mp hierarchies, which would complicate IPC routing and lifecycle management.

### 10.5 Tree Operations

**`registry.get_subtree(root_key)`** — Returns all descendants of a session in depth-first order. Used by `kill_recursive()` and `reset_children()`.

**`manager.kill_recursive(agent_id)`** — Gets the subtree, reverses it (deepest first), and kills each active agent. Returns total count killed. Bottom-up order prevents orphans.

**`manager.reset_children(parent_key)`** — For each direct child: recursively kill if active, delete session memory files, delete session from store, remove from registry.

### 10.6 Announce Routing for Nested Agents

Results from nested agents are announced to their **immediate parent**, not the root. The `_announce()` method in `AgentManager` has special handling: if the parent is an active mpAgent, it sends a `MSG_CHILD_COMPLETED` IPC message directly to the parent's child process via `runner.notify_child_completed()`. This lets the mpAgent's `_SpawnProxy` resolve its `wait_child()` future.

The root agent only sees results from its direct children. If a child summarizes its grandchildren's work in its own output, that summary flows up naturally.

### 10.7 Practical Guidance

**When depth > 1 makes sense:**
- A coordinator agent that spawns specialists, each needing their own helpers
- Multi-stage pipelines where each stage may dynamically fork
- Research tasks that discover they need to explore multiple branches

**When to stay at depth 1:**
- Most task delegation scenarios (the vast majority)
- Resource-constrained environments
- When you can't predict the maximum depth

Start with the default (`max_spawn_depth: 1`) and increase only when you have a concrete need. Depth 2–3 covers nearly all hierarchical decomposition patterns.

---

## 11. Comparison with Other Systems

### vs. OpenClaw sessions_spawn

| | March | OpenClaw |
|---|-------|----------|
| Execution modes | Dual: mt (asyncio) + mp (process) | Single: sub-agent session |
| Process isolation | mpAgent: full (own process group) | None (shares gateway process) |
| Heartbeat | Built-in: RSS, tokens, cost, tool history | External: claw-guard monitors PIDs |
| IPC | Unix socketpair + msgpack | Session messages |
| Steering | Native (mt: Queue, mp: IPC) | Steer API injection |
| Result delivery | 3-tier cascade (steer → queue → direct) | Auto-announce to requester |
| Failure recovery | 5-layer guarantee in wait_result() | claw-guard notifications + manual checks |
| Concurrency | Lane-based TaskQueue | No built-in limits |

### vs. LangGraph

| | March | LangGraph |
|---|-------|-----------|
| Execution | Independent agent instances | Graph node execution |
| Isolation | Process-level (mpAgent) | None (same process) |
| Communication | Binary IPC (msgpack over Unix socket) | Graph state passing |
| Monitoring | Real-time heartbeat with resource metrics | None built-in |
| Failure handling | Guaranteed result delivery | Graph error handling |
| Runtime steering | Dynamic message injection | Not supported |
| Concurrency | Independent lane-based control | External orchestration |

### What's Distinctive About March

1. **Process-group isolation.** `os.setpgrp()` + `os.killpg()` means the parent can cleanly kill a child and all its descendants. OOM only hits the child group.

2. **Structured heartbeats.** Not just "alive/dead" — each heartbeat carries memory usage, token consumption, cost, current tool, and recent tool history. The parent knows what the child is doing at all times.

3. **IPC steering.** The parent can redirect a running child mid-task by injecting steering messages through the HeartbeatThread → steering pump → Agent pipeline.

4. **Guaranteed result delivery.** Five independent detection mechanisms ensure `wait_result()` always returns, covering every failure mode from clean exit to OOM to hung process to broken pipe.

5. **Orphan protection.** If the parent dies, the child's HeartbeatThread detects `BrokenPipeError` on the next send and exits cleanly. No zombies.

6. **Nested agent support.** `_SpawnProxy` transparently proxies spawn/steer/kill/wait operations over IPC, letting mpAgent children spawn their own children without direct access to `AgentManager`. All agents register in the same central `AgentRegistry`.

---

## Appendix: Quick Reference

```python
# Spawn an mtAgent
result = await manager.spawn(
    SpawnParams(task="Summarize the Python 3.13 changelog", execution="mt", label="changelog"),
    SpawnContext(requester_session="main-session", origin="terminal"),
)

# Spawn an mpAgent
result = await manager.spawn(
    SpawnParams(task="Process 500MB CSV dataset", execution="mp", model="litellm/claude-sonnet-4-20250514"),
    SpawnContext(requester_session="main-session", origin="terminal"),
)

# Steer a running agent
await manager.send("agent-abc:mpagent:def123", "Focus on the revenue column outliers")

# Kill one agent
await manager.kill("agent-abc:mpagent:def123")

# Kill an agent and all its descendants
count = await manager.kill_recursive("agent-abc:mpagent:def123")

# List all agents
for s in await manager.list():
    print(f"{s.child_key}: {s.status} ({s.duration_seconds:.1f}s)")
    if s.heartbeat:
        print(f"  RSS={s.heartbeat['memory_rss_mb']}MB tokens={s.heartbeat['tokens_used']}")

# Get full descendant tree
for depth, record in manager.registry.get_tree_with_depth("main-session"):
    print(f"{'  ' * depth}{record.child_key}: {record.task[:50]}")
```
