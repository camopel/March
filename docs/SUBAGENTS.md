# SUBAGENTS.md — March Agent 系统技术文档

> March 框架的 agent 子系统：mtAgent（asyncio 协程）与 mpAgent（隔离进程）的完整技术参考。

---

## 目录

1. [概述](#1-概述)
2. [两种执行模式](#2-两种执行模式)
3. [架构图](#3-架构图)
4. [核心组件](#4-核心组件)
5. [生命周期](#5-生命周期)
6. [故障处理](#6-故障处理)
7. [心跳与监控](#7-心跳与监控)
8. [配置参考](#8-配置参考)
9. [Session ID 和日志](#9-session-id-和日志)
10. [与其他框架的不同](#10-与其他框架的不同)

---

## 1. 概述

March 的 agent 系统允许主 agent 将复杂任务分解为多个子任务，并行派发给独立的 sub-agent 执行。它解决的核心问题是：

- **并行执行**：多个子任务同时运行，不阻塞主 agent 的交互循环
- **故障隔离**：子任务崩溃（OOM、死循环、异常）不影响主进程
- **资源控制**：通过 lane 并发限制防止资源耗尽
- **结果推送**：子任务完成后自动将结果推送给父 agent，无需轮询

系统提供两种执行模式——**mtAgent**（asyncio 协程，轻量快速）和 **mpAgent**（独立进程，完全隔离）——由调用方根据任务特性选择。

### 设计哲学

- **Push-based**：所有结果通过 `AgentAnnouncer` 主动推送，父 agent 永远不需要轮询
- **保证交付**：`MpRunner.wait_result()` 承诺**永远返回** `RunOutcome`，无论子进程如何崩溃
- **纯内存追踪**：`AgentRegistry` 不做磁盘持久化，crash recovery 依赖日志

---

## 2. 两种执行模式

### 2.1 mtAgent（Multi-Thread / Asyncio）

mtAgent 在主进程的 asyncio 事件循环中作为一个 `asyncio.Task` 运行。它与主 agent 共享进程内存空间。

**原理：**

```
主进程 event loop
  ├── 主 agent turn（处理用户消息）
  ├── mtAgent-1（asyncio.Task）
  ├── mtAgent-2（asyncio.Task）
  └── ...
```

每个 mtAgent 通过 `TaskQueue` 的 `mt` lane 排队执行。当 lane 有空闲 slot 时，task 被 drain 出来并创建为 `asyncio.Task`。

**适用场景：**
- 对话式任务（聊天、问答）
- 文件 I/O 操作（读写、编辑）
- API 调用（web search、LLM 调用）
- 消息发送
- 任何 I/O-bound 的轻量任务

**限制：**
- 共享进程内存 → 子任务崩溃会拖垮主进程
- Python GIL → 无法利用多核 CPU 做计算密集任务
- 无进程级隔离 → OOM 会杀死整个主进程
- 无心跳监控 → 只能通过 asyncio 超时检测问题

**Steering 机制：**
mtAgent 通过 `asyncio.Queue` 接收 steering 消息。父 agent 调用 `AgentManager.send()` 将消息放入队列，mtAgent 在下一次 LLM 调用前消费。

### 2.2 mpAgent（Multi-Process / Isolated）

mpAgent 通过 `multiprocessing.Process(start_method="spawn")` 创建独立子进程。父子进程通过 Unix socketpair + msgpack 序列化的 IPC 协议通信。

**原理：**

```
主进程                          子进程（独立 process group）
  │                                │
  ├── MpRunner                     ├── os.setpgrp()
  │   ├── _recv_loop (IPC 接收)    ├── Agent 实例（独立 LLM/Tools）
  │   ├── _monitor (心跳监控)      ├── HeartbeatThread（心跳+steering 接收）
  │   └── parent_sock ◄──────────► child_sock
  │       (Unix socketpair, msgpack + 4-byte length prefix)
```

**适用场景：**
- 大数据处理（文件解析、ETL）
- GPU 任务（ML 训练、推理）
- 仿真任务（物理模拟、机器人仿真）
- 3D 资产处理（模型转换、渲染）
- 任何可能 OOM 或 hang 的任务

**限制：**
- 启动开销较高（需要重新加载 config、初始化 LLM provider）
- 子进程**不访问 SessionStore**（结果通过 IPC 返回，由父进程持久化）
- 子进程有独立的 session memory（存储在自己的 session ID 目录下）
- 无法直接共享父进程的内存状态

**选择规则：不确定用哪个 → 用 mpAgent。安全第一。**

### 2.3 对比表格

| 特性 | mtAgent | mpAgent |
|------|---------|---------|
| **执行方式** | asyncio.Task（主进程内） | multiprocessing.Process（独立进程） |
| **隔离级别** | 无（共享内存） | 进程级（独立 process group） |
| **启动开销** | 极低（创建 coroutine） | 较高（spawn 进程 + 初始化 Agent） |
| **故障影响** | 崩溃影响主进程 | 崩溃仅影响子进程 |
| **OOM 保护** | 无 | 有（SIGKILL 仅杀子进程组） |
| **心跳监控** | 无 | 有（周期性 heartbeat + 超时检测） |
| **IPC 方式** | asyncio.Queue（内存） | Unix socketpair + msgpack |
| **Steering** | asyncio.Queue 直接注入 | IPC `steer` 消息 → HeartbeatThread → Agent |
| **Kill 方式** | asyncio.Task.cancel() | os.killpg(SIGTERM → SIGKILL) |
| **SessionStore** | 共享主进程的 | 无（结果由父进程持久化） |
| **Session ID 格式** | `{agent_id}:mtagent:{uuid[:12]}` | `{agent_id}:mpagent:{uuid[:12]}` |
| **默认并发上限** | 8 | 8 |
| **适用任务** | I/O-bound、轻量、低风险 | CPU-bound、高风险、可能 OOM |

---

## 3. 架构图

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentManager                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ spawn()  │  │   kill()     │  │      send() / steer      │  │
│  │ list()   │  │   logs()     │  │  reset_children()        │  │
│  └────┬─────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│       │               │                       │                 │
│  ┌────▼───────────────▼───────────────────────▼──────────────┐  │
│  │                    TaskQueue                               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│  │  │ main    │  │ mt      │  │ mp      │  │ cron        │  │  │
│  │  │ lane    │  │ lane    │  │ lane    │  │ lane        │  │  │
│  │  │ (auto)  │  │ (max:8) │  │ (max:8) │  │ (max:1)    │  │  │
│  │  └─────────┘  └────┬────┘  └────┬────┘  └─────────────┘  │  │
│  └─────────────────────┼───────────┼─────────────────────────┘  │
│                        │           │                            │
│  ┌─────────────────────┼───────────┼─────────────────────────┐  │
│  │              AgentRegistry (纯内存)                        │  │
│  │  RunRecord: run_id, child_key, status, outcome, pid ...   │  │
│  └─────────────────────┼───────────┼─────────────────────────┘  │
│                        │           │                            │
│  ┌─────────────────────▼───────────▼─────────────────────────┐  │
│  │              AgentAnnouncer                                │  │
│  │  策略: steer → queue → direct → pending_queue             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 mtAgent 数据流

```
父 Agent (主进程)
    │
    │ spawn(execution="mt")
    ▼
AgentManager.spawn()
    │
    │ 1. 创建 RunRecord → AgentRegistry.register()
    │ 2. 创建 asyncio.Queue (steer)
    │ 3. TaskQueue.enqueue("mt", _execute_child)
    ▼
TaskQueue (mt lane)
    │
    │ drain → asyncio.create_task()
    ▼
_execute_child()                     父 Agent
    │                                    │
    │ agent_factory(task, model, ...)    │ send(message)
    │         │                          │     │
    │         ▼                          │     ▼
    │   Child Agent.run()                │ asyncio.Queue.put()
    │         │                          │     │
    │         │ (完成)                    │     │
    │         ▼                          │     │
    │   RunOutcome                       │     │
    │         │                          │     │
    ▼         ▼                          │     │
AgentRegistry.complete()               │     │
    │                                    │     │
    ▼                                    │     │
AgentAnnouncer.announce_completion()    │     │
    │                                    │     │
    │ try_steer ──────────────────────►  │     │
    │ try_queue ──────────────────────►  │     │
    │ send_direct ────────────────────►  │     │
    ▼                                    ▼     ▼
  (结果推送到父 session)
```

### 3.3 mpAgent 数据流（含 IPC）

```
父进程 (MpRunner)                    子进程 (mp_child_main)
    │                                    │
    │ spawn(execution="mp")              │
    ▼                                    │
create_socket_pair()                     │
    │                                    │
    ├── parent_sock (non-blocking)       │
    └── child_sock ──────────────────►   │
                                         │
multiprocessing.Process.start() ─────►   │ os.setpgrp()
    │                                    │ 加载 config.yaml
    │                                    │ 初始化 Agent (LLM, Tools, Memory)
    │                                    │ 创建 in-memory Session
    │                                    │
    │                                    ▼
    │                              HeartbeatThread.start()
    │                                    │
    │   ┌────────────────────────────────┤
    │   │                                │
    │   │  IPC 消息流                     │
    │   │                                │
    │   │  heartbeat ◄───────────────── 周期性心跳 (每 60s)
    │   │  {type:"heartbeat",            │ {memory_rss_mb, elapsed_seconds,
    │   │   ts:..., data:{...}}          │  tokens_used, total_cost,
    │   │                                │  tool_calls_made, summary, ...}
    │   │                                │
    │   │  steer ──────────────────────► 注入 steering 消息
    │   │  {type:"steer",                │ → Agent.steer(session_id, msg)
    │   │   message:"..."}               │
    │   │                                │
    │   │  kill ───────────────────────► 请求优雅退出
    │   │  {type:"kill"}                 │ → kill_requested = True
    │   │                                │
    │   │  progress ◄──────────────────  工具执行进度
    │   │  {type:"progress",             │
    │   │   tool_name, status, ...}      │
    │   │                                │
    │   │  result ◄────────────────────  最终结果
    │   │  {type:"result",               │ Agent.run() 完成
    │   │   status, output, error,       │
    │   │   tokens, cost}                │
    │   │                                │
    │   └────────────────────────────────┘
    │                                    │
    ▼                                    ▼
_recv_loop() 处理消息               子进程退出
_monitor() 检测心跳超时
    │
    ▼
RunOutcome → AgentRegistry.complete()
    │
    ▼
AgentAnnouncer.announce_completion()
    │
    ▼
(可选) SessionStore 持久化子 agent 结果
```

---

## 4. 核心组件

### 4.1 AgentManager — 统一入口

`AgentManager` 是 agent 子系统的唯一公开接口，封装了 spawn、kill、steer、list 等所有操作。

**文件：** `src/march/agents/manager.py`

**核心方法：**

```python
class AgentManager:
    async def spawn(self, params: SpawnParams, ctx: SpawnContext) -> SpawnResult
    async def list(self) -> list[AgentStatus]
    async def kill(self, agent_id: str) -> bool
    async def send(self, agent_id: str, message: str) -> bool
    async def logs(self, agent_id: str, tail: int = 50) -> list[str]
    async def reset_children(self, parent_session_key: str) -> int
    def get_child_sessions(self, parent_session_key: str) -> list[RunRecord]
```

**SpawnParams 参数：**

```python
@dataclass
class SpawnParams:
    task: str                    # 任务描述
    agent_id: str = ""           # 自动生成 "agent-{uuid[:8]}"
    model: str = ""              # LLM 模型（空则用默认）
    tools: list[str] | None = None  # 可用工具列表
    timeout: int = 0             # 超时秒数（0=无限）
    mode: str = "run"            # "run"（一次性）或 "session"（保持活跃）
    cleanup: str = "keep"        # "keep" 或 "delete"
    label: str = ""              # 可读标签
    execution: str = "mt"        # "mt" 或 "mp"
```

**SpawnContext 上下文：**

```python
@dataclass
class SpawnContext:
    requester_session: str       # 父 session key
    origin: str = ""             # 父的 channel/source
    caller_depth: int = 0        # 当前 spawn 深度
```

**关键行为：**
- `spawn()` 检查 `caller_depth >= max_spawn_depth` 防止无限递归
- mtAgent 和 mpAgent 走不同的执行路径（`_execute_child` vs `_execute_child_mp`）
- `kill()` 对 mtAgent 调用 `task.cancel()`，对 mpAgent 调用 `runner.kill()`（killpg 整个进程组）
- `send()` 对 mtAgent 写入 `asyncio.Queue`，对 mpAgent 通过 `MpRunner.send_steer()` 发送 IPC 消息
- `reset_children()` 在父 session `/reset` 时清理所有子 agent（先 kill 活跃的，再删除 session 数据）

### 4.2 AgentRegistry — 内存追踪

`AgentRegistry` 是纯内存的 agent 运行记录存储。不做磁盘持久化。

**文件：** `src/march/agents/registry.py`

**核心数据结构：**

```python
@dataclass
class RunRecord:
    run_id: str              # UUID，唯一标识
    child_key: str           # 子 agent 的 session key
    requester_key: str       # 父 session key
    requester_origin: str    # 父的 channel（用于 announce 投递）
    task: str                # 任务描述
    started_at: float        # Unix 时间戳
    ended_at: float          # 0.0 = 仍在运行
    mode: str                # "run" | "session"
    cleanup: str             # "delete" | "keep"
    outcome: RunOutcome | None  # 完成后的结果
    cleanup_done: bool       # announce 是否已投递
    execution: str           # "mt" | "mp"
    pid: int                 # mpAgent 的进程 PID（mtAgent 为 0）
    log_path: str            # 日志目录路径

@dataclass
class RunOutcome:
    status: str              # "ok" | "error" | "timeout" | "cancelled"
    error: str = ""
    output: str = ""
    duration_ms: float = 0.0
```

**关键属性：**
- `RunRecord.is_active` → `ended_at == 0.0`（判断是否仍在运行）
- `RunRecord.duration_seconds` → 实时计算已运行时长

**清理策略：**
- `cleanup_old()` 只移除 `cleanup_done=True` 且超过 `max_age_seconds` 的记录（安全网，处理孤儿记录）
- 正常清理通过 `AgentManager.reset_children()` 在父 session `/reset` 时触发

### 4.3 AgentAnnouncer — 结果推送

`AgentAnnouncer` 负责将子 agent 的完成结果推送给父 session。采用三级降级策略：

**文件：** `src/march/agents/announce.py`

**投递策略（按优先级）：**

| 优先级 | 策略 | 说明 |
|--------|------|------|
| 1 | **Steer** | 注入到父 agent 当前正在执行的 turn 中 |
| 2 | **Queue** | 排队到父 agent 当前 turn 结束后投递 |
| 3 | **Direct** | 启动一个新的 agent turn 来处理结果 |
| 4 | **Pending** | 所有方式都失败时，存入内存 pending queue |

**消息格式：**

```
✅ mtAgent `agent-abc12345:mtagent:def678901234` finished
<output content>

❌ mpAgent `agent-xyz:mpagent:abc123456789` failed
**Error:** OOM killed by SIGKILL

⏱️ mpAgent `agent-xyz:mpagent:abc123456789` timed out
**Error:** No heartbeat for 300s (timeout: 300s)

🚫 mtAgent `agent-abc:mtagent:def678` was cancelled
**Error:** killed by user
```

**Pending 恢复：**
当父 session 开始新的 turn 时，调用 `announcer.get_pending(requester_key)` 获取并清空积压的通知。

**关键设计：**
- Announcer 通过回调函数连接 session 基础设施，避免循环依赖
- Agent session 在完成后**不会被删除**，持续存在直到父 session `/reset`
- `cleanup_done` 标记仅表示 announce 已投递，不代表 session 被删除

### 4.4 TaskQueue — Lane 并发控制

`TaskQueue` 提供基于 lane 的异步任务队列，每个 lane 有独立的并发限制。

**文件：** `src/march/agents/task_queue.py`

**默认 Lane 配置：**

| Lane | 用途 | 默认并发数 |
|------|------|-----------|
| `main` | 用户会话 | `os.cpu_count()` (auto) |
| `mt` | mtAgent 运行 | 8 |
| `mp` | mpAgent 运行 | 8 |
| `cron` | 定时任务 | 1 |

**工作原理：**

```python
# 入队并等待完成
result = await task_queue.enqueue("mt", my_async_fn)

# Fire-and-forget（不等待）
task_id = task_queue.enqueue_fire_and_forget("mp", my_async_fn)
```

内部使用 `deque` 作为等待队列，`set` 追踪活跃任务。`_drain()` 方法在每次任务完成或入队时被调用，将等待队列中的任务提升为活跃任务（直到达到 `max_concurrent`）。

**Introspection API：**

```python
task_queue.lane_stats("mt")
# → {"name": "mt", "max_concurrent": 8, "active": 3, "queued": 2}

task_queue.all_stats()
# → {"main": {...}, "mt": {...}, "mp": {...}, "cron": {...}}

task_queue.total_active   # 所有 lane 的活跃任务总数
task_queue.total_queued   # 所有 lane 的排队任务总数
```

### 4.5 MpRunner — 进程生命周期管理

`MpRunner` 管理单个 mpAgent 子进程的完整生命周期：spawn、heartbeat 监控、steering、kill、结果收集。

**文件：** `src/march/agents/mp_runner.py`

**核心保证：`wait_result()` 永远返回 `RunOutcome`，永远不会 hang。**

所有故障模式（crash、OOM、timeout、IPC 断开）都会被捕获并产生对应的 `RunOutcome`。

**内部组件：**

```
MpRunner
  ├── _recv_loop (asyncio.Task)     # 持续接收子进程 IPC 消息
  ├── _monitor (asyncio.Task)       # 心跳超时检测 + 进程存活检查
  ├── _parent_sock                  # Unix socket（非阻塞，asyncio 驱动）
  ├── _result_future                # asyncio.Future[RunOutcome]
  └── _latest_heartbeat             # 最新心跳数据
```

**Spawn 流程：**

1. 创建 Unix socketpair（`create_socket_pair()`）
2. 设置 parent socket 为非阻塞
3. 通过 `multiprocessing.get_context("spawn").Process()` 创建子进程
4. 关闭 parent 端的 child socket
5. 记录 PID 和 PGID（`pgid == pid`，因为子进程调用 `os.setpgrp()`）
6. 启动 `_recv_loop` 和 `_monitor` 两个 asyncio.Task

**Kill 流程：**

1. 尝试通过 IPC 发送 `{type: "kill"}` 消息（优雅通知）
2. 调用 `_kill_process_group(graceful=False)` → `os.killpg(pgid, SIGKILL)`
3. 设置 `RunOutcome(status="cancelled", error="Killed by parent")`

**Graceful Kill 流程：**

1. `os.killpg(pgid, SIGTERM)`
2. 等待 `kill_grace_seconds`（默认 10s）
3. 如果子进程仍然存活 → `os.killpg(pgid, SIGKILL)`
4. `process.join(timeout=5.0)` 等待回收

### 4.6 IPC 协议 — 消息类型和格式

父子进程通过 Unix socketpair 通信，使用 **msgpack 序列化 + 4 字节大端长度前缀** 的帧协议。

**文件：** `src/march/agents/ipc.py`

**帧格式：**

```
┌──────────────┬──────────────────────────┐
│ 4 bytes (BE) │ msgpack payload          │
│ payload len  │ (max 64 MB)             │
└──────────────┴──────────────────────────┘
```

**消息类型：**

| 方向 | 类型 | 常量 | 说明 |
|------|------|------|------|
| Parent → Child | steer | `MSG_STEER` | 注入 steering 消息 |
| Parent → Child | kill | `MSG_KILL` | 请求优雅退出 |
| Child → Parent | heartbeat | `MSG_HEARTBEAT` | 周期性心跳 + 资源指标 |
| Child → Parent | progress | `MSG_PROGRESS` | 工具执行进度 |
| Child → Parent | result | `MSG_RESULT` | 最终运行结果 |
| Child → Parent | log | `MSG_LOG` | 日志条目 |

**消息 Schema：**

```python
# Parent → Child: Steer
{"type": "steer", "message": "请优先处理 API 部分"}

# Parent → Child: Kill
{"type": "kill"}

# Child → Parent: Heartbeat
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
            {"name": "edit", "status": "done", "ms": 45, "summary": ""},
            {"name": "exec", "status": "done", "ms": 3200, "summary": ""}
        ]
    }
}

# Child → Parent: Progress
{
    "type": "progress",
    "tool_name": "exec",
    "status": "running",
    "summary": "Running pytest...",
    "duration_ms": 1500.0
}

# Child → Parent: Result
{
    "type": "result",
    "status": "ok",          # "ok" | "error"
    "output": "任务完成：已创建 12 个测试文件...",
    "error": "",
    "tokens": 25000,
    "cost": 0.0768
}

# Child → Parent: Log
{"type": "log", "level": "info", "message": "Starting task execution"}
```

**序列化后端：**
- 优先使用 `msgpack`（高效二进制）
- 如果 msgpack 未安装，自动降级为 JSON（带 warning 日志）

**同步 vs 异步 API：**
- 父进程（asyncio）使用 `send_message()` / `recv_message()`（async）
- 子进程的 HeartbeatThread 使用 `send_message_sync()` / `recv_message_sync()`（blocking）

**Socket 继承：**
两个 socket 都设置 `set_inheritable(True)`，确保在 `multiprocessing.Process(start_method="spawn")` 跨进程时 fd 可继承。

---

## 5. 生命周期

### 5.1 通用生命周期

```
spawn → enqueue → execute → result → announce → cleanup
  │        │         │         │         │          │
  │        │         │         │         │          └── Registry 标记 cleanup_done
  │        │         │         │         └── Announcer 推送到父 session
  │        │         │         └── Registry.complete(outcome)
  │        │         └── Agent 执行任务
  │        └── TaskQueue 排队等待 lane slot
  └── Registry.register(record)
```

### 5.2 mtAgent 生命周期

```
1. AgentManager.spawn(execution="mt")
   ├── 验证 spawn depth
   ├── 生成 run_id (UUID) 和 child_key
   ├── 创建 RunRecord → Registry.register()
   ├── 创建 steer Queue (asyncio.Queue)
   └── TaskQueue.enqueue("mt", _execute_child)

2. TaskQueue drain → asyncio.create_task()

3. _execute_child()
   ├── agent_factory(task, model, tools, child_key, parent_key)
   │   └── 创建 Child Agent → Agent.run(task)
   ├── 捕获所有异常 → RunOutcome
   │   ├── CancelledError → status="cancelled"
   │   ├── TimeoutError → status="timeout"
   │   └── Exception → status="error"
   └── 正常完成 → RunOutcome(status="ok", output=result)

4. Registry.complete(run_id, outcome)

5. 清理
   ├── 移除 _active_tasks[run_id]
   └── 移除 _steer_queues[child_key]

6. AgentAnnouncer.announce_completion(record, outcome)
   └── steer → queue → direct → pending
```

### 5.3 mpAgent 生命周期

```
1. AgentManager.spawn(execution="mp")
   ├── 验证 spawn depth
   ├── 生成 run_id 和 child_key
   ├── 创建 RunRecord → Registry.register()
   └── TaskQueue.enqueue("mp", _execute_child_mp)

2. _execute_child_mp()
   ├── 创建 MpRunner + MpConfig
   ├── runner.spawn(task, session_id, config_path, mp_config)
   │   ├── create_socket_pair()
   │   ├── multiprocessing.Process.start()
   │   │   └── 子进程: mp_child_main()
   │   │       ├── os.setpgrp()  (新进程组)
   │   │       ├── socket.fromfd(child_sock_fd)
   │   │       ├── _setup_child_logging()
   │   │       ├── load_config() + 初始化 Agent
   │   │       ├── HeartbeatThread.start()
   │   │       ├── steering_pump (asyncio.Task)
   │   │       ├── Agent.run(task, session)
   │   │       ├── HeartbeatThread.stop()
   │   │       └── IPC 发送 result 消息
   │   ├── 关闭 parent 端的 child_sock
   │   ├── 启动 _recv_loop (asyncio.Task)
   │   └── 启动 _monitor (asyncio.Task)
   │
   ├── 记录 PID 到 RunRecord
   └── runner.wait_result()  ← 保证返回

3. 父进程并行运行:
   ├── _recv_loop: 接收 heartbeat/progress/log/result
   └── _monitor: 检测心跳超时 + 进程存活

4. 子进程完成 → IPC result 消息
   └── _recv_loop 收到 → _resolve_result(outcome)

5. wait_result() 返回 RunOutcome

6. 清理
   ├── 移除 _active_runners[run_id]
   ├── Registry.complete(run_id, outcome)
   ├── 移除 _active_tasks[run_id]
   └── (可选) SessionStore 持久化子 agent 结果

7. AgentAnnouncer.announce_completion(record, outcome)
```

**子进程内部生命周期：**

```
mp_child_main(child_sock_fd, config_path, task, session_id, log_dir, hb_interval)
  │
  ├── os.setpgrp()                    # 独立进程组
  ├── socket.fromfd(child_sock_fd)    # 重建 IPC socket
  ├── os.close(child_sock_fd)         # 关闭 dup 的 fd
  │
  └── asyncio.run(_async_child_main())
        │
        ├── load_config(config_path)
        ├── 创建 LLM Router + Providers
        ├── 创建 Tool Registry + Builtin Tools + Skills
        ├── 创建 MemoryStore
        ├── 创建 Agent（无 SessionStore）
        ├── 创建 in-memory Session
        │
        ├── HeartbeatThread.start()
        │   └── 循环: 发送心跳 → 接收 steer/kill
        │
        ├── steering_pump (asyncio.Task)
        │   └── 循环: drain steer messages → Agent.steer()
        │
        ├── 包装 tools.execute 为 _tracked_execute
        │   └── 追踪: current_tool, recent_tools, tool_calls_made
        │
        ├── Agent.run(task, session)
        │   └── (正常执行或异常)
        │
        ├── steering_pump.cancel()
        ├── HeartbeatThread.stop() + join()
        │
        └── IPC 发送 {type:"result", status, output, error, tokens, cost}
```

---

## 6. 故障处理

### 6.1 故障矩阵

| 故障场景 | 影响范围 | 检测方式 | 处理策略 | 最终 RunOutcome |
|----------|----------|----------|----------|-----------------|
| **mtAgent 异常** | 仅该 task | try/except | 捕获 → outcome | `error` + 异常信息 |
| **mtAgent cancel** | 仅该 task | CancelledError | 捕获 → outcome | `cancelled` |
| **mtAgent 超时** | 仅该 task | TimeoutError | 捕获 → outcome | `timeout` |
| **mpAgent 正常退出** | 仅子进程 | IPC result 消息 | _recv_loop 处理 | `ok` + output |
| **mpAgent 异常退出** | 仅子进程 | IPC result(error) | _recv_loop 处理 | `error` + 异常信息 |
| **mpAgent OOM (SIGKILL)** | 仅子进程组 | process.is_alive()=False, exitcode=-9 | _monitor → _handle_process_exit | `error` + "killed by SIGKILL (likely OOM)" |
| **mpAgent 信号杀死** | 仅子进程组 | exitcode < 0 | _handle_process_exit | `error` + 信号名 |
| **mpAgent 心跳超时** | 仅子进程组 | _monitor 检测 elapsed > timeout | SIGTERM → grace → SIGKILL | `timeout` + 超时详情 |
| **mpAgent IPC 断开** | 仅子进程 | ConnectionError in _recv_loop | _resolve_result | `error` + "IPC connection lost" |
| **mpAgent 父进程消失** | 子进程变孤儿 | HeartbeatThread send 失败 | kill_requested → 子进程自行退出 | N/A（父已不在） |
| **mpAgent 正常退出但无 result** | 仅子进程 | exitcode=0 但 _done=False | 等 0.5s → error | `error` + "no result received" |
| **announce 投递失败** | 结果未送达 | 三级策略全部失败 | 存入 pending_queue | 下次 turn 时重试 |
| **announce 超时** | 结果延迟 | wait_for timeout | 记录 warning | 结果仍在 registry |
| **wait_result safety timeout** | 极端情况 | asyncio.wait_for | force kill → outcome | `timeout` + "safety timeout" |

### 6.2 核心保证

**父进程一定能拿到结果。** 这是通过多层防御实现的：

1. **IPC result 消息**：正常路径，子进程主动发送结果
2. **进程退出检测**：`_monitor` 定期检查 `process.is_alive()`，如果子进程已退出但没收到 result，根据 exitcode 生成 outcome
3. **心跳超时**：如果子进程既不发心跳也不退出（hang），超时后 SIGTERM → SIGKILL
4. **IPC 断开检测**：`_recv_loop` 捕获 `ConnectionError`，立即生成 error outcome
5. **Safety timeout**：`wait_result()` 有最终安全超时（`heartbeat_timeout + kill_grace + 60s`），防止所有其他机制都失败时永远 hang
6. **`_resolve_result()` 幂等**：多个路径可能同时尝试设置结果，`_done` flag 确保只有第一个生效

```python
# wait_result 的安全超时计算
safety_timeout = (
    heartbeat_timeout_seconds    # 300s (默认)
    + kill_grace_seconds         # 10s
    + 60                         # 额外缓冲
)
# = 370s，足够覆盖所有正常超时路径
```

---

## 7. 心跳与监控

### 7.1 心跳格式和内容

心跳由子进程的 `_HeartbeatThread`（daemon thread）周期性发送，包含丰富的运行时指标：

```python
{
    "type": "heartbeat",
    "ts": 1709913600.0,          # Unix 时间戳
    "data": {
        "memory_rss_mb": 256.3,           # 进程 RSS 内存 (MB)
        "elapsed_seconds": 45.2,          # 已运行时间
        "tokens_used": 12500,             # 已消耗 token 数
        "total_cost": 0.0384,             # 已消耗费用 (USD)
        "tool_calls_made": 7,             # 工具调用次数
        "llm_calls_made": 3,              # LLM 调用次数
        "summary": "Executing tool: exec",# 当前状态摘要
        "current_tool": "exec",           # 正在执行的工具
        "current_tool_detail": "{'command': 'pytest'}",  # 工具参数
        "recent_tools": [                 # 最近 3 次工具调用
            {"name": "read", "status": "done", "ms": 12, "summary": ""},
            {"name": "edit", "status": "done", "ms": 45, "summary": ""}
        ]
    }
}
```

**内存计算：**
- Linux：`resource.getrusage(RUSAGE_SELF).ru_maxrss` 返回 KB，除以 1024 得到 MB
- macOS：返回 bytes，除以 1024² 得到 MB

**工具追踪：**
子进程通过 monkey-patch `agent.tools.execute` 为 `_tracked_execute`，自动追踪每次工具调用的名称、状态、耗时，并更新 HeartbeatThread 的统计字段。

### 7.2 进度查询

父 agent 通过 `AgentManager.logs(agent_id)` 查询 mpAgent 的实时状态：

```python
lines = await manager.logs("agent-abc12345")
# 输出示例:
# Run ID: 550e8400-e29b-41d4-a716-446655440000
# Child Key: agent-abc12345:mpagent:def678901234
# Execution: mp
# Task: 重构数据库层
# Started: Sun Mar  8 16:00:00 2026
# Status: running
# Duration: 45.2s
# PID: 12345
# Log Path: /home/user/.march/logs/agent-abc12345:mpagent:def678901234
# --- Latest Heartbeat ---
#   Memory RSS: 256.3 MB
#   Elapsed: 45.2s
#   Tokens: 12500
#   Cost: $0.0384
#   Tool Calls: 7
#   Summary: Executing tool: exec
#   Current Tool: exec
#   Detail: {'command': 'pytest tests/'}
```

也可以通过 `AgentManager.list()` 获取所有 agent 的状态，包含最新心跳数据。

### 7.3 超时检测和 Kill 流程

**心跳监控循环（`_monitor`）：**

```
每 min(timeout/4, 15s) 检查一次:
  │
  ├── process.is_alive() == False?
  │   └── YES → _handle_process_exit()
  │             根据 exitcode 生成 RunOutcome
  │
  └── elapsed since last heartbeat > heartbeat_timeout_seconds?
      └── YES → 心跳超时
          ├── os.killpg(pgid, SIGTERM)
          ├── 等待 kill_grace_seconds (默认 10s)
          ├── 如果仍然存活 → os.killpg(pgid, SIGKILL)
          └── RunOutcome(status="timeout")
```

**孤儿检测（子进程侧）：**

HeartbeatThread 在每次发送心跳时检测 `BrokenPipeError`。如果父进程已消失（socket 断开），子进程设置 `kill_requested = True` 并自行退出，防止成为孤儿进程。

---

## 8. 配置参考

### 8.1 完整 config.yaml 示例

```yaml
# ~/.march/config.yaml

agents:
  max_concurrent: 4              # (未使用，由各 lane 独立控制)

  mt:
    max_concurrent: 8            # mt lane 最大并发 mtAgent 数

  mp:
    max_concurrent: 8            # mp lane 最大并发 mpAgent 数
    heartbeat_interval_seconds: 60   # 心跳发送间隔（秒）
    heartbeat_timeout_seconds: 300   # 心跳超时阈值（秒）
    kill_grace_seconds: 10           # SIGTERM → SIGKILL 等待时间（秒）
    spawn_method: spawn              # "spawn" | "forkserver"

  subagents:
    max_spawn_depth: 1           # 最大 spawn 嵌套深度
```

### 8.2 参数说明

#### MtConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_concurrent` | int | 8 | mt lane 最大并发 mtAgent 数量。超出的任务排队等待。 |

#### MpConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_concurrent` | int | 8 | mp lane 最大并发 mpAgent 数量。超出的任务排队等待。 |
| `heartbeat_interval_seconds` | int | 60 | 子进程发送心跳的间隔（秒）。较小的值提高监控精度但增加 IPC 开销。 |
| `heartbeat_timeout_seconds` | int | 300 | 无心跳超时阈值（秒）。超过此时间未收到心跳，父进程将 kill 子进程。 |
| `kill_grace_seconds` | int | 10 | 发送 SIGTERM 后等待子进程优雅退出的时间（秒）。超时后发送 SIGKILL。 |
| `spawn_method` | str | "spawn" | multiprocessing 的 start method。`"spawn"` 最安全（完全重新初始化），`"forkserver"` 启动更快但可能继承父进程状态。 |

#### SubagentsCommonConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_spawn_depth` | int | 1 | 最大 spawn 嵌套深度。防止 agent 无限递归 spawn 子 agent。值为 1 表示只允许一层子 agent。 |

#### AgentManagerConfig（代码内部）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_spawn_depth` | int | 1 | 同上，从 SubagentsCommonConfig 映射。 |
| `reset_after_complete_minutes` | int | 60 | 已完成记录的最大保留时间（分钟）。仅清理 `cleanup_done=True` 的记录。 |
| `announce_timeout_seconds` | int | 60 | announce 投递的超时时间（秒）。 |

---

## 9. Session ID 和日志

### 9.1 命名格式

**Session ID（child_key）格式：**

```
{agent_id}:{execution_suffix}:{uuid_hex[:12]}
```

- `agent_id`：由调用方指定或自动生成为 `agent-{uuid[:8]}`
- `execution_suffix`：`mtagent`（mtAgent）或 `mpagent`（mpAgent）
- `uuid_hex[:12]`：12 位十六进制随机字符串

**示例：**

```
agent-a1b2c3d4:mtagent:e5f6a7b8c9d0     # mtAgent
agent-a1b2c3d4:mpagent:e5f6a7b8c9d0     # mpAgent
refactor-db:mtagent:abc123def456          # 自定义 agent_id
```

**Run ID：** 完整 UUID v4，用于内部追踪（不暴露给用户）。

### 9.2 日志目录结构

```
~/.march/
  ├── config.yaml
  ├── AGENT.md
  └── logs/
      ├── {session_id}/                    # 每个 session 一个目录
      │   ├── 2026-03-08.log              # 按日期分割的日志文件
      │   ├── 2026-03-09.log
      │   └── heartbeats.jsonl            # mpAgent 心跳记录
      │
      ├── agent-abc:mtagent:def123/       # mtAgent 日志
      │   └── 2026-03-08.log
      │
      └── agent-xyz:mpagent:abc456/       # mpAgent 日志
          ├── 2026-03-08.log              # 子进程文件日志
          └── heartbeats.jsonl
```

**mpAgent 子进程日志：**
子进程通过 `_setup_child_logging()` 配置独立的文件日志，写入 `{log_dir}/{date}.log`。日志格式：

```
2026-03-08T16:00:00 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — mpAgent child started: pid=12345 pgid=12345 session=agent-xyz:mpagent:abc456
2026-03-08T16:00:01 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — Loading config from /home/user/.march/config.yaml
2026-03-08T16:00:02 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — Agent initialized, starting task execution
2026-03-08T16:00:45 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — Task completed: tokens=25000 cost=0.0768 tool_calls=12 duration_ms=43000
2026-03-08T16:00:45 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — Result sent via IPC: status=ok
2026-03-08T16:00:45 [INFO] march.mpchild.agent-xyz:mpagent:abc456 — mpAgent child exiting: pid=12345
```

**查询日志：**

```bash
# 查看某个 agent 的日志
cat ~/.march/logs/agent-xyz:mpagent:abc456/2026-03-08.log

# 搜索错误
grep -r "ERROR" ~/.march/logs/agent-xyz:mpagent:abc456/

# 查看心跳
jq . ~/.march/logs/agent-xyz:mpagent:abc456/heartbeats.jsonl | tail
```

---

## 10. 与其他框架的不同

### 10.1 对比 OpenClaw 的 sessions_spawn

| 维度 | March Agent System | OpenClaw sessions_spawn |
|------|-------------------|------------------------|
| **执行模式** | 双模式：mtAgent (asyncio) + mpAgent (进程) | 单一模式：sub-agent session |
| **进程隔离** | mpAgent 有完整进程级隔离（独立 process group） | 无进程隔离，共享 gateway 进程 |
| **心跳监控** | mpAgent 内置心跳 + 资源指标（RSS、tokens、cost） | 依赖外部 claw-guard 监控 PID |
| **IPC** | 内置 Unix socketpair + msgpack 协议 | 无内置 IPC，通过 session 消息传递 |
| **Steering** | 内置 steer 支持（mt: Queue, mp: IPC） | 通过 steer API 注入消息 |
| **结果推送** | 三级降级策略（steer → queue → direct） | 自动 announce 到 requester session |
| **故障恢复** | 多层防御保证 `wait_result()` 必返回 | 依赖 claw-guard 通知 + 手动检查 |
| **并发控制** | Lane-based TaskQueue（独立并发限制） | 无内置并发限制 |

### 10.2 对比 LangGraph 的 Agent 模型

| 维度 | March Agent System | LangGraph |
|------|-------------------|-----------|
| **执行模型** | 独立 agent 实例（mt 或 mp） | Graph node 执行 |
| **隔离** | 进程级隔离（mpAgent） | 无隔离，同一进程内 |
| **通信** | IPC 协议（msgpack over Unix socket） | Graph state 传递 |
| **监控** | 实时心跳 + 资源指标 | 无内置监控 |
| **故障处理** | 多层防御 + 保证结果交付 | 依赖 Graph 的 error handling |
| **Steering** | 运行时动态注入消息 | 无运行时 steering |
| **并发** | Lane-based 并发控制 | 依赖外部编排 |

### 10.3 March 的独特之处

**1. 进程级隔离（Process-Level Isolation）**

mpAgent 通过 `os.setpgrp()` 创建独立进程组，`os.killpg()` 可以一次性杀死子进程及其所有子孙进程。OOM killer 只影响子进程组，主进程安全。

**2. 心跳监控（Heartbeat Monitoring）**

不是简单的"进程还活着吗"检查，而是包含丰富运行时指标的结构化心跳：内存使用、token 消耗、费用、工具调用历史、当前执行状态。父进程可以实时了解子 agent 在做什么。

**3. IPC Steering**

父 agent 可以在子 agent 运行过程中注入新的指令（steering message），改变子 agent 的行为方向。这通过 HeartbeatThread 接收 steer 消息，再通过 steering pump 注入到 Agent 的 steering queue 实现。

**4. 保证结果交付（Guaranteed Result Delivery）**

`MpRunner.wait_result()` 通过五层防御机制保证**永远返回** `RunOutcome`：
- IPC result 消息（正常路径）
- 进程退出检测（异常退出）
- 心跳超时（hang 检测）
- IPC 断开检测（连接丢失）
- Safety timeout（最终兜底）

**5. 孤儿保护（Orphan Protection）**

如果父进程意外消失，子进程的 HeartbeatThread 会在下一次心跳发送时检测到 `BrokenPipeError`，自动设置 `kill_requested` 并退出，防止成为僵尸进程。

**6. Lane-Based 并发控制**

四条独立 lane（main、mt、mp、cron）各有独立的并发限制和排队机制，互不干扰。mpAgent 不会抢占 mtAgent 的 slot，反之亦然。

---

## 附录：快速参考

### Spawn 一个 mtAgent

```python
result = await manager.spawn(
    SpawnParams(
        task="搜索最新的 Python 3.13 变更日志并总结",
        execution="mt",
        label="changelog-search",
    ),
    SpawnContext(
        requester_session="main-session-key",
        origin="terminal",
    ),
)
# result.child_key = "agent-a1b2c3d4:mtagent:e5f6a7b8c9d0"
```

### Spawn 一个 mpAgent

```python
result = await manager.spawn(
    SpawnParams(
        task="处理 500MB 的 CSV 数据集，生成统计报告",
        execution="mp",
        model="litellm/claude-sonnet-4-20250514",
        label="data-processing",
    ),
    SpawnContext(
        requester_session="main-session-key",
        origin="terminal",
    ),
)
# result.child_key = "agent-a1b2c3d4:mpagent:e5f6a7b8c9d0"
```

### Steer 一个运行中的 agent

```python
await manager.send("agent-a1b2c3d4:mpagent:e5f6a7b8c9d0", "优先处理 revenue 列的异常值")
```

### Kill 一个 agent

```python
await manager.kill("agent-a1b2c3d4:mpagent:e5f6a7b8c9d0")
```

### 查看所有 agent 状态

```python
statuses = await manager.list()
for s in statuses:
    print(f"{s.child_key}: {s.status} ({s.duration_seconds:.1f}s)")
    if s.heartbeat:
        print(f"  RSS: {s.heartbeat['memory_rss_mb']}MB, Tokens: {s.heartbeat['tokens_used']}")
```
