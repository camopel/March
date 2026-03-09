"""mpAgent child process entry point.

Runs inside a spawned ``multiprocessing.Process``. Responsibilities:

1. Create an independent process group (``os.setpgrp()``) so the parent can
   ``os.killpg()`` the entire tree.
2. Initialize an ``Agent`` from the config file — **without** a ``SessionStore``
   (the child never persists sessions; results go back via IPC).
3. Run the task via ``Agent.run(task, session)`` with a pure in-memory ``Session``.
4. Heartbeat thread: periodically sends resource/progress metrics to the parent.
5. Steering receiver thread: listens for ``steer`` / ``kill`` messages from the parent.
6. Report the final result (success or failure) via IPC.
7. Orphan protection: if an IPC write fails (parent gone), exit immediately.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from march.agents.ipc import (
    MSG_HEARTBEAT,
    MSG_KILL,
    MSG_LOG,
    MSG_RESULT,
    MSG_STEER,
    MSG_SPAWN_REQUEST,
    MSG_SPAWN_RESULT,
    MSG_SPAWN_STEER,
    MSG_SPAWN_KILL,
    MSG_CHILD_COMPLETED,
    recv_message_sync,
    send_message_sync,
)


# ── Logging setup (child-local, file-based) ─────────────────────────


def _setup_child_logging(log_dir: str, session_id: str) -> logging.Logger:
    """Configure file-based logging for the child process.

    Logs are written to ``{log_dir}/{date}.log``.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_path / f"{date_str}.log"

    logger = logging.getLogger(f"march.mpchild.{session_id}")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    # Also capture root-level march logs to the same file
    root_march = logging.getLogger("march")
    root_march.addHandler(handler)

    return logger


# ── IPC helpers with orphan protection ───────────────────────────────


def _safe_send(sock: socket.socket, msg: dict, logger: logging.Logger | None = None) -> bool:
    """Send a message via IPC. Returns False if the pipe is broken (orphan)."""
    try:
        send_message_sync(sock, msg)
        return True
    except (BrokenPipeError, ConnectionError, OSError) as exc:
        if logger:
            logger.error("IPC send failed (parent gone?): %s", exc)
        return False


# ── Spawn proxy (delegates spawn to parent via IPC) ──────────────────


class _SpawnProxy:
    """Proxy spawn requests to the parent process via IPC.

    The child process has no AgentManager. When it needs to spawn a
    grandchild agent, it sends a request to the parent via IPC and
    waits for the result. The parent's AgentManager handles the actual
    spawn and notifies the child when the grandchild completes.
    """

    def __init__(
        self,
        sock: socket.socket,
        session_id: str,
        execution: str,
        logger: logging.Logger,
    ) -> None:
        self._sock = sock
        self._session_id = session_id
        self._execution = execution
        self._logger = logger
        # request_id → Future[tuple[str, str, str, str]]  (status, child_key, run_id, error)
        self._pending_spawns: dict[str, asyncio.Future[tuple[str, str, str, str]]] = {}
        # child_key → Future[tuple[str, str, str]]  (status, output, error)
        self._child_results: dict[str, asyncio.Future[tuple[str, str, str]]] = {}
        self._lock = threading.Lock()

    async def spawn(
        self, task: str, agent_id: str = "", model: str = "", timeout: int = 0
    ) -> tuple[str, str]:
        """Request the parent to spawn a grandchild agent.

        Args:
            task: Task description for the grandchild.
            agent_id: Optional agent ID (empty = auto-generate).
            model: Optional model override.
            timeout: Optional timeout in seconds.

        Returns:
            (child_key, run_id) of the spawned grandchild.

        Raises:
            RuntimeError: If the spawn request was rejected or failed.
        """
        import uuid

        request_id = uuid.uuid4().hex[:16]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, str, str, str]] = loop.create_future()

        with self._lock:
            self._pending_spawns[request_id] = future

        msg = {
            "type": MSG_SPAWN_REQUEST,
            "task": task,
            "agent_id": agent_id,
            "model": model,
            "timeout": timeout,
            "request_id": request_id,
        }

        if not _safe_send(self._sock, msg, self._logger):
            with self._lock:
                self._pending_spawns.pop(request_id, None)
            raise RuntimeError("IPC send failed — parent may be gone")

        status, child_key, run_id, error = await future

        if status != "accepted":
            raise RuntimeError(f"Spawn rejected: {error}")

        return child_key, run_id

    async def wait_child(self, child_key: str) -> tuple[str, str]:
        """Wait for a grandchild agent to complete.

        Args:
            child_key: Session key of the grandchild to wait for.

        Returns:
            (status, output) of the completed grandchild.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, str, str]] = loop.create_future()

        with self._lock:
            self._child_results[child_key] = future

        status, output, error = await future

        if status not in ("ok",):
            if error:
                return status, f"Error: {error}"
        return status, output

    async def steer_child(self, child_key: str, message: str) -> None:
        """Send a steer message to a grandchild via the parent.

        Args:
            child_key: Session key of the grandchild.
            message: Steering message text.
        """
        msg = {
            "type": MSG_SPAWN_STEER,
            "child_key": child_key,
            "message": message,
        }
        if not _safe_send(self._sock, msg, self._logger):
            raise RuntimeError("IPC send failed — parent may be gone")

    async def kill_child(self, child_key: str) -> None:
        """Request the parent to kill a grandchild.

        Args:
            child_key: Session key of the grandchild to kill.
        """
        msg = {
            "type": MSG_SPAWN_KILL,
            "child_key": child_key,
        }
        if not _safe_send(self._sock, msg, self._logger):
            raise RuntimeError("IPC send failed — parent may be gone")

    def handle_spawn_result(self, msg: dict) -> None:
        """Handle a spawn_result message from the parent.

        Called from the heartbeat thread when a MSG_SPAWN_RESULT is received.
        Resolves the corresponding pending Future.
        """
        request_id = msg.get("request_id", "")
        with self._lock:
            future = self._pending_spawns.pop(request_id, None)

        if future is None:
            self._logger.warning("spawn_result for unknown request_id: %s", request_id)
            return

        result = (
            msg.get("status", "error"),
            msg.get("child_key", ""),
            msg.get("run_id", ""),
            msg.get("error", ""),
        )

        # Resolve from the event loop thread (future belongs to asyncio)
        try:
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, result)
        except Exception as exc:
            self._logger.error("Failed to resolve spawn_result future: %s", exc)

    def handle_child_completed(self, msg: dict) -> None:
        """Handle a child_completed message from the parent.

        Called from the heartbeat thread when a MSG_CHILD_COMPLETED is received.
        Resolves the corresponding wait_child Future.
        """
        child_key = msg.get("child_key", "")
        with self._lock:
            future = self._child_results.pop(child_key, None)

        if future is None:
            self._logger.warning("child_completed for unknown child_key: %s", child_key)
            return

        result = (
            msg.get("status", "error"),
            msg.get("output", ""),
            msg.get("error", ""),
        )

        try:
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, result)
        except Exception as exc:
            self._logger.error("Failed to resolve child_completed future: %s", exc)


# ── Heartbeat thread ─────────────────────────────────────────────────


class _HeartbeatThread(threading.Thread):
    """Sends periodic heartbeat messages to the parent via IPC.

    Also listens for incoming ``steer`` / ``kill`` messages from the parent
    using a short recv timeout between heartbeats.
    """

    def __init__(
        self,
        sock: socket.socket,
        interval_seconds: float,
        session_id: str,
        start_time: float,
        logger: logging.Logger,
        spawn_proxy: _SpawnProxy | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"hb-{session_id[:12]}")
        self._sock = sock
        self._interval = interval_seconds
        self._session_id = session_id
        self._start_time = start_time
        self._logger = logger
        self._stop_event = threading.Event()
        self._spawn_proxy = spawn_proxy

        # Mutable stats updated by the agent loop (via thread-safe attrs)
        self.tokens_used: int = 0
        self.total_cost: float = 0.0
        self.tool_calls_made: int = 0
        self.llm_calls_made: int = 0
        self.summary: str = ""
        self.current_tool: str = ""
        self.current_tool_detail: str = ""
        self.recent_tools: list[dict[str, Any]] = []

        # Steering messages received from parent, consumed by the agent loop
        self._steer_messages: list[str] = []
        self._steer_lock = threading.Lock()

        # Kill flag
        self.kill_requested = False

    def stop(self) -> None:
        """Signal the heartbeat thread to stop."""
        self._stop_event.set()

    def drain_steer_messages(self) -> list[str]:
        """Drain and return all pending steering messages (thread-safe)."""
        with self._steer_lock:
            msgs = list(self._steer_messages)
            self._steer_messages.clear()
            return msgs

    def run(self) -> None:
        """Thread main: send heartbeats, receive steering messages."""
        while not self._stop_event.is_set():
            # Send heartbeat
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux ru_maxrss is in KB; on macOS it's in bytes
            rss_mb = rss_bytes / 1024 if sys.platform == "linux" else rss_bytes / (1024 * 1024)

            hb_msg = {
                "type": MSG_HEARTBEAT,
                "ts": time.time(),
                "data": {
                    "memory_rss_mb": round(rss_mb, 1),
                    "elapsed_seconds": round(time.time() - self._start_time, 1),
                    "tokens_used": self.tokens_used,
                    "total_cost": self.total_cost,
                    "tool_calls_made": self.tool_calls_made,
                    "llm_calls_made": self.llm_calls_made,
                    "summary": self.summary,
                    "current_tool": self.current_tool,
                    "current_tool_detail": self.current_tool_detail,
                    "recent_tools": list(self.recent_tools),
                },
            }

            if not _safe_send(self._sock, hb_msg, self._logger):
                # Parent is gone — trigger orphan exit
                self._logger.warning("Orphan detected (heartbeat send failed), requesting exit")
                self.kill_requested = True
                self._stop_event.set()
                return

            # Listen for incoming messages for a short window, then loop
            # Split the interval into small recv windows to stay responsive
            deadline = time.monotonic() + self._interval
            while time.monotonic() < deadline and not self._stop_event.is_set():
                remaining = max(0.1, deadline - time.monotonic())
                timeout = min(remaining, 1.0)  # Check at most every 1s
                try:
                    incoming = recv_message_sync(self._sock, timeout=timeout)
                except (ConnectionError, OSError):
                    self._logger.warning("IPC recv failed in heartbeat thread, orphan exit")
                    self.kill_requested = True
                    self._stop_event.set()
                    return

                if incoming is None:
                    continue  # Timeout, loop again

                msg_type = incoming.get("type")
                if msg_type == MSG_STEER:
                    steer_text = incoming.get("message", "")
                    self._logger.info("Received steer message: %s", steer_text[:100])
                    with self._steer_lock:
                        self._steer_messages.append(steer_text)
                elif msg_type == MSG_KILL:
                    self._logger.info("Received kill request from parent")
                    self.kill_requested = True
                    self._stop_event.set()
                    return
                elif msg_type == MSG_SPAWN_RESULT:
                    if self._spawn_proxy:
                        self._spawn_proxy.handle_spawn_result(incoming)
                    else:
                        self._logger.warning("Received spawn_result but no spawn_proxy")
                elif msg_type == MSG_CHILD_COMPLETED:
                    if self._spawn_proxy:
                        self._spawn_proxy.handle_child_completed(incoming)
                    else:
                        self._logger.warning("Received child_completed but no spawn_proxy")
                else:
                    self._logger.warning("Unknown IPC message type: %s", msg_type)


# ── Main child entry point ───────────────────────────────────────────


def mp_child_main(
    child_sock_fd: int,
    config_path: str,
    task: str,
    session_id: str,
    log_dir: str,
    heartbeat_interval_seconds: float = 60.0,
) -> None:
    """Entry point for the mpAgent child process.

    Called as the ``target`` of ``multiprocessing.Process(start_method="spawn")``.
    All arguments must be picklable primitives.

    Args:
        child_sock_fd: File descriptor of the child end of the IPC socketpair.
        config_path: Path to the March config YAML file.
        task: The task string to execute.
        session_id: Session ID for this mpAgent run.
        log_dir: Directory for log files (``{log_dir}/{date}.log``).
        heartbeat_interval_seconds: Seconds between heartbeats.
    """
    # ── 1. Process isolation ──
    os.setpgrp()  # New process group for killpg cleanup

    # Reconstruct the socket from the file descriptor
    child_sock = socket.fromfd(child_sock_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    # Close the duplicated fd (fromfd dups it)
    os.close(child_sock_fd)

    # ── 2. Logging ──
    logger = _setup_child_logging(log_dir, session_id)
    logger.info(
        "mpAgent child started: pid=%d pgid=%d session=%s",
        os.getpid(),
        os.getpgrp(),
        session_id,
    )

    start_time = time.time()

    # ── 3. Run the async main ──
    try:
        asyncio.run(_async_child_main(
            child_sock=child_sock,
            config_path=config_path,
            task=task,
            session_id=session_id,
            log_dir=log_dir,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            logger=logger,
            start_time=start_time,
        ))
    except Exception as exc:
        logger.error("Fatal error in mpAgent child: %s", exc, exc_info=True)
        # Try to report via IPC
        _safe_send(child_sock, {
            "type": MSG_RESULT,
            "status": "error",
            "output": "",
            "error": f"Fatal child error: {exc}",
            "tokens": 0,
            "cost": 0.0,
        }, logger)
    finally:
        try:
            child_sock.close()
        except OSError:
            pass
        logger.info("mpAgent child exiting: pid=%d", os.getpid())


async def _async_child_main(
    child_sock: socket.socket,
    config_path: str,
    task: str,
    session_id: str,
    log_dir: str,
    heartbeat_interval_seconds: float,
    logger: logging.Logger,
    start_time: float,
) -> None:
    """Async portion of the child process: initialize Agent, run task, report result."""

    # ── 1. Load config and initialize components ──
    from march.config.loader import load_config
    from march.core.agent import Agent
    from march.core.session import Session
    from march.llm.router import LLMRouter, RouterConfig
    from march.memory.store import MemoryStore
    from march.plugins._manager import PluginManager
    from march.tools.registry import ToolRegistry

    logger.info("Loading config from %s", config_path)
    config = load_config(Path(config_path), use_cache=False)

    # LLM Router
    router_config = RouterConfig(
        fallback_chain=list(config.llm.fallback_chain),
        default_provider=config.llm.default,
    )
    llm_router = LLMRouter(config=router_config, providers={})

    # Create LLM providers from config (same logic as MarchApp._create_providers)
    _create_providers_for_child(llm_router, config, logger)

    # Tool registry
    tool_registry = ToolRegistry()

    # Register builtin tools
    from march.tools.builtin import register_all_builtin_tools
    register_all_builtin_tools(tool_registry)

    # Load skills
    from march.tools.skills.loader import SkillLoader
    skills_dir = Path.cwd() / "skills"
    skill_loader = SkillLoader()
    if skills_dir.is_dir():
        skill_loader.load_directory(skills_dir, registry=tool_registry)

    # Plugin manager (empty — mpAgent runs without plugins for isolation)
    plugin_manager = PluginManager()

    # Memory store
    memory_store = MemoryStore(
        workspace=Path.cwd(),
        config_dir=Path.home() / ".march",
        system_rules_path=config.memory.system_rules,
        agent_profile_path=config.memory.agent_profile,
        tool_rules_path=config.memory.tool_rules,
        memory_path=config.memory.memory_path,
    )
    await memory_store.initialize()

    # Build Agent — NO SessionStore
    agent = Agent(
        llm_router=llm_router,
        tool_registry=tool_registry,
        plugin_manager=plugin_manager,
        memory_store=memory_store,
        config=config,
    )

    # Create in-memory Session (not persisted)
    session = Session(
        id=session_id,
        source_type="mpagent",
        name=f"mpAgent {session_id[:12]}",
    )

    logger.info("Agent initialized, starting task execution")

    # ── 2. Create spawn proxy for grandchild delegation ──
    spawn_proxy = _SpawnProxy(
        sock=child_sock,
        session_id=session_id,
        execution="mp",
        logger=logger,
    )

    # ── 3. Start heartbeat thread ──
    hb_thread = _HeartbeatThread(
        sock=child_sock,
        interval_seconds=heartbeat_interval_seconds,
        session_id=session_id,
        start_time=start_time,
        logger=logger,
        spawn_proxy=spawn_proxy,
    )
    hb_thread.start()

    # ── 4. Register spawn_agent tool for grandchild delegation ──
    from march.tools.base import Tool as _Tool

    async def _spawn_agent_tool(
        task: str,
        agent_id: str = "",
        model: str = "",
        timeout: int = 0,
        wait: bool = True,
    ) -> str:
        """Spawn a child agent via the parent process.

        Args:
            task: Task description for the child agent.
            agent_id: Optional agent ID (empty = auto-generate).
            model: Optional model override.
            timeout: Optional timeout in seconds.
            wait: If True, wait for the child to complete and return its output.

        Returns:
            If wait=True: the child's output.
            If wait=False: a JSON string with child_key and run_id.
        """
        import json as _json

        child_key, run_id = await spawn_proxy.spawn(task, agent_id, model, timeout)

        if not wait:
            return _json.dumps({"child_key": child_key, "run_id": run_id})

        status, output = await spawn_proxy.wait_child(child_key)
        return output

    tool_registry.register(_Tool(
        name="spawn_agent",
        description="Spawn a child agent to handle a subtask. The child runs in an isolated process.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description for the child agent"},
                "agent_id": {"type": "string", "description": "Optional agent ID", "default": ""},
                "model": {"type": "string", "description": "Optional model override", "default": ""},
                "timeout": {"type": "integer", "description": "Timeout in seconds (0 = no timeout)", "default": 0},
                "wait": {"type": "boolean", "description": "Wait for completion", "default": True},
            },
            "required": ["task"],
        },
        fn=_spawn_agent_tool,
        source="builtin",
    ))

    # ── 5. Hook into agent metrics for heartbeat reporting ──
    # We wrap the agent's tool execution to track progress
    _original_tools_execute = agent.tools.execute

    _recent_tools_buf: list[dict[str, Any]] = []

    async def _tracked_execute(tool_call: Any) -> Any:
        hb_thread.current_tool = tool_call.name
        hb_thread.current_tool_detail = str(tool_call.args)[:200]
        hb_thread.summary = f"Executing tool: {tool_call.name}"

        t0 = time.monotonic()
        try:
            result = await _original_tools_execute(tool_call)
            dur_ms = (time.monotonic() - t0) * 1000
            entry = {
                "name": tool_call.name,
                "status": "done" if not result.is_error else "error",
                "ms": round(dur_ms),
                "summary": (result.summary[:100] if hasattr(result, "summary") else ""),
            }
        except Exception as exc:
            dur_ms = (time.monotonic() - t0) * 1000
            entry = {
                "name": tool_call.name,
                "status": "error",
                "ms": round(dur_ms),
                "summary": str(exc)[:100],
            }
            raise
        finally:
            _recent_tools_buf.append(entry)
            # Keep only last 3
            while len(_recent_tools_buf) > 3:
                _recent_tools_buf.pop(0)
            hb_thread.recent_tools = list(_recent_tools_buf)
            hb_thread.tool_calls_made += 1
            hb_thread.current_tool = ""
            hb_thread.current_tool_detail = ""

        return result

    agent.tools.execute = _tracked_execute  # type: ignore[assignment]

    # ── 6. Inject steering messages into the agent ──
    # The heartbeat thread collects steer messages; we inject them into
    # the agent's steering queue before each LLM call via a plugin-like hook.
    # Since we're using the standard Agent.run(), we feed steering via
    # the agent's existing steering API.
    async def _steering_pump() -> None:
        """Periodically drain steer messages from the heartbeat thread
        and inject them into the agent's steering queue."""
        while not hb_thread.kill_requested and hb_thread.is_alive():
            steer_msgs = hb_thread.drain_steer_messages()
            for msg in steer_msgs:
                agent.steer(session_id, msg)
            await asyncio.sleep(0.5)

    steering_task = asyncio.create_task(_steering_pump())

    # ── 7. Run the task ──
    result_status = "ok"
    result_output = ""
    result_error = ""
    result_tokens = 0
    result_cost = 0.0

    try:
        # Check for kill before starting
        if hb_thread.kill_requested:
            result_status = "error"
            result_error = "Kill requested before task started"
        else:
            response = await agent.run(task, session)
            result_output = response.content
            result_tokens = response.total_tokens
            result_cost = response.total_cost
            hb_thread.tokens_used = result_tokens
            hb_thread.total_cost = result_cost
            hb_thread.summary = "Task completed"
            logger.info(
                "Task completed: tokens=%d cost=%.4f tool_calls=%d duration_ms=%.0f",
                response.total_tokens,
                response.total_cost,
                response.tool_calls_made,
                response.duration_ms,
            )
    except asyncio.CancelledError:
        result_status = "error"
        result_error = "Task was cancelled"
        logger.warning("Task cancelled")
    except Exception as exc:
        result_status = "error"
        result_error = str(exc)
        logger.error("Task failed: %s", exc, exc_info=True)

    # ── 8. Stop steering pump ──
    steering_task.cancel()
    try:
        await steering_task
    except asyncio.CancelledError:
        pass

    # ── 9. Stop heartbeat thread ──
    hb_thread.stop()
    hb_thread.join(timeout=5.0)

    # ── 10. Send result via IPC ──
    result_msg = {
        "type": MSG_RESULT,
        "status": result_status,
        "output": result_output,
        "error": result_error,
        "tokens": result_tokens,
        "cost": result_cost,
    }

    if not _safe_send(child_sock, result_msg, logger):
        logger.error("Failed to send result via IPC — parent may be gone")
        sys.exit(1)

    logger.info("Result sent via IPC: status=%s", result_status)


def _create_providers_for_child(
    llm_router: Any,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Create LLM providers from config and register them with the router.

    Mirrors ``MarchApp._create_providers()`` but runs in the child process.
    """
    import importlib

    from march.llm.router import ProviderHealth

    provider_map = {
        "litellm": "march.llm.litellm_provider.LiteLLMProvider",
        "ollama": "march.llm.ollama.OllamaProvider",
        "openai": "march.llm.openai_provider.OpenAIProvider",
        "anthropic": "march.llm.anthropic_provider.AnthropicProvider",
        "bedrock": "march.llm.bedrock.BedrockProvider",
        "openrouter": "march.llm.openrouter.OpenRouterProvider",
    }

    for name, pcfg in config.llm.providers.items():
        provider_cls_path = provider_map.get(name)
        if not provider_cls_path:
            logger.warning("Unknown LLM provider: %s — skipping", name)
            continue

        try:
            module_path, cls_name = provider_cls_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)

            if name == "bedrock":
                kwargs: dict[str, Any] = {
                    "model": pcfg.model,
                    "region": pcfg.region or "us-west-2",
                    "max_tokens": pcfg.max_tokens,
                    "temperature": pcfg.temperature,
                    "timeout": float(pcfg.timeout),
                    "input_price": pcfg.cost.input,
                    "output_price": pcfg.cost.output,
                }
                if pcfg.profile:
                    kwargs["profile"] = pcfg.profile
            else:
                kwargs = {
                    "model": pcfg.model,
                    "url": pcfg.url,
                    "max_tokens": pcfg.max_tokens,
                    "temperature": pcfg.temperature,
                    "timeout": float(pcfg.timeout),
                    "input_price": pcfg.cost.input,
                    "output_price": pcfg.cost.output,
                }
                if pcfg.api_key:
                    kwargs["api_key"] = pcfg.api_key

            provider = cls(**kwargs)
            llm_router.providers[name] = provider
            llm_router._health[name] = ProviderHealth(name=name)
            logger.info("Registered LLM provider: %s (model=%s)", name, pcfg.model)
        except Exception as exc:
            logger.warning("Failed to create LLM provider %s: %s", name, exc)
