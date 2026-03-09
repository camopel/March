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
    ) -> None:
        super().__init__(daemon=True, name=f"hb-{session_id[:12]}")
        self._sock = sock
        self._interval = interval_seconds
        self._session_id = session_id
        self._start_time = start_time
        self._logger = logger
        self._stop_event = threading.Event()

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

    # ── 2. Start heartbeat thread ──
    hb_thread = _HeartbeatThread(
        sock=child_sock,
        interval_seconds=heartbeat_interval_seconds,
        session_id=session_id,
        start_time=start_time,
        logger=logger,
    )
    hb_thread.start()

    # ── 3. Hook into agent metrics for heartbeat reporting ──
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

    # ── 4. Inject steering messages into the agent ──
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

    # ── 5. Run the task ──
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

    # ── 6. Stop steering pump ──
    steering_task.cancel()
    try:
        await steering_task
    except asyncio.CancelledError:
        pass

    # ── 7. Stop heartbeat thread ──
    hb_thread.stop()
    hb_thread.join(timeout=5.0)

    # ── 8. Send result via IPC ──
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
