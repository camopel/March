"""Shell execution via asyncio subprocess with pty, timeout, and background mode."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.exec_tool")

_MAX_OUTPUT = 100_000  # 100KB output cap


@dataclass
class ExecSession:
    """A background exec session."""

    id: str
    command: str
    process: asyncio.subprocess.Process
    start_time: float
    output: list[str] = field(default_factory=list)
    master_fd: int | None = None
    _finished: bool = False

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def finished(self) -> bool:
        return self._finished or self.process.returncode is not None

    def get_output(self) -> str:
        return "".join(self.output)


# Global session registry for background processes
_sessions: dict[str, ExecSession] = {}


def get_sessions() -> dict[str, ExecSession]:
    """Get the background session registry."""
    return _sessions


@tool(name="exec", description="Execute a shell command. Supports timeout, background mode, and working directory.")
async def exec_tool(
    command: str,
    workdir: str = "",
    timeout: int = 30,
    background: bool = False,
    env: dict = None,
) -> str:
    """Execute a shell command.

    Args:
        command: Shell command to execute.
        workdir: Working directory (defaults to cwd).
        timeout: Timeout in seconds (0 for no timeout, default 30).
        background: Run in background and return session ID immediately.
        env: Additional environment variables.
    """
    cwd = workdir or None
    if cwd:
        from pathlib import Path
        cwd_path = Path(cwd).expanduser().resolve()
        if not cwd_path.is_dir():
            return f"Error: Working directory not found: {workdir}"
        cwd = str(cwd_path)

    # Merge environment
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=proc_env,
        )
    except Exception as e:
        return f"Error starting command: {e}"

    if background:
        session_id = uuid.uuid4().hex[:12]
        session = ExecSession(
            id=session_id,
            command=command,
            process=proc,
            start_time=time.monotonic(),
        )
        _sessions[session_id] = session

        # Start output collector in background
        asyncio.create_task(_collect_output(session))

        return (
            f"Background session started: {session_id}\n"
            f"PID: {proc.pid}\n"
            f"Use process tool to manage."
        )

    # Foreground execution with timeout
    try:
        t = timeout if timeout > 0 else None
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=t)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            stdout = b""
        output = stdout.decode(errors="replace") if stdout else ""
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n[output truncated]"
        return f"[Timed out after {timeout}s, process killed]\n{output}"
    except Exception as e:
        return f"Error during execution: {e}"

    output = stdout.decode(errors="replace") if stdout else ""
    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + "\n[output truncated]"

    rc = proc.returncode
    if rc != 0:
        return f"[Exit code: {rc}]\n{output}"
    return output if output else "(no output)"


async def _collect_output(session: ExecSession) -> None:
    """Collect output from a background process."""
    try:
        while True:
            if session.process.stdout is None:
                break
            data = await session.process.stdout.read(8192)
            if not data:
                break
            session.output.append(data.decode(errors="replace"))
    except Exception as e:
        session.output.append(f"\n[Output collection error: {e}]")
    finally:
        session._finished = True
        await session.process.wait()
