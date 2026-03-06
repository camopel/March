"""March logging — structlog-based structured logging with subsystem tags.

get_logger() returns a structlog lazy proxy that supports kwargs:
    logger.info("event.name", key=value, another=123)

Call configure_logging() during app init for full setup (file + console).
Before configuration, logs go to stderr with basic formatting.

Subsystems: agent, llm, tools, ws_proxy, stream, compaction, metrics, system
"""

from __future__ import annotations

from typing import Any

import structlog


def get_logger(name: str = "march", subsystem: str = "system", **initial_context: Any) -> Any:
    """Get a structlog logger with a subsystem tag and optional initial context.

    Args:
        name: Logger name (for stdlib routing).
        subsystem: Subsystem tag (agent, llm, tools, ws_proxy, stream, compaction, metrics).
        **initial_context: Key-value pairs bound to every log entry.

    Returns:
        A bound structlog logger with the subsystem pre-bound.
    """
    return structlog.get_logger(name, subsystem=subsystem, **initial_context)
