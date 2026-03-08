"""March Agent Manager — agent orchestration (mt + mp), task queue, and registry."""

from march.agents.manager import AgentManager, AgentManagerConfig, SpawnParams, SpawnContext, SpawnResult
from march.agents.task_queue import TaskQueue
from march.agents.registry import AgentRegistry, RunRecord, RunOutcome
from march.agents.announce import AgentAnnouncer

# Backward-compat aliases
SubagentRegistry = AgentRegistry
SubagentAnnouncer = AgentAnnouncer

__all__ = [
    "AgentManager",
    "AgentManagerConfig",
    "SpawnParams",
    "SpawnContext",
    "SpawnResult",
    "TaskQueue",
    "AgentRegistry",
    "SubagentRegistry",
    "RunRecord",
    "RunOutcome",
    "AgentAnnouncer",
    "SubagentAnnouncer",
]
