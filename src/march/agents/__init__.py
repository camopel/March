"""March Agent Manager — sub-agent orchestration, task queue, and registry."""

from march.agents.manager import AgentManager, AgentManagerConfig, SpawnParams, SpawnContext, SpawnResult
from march.agents.task_queue import TaskQueue
from march.agents.registry import SubagentRegistry, RunRecord, RunOutcome
from march.agents.announce import SubagentAnnouncer
from march.agents.protocol import IPCMessage, IPCReader, IPCWriter, MessageType
from march.agents.guardian import Guardian, GuardianConfig

__all__ = [
    "AgentManager",
    "AgentManagerConfig",
    "SpawnParams",
    "SpawnContext",
    "SpawnResult",
    "TaskQueue",
    "SubagentRegistry",
    "RunRecord",
    "RunOutcome",
    "SubagentAnnouncer",
    "IPCMessage",
    "IPCReader",
    "IPCWriter",
    "MessageType",
    "Guardian",
    "GuardianConfig",
]
