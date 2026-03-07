"""Channel base class for the March agent framework.

Defines the abstract interface that all communication channels must implement.
Channels are how users interact with the agent (terminal, WebSocket, Matrix, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.core.session import Session
    from march.llm.base import StreamChunk


class Channel(ABC):
    """Abstract interface for all communication channels.

    Each channel handles:
    - Receiving user input
    - Sending agent responses (streaming and non-streaming)
    - Session management
    - Lifecycle (start/stop)
    """

    name: str = "base"

    @abstractmethod
    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the channel and begin accepting user input.

        This method should run the channel's main loop (e.g., read input,
        accept connections, etc.) and only return when the channel is stopped.

        Args:
            agent: The agent instance to route messages to.
            **kwargs: Channel-specific configuration.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        ...

    @abstractmethod
    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a complete (non-streaming) response to the user.

        Args:
            content: The text content to send.
            **kwargs: Channel-specific options.
        """
        ...

    @abstractmethod
    async def send_stream(
        self, chunks: AsyncIterator["StreamChunk"], **kwargs: Any
    ) -> None:
        """Send a streaming response to the user.

        Iterates over stream chunks and delivers them in real-time.

        Args:
            chunks: Async iterator of StreamChunk objects.
            **kwargs: Channel-specific options.
        """
        ...
