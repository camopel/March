"""Tests for agent steering mechanism."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from march.core.agent import Agent
from march.core.session import Session


class TestSteeringQueue:
    def setup_method(self):
        self.agent = Agent(
            llm_router=MagicMock(),
            tool_registry=MagicMock(),
            plugin_manager=MagicMock(),
            memory_store=MagicMock(),
        )

    def test_get_steering_queue_creates_new(self):
        q = self.agent.get_steering_queue("session-1")
        assert isinstance(q, asyncio.Queue)

    def test_get_steering_queue_returns_same(self):
        q1 = self.agent.get_steering_queue("session-1")
        q2 = self.agent.get_steering_queue("session-1")
        assert q1 is q2

    def test_steer_returns_false_no_queue(self):
        assert self.agent.steer("nonexistent", "hello") is False

    def test_steer_returns_true_with_queue(self):
        self.agent.get_steering_queue("session-1")
        assert self.agent.steer("session-1", "new direction") is True

    def test_steer_message_in_queue(self):
        self.agent.get_steering_queue("session-1")
        self.agent.steer("session-1", "go left")
        msgs = self.agent._drain_steering("session-1")
        assert msgs == ["go left"]

    def test_drain_empty_queue(self):
        self.agent.get_steering_queue("session-1")
        msgs = self.agent._drain_steering("session-1")
        assert msgs == []

    def test_drain_no_queue(self):
        msgs = self.agent._drain_steering("nonexistent")
        assert msgs == []

    def test_drain_multiple_messages(self):
        self.agent.get_steering_queue("session-1")
        self.agent.steer("session-1", "msg1")
        self.agent.steer("session-1", "msg2")
        self.agent.steer("session-1", "msg3")
        msgs = self.agent._drain_steering("session-1")
        assert msgs == ["msg1", "msg2", "msg3"]

    def test_drain_clears_queue(self):
        self.agent.get_steering_queue("session-1")
        self.agent.steer("session-1", "msg1")
        self.agent._drain_steering("session-1")
        msgs = self.agent._drain_steering("session-1")
        assert msgs == []

    def test_cleanup_on_finalize(self):
        """Steering queue should be cleaned up after turn ends."""
        self.agent._steering_queues["session-1"] = asyncio.Queue()
        self.agent._steering_queues.pop("session-1", None)
        assert "session-1" not in self.agent._steering_queues

    def test_multiple_sessions_independent(self):
        """Steering queues for different sessions are independent."""
        self.agent.get_steering_queue("sess-a")
        self.agent.get_steering_queue("sess-b")
        self.agent.steer("sess-a", "for a")
        self.agent.steer("sess-b", "for b")
        assert self.agent._drain_steering("sess-a") == ["for a"]
        assert self.agent._drain_steering("sess-b") == ["for b"]

    def test_steer_after_queue_removed(self):
        """Steer returns False after queue is removed (turn ended)."""
        self.agent.get_steering_queue("session-1")
        assert self.agent.steer("session-1", "msg") is True
        self.agent._steering_queues.pop("session-1", None)
        assert self.agent.steer("session-1", "msg") is False
