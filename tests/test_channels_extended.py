"""Extended tests for channels and edge cases."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.channels.base import Channel
from march.channels.websocket import WebSocketChannel, _WSConnection
from march.channels.acp import ACPChannel, JSONRPC_VERSION, ACP_PROTOCOL_VERSION
from march.channels.matrix_channel import MatrixChannel
from march.channels.vscode import VSCodeChannel
from march.channels.terminal import TerminalChannel
from march.core.session import Session, deterministic_session_id
from march.core.message import Message, Role
from march.agents.task_queue import TaskQueue


# ─────────────────────────────────────────────────────────────
# Terminal Channel
# ─────────────────────────────────────────────────────────────

class TestTerminalChannel:
    def test_init_defaults(self):
        ch = TerminalChannel()
        assert ch.name == "terminal"
        assert ch.streaming is True

    def test_init_custom(self):
        ch = TerminalChannel(streaming=False, theme="dark")
        assert ch.streaming is False
        assert ch.theme == "dark"

    def test_is_channel(self):
        ch = TerminalChannel()
        assert isinstance(ch, Channel)

    async def test_stop(self):
        ch = TerminalChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running


# ─────────────────────────────────────────────────────────────
# WebSocket Channel Extended
# ─────────────────────────────────────────────────────────────

class TestWebSocketChannelExtended:
    def test_connection_count(self):
        ch = WebSocketChannel()
        assert ch.connection_count == 0

    async def test_multiple_connections(self):
        ch = WebSocketChannel()
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.close = AsyncMock()
            conn = _WSConnection(ws=mock_ws, conn_id=f"conn-{i}")
            ch._connections[f"conn-{i}"] = conn
        assert ch.connection_count == 3
        await ch.stop()
        assert ch.connection_count == 0

    async def test_session_create_with_source(self):
        ch = WebSocketChannel()
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        conn.conn_id = "test-conn"
        conn.session = None

        await ch._handle_session_create(conn, {"source_id": "my-app"})
        assert conn.session is not None
        assert conn.session.source_type == "homehub"

    async def test_session_history_with_data(self):
        ch = WebSocketChannel()
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        session = Session()
        session.add_exchange("hello", "world")
        conn.session = session

        await ch._handle_session_history(conn, {})
        data = conn.send_json.call_args[0][0]
        assert len(data["data"]["history"]) > 0


# ─────────────────────────────────────────────────────────────
# ACP Channel Extended
# ─────────────────────────────────────────────────────────────

class TestACPChannelExtended:
    async def test_handle_message_initialized(self):
        ch = ACPChannel()
        ch._agent = MagicMock()
        ch._write_message = AsyncMock()
        ch._initialized = True
        ch._session = Session()

        # Mock agent.run to return a response
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_calls_made = 0
        mock_response.total_tokens = 10
        mock_response.total_cost = 0.01
        ch._agent.run = AsyncMock(return_value=mock_response)

        await ch._handle_agent_message(1, {"message": "hello"})
        response = ch._write_message.call_args[0][0]
        assert "result" in response

    async def test_handle_shutdown_sets_not_running(self):
        ch = ACPChannel()
        ch._write_message = AsyncMock()
        ch._running = True

        await ch._handle_shutdown(2, {})
        assert not ch._running
        response = ch._write_message.call_args[0][0]
        assert response["id"] == 2


# ─────────────────────────────────────────────────────────────
# Matrix Channel Extended
# ─────────────────────────────────────────────────────────────

class TestMatrixChannelExtended:
    def test_session_per_room(self):
        ch = MatrixChannel()
        s1 = ch._get_or_create_session("!room1:server")
        s2 = ch._get_or_create_session("!room2:server")
        assert s1.id != s2.id
        assert s1.source_id == "!room1:server"
        assert s2.source_id == "!room2:server"

    def test_session_deterministic(self):
        ch = MatrixChannel()
        s = ch._get_or_create_session("!stable:server")
        expected_id = deterministic_session_id("matrix", "!stable:server")
        assert s.id == expected_id

    def test_markdown_simple(self):
        html = MatrixChannel._markdown_to_html("Simple text")
        assert "Simple" in html

    def test_markdown_code_block(self):
        html = MatrixChannel._markdown_to_html("```python\nprint('hi')\n```")
        assert "print" in html

    def test_markdown_bold(self):
        html = MatrixChannel._markdown_to_html("**bold text**")
        assert "bold" in html


# ─────────────────────────────────────────────────────────────
# VS Code Channel Extended
# ─────────────────────────────────────────────────────────────

class TestVSCodeChannelExtended:
    def test_session_from_workspace(self):
        ch = VSCodeChannel(workspace_path="/home/user/project")
        assert ch.workspace_path == "/home/user/project"

    async def test_stop_no_ws(self):
        ch = VSCodeChannel()
        ch._running = True
        ch._ws = None
        await ch.stop()
        assert not ch._running


# ─────────────────────────────────────────────────────────────
# Edge Cases: Empty/Malformed Inputs
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_session_empty_history_to_llm(self):
        s = Session()
        msgs = s.get_messages_for_llm()
        assert msgs == []

    def test_message_empty_content(self):
        msg = Message.user("")
        assert msg.content == ""
        llm = msg.to_llm_messages()
        assert llm[0]["content"] == ""

    def test_message_none_tool_calls(self):
        msg = Message.assistant("hi", tool_calls=None)
        assert not msg.has_tool_calls

    def test_tool_result_empty_content(self):
        from march.core.message import ToolResult
        tr = ToolResult(id="tc1", content="")
        assert not tr.is_error
        assert tr.summary == ""

    def test_session_add_many_messages(self):
        s = Session()
        for i in range(100):
            s.add_exchange(f"msg {i}", f"reply {i}")
        assert len(s.history) == 200
        msgs = s.get_messages_for_llm()
        assert len(msgs) == 200

    def test_tool_call_empty_args(self):
        from march.core.message import ToolCall
        tc = ToolCall(id="tc1", name="test", args={})
        d = tc.to_dict()
        tc2 = ToolCall.from_dict(d)
        assert tc2.args == {}

    def test_context_empty_session_context(self):
        from march.core.context import Context
        ctx = Context(session_context={})
        prompt = ctx.build_system_prompt()
        assert isinstance(prompt, str)

    def test_context_very_long_extra(self):
        from march.core.context import Context
        ctx = Context()
        ctx.add("x" * 100000)
        prompt = ctx.build_system_prompt()
        assert len(prompt) > 0

    async def test_session_store_save_empty_session(self, tmp_path):
        from march.core.session import SessionStore
        store = SessionStore(tmp_path / "edge.db")
        await store.initialize()
        try:
            s = Session(id="empty-sess")
            await store.save_session(s)
            loaded = await store.load_session("empty-sess")
            assert loaded is not None
            assert len(loaded.history) == 0
        finally:
            await store.close()

    async def test_session_store_save_large_history(self, tmp_path):
        from march.core.session import SessionStore
        store = SessionStore(tmp_path / "large.db")
        await store.initialize()
        try:
            s = Session(id="large-sess")
            for i in range(200):
                s.add_exchange(f"message {i} " + "x" * 100, f"reply {i} " + "y" * 100)
            await store.save_session(s)
            loaded = await store.load_session("large-sess")
            assert loaded is not None
            assert len(loaded.history) == 400
        finally:
            await store.close()


# ─────────────────────────────────────────────────────────────
# Concurrent Operations
# ─────────────────────────────────────────────────────────────

class TestConcurrentOps:
    async def test_concurrent_session_saves(self, tmp_path):
        from march.core.session import SessionStore
        store = SessionStore(tmp_path / "concurrent.db")
        await store.initialize()
        try:
            async def create_and_save(i):
                s = Session(id=f"conc-{i}", source_type="test", source_id=f"s{i}")
                s.add_exchange(f"hi {i}", f"hello {i}")
                await store.save_session(s)
                return i

            results = await asyncio.gather(*[create_and_save(i) for i in range(20)])
            assert len(results) == 20

            sessions = await store.list_sessions()
            assert len(sessions) == 20
        finally:
            await store.close()

    async def test_concurrent_task_queue_ops(self):
        tq = TaskQueue()
        results = []

        async def task(n):
            await asyncio.sleep(0.01)
            results.append(n)
            return n

        tasks = [tq.enqueue("main", lambda i=i: task(i)) for i in range(50)]
        await asyncio.gather(*tasks)
        assert len(results) == 50


# ─────────────────────────────────────────────────────────────
# Resource Cleanup
# ─────────────────────────────────────────────────────────────

class TestResourceCleanup:
    async def test_session_store_close_idempotent(self, tmp_path):
        from march.core.session import SessionStore
        store = SessionStore(tmp_path / "cleanup.db")
        await store.initialize()
        await store.close()
        await store.close()  # Second close should not crash

    async def test_memory_store_close_twice(self, tmp_path):
        from march.memory.store import MemoryStore
        from march.memory.vector_store import VectorStore
        store = MemoryStore(
            workspace=tmp_path, index_dir=tmp_path / "idx", db_path=tmp_path / "db.db",
        )
        mock_embedder = AsyncMock()
        mock_embedder.dim = 8
        store.embedder = mock_embedder
        store.vectors = VectorStore(index_dir=tmp_path / "idx", dim=8)
        await store.initialize()
        await store.close()
        # Second close should be safe since _initialized is False
        assert not store._initialized
