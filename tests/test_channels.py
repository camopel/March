"""Tests for March channels: WebSocket, ACP, Matrix, VS Code."""

from __future__ import annotations

import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from march.channels.base import Channel
from march.channels.websocket import WebSocketChannel, _WSConnection
from march.channels.acp import ACPChannel, JSONRPC_VERSION, ACP_PROTOCOL_VERSION
from march.channels.matrix_channel import MatrixChannel, CREDENTIALS_PATH
from march.channels.vscode import VSCodeChannel
from march.core.session import Session


# ── Channel Interface Tests ──────────────────────────────────────────────

class TestChannelInterface:
    """Verify all channels implement the Channel ABC."""

    def test_websocket_is_channel(self):
        ch = WebSocketChannel()
        assert isinstance(ch, Channel)
        assert ch.name == "homehub"

    def test_acp_is_channel(self):
        ch = ACPChannel()
        assert isinstance(ch, Channel)
        assert ch.name == "acp"

    def test_matrix_is_channel(self):
        ch = MatrixChannel()
        assert isinstance(ch, Channel)
        assert ch.name == "matrix"

    def test_vscode_is_channel(self):
        ch = VSCodeChannel()
        assert isinstance(ch, Channel)
        assert ch.name == "vscode"


# ── WebSocket Channel Tests ─────────────────────────────────────────────

class TestWebSocketChannel:
    """Tests for the HomeHub WebSocket channel."""

    def test_init_defaults(self):
        ch = WebSocketChannel()
        assert ch.host == "0.0.0.0"
        assert ch.port == 8100
        assert ch.cors_origins == []
        assert ch.connection_count == 0

    def test_init_custom(self):
        ch = WebSocketChannel(host="127.0.0.1", port=9000, cors_origins=["http://localhost:3000"])
        assert ch.host == "127.0.0.1"
        assert ch.port == 9000
        assert ch.cors_origins == ["http://localhost:3000"]

    @pytest.mark.asyncio
    async def test_stop_clears_connections(self):
        ch = WebSocketChannel()
        ch._running = True
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()
        conn = _WSConnection(ws=mock_ws, conn_id="test-1")
        ch._connections["test-1"] = conn

        await ch.stop()
        assert not ch._running
        assert len(ch._connections) == 0

    @pytest.mark.asyncio
    async def test_handle_message_routing(self):
        ch = WebSocketChannel()
        ch._agent = MagicMock()
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        conn.session = None
        conn.conn_id = "test-conn"

        # Unknown message type → error
        await ch._handle_message(conn, {"type": "unknown.type", "data": {}})
        conn.send_json.assert_called()
        call_args = conn.send_json.call_args[0][0]
        assert call_args["type"] == "error"

    @pytest.mark.asyncio
    async def test_session_create(self):
        ch = WebSocketChannel()
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        conn.conn_id = "test-conn"
        conn.session = None

        await ch._handle_session_create(conn, {})
        assert conn.session is not None
        assert conn.session.source_type == "homehub"

    @pytest.mark.asyncio
    async def test_session_history_empty(self):
        ch = WebSocketChannel()
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        conn.session = None

        await ch._handle_session_history(conn, {})
        conn.send_json.assert_called_once()
        data = conn.send_json.call_args[0][0]
        assert data["data"]["history"] == []

    @pytest.mark.asyncio
    async def test_status_get(self):
        ch = WebSocketChannel()
        ch._agent = MagicMock()
        ch._agent.tools.tool_count = 5
        conn = MagicMock(spec=_WSConnection)
        conn.send_json = AsyncMock()
        conn.conn_id = "test-conn"
        conn.session = None

        await ch._handle_status_get(conn, {})
        data = conn.send_json.call_args[0][0]
        assert data["type"] == "status"
        assert data["data"]["tools"] == 5


# ── ACP Channel Tests ────────────────────────────────────────────────────

class TestACPChannel:
    """Tests for the ACP (Agent Client Protocol) channel."""

    def test_init(self):
        ch = ACPChannel()
        assert ch.name == "acp"
        assert not ch._initialized

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        ch = ACPChannel()
        ch._writer = MagicMock()
        ch._write_message = AsyncMock()

        await ch._handle_initialize(1, {
            "capabilities": {"streaming": True},
            "workspacePath": "/home/user/project",
        })

        assert ch._initialized
        assert ch._session is not None
        assert ch._session.source_type == "acp"
        assert ch._session.source_id == "/home/user/project"

        # Check response
        ch._write_message.assert_called_once()
        response = ch._write_message.call_args[0][0]
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_handle_message_not_initialized(self):
        ch = ACPChannel()
        ch._agent = MagicMock()
        ch._write_message = AsyncMock()
        ch._initialized = False

        await ch._handle_agent_message(1, {"message": "hello"})
        response = ch._write_message.call_args[0][0]
        assert "error" in response
        assert response["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_handle_shutdown(self):
        ch = ACPChannel()
        ch._write_message = AsyncMock()
        ch._running = True

        await ch._handle_shutdown(99, {})
        assert not ch._running
        response = ch._write_message.call_args[0][0]
        assert response["result"]["status"] == "shutdown"


# ── Matrix Channel Tests ────────────────────────────────────────────────

class TestMatrixChannel:
    """Tests for the Matrix channel."""

    def test_init_defaults(self):
        ch = MatrixChannel()
        assert ch.name == "matrix"
        assert ch.homeserver == "auto"
        assert ch.e2ee is False
        assert ch.auto_setup is True

    def test_init_custom(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            rooms=["#general:example.com"],
            e2ee=True,
        )
        assert ch.homeserver == "https://matrix.example.com"
        assert ch.user_id == "@bot:example.com"
        assert ch.rooms == ["#general:example.com"]
        assert ch.e2ee is True

    def test_get_or_create_session(self):
        ch = MatrixChannel()
        s1 = ch._get_or_create_session("!room1:server")
        s2 = ch._get_or_create_session("!room1:server")
        s3 = ch._get_or_create_session("!room2:server")

        assert s1 is s2  # Same room = same session
        assert s1 is not s3  # Different room = different session
        assert s1.source_type == "matrix"
        assert s1.source_id == "!room1:server"

    def test_markdown_to_html(self):
        result = MatrixChannel._markdown_to_html("Hello **world**")
        assert "Hello" in result
        assert "world" in result


# ── VS Code Channel Tests ───────────────────────────────────────────────

class TestVSCodeChannel:
    """Tests for the VS Code extension bridge."""

    def test_init_defaults(self):
        ch = VSCodeChannel()
        assert ch.name == "vscode"
        assert ch.ws_url == "ws://localhost:8100"
        assert ch.workspace_path == ""

    def test_init_custom(self):
        ch = VSCodeChannel(ws_url="ws://localhost:9000", workspace_path="/home/user/project")
        assert ch.ws_url == "ws://localhost:9000"
        assert ch.workspace_path == "/home/user/project"

    def test_is_connected_false(self):
        ch = VSCodeChannel()
        assert not ch.is_connected

    @pytest.mark.asyncio
    async def test_stop(self):
        ch = VSCodeChannel()
        ch._running = True
        ch._ws = AsyncMock()
        await ch.stop()
        assert not ch._running
