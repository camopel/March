"""Tests for Matrix channel as I/O adapter.

All tests run without external services — nio client and Orchestrator are
fully mocked.  Each test is independent.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.channels.matrix_channel import MatrixChannel
from march.core.orchestrator import (
    Cancelled,
    Error,
    FinalResponse,
    OrchestratorEvent,
    TextDelta,
    ToolProgress,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_channel(orchestrator: Any = None) -> MatrixChannel:
    """Create a MatrixChannel wired to a mock orchestrator and nio client."""
    orch = orchestrator or AsyncMock()
    ch = MatrixChannel(orchestrator=orch)
    ch._orchestrator = orch
    ch._running = True
    ch._start_ts = 0  # accept all events by default

    # Mock nio client
    client = AsyncMock()
    client.user_id = "@march:localhost"
    client.room_read_markers = AsyncMock()
    client.room_typing = AsyncMock()
    client.room_send = AsyncMock()
    ch._client = client

    return ch


def _text_event(
    body: str,
    sender: str = "@user:localhost",
    room_id: str = "!room1:localhost",
    server_timestamp: int | None = None,
    event_id: str = "$evt1",
) -> SimpleNamespace:
    """Build a fake RoomMessageText event."""
    return SimpleNamespace(
        body=body,
        sender=sender,
        server_timestamp=server_timestamp or int(time.time() * 1000) + 10_000,
        event_id=event_id,
    )


def _room(room_id: str = "!room1:localhost") -> SimpleNamespace:
    """Build a fake Room object."""
    return SimpleNamespace(room_id=room_id)


def _image_event(
    filename: str = "photo.jpg",
    sender: str = "@user:localhost",
    url: str = "mxc://localhost/abc123",
    server_timestamp: int | None = None,
    event_id: str = "$img1",
) -> SimpleNamespace:
    """Build a fake RoomMessageImage event."""
    return SimpleNamespace(
        body=filename,
        sender=sender,
        url=url,
        server_timestamp=server_timestamp or int(time.time() * 1000) + 10_000,
        event_id=event_id,
        key=None,
        hashes=None,
        iv=None,
        source={},
        mimetype="image/jpeg",
    )


def _audio_event(
    filename: str = "voice.ogg",
    sender: str = "@user:localhost",
    url: str = "mxc://localhost/audio1",
    server_timestamp: int | None = None,
    event_id: str = "$aud1",
) -> SimpleNamespace:
    """Build a fake RoomMessageAudio event."""
    return SimpleNamespace(
        body=filename,
        sender=sender,
        url=url,
        server_timestamp=server_timestamp or int(time.time() * 1000) + 10_000,
        event_id=event_id,
        key=None,
        hashes=None,
        iv=None,
        mimetype="audio/ogg",
    )


def _file_event(
    filename: str = "doc.pdf",
    sender: str = "@user:localhost",
    url: str = "mxc://localhost/file1",
    mime_type: str = "application/pdf",
    server_timestamp: int | None = None,
    event_id: str = "$file1",
) -> SimpleNamespace:
    """Build a fake RoomMessageFile event."""
    return SimpleNamespace(
        body=filename,
        sender=sender,
        url=url,
        server_timestamp=server_timestamp or int(time.time() * 1000) + 10_000,
        event_id=event_id,
        key=None,
        hashes=None,
        iv=None,
        mimetype=mime_type,
    )


async def _drain_orchestrator_events(
    events: list[OrchestratorEvent],
) -> AsyncIterator[OrchestratorEvent]:
    """Async generator that yields pre-built orchestrator events."""
    for ev in events:
        yield ev


# ═══════════════════════════════════════════════════════════════════════════════
# TestMatrixTextMessage
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixTextMessage:
    """Text event → orchestrator.handle_message called correctly."""

    @pytest.mark.asyncio
    async def test_text_message_processed(self):
        """Text event → orchestrator.handle_message called with correct content."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="ok")])
        )
        ch = _make_channel(orch)

        room = _room()
        event = _text_event("Hello agent")

        await ch._on_message(room, event)
        # _on_message spawns a background task; give it time to run
        await asyncio.sleep(0.1)

        orch.handle_message.assert_called_once()
        call_kwargs = orch.handle_message.call_args
        assert call_kwargs.kwargs["content"] == "Hello agent"
        assert call_kwargs.kwargs["source"] == "matrix"

    @pytest.mark.asyncio
    async def test_response_sent_to_room(self):
        """Orchestrator returns text → send() called with room_id."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events(
                [FinalResponse(content="Here is my reply")]
            )
        )
        ch = _make_channel(orch)

        room = _room("!testroom:localhost")
        event = _text_event("Hi", room_id="!testroom:localhost")

        await ch._on_message(room, event)
        await asyncio.sleep(0.1)

        # send() calls room_send on the nio client
        ch._client.room_send.assert_called()
        call_args = ch._client.room_send.call_args
        assert call_args.kwargs["room_id"] == "!testroom:localhost"
        body = call_args.kwargs["content"]["body"]
        assert "Here is my reply" in body


# ═══════════════════════════════════════════════════════════════════════════════
# TestMatrixMultiModal
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixMultiModal:
    """Multi-modal message handling: image, audio, PDF, text files."""

    @pytest.mark.asyncio
    async def test_image_message(self):
        """Image event → download + resize → orchestrator called with image content."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="nice pic")])
        )
        ch = _make_channel(orch)

        # Mock download to return a tiny valid JPEG
        # 1x1 red pixel JPEG
        import base64
        tiny_jpeg = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf"
            "/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAA"
            "AAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
        )
        download_resp = MagicMock()
        download_resp.body = tiny_jpeg
        download_resp.content_type = "image/jpeg"
        ch._client.download = AsyncMock(return_value=download_resp)

        room = _room()
        event = _image_event()

        with patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="abcd1234")
            await ch._on_image(room, event)
            await asyncio.sleep(0.2)

        orch.handle_message.assert_called_once()
        call_kwargs = orch.handle_message.call_args.kwargs
        content = call_kwargs["content"]
        # Content should be a list with image + text blocks
        assert isinstance(content, list)
        types = [block["type"] for block in content]
        assert "image" in types
        assert "text" in types

    @pytest.mark.asyncio
    async def test_audio_message(self):
        """Audio event → download + transcribe → orchestrator called with transcription."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="got it")])
        )
        ch = _make_channel(orch)

        # Mock download
        download_resp = MagicMock()
        download_resp.body = b"\x00" * 100  # fake audio bytes
        download_resp.content_type = "audio/ogg"
        ch._client.download = AsyncMock(return_value=download_resp)

        # Mock agent with tools that return transcription
        mock_agent = MagicMock()
        mock_agent.app = None
        tool_result = MagicMock()
        tool_result.is_error = False
        tool_result.content = "Hello from voice"
        mock_agent.tools = MagicMock()
        mock_agent.tools.execute = AsyncMock(return_value=tool_result)
        ch._agent = mock_agent

        room = _room()
        event = _audio_event()

        with patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="abcd1234")
            await ch._on_audio(room, event)
            await asyncio.sleep(0.2)

        orch.handle_message.assert_called_once()
        call_kwargs = orch.handle_message.call_args.kwargs
        # The transcribed text should be passed as content
        assert call_kwargs["content"] == "Hello from voice"

    @pytest.mark.asyncio
    async def test_pdf_file_message(self):
        """File event (PDF) → download + extract → orchestrator called with PDF text."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="read it")])
        )
        ch = _make_channel(orch)

        # Mock download
        download_resp = MagicMock()
        download_resp.body = b"%PDF-fake"
        download_resp.content_type = "application/pdf"
        ch._client.download = AsyncMock(return_value=download_resp)

        room = _room()
        event = _file_event("report.pdf", mime_type="application/pdf")

        # Mock PDF extraction
        with patch.object(
            MatrixChannel,
            "_extract_pdf_text",
            return_value="[PDF: report.pdf (3 pages, 10KB)]\n\nExtracted PDF content here",
        ), patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="abcd1234")
            await ch._on_file(room, event)
            await asyncio.sleep(0.2)

        orch.handle_message.assert_called_once()
        call_kwargs = orch.handle_message.call_args.kwargs
        assert "Extracted PDF content here" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_text_file_message(self):
        """File event (.py/.txt) → download → orchestrator called with file content."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="nice code")])
        )
        ch = _make_channel(orch)

        # Mock download with Python file content
        file_content = b'def hello():\n    print("world")\n'
        download_resp = MagicMock()
        download_resp.body = file_content
        download_resp.content_type = "text/x-python"
        ch._client.download = AsyncMock(return_value=download_resp)

        room = _room()
        event = _file_event("script.py", mime_type="text/x-python")

        with patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="abcd1234")
            await ch._on_file(room, event)
            await asyncio.sleep(0.2)

        orch.handle_message.assert_called_once()
        call_kwargs = orch.handle_message.call_args.kwargs
        assert 'def hello()' in call_kwargs["content"]
        assert "script.py" in call_kwargs["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestMatrixCommands
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixCommands:
    """Command handling: /stop, 停止, /reset."""

    @pytest.mark.asyncio
    async def test_stop_command(self):
        """User sends '/stop' → cancel_event set, no orchestrator call."""
        orch = AsyncMock()
        orch.handle_message = MagicMock()
        ch = _make_channel(orch)

        room_id = "!room1:localhost"
        # Set up an active cancel event (simulating an in-progress turn)
        cancel_ev = ch._reset_cancel_event(room_id)

        room = _room(room_id)
        event = _text_event("/stop")

        await ch._on_message(room, event)
        await asyncio.sleep(0.05)

        # Cancel event should be set
        assert cancel_ev.is_set()
        # Orchestrator should NOT have been called
        orch.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_chinese(self):
        """User sends '停止' → cancel_event set."""
        orch = AsyncMock()
        orch.handle_message = MagicMock()
        ch = _make_channel(orch)

        room_id = "!room1:localhost"
        cancel_ev = ch._reset_cancel_event(room_id)

        room = _room(room_id)
        event = _text_event("停止")

        await ch._on_message(room, event)
        await asyncio.sleep(0.05)

        assert cancel_ev.is_set()
        orch.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_command(self):
        """User sends '/reset' → orchestrator.reset_session called."""
        orch = AsyncMock()
        orch.reset_session = AsyncMock(return_value={"memory": {}})
        orch.handle_message = MagicMock()
        ch = _make_channel(orch)

        room = _room("!resetroom:localhost")
        event = _text_event("/reset")

        await ch._on_message(room, event)
        await asyncio.sleep(0.1)

        orch.reset_session.assert_called_once()
        # Should have been called with the session_id for this room
        session_id = MatrixChannel._session_id_for_room("!resetroom:localhost")
        orch.reset_session.assert_called_with(session_id)
        # handle_message should NOT be called for /reset
        orch.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_clears_cancel_event(self):
        """After reset, cancel event for room is cleared."""
        orch = AsyncMock()
        orch.reset_session = AsyncMock(return_value={"memory": {}})
        ch = _make_channel(orch)

        room_id = "!resetroom:localhost"
        # Set a cancel event
        cancel_ev = ch._reset_cancel_event(room_id)
        cancel_ev.set()
        assert room_id in ch._cancel_events

        room = _room(room_id)
        event = _text_event("/reset")

        await ch._on_message(room, event)
        await asyncio.sleep(0.1)

        # After reset, the cancel event for this room should be removed
        assert room_id not in ch._cancel_events


# ═══════════════════════════════════════════════════════════════════════════════
# TestMatrixSessionManagement
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixSessionManagement:
    """Session ID generation and event filtering."""

    def test_session_id_deterministic(self):
        """Same room_id always produces same session_id."""
        room_id = "!stable:localhost"
        sid1 = MatrixChannel._session_id_for_room(room_id)
        sid2 = MatrixChannel._session_id_for_room(room_id)
        assert sid1 == sid2

    def test_different_rooms_different_sessions(self):
        """Two rooms → two different session_ids."""
        sid1 = MatrixChannel._session_id_for_room("!roomA:localhost")
        sid2 = MatrixChannel._session_id_for_room("!roomB:localhost")
        assert sid1 != sid2

    @pytest.mark.asyncio
    async def test_start_ts_guard(self):
        """Events before _start_ts are ignored (initial sync)."""
        orch = AsyncMock()
        orch.handle_message = MagicMock()
        ch = _make_channel(orch)

        # Set start_ts to "now" so old events are filtered
        ch._start_ts = int(time.time() * 1000)

        room = _room()
        # Event with timestamp before _start_ts
        old_event = _text_event(
            "old message",
            server_timestamp=ch._start_ts - 5000,
        )

        await ch._on_message(room, old_event)
        await asyncio.sleep(0.05)

        # Should have been silently ignored
        orch.handle_message.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TestMatrixConnection
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatrixConnection:
    """Connection-level behavior: multi-type messages, typing indicators."""

    @pytest.mark.asyncio
    async def test_same_room_multiple_message_types(self):
        """Text + image + audio in same room → all same session."""
        orch = AsyncMock()
        orch.handle_message = MagicMock(
            return_value=_drain_orchestrator_events([FinalResponse(content="ok")])
        )
        ch = _make_channel(orch)

        room_id = "!multiroom:localhost"
        room = _room(room_id)

        # Mock download for image/audio
        download_resp = MagicMock()
        download_resp.body = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header
        download_resp.content_type = "image/jpeg"
        ch._client.download = AsyncMock(return_value=download_resp)

        # Mock agent for audio transcription
        mock_agent = MagicMock()
        mock_agent.app = None
        tool_result = MagicMock()
        tool_result.is_error = False
        tool_result.content = "transcribed audio"
        mock_agent.tools = MagicMock()
        mock_agent.tools.execute = AsyncMock(return_value=tool_result)
        ch._agent = mock_agent

        expected_session = MatrixChannel._session_id_for_room(room_id)

        # Send text
        text_event = _text_event("hello", room_id=room_id)
        await ch._on_message(room, text_event)
        await asyncio.sleep(0.1)

        # Send image
        with patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="abcd1234")
            img_event = _image_event()
            await ch._on_image(room, img_event)
            await asyncio.sleep(0.2)

        # Send audio
        download_resp_audio = MagicMock()
        download_resp_audio.body = b"\x00" * 50
        download_resp_audio.content_type = "audio/ogg"
        ch._client.download = AsyncMock(return_value=download_resp_audio)

        with patch("march.channels.matrix_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = MagicMock(hex="efgh5678")
            aud_event = _audio_event()
            await ch._on_audio(room, aud_event)
            await asyncio.sleep(0.2)

        # All three calls should use the same session_id
        assert orch.handle_message.call_count == 3
        for call in orch.handle_message.call_args_list:
            assert call.kwargs["session_id"] == expected_session

    @pytest.mark.asyncio
    async def test_typing_indicator_during_processing(self):
        """While orchestrator runs, typing indicator is sent."""
        orch = AsyncMock()

        async def slow_handle(**kwargs):
            """Simulate slow orchestrator that yields after a delay."""
            await asyncio.sleep(0.1)
            yield FinalResponse(content="done")

        orch.handle_message = MagicMock(side_effect=slow_handle)
        ch = _make_channel(orch)

        room = _room("!typingroom:localhost")
        event = _text_event("think hard")

        await ch._on_message(room, event)
        # Check typing was turned on immediately (before background task completes)
        await asyncio.sleep(0.05)

        # Typing indicator should have been sent (typing_state=True)
        typing_calls = ch._client.room_typing.call_args_list
        assert any(
            call.args == ("!typingroom:localhost",)
            and call.kwargs.get("typing_state") is True
            for call in typing_calls
        ) or any(
            len(call.args) >= 1
            and call.args[0] == "!typingroom:localhost"
            for call in typing_calls
        ), f"Expected typing indicator call for room, got: {typing_calls}"

        # Wait for completion
        await asyncio.sleep(0.2)

        # After processing, typing should be turned off
        assert any(
            call.kwargs.get("typing_state") is False
            for call in ch._client.room_typing.call_args_list
        ), "Expected typing_state=False after processing"
