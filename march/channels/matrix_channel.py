"""Matrix channel for March.

Bot joins Matrix rooms, sends/receives messages. One room = one session.
Uses matrix-nio for Matrix client functionality.

Auto-setup: detect homeserver, create user, persist credentials.
E2EE support is optional and config-driven.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from march.channels.base import Channel
from march.core.session import Session
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent, AgentResponse
    from march.llm.base import StreamChunk

logger = get_logger("march.matrix")

CREDENTIALS_PATH = Path.home() / ".march" / "credentials"


class MatrixChannel(Channel):
    """Matrix chat channel using matrix-nio.

    Each room March is in = one session. Messages in room A don't mix
    with room B. Each room has its own conversation history.
    """

    name: str = "matrix"

    def __init__(
        self,
        homeserver: str = "auto",
        user_id: str = "",
        password: str = "",
        rooms: list[str] | None = None,
        e2ee: bool = False,
        auto_setup: bool = True,
        display_name: str = "March",
    ) -> None:
        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.rooms = rooms or []
        self.e2ee = e2ee
        self.auto_setup = auto_setup
        self.display_name = display_name

        self._agent: Agent | None = None
        self._client: Any = None  # nio.AsyncClient
        self._sessions: dict[str, Session] = {}  # room_id → session
        self._running = False

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the Matrix client, join rooms, and begin listening."""
        self._agent = agent
        self._running = True

        try:
            from nio import AsyncClient, RoomMessageText, InviteMemberEvent
        except ImportError:
            logger.error("matrix-nio not installed. Install with: pip install matrix-nio")
            return

        # Resolve homeserver
        homeserver = await self._resolve_homeserver()
        if not homeserver:
            logger.error("Could not determine Matrix homeserver")
            return

        # Load or create credentials
        creds = await self._load_or_create_credentials(homeserver)
        if not creds:
            logger.error("Failed to set up Matrix credentials")
            return

        # Create client
        store_path = str(Path.home() / ".march" / "matrix_store")
        Path(store_path).mkdir(parents=True, exist_ok=True)

        if self.e2ee:
            from nio import AsyncClient
            self._client = AsyncClient(
                homeserver,
                creds["user_id"],
                store_path=store_path,
            )
        else:
            self._client = AsyncClient(homeserver, creds["user_id"])

        # Login
        response = await self._client.login(creds["password"])
        if hasattr(response, "access_token"):
            logger.info("matrix: logged in as %s", creds["user_id"])
        else:
            logger.error("matrix: login failed: %s", response)
            return

        # Set display name
        try:
            await self._client.set_displayname(self.display_name)
        except Exception:
            pass

        # Join configured rooms
        for room in self.rooms:
            try:
                await self._client.join(room)
                logger.info("matrix: joined room %s", room)
            except Exception as e:
                logger.warning("matrix: failed to join %s: %s", room, e)

        # Set up callbacks
        self._client.add_event_callback(
            self._on_message, RoomMessageText
        )
        self._client.add_event_callback(
            self._on_invite, InviteMemberEvent
        )

        # Sync loop
        try:
            await self._client.sync_forever(timeout=30000, full_state=True)
        except asyncio.CancelledError:
            pass
        finally:
            await self._client.close()

    async def stop(self) -> None:
        """Stop the Matrix client."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a message to a Matrix room."""
        room_id = kwargs.get("room_id", "")
        if not room_id or not self._client:
            return

        try:
            from nio import RoomSendResponse
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": content,
                    "format": "org.matrix.custom.html",
                    "formatted_body": self._markdown_to_html(content),
                },
            )
        except Exception as e:
            logger.error("matrix: failed to send to %s: %s", room_id, e)

    async def send_stream(
        self, chunks: AsyncIterator["StreamChunk"], **kwargs: Any
    ) -> None:
        """Send a streaming response to a Matrix room.

        Matrix doesn't natively support streaming, so we collect
        the full response and send it as a single message.
        """
        room_id = kwargs.get("room_id", "")
        collected = ""
        async for chunk in chunks:
            if hasattr(chunk, "delta") and chunk.delta:
                collected += chunk.delta
        if collected:
            await self.send(collected, room_id=room_id)

    # ── Event Callbacks ──────────────────────────────────────────────────

    async def _on_message(self, room: Any, event: Any) -> None:
        """Handle incoming messages from Matrix rooms."""
        if not self._agent or not self._running:
            return

        # Ignore our own messages
        if self._client and event.sender == self._client.user_id:
            return

        text = event.body.strip()
        if not text:
            return

        room_id = room.room_id
        logger.info("matrix: message in %s from %s: %s", room_id, event.sender, text[:80])

        # Get or create session for this room
        session = self._get_or_create_session(room_id)

        # Process the message
        try:
            from march.core.agent import AgentResponse
            collected = ""
            async for item in self._agent.run_stream(text, session):
                if isinstance(item, AgentResponse):
                    break
                if hasattr(item, "delta") and item.delta:
                    collected += item.delta

            if collected:
                await self.send(collected, room_id=room_id)
        except Exception as e:
            logger.error("matrix: error processing message in %s: %s", room_id, e)
            await self.send(f"Error: {e}", room_id=room_id)

    async def _on_invite(self, room: Any, event: Any) -> None:
        """Handle room invites — auto-join."""
        if not self._client:
            return
        try:
            room_id = room.room_id
            await self._client.join(room_id)
            logger.info("matrix: auto-joined room %s", room_id)
        except Exception as e:
            logger.warning("matrix: failed to join invited room: %s", e)

    # ── Session Management ───────────────────────────────────────────────

    def _get_or_create_session(self, room_id: str) -> Session:
        """Get the session for a room, or create a new one."""
        if room_id not in self._sessions:
            self._sessions[room_id] = Session(
                source_type="matrix",
                source_id=room_id,
            )
        return self._sessions[room_id]

    # ── Setup Helpers ────────────────────────────────────────────────────

    async def _resolve_homeserver(self) -> str:
        """Resolve the Matrix homeserver URL."""
        if self.homeserver and self.homeserver != "auto":
            return self.homeserver

        # Try common local homeservers
        import httpx

        candidates = [
            "http://localhost:8008",
            "http://localhost:8448",
            "https://localhost:8448",
        ]
        async with httpx.AsyncClient(verify=False) as client:
            for url in candidates:
                try:
                    resp = await client.get(
                        f"{url}/_matrix/client/versions",
                        timeout=3.0,
                    )
                    if resp.status_code == 200:
                        logger.info("matrix: detected homeserver at %s", url)
                        return url
                except Exception:
                    continue

        return ""

    async def _load_or_create_credentials(self, homeserver: str) -> dict[str, str] | None:
        """Load credentials from disk, or create user if auto_setup is enabled."""
        creds_file = CREDENTIALS_PATH / "matrix.json"

        # Try loading existing credentials
        if creds_file.exists():
            try:
                creds = json.loads(creds_file.read_text())
                if creds.get("user_id") and creds.get("password"):
                    self.user_id = creds["user_id"]
                    self.password = creds["password"]
                    return creds
            except (json.JSONDecodeError, KeyError):
                pass

        # Use configured credentials if available
        if self.user_id and self.password and self.password != "auto":
            creds = {
                "user_id": self.user_id,
                "password": self.password,
                "homeserver": homeserver,
            }
            self._save_credentials(creds)
            return creds

        # Auto-setup: generate credentials
        if self.auto_setup:
            import secrets
            user_id = self.user_id or "@march:localhost"
            password = secrets.token_urlsafe(32)
            creds = {
                "user_id": user_id,
                "password": password,
                "homeserver": homeserver,
            }
            self._save_credentials(creds)
            return creds

        return None

    def _save_credentials(self, creds: dict[str, str]) -> None:
        """Persist credentials to disk."""
        CREDENTIALS_PATH.mkdir(parents=True, exist_ok=True)
        creds_file = CREDENTIALS_PATH / "matrix.json"
        creds_file.write_text(json.dumps(creds, indent=2))
        creds_file.chmod(0o600)

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Simple markdown-to-HTML conversion for Matrix messages."""
        # Basic conversion: just use the text as-is for now
        # A proper implementation would use a markdown parser
        import html
        escaped = html.escape(text)
        # Convert basic markdown
        lines = escaped.split("\n")
        result = "<br>".join(lines)
        return result
