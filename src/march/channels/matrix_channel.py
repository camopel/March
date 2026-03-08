"""Matrix channel for March — pure I/O adapter.

Bot joins Matrix rooms, sends/receives messages. One room = one session.
Uses matrix-nio for Matrix client functionality.

This channel is a thin I/O adapter: all session management, agent calls,
and message persistence are delegated to the Orchestrator. The channel
only handles:
  - Matrix nio client setup, login, sync_forever
  - Converting Matrix events → Orchestrator.handle_message()
  - Converting OrchestratorEvents → Matrix messages
  - E2EE setup and key management
  - Read receipts and typing indicators

Auto-setup: detect homeserver, create user, persist credentials.
E2EE support is optional and config-driven.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from march.channels.base import Channel
from march.core.orchestrator import (
    Cancelled,
    Error,
    FinalResponse,
    OrchestratorEvent,
    TextDelta,
    ToolProgress,
)
from march.logging import get_logger

if TYPE_CHECKING:
    from march.core.agent import Agent
    from march.core.orchestrator import Orchestrator
    from march.llm.base import StreamChunk

logger = get_logger("march.matrix")

CREDENTIALS_PATH = Path.home() / ".march" / "credentials"


class MatrixChannel(Channel):
    """Matrix chat channel using matrix-nio.

    Each room March is in = one session. Messages in room A don't mix
    with room B. Each room has its own conversation history.

    The channel delegates all agent interaction to an Orchestrator instance.
    It only handles Matrix I/O: connecting, sending, receiving, E2EE.
    """

    name: str = "matrix"

    def __init__(
        self,
        homeserver: str = "auto",
        user_id: str = "",
        password: str = "",
        access_token: str = "",
        rooms: list[str] | None = None,
        e2ee: bool = False,
        auto_setup: bool = True,
        display_name: str = "March",
        orchestrator: "Orchestrator | None" = None,
    ) -> None:
        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.access_token = access_token
        self.rooms = rooms or []
        self.e2ee = e2ee
        self.auto_setup = auto_setup
        self.display_name = display_name

        self._agent: Agent | None = None
        self._orchestrator: Orchestrator | None = orchestrator
        self._client: Any = None  # nio.AsyncClient
        self._running = False
        self._start_ts: int = 0  # server_timestamp threshold — ignore events before this

        # Per-room cancel events — set to signal the current turn should stop
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the Matrix client, join rooms, and begin listening."""
        self._agent = agent

        # Resolve orchestrator: explicit > kwarg > build from agent
        if self._orchestrator is None:
            self._orchestrator = kwargs.get("orchestrator")
        if self._orchestrator is None:
            # Build an Orchestrator from the agent (agent must have session_store)
            from march.core.orchestrator import Orchestrator
            if not hasattr(agent, "session_store") or agent.session_store is None:
                logger.error("matrix: agent has no session_store — cannot create Orchestrator")
                return
            self._orchestrator = Orchestrator(agent, agent.session_store)
            logger.info("matrix: created Orchestrator from agent")

        self._running = True

        try:
            from nio import (
                AsyncClient, RoomMessageText, RoomMessageImage, RoomEncryptedImage,
                RoomMessageAudio, RoomEncryptedAudio,
                RoomMessageFile, RoomEncryptedFile,
                InviteMemberEvent, MegolmEvent,
            )
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
            import nio as nio
            from nio import AsyncClient
            from nio.store import SqliteStore
            self._client = AsyncClient(
                homeserver,
                creds["user_id"],
                store_path=store_path,
                config=nio.AsyncClientConfig(
                    store=SqliteStore,
                    store_name="march_crypto",
                    encryption_enabled=True,
                    store_sync_tokens=True,
                ),
            )
        else:
            self._client = AsyncClient(homeserver, creds["user_id"])

        # Login — prefer access_token if provided, otherwise password
        if self.access_token or creds.get("access_token"):
            token = self.access_token or creds["access_token"]
            # Fetch device_id via whoami before setting up client state
            device_id = creds.get("device_id", "")
            if not device_id:
                try:
                    import httpx
                    async with httpx.AsyncClient() as http:
                        r = await http.get(
                            f"{homeserver}/_matrix/client/v3/account/whoami",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        if r.status_code == 200:
                            device_id = r.json().get("device_id", "")
                except Exception:
                    pass
            # Use restore_login — sets user_id, device_id, access_token
            # AND loads the olm store for E2EE in one correct step
            self._client.restore_login(
                user_id=creds["user_id"],
                device_id=device_id or self._client.device_id,
                access_token=token,
            )
            logger.info("matrix: authenticated via access token as %s (device=%s)",
                        creds["user_id"], device_id or "unknown")
        else:
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
            self._on_image, RoomMessageImage
        )
        self._client.add_event_callback(
            self._on_image, RoomEncryptedImage
        )
        self._client.add_event_callback(
            self._on_audio, RoomMessageAudio
        )
        self._client.add_event_callback(
            self._on_audio, RoomEncryptedAudio
        )
        self._client.add_event_callback(
            self._on_file, RoomMessageFile
        )
        self._client.add_event_callback(
            self._on_file, RoomEncryptedFile
        )
        self._client.add_event_callback(
            self._on_invite, InviteMemberEvent
        )

        # E2EE: trust all devices and handle undecryptable events
        logger.info("matrix: e2ee=%s, olm=%s", self.e2ee, self._client.olm is not None)
        if self.e2ee:
            self._client.add_event_callback(
                self._on_megolm, MegolmEvent
            )
            from nio import SyncResponse as _SyncResponse
            # After each sync: upload keys if needed, then trust all devices
            async def _post_sync_e2ee_setup(_resp):
                logger.info("matrix: post-sync e2ee check (olm=%s, upload=%s, claim=%s, query=%s)",
                            self._client.olm is not None,
                            self._client.should_upload_keys,
                            self._client.should_claim_keys,
                            self._client.should_query_keys)
                try:
                    if self._client.should_upload_keys:
                        await self._client.keys_upload()
                        logger.info("matrix: uploaded device keys")
                    if self._client.should_claim_keys:
                        users = self._client.get_users_for_key_claiming()
                        if users:
                            await self._client.keys_claim(users)
                            logger.info("matrix: claimed keys for %d users", len(users))
                    if self._client.should_query_keys:
                        await self._client.keys_query()
                        logger.info("matrix: queried device keys")
                    if hasattr(self._client, 'olm') and self._client.olm:
                        for uid, devices in self._client.device_store.items():
                            for did, olm_device in devices.items():
                                if not self._client.olm.is_device_verified(olm_device):
                                    self._client.verify_device(olm_device)
                                    logger.info("matrix: verified device %s/%s", uid, did)
                except Exception as e:
                    logger.warning("matrix: e2ee post-sync error: %s", e)
            self._client.add_response_callback(_post_sync_e2ee_setup, _SyncResponse)

        # Record startup time — ignore events with server_timestamp before this
        # to prevent replaying historical commands (like /reset) on initial sync.
        import time
        self._start_ts = int(time.time() * 1000)

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

    # ── Cancel support ───────────────────────────────────────────────────

    def _get_cancel_event(self, room_id: str) -> asyncio.Event:
        """Get or create a cancel event for a room."""
        if room_id not in self._cancel_events:
            self._cancel_events[room_id] = asyncio.Event()
        return self._cancel_events[room_id]

    def _reset_cancel_event(self, room_id: str) -> asyncio.Event:
        """Create a fresh (unset) cancel event for a room, replacing any old one."""
        self._cancel_events[room_id] = asyncio.Event()
        return self._cancel_events[room_id]

    # ── Orchestrator event processing ────────────────────────────────────

    async def _process_orchestrator_events(
        self,
        room_id: str,
        content: str | list,
        session_id: str,
    ) -> None:
        """Send content to the Orchestrator and convert events to Matrix messages.

        Collects all TextDelta events into a single message (Matrix doesn't
        stream). Sends typing indicators while processing. Handles cancel,
        error, and final response events.
        """
        if self._orchestrator is None:
            logger.error("matrix: _process_orchestrator_events called but orchestrator is None")
            await self.send("Error: Orchestrator not initialised", room_id=room_id)
            return

        cancel_event = self._reset_cancel_event(room_id)

        try:
            collected = ""
            async for event in self._orchestrator.handle_message(
                session_id=session_id,
                content=content,
                source="matrix",
                cancel_event=cancel_event,
            ):
                if isinstance(event, TextDelta):
                    collected += event.delta

                elif isinstance(event, ToolProgress):
                    logger.debug(
                        "matrix: tool %s %s in %s",
                        event.name, event.status, room_id,
                    )

                elif isinstance(event, FinalResponse):
                    if event.content:
                        await self.send(event.content, room_id=room_id)
                    return

                elif isinstance(event, Cancelled):
                    # Send whatever partial content we have
                    text = event.partial_content or collected
                    if text:
                        await self.send(text + "\n\n⚠️ _Cancelled_", room_id=room_id)
                    else:
                        await self.send("⚠️ Cancelled", room_id=room_id)
                    return

                elif isinstance(event, Error):
                    await self.send(f"Error: {event.message}", room_id=room_id)
                    return

            # If we exit the loop without a terminal event, send collected text
            if collected:
                await self.send(collected, room_id=room_id)

        except Exception as e:
            logger.error("matrix: error processing in %s: %s", room_id, e)
            await self.send(f"Error: {e}", room_id=room_id)
        finally:
            # Clear typing indicator
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    # ── Session ID helper ────────────────────────────────────────────────

    @staticmethod
    def _session_id_for_room(room_id: str) -> str:
        """Deterministic session ID from a Matrix room ID."""
        from march.core.session import deterministic_session_id
        return deterministic_session_id("matrix", room_id)

    # ── Event Callbacks ──────────────────────────────────────────────────

    async def _on_message(self, room: Any, event: Any) -> None:
        """Handle incoming messages from Matrix rooms.

        Sends read receipt + typing indicator immediately, then spawns
        the Orchestrator processing as a background task so the sync loop
        isn't blocked and the ack is visible to the sender right away.
        """
        if not self._orchestrator or not self._running:
            return

        # Ignore our own messages
        if self._client and event.sender == self._client.user_id:
            return

        # Skip messages from before startup (initial sync replays history)
        if self._start_ts and event.server_timestamp < self._start_ts:
            logger.debug("matrix: skipping pre-startup message %s (ts=%d < start=%d)",
                         event.event_id, event.server_timestamp, self._start_ts)
            return

        text = event.body.strip()
        if not text:
            return

        room_id = room.room_id
        logger.info("matrix: message in %s from %s: %s", room_id, event.sender, text[:80])

        # ── Immediate acknowledgment ─────────────────────────────────
        try:
            await self._client.room_read_markers(
                room_id,
                fully_read_event=event.event_id,
                read_event=event.event_id,
            )
        except Exception as e:
            logger.debug("matrix: failed to send read receipt: %s", e)

        try:
            await self._client.room_typing(room_id, typing_state=True, timeout=30000)
        except Exception as e:
            logger.debug("matrix: failed to send typing indicator: %s", e)

        # ── Handle cancel commands ───────────────────────────────────
        text_lower = text.strip().lower()
        if text_lower in ("/stop", "停止"):
            cancel_ev = self._cancel_events.get(room_id)
            if cancel_ev and not cancel_ev.is_set():
                cancel_ev.set()
                await self.send("⚠️ Stopping…", room_id=room_id)
            else:
                await self.send("Nothing to cancel.", room_id=room_id)
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass
            return

        # ── Handle /reset — delegate to Orchestrator ─────────────────
        # /reset MUST complete before any new messages are processed for this
        # room, otherwise a race condition allows the next message to pick up
        # the old (not-yet-cleared) session history.  We await it directly so
        # the sync loop blocks until cleanup is done.
        if text_lower == "/reset":
            await self._handle_matrix_reset(room_id)
            return

        # ── Normal message — background task via Orchestrator ────────
        session_id = self._session_id_for_room(room_id)
        asyncio.create_task(
            self._process_orchestrator_events(room_id, text, session_id)
        )

    async def _handle_matrix_reset(self, room_id: str) -> None:
        """Handle /reset: delegate to Orchestrator.reset_session().

        Also cleans up Matrix-specific resources (attachments on disk).
        """
        import shutil

        if self._orchestrator is None:
            logger.error("matrix: _handle_matrix_reset called but orchestrator is None")
            await self.send("Error: Orchestrator not initialised", room_id=room_id)
            return

        session_id = self._session_id_for_room(room_id)
        cleaned = []

        try:
            # 1. Delegate full session reset to Orchestrator
            #    (clears cache + DB + memory store + session memory files)
            result = await self._orchestrator.reset_session(session_id)
            cleaned.append("session history")
            mem_result = result.get("memory", {})
            db_count = mem_result.get("sqlite_entries", 0) if isinstance(mem_result, dict) else 0
            if db_count:
                cleaned.append(f"{db_count} DB entries")
            if result.get("session_memory_deleted"):
                cleaned.append("session memory")

            # 2. Clean up attachments for THIS session only (Matrix-specific)
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session_id
            if attach_dir.exists():
                file_count = sum(1 for _ in attach_dir.iterdir())
                if file_count > 0:
                    shutil.rmtree(str(attach_dir))
                    cleaned.append(f"{file_count} attachments")

            # 3. Clear cancel event for this room
            self._cancel_events.pop(room_id, None)

            msg = f"✓ Session reset. Cleaned: {', '.join(cleaned)}."
            logger.info("matrix: reset session %s: %s", session_id[:8], msg)
            await self.send(msg, room_id=room_id)

        except Exception as e:
            logger.error("matrix: reset failed: %s", e)
            await self.send(f"Reset error: {e}", room_id=room_id)
        finally:
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_image(self, room: Any, event: Any) -> None:
        """Handle incoming image messages from Matrix rooms.

        Downloads the image, resizes for LLM, and sends as multimodal content
        via the Orchestrator.
        """
        if not self._orchestrator or not self._running:
            return

        # Ignore our own messages
        if self._client and event.sender == self._client.user_id:
            return

        # Skip messages from before startup (initial sync replays history)
        if self._start_ts and event.server_timestamp < self._start_ts:
            return

        room_id = room.room_id
        logger.info("matrix: image in %s from %s: %s", room_id, event.sender, getattr(event, "body", "image"))

        # Immediate acknowledgment
        try:
            await self._client.room_read_markers(
                room_id,
                fully_read_event=event.event_id,
                read_event=event.event_id,
            )
        except Exception:
            pass

        try:
            await self._client.room_typing(room_id, typing_state=True, timeout=30000)
        except Exception:
            pass

        asyncio.create_task(self._process_image(room_id, event))

    async def _process_image(self, room_id: str, event: Any) -> None:
        """Download and process an image message via the Orchestrator."""
        session_id = self._session_id_for_room(room_id)

        try:
            # Download the image from Matrix
            mxc_url = event.url
            if not mxc_url:
                logger.warning("matrix: image event has no URL")
                return

            from nio import DownloadError
            response = await self._client.download(mxc_url)
            if isinstance(response, DownloadError):
                logger.warning("matrix: failed to download image: %s", response)
                return

            raw_bytes = response.body
            mime_type = response.content_type or "image/jpeg"
            filename = getattr(event, "body", "image.jpg") or "image.jpg"

            # Decrypt if E2EE — RoomEncryptedImage has key/hashes/iv directly,
            # RoomMessageImage from non-E2EE rooms does not.
            enc_key = getattr(event, "key", None)
            enc_hashes = getattr(event, "hashes", None)
            enc_iv = getattr(event, "iv", None)
            if enc_key and enc_hashes and enc_iv:
                k = enc_key.get("k", "") if isinstance(enc_key, dict) else ""
                sha256_hash = enc_hashes.get("sha256", "") if isinstance(enc_hashes, dict) else ""
                if k and sha256_hash and enc_iv:
                    try:
                        from nio.crypto import decrypt_attachment
                        raw_bytes = decrypt_attachment(raw_bytes, k, sha256_hash, enc_iv)
                        mime_type = getattr(event, "mimetype", mime_type) or mime_type
                        logger.info("matrix: decrypted image attachment (%dKB)", len(raw_bytes) // 1024)
                    except Exception as e:
                        logger.warning("matrix: image decryption failed: %s", e)
            else:
                # Fallback: check source dict for file encryption info (older events)
                source = getattr(event, "source", {})
                file_info = source.get("content", {}).get("file")
                if file_info and isinstance(file_info, dict):
                    key_info = file_info.get("key", {})
                    iv = file_info.get("iv", "")
                    hashes = file_info.get("hashes", {})
                    sha256_hash = hashes.get("sha256", "")
                    k = key_info.get("k", "")
                    if k and iv and sha256_hash:
                        try:
                            from nio.crypto import decrypt_attachment
                            raw_bytes = decrypt_attachment(raw_bytes, k, sha256_hash, iv)
                            logger.info("matrix: decrypted image attachment (fallback)")
                        except Exception as e:
                            logger.warning("matrix: image decryption failed: %s", e)

            # Resize for LLM (max 512px, JPEG)
            max_dim = 512
            quality = 85
            resized_bytes = raw_bytes
            out_mime = mime_type
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes))
                w, h = img.size
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if max(w, h) > max_dim:
                    ratio = max_dim / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                resized_bytes = buf.getvalue()
                out_mime = "image/jpeg"
                logger.info("matrix: image resized %dx%d → %dx%d (%dKB)",
                            w, h, img.width, img.height, len(resized_bytes) // 1024)
            except ImportError:
                logger.warning("matrix: Pillow not installed, sending raw image")
            except Exception as e:
                logger.warning("matrix: image resize failed: %s", e)

            # Save resized version to disk (not the original)
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session_id
            attach_dir.mkdir(parents=True, exist_ok=True)
            file_path = attach_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path.write_bytes(resized_bytes)

            # Build multimodal content for the Orchestrator
            b64_str = base64.b64encode(resized_bytes).decode("ascii")
            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": out_mime,
                        "data": b64_str,
                    },
                },
                {
                    "type": "text",
                    "text": f"[User sent image: {filename}]",
                },
            ]

            # Check if there's a caption/body text beyond the filename
            body_text = getattr(event, "body", "")
            if body_text and body_text != filename:
                content.append({"type": "text", "text": body_text})

            await self._process_orchestrator_events(room_id, content, session_id)

        except Exception as e:
            logger.error("matrix: error processing image in %s: %s", room_id, e)
            await self.send(f"Error processing image: {e}", room_id=room_id)
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_audio(self, room: Any, event: Any) -> None:
        """Handle incoming audio/voice messages from Matrix rooms.

        Downloads the audio, transcribes via voice_to_text tool, then
        processes the transcribed text via the Orchestrator.
        """
        if not self._orchestrator or not self._running:
            return

        if self._client and event.sender == self._client.user_id:
            return

        # Skip messages from before startup (initial sync replays history)
        if self._start_ts and event.server_timestamp < self._start_ts:
            return

        room_id = room.room_id
        logger.info("matrix: audio in %s from %s: %s", room_id, event.sender, getattr(event, "body", "audio"))

        try:
            await self._client.room_read_markers(
                room_id,
                fully_read_event=event.event_id,
                read_event=event.event_id,
            )
        except Exception:
            pass

        try:
            await self._client.room_typing(room_id, typing_state=True, timeout=60000)
        except Exception:
            pass

        asyncio.create_task(self._process_audio(room_id, event))

    async def _process_audio(self, room_id: str, event: Any) -> None:
        """Download, decrypt, transcribe, and process an audio message via the Orchestrator."""
        session_id = self._session_id_for_room(room_id)

        try:
            mxc_url = event.url
            if not mxc_url:
                logger.warning("matrix: audio event has no URL")
                return

            from nio import DownloadError
            response = await self._client.download(mxc_url)
            if isinstance(response, DownloadError):
                logger.warning("matrix: failed to download audio: %s", response)
                return

            raw_bytes = response.body
            mime_type = getattr(event, "mimetype", None) or response.content_type or "audio/ogg"
            filename = getattr(event, "body", "voice.ogg") or "voice.ogg"

            # Decrypt if E2EE
            enc_key = getattr(event, "key", None)
            enc_hashes = getattr(event, "hashes", None)
            enc_iv = getattr(event, "iv", None)
            if enc_key and enc_hashes and enc_iv:
                k = enc_key.get("k", "") if isinstance(enc_key, dict) else ""
                sha256_hash = enc_hashes.get("sha256", "") if isinstance(enc_hashes, dict) else ""
                if k and sha256_hash and enc_iv:
                    try:
                        from nio.crypto import decrypt_attachment
                        raw_bytes = decrypt_attachment(raw_bytes, k, sha256_hash, enc_iv)
                        logger.info("matrix: decrypted audio attachment (%dKB)", len(raw_bytes) // 1024)
                    except Exception as e:
                        logger.warning("matrix: audio decryption failed: %s", e)

            # Save to disk for transcription
            ext_map = {
                "audio/ogg": ".ogg", "audio/webm": ".webm", "audio/mp4": ".m4a",
                "audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/aac": ".aac",
            }
            ext = ext_map.get(mime_type, ".ogg")
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session_id
            attach_dir.mkdir(parents=True, exist_ok=True)
            voice_path = attach_dir / f"voice_{uuid.uuid4().hex[:8]}{ext}"
            voice_path.write_bytes(raw_bytes)

            logger.info("matrix: audio saved %s (%dKB)", voice_path.name, len(raw_bytes) // 1024)

            # Transcribe using voice_to_text tool
            transcription = ""
            try:
                from march.core.message import ToolCall as MarchToolCall

                vtt_args: dict[str, Any] = {"path": str(voice_path)}
                # Read voice_to_text config if available
                if self._agent and hasattr(self._agent, "app") and self._agent.app:
                    app = self._agent.app
                    if hasattr(app, "config") and app.config:
                        vtt_cfg = getattr(app.config.tools, "voice_to_text", None)
                        if vtt_cfg:
                            if getattr(vtt_cfg, "model", ""):
                                vtt_args["model_size"] = vtt_cfg.model
                            if getattr(vtt_cfg, "language", ""):
                                vtt_args["language"] = vtt_cfg.language

                tool_call = MarchToolCall(
                    id=f"voice-{uuid.uuid4().hex[:8]}",
                    name="voice_to_text",
                    args=vtt_args,
                )
                result = await self._agent.tools.execute(tool_call)
                if result.is_error:
                    logger.error("matrix: voice transcription failed: %s", result.error)
                else:
                    transcription = result.content.strip()
            except Exception as e:
                logger.error("matrix: voice transcription error: %s", e)

            if not transcription:
                await self.send("⚠️ Could not transcribe voice message", room_id=room_id)
                return

            logger.info("matrix: voice transcribed (%d chars): %s", len(transcription), transcription[:80])

            # Process transcribed text via Orchestrator
            await self._process_orchestrator_events(room_id, transcription, session_id)

        except Exception as e:
            logger.error("matrix: error processing audio in %s: %s", room_id, e)
            await self.send(f"Error processing voice message: {e}", room_id=room_id)
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_file(self, room: Any, event: Any) -> None:
        """Handle incoming file attachments from Matrix rooms.

        Downloads the file, processes based on type (PDF → extract text,
        text files → read content, binary → note), then sends to Orchestrator.
        """
        if not self._orchestrator or not self._running:
            return

        if self._client and event.sender == self._client.user_id:
            return

        # Skip messages from before startup (initial sync replays history)
        if self._start_ts and event.server_timestamp < self._start_ts:
            return

        room_id = room.room_id
        logger.info("matrix: file in %s from %s: %s", room_id, event.sender, getattr(event, "body", "file"))

        try:
            await self._client.room_read_markers(
                room_id,
                fully_read_event=event.event_id,
                read_event=event.event_id,
            )
        except Exception:
            pass

        try:
            await self._client.room_typing(room_id, typing_state=True, timeout=60000)
        except Exception:
            pass

        asyncio.create_task(self._process_file(room_id, event))

    async def _process_file(self, room_id: str, event: Any) -> None:
        """Download, decrypt, and process a file attachment via the Orchestrator."""
        session_id = self._session_id_for_room(room_id)

        try:
            mxc_url = event.url
            if not mxc_url:
                logger.warning("matrix: file event has no URL")
                return

            from nio import DownloadError
            response = await self._client.download(mxc_url)
            if isinstance(response, DownloadError):
                logger.warning("matrix: failed to download file: %s", response)
                return

            raw_bytes = response.body
            mime_type = getattr(event, "mimetype", None) or response.content_type or "application/octet-stream"
            filename = getattr(event, "body", "file") or "file"

            # Decrypt if E2EE
            enc_key = getattr(event, "key", None)
            enc_hashes = getattr(event, "hashes", None)
            enc_iv = getattr(event, "iv", None)
            if enc_key and enc_hashes and enc_iv:
                k = enc_key.get("k", "") if isinstance(enc_key, dict) else ""
                sha256_hash = enc_hashes.get("sha256", "") if isinstance(enc_hashes, dict) else ""
                if k and sha256_hash and enc_iv:
                    try:
                        from nio.crypto import decrypt_attachment
                        raw_bytes = decrypt_attachment(raw_bytes, k, sha256_hash, enc_iv)
                        logger.info("matrix: decrypted file attachment (%dKB)", len(raw_bytes) // 1024)
                    except Exception as e:
                        logger.warning("matrix: file decryption failed: %s", e)

            # Save to disk
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session_id
            attach_dir.mkdir(parents=True, exist_ok=True)
            file_path = attach_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path.write_bytes(raw_bytes)
            size_kb = len(raw_bytes) // 1024

            logger.info("matrix: file saved %s (%s, %dKB)", file_path.name, mime_type, size_kb)

            # Process based on file type
            content_text = ""

            # PDF
            if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
                content_text = self._extract_pdf_text(raw_bytes, filename, size_kb)

            # Text-based files
            elif self._is_text_file(filename, mime_type):
                try:
                    text = raw_bytes.decode("utf-8")
                    # Truncate if too long (keep first 8000 chars for context)
                    if len(text) > 8000:
                        content_text = (
                            f"[File: {filename} ({mime_type}, {size_kb}KB) — truncated to first 8000 chars]\n\n"
                            f"{text[:8000]}\n\n[... truncated ...]"
                        )
                    else:
                        content_text = f"[File: {filename} ({mime_type}, {size_kb}KB)]\n\n{text}"
                except UnicodeDecodeError:
                    content_text = f"[File: {filename} ({mime_type}, {size_kb}KB) — binary, could not decode as text]"

            # Binary/unknown
            else:
                content_text = f"[File: {filename} ({mime_type}, {size_kb}KB) — binary file saved to disk]"

            # Send to Orchestrator
            await self._process_orchestrator_events(room_id, content_text, session_id)

        except Exception as e:
            logger.error("matrix: error processing file in %s: %s", room_id, e)
            await self.send(f"Error processing file: {e}", room_id=room_id)
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    @staticmethod
    def _extract_pdf_text(raw_bytes: bytes, filename: str, size_kb: int) -> str:
        """Extract text from a PDF using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = []
            for page in doc:
                text = page.get_text().strip()
                if text:
                    pages.append(text)
            num_pages = len(doc)
            doc.close()

            if pages:
                full_text = "\n\n".join(pages)
                # Truncate if too long
                if len(full_text) > 12000:
                    full_text = full_text[:12000] + "\n\n[... truncated ...]"
                return f"[PDF: {filename} ({num_pages} pages, {size_kb}KB)]\n\n{full_text}"
            return f"[PDF: {filename} ({num_pages} pages, {size_kb}KB) — no extractable text]"
        except ImportError:
            return f"[PDF: {filename} ({size_kb}KB) — PyMuPDF not installed, cannot extract text]"
        except Exception as e:
            return f"[PDF: {filename} ({size_kb}KB) — extraction failed: {e}]"

    @staticmethod
    def _is_text_file(filename: str, mime_type: str) -> bool:
        """Check if a file is likely a text file."""
        text_extensions = {
            ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".csv",
            ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
            ".sh", ".bash", ".zsh", ".conf", ".cfg", ".ini",
            ".xml", ".svg", ".sql", ".rs", ".go", ".java", ".c", ".cpp",
            ".h", ".hpp", ".rb", ".lua", ".r", ".swift", ".kt",
            ".dockerfile", ".env", ".gitignore", ".log",
        }
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()
        return (
            mime_type.startswith("text/")
            or mime_type in ("application/json", "application/xml", "application/yaml")
            or ext in text_extensions
        )

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

    async def _on_megolm(self, room: Any, event: Any) -> None:
        """Handle undecryptable Megolm events — request keys."""
        if not self._client:
            return
        logger.warning("matrix: could not decrypt event %s in %s (session_id=%s)",
                        event.event_id, room.room_id,
                        getattr(event, "session_id", "?"))

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
                if creds.get("user_id") and (creds.get("password") or creds.get("access_token")):
                    self.user_id = creds["user_id"]
                    if creds.get("password"):
                        self.password = creds["password"]
                    if creds.get("access_token"):
                        self.access_token = creds["access_token"]
                    return creds
            except (json.JSONDecodeError, KeyError):
                pass

        # Use configured credentials if available (access_token or password)
        if self.user_id and (self.access_token or (self.password and self.password != "auto")):
            creds = {
                "user_id": self.user_id,
                "homeserver": homeserver,
            }
            if self.access_token:
                creds["access_token"] = self.access_token
            if self.password:
                creds["password"] = self.password
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
        """Convert Markdown to HTML for Matrix formatted_body.

        Uses the Python `markdown` library with extensions for
        fenced code blocks, tables, etc. Falls back to basic
        HTML-escaped text if the library is unavailable.

        Important: Element (and other Matrix clients) sanitize HTML
        aggressively — they strip all `style` attributes (except on
        img), and only allow `class` values starting with `language-`
        on `<code>` tags. So we avoid codehilite (inline styles) and
        nl2br (breaks normal paragraph spacing).
        """
        try:
            import markdown
            import re

            html_output = markdown.markdown(
                text,
                extensions=[
                    "fenced_code",
                    "tables",
                    "sane_lists",
                ],
            )

            # fenced_code generates <code class="language-python"> which
            # Element allows. But if it generates other class formats,
            # normalize them to language- prefix so Element doesn't strip them.
            def _fix_code_class(m: re.Match) -> str:
                cls = m.group(1)
                # Already has language- prefix — keep as-is
                if cls.startswith("language-"):
                    return m.group(0)
                # Convert e.g. class="python" to class="language-python"
                return f'<code class="language-{cls}">'

            html_output = re.sub(
                r'<code class="([^"]+)">',
                _fix_code_class,
                html_output,
            )

            return html_output
        except ImportError:
            logger.warning("markdown library not installed, falling back to plain text HTML")
            import html as html_mod
            escaped = html_mod.escape(text)
            # Preserve paragraph breaks (double newline) and line breaks
            escaped = escaped.replace("\n\n", "</p><p>")
            escaped = escaped.replace("\n", "<br>")
            return f"<p>{escaped}</p>"
