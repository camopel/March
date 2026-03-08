"""Matrix channel for March.

Bot joins Matrix rooms, sends/receives messages. One room = one session.
Uses matrix-nio for Matrix client functionality.

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
        access_token: str = "",
        rooms: list[str] | None = None,
        e2ee: bool = False,
        auto_setup: bool = True,
        display_name: str = "March",
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
        self._client: Any = None  # nio.AsyncClient
        self._sessions: dict[str, Session] = {}  # room_id → session
        self._running = False

    async def start(self, agent: "Agent", **kwargs: Any) -> None:
        """Start the Matrix client, join rooms, and begin listening."""
        self._agent = agent
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
        """Handle incoming messages from Matrix rooms.

        Sends read receipt + typing indicator immediately, then spawns
        the LLM processing as a background task so the sync loop isn't
        blocked and the ack is visible to the sender right away.
        """
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

        # ── Immediate acknowledgment ─────────────────────────────────
        # Send read receipt + typing indicator BEFORE processing so the
        # sender knows the message was received instantly.
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

        # ── Spawn LLM processing as background task ──────────────────
        # This returns control to the sync loop immediately so the read
        # receipt and typing indicator actually get flushed to the server
        # without waiting for the full LLM response.

        # Handle /reset directly — clean up everything
        if text.strip().lower() == "/reset":
            asyncio.create_task(self._handle_matrix_reset(room_id))
        else:
            asyncio.create_task(self._process_message(room_id, text, event))

    async def _handle_matrix_reset(self, room_id: str) -> None:
        """Handle /reset: clear session history, attachments, and DB entries."""
        import shutil

        session = self._get_or_create_session(room_id)
        session_id = session.id
        cleaned = []

        try:
            # 1. Clear in-memory session
            session.clear()
            cleaned.append("history")

            # 2. Clear session from unified SessionStore (if it exists there)
            if self._agent and hasattr(self._agent, "memory"):
                result = await self._agent.memory.reset_session(session_id)
                db_count = result.get("sqlite_entries", 0)
                if db_count:
                    cleaned.append(f"{db_count} DB entries")

            # 3. Delete session memory files (facts.md, plan.md, etc.)
            try:
                from march.core.compaction import delete_session_memory
                if delete_session_memory(session_id):
                    cleaned.append("session memory")
            except Exception as e:
                logger.warning("matrix: reset - failed to delete session memory: %s", e)

            # 4. Clean up attachments for THIS session only
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session_id
            if attach_dir.exists():
                file_count = sum(1 for _ in attach_dir.iterdir())
                if file_count > 0:
                    shutil.rmtree(str(attach_dir))
                    cleaned.append(f"{file_count} attachments")
                    cleaned.append(f"{file_count} attachments")

            # 5. Remove from in-memory sessions dict so next message creates fresh
            self._sessions.pop(room_id, None)

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

    async def _process_message(self, room_id: str, text: str, event: Any) -> None:
        """Process a message with the LLM in the background."""
        session = self._get_or_create_session(room_id)

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
        finally:
            # ── Clear typing indicator ───────────────────────────────
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_image(self, room: Any, event: Any) -> None:
        """Handle incoming image messages from Matrix rooms.

        Downloads the image, resizes for LLM, and sends as multimodal content.
        """
        if not self._agent or not self._running:
            return

        # Ignore our own messages
        if self._client and event.sender == self._client.user_id:
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
        """Download and process an image message with the LLM."""
        session = self._get_or_create_session(room_id)

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
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session.id
            attach_dir.mkdir(parents=True, exist_ok=True)
            file_path = attach_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path.write_bytes(resized_bytes)

            # Build multimodal content for the agent
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

            from march.core.agent import AgentResponse
            collected = ""
            async for item in self._agent.run_stream(content, session):
                if isinstance(item, AgentResponse):
                    break
                if hasattr(item, "delta") and item.delta:
                    collected += item.delta

            if collected:
                await self.send(collected, room_id=room_id)

        except Exception as e:
            logger.error("matrix: error processing image in %s: %s", room_id, e)
            await self.send(f"Error processing image: {e}", room_id=room_id)
        finally:
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_audio(self, room: Any, event: Any) -> None:
        """Handle incoming audio/voice messages from Matrix rooms.

        Downloads the audio, transcribes via voice_to_text tool, then
        processes the transcribed text as a regular message.
        """
        if not self._agent or not self._running:
            return

        if self._client and event.sender == self._client.user_id:
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
        """Download, decrypt, transcribe, and process an audio message."""
        session = self._get_or_create_session(room_id)

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
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session.id
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

            # Process transcribed text as a regular message
            from march.core.agent import AgentResponse
            collected = ""
            async for item in self._agent.run_stream(transcription, session):
                if isinstance(item, AgentResponse):
                    break
                if hasattr(item, "delta") and item.delta:
                    collected += item.delta

            if collected:
                await self.send(collected, room_id=room_id)

        except Exception as e:
            logger.error("matrix: error processing audio in %s: %s", room_id, e)
            await self.send(f"Error processing voice message: {e}", room_id=room_id)
        finally:
            try:
                await self._client.room_typing(room_id, typing_state=False)
            except Exception:
                pass

    async def _on_file(self, room: Any, event: Any) -> None:
        """Handle incoming file attachments from Matrix rooms.

        Downloads the file, processes based on type (PDF → extract text,
        text files → read content, binary → note), then sends to agent.
        """
        if not self._agent or not self._running:
            return

        if self._client and event.sender == self._client.user_id:
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
        """Download, decrypt, and process a file attachment."""
        session = self._get_or_create_session(room_id)

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
            attach_dir = Path.home() / ".march" / "attachments" / "matrix" / session.id
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

            # Send to agent
            from march.core.agent import AgentResponse
            collected = ""
            async for item in self._agent.run_stream(content_text, session):
                if isinstance(item, AgentResponse):
                    break
                if hasattr(item, "delta") and item.delta:
                    collected += item.delta

            if collected:
                await self.send(collected, room_id=room_id)

        except Exception as e:
            logger.error("matrix: error processing file in %s: %s", room_id, e)
            await self.send(f"Error processing file: {e}", room_id=room_id)
        finally:
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
        """Simple markdown-to-HTML conversion for Matrix messages."""
        # Basic conversion: just use the text as-is for now
        # A proper implementation would use a markdown parser
        import html
        escaped = html.escape(text)
        # Convert basic markdown
        lines = escaped.split("\n")
        result = "<br>".join(lines)
        return result
