"""WSProxyPlugin — Embedded HTTP/WS server for the frontend chat.

Owns the conversation SQLite DB, runs an aiohttp server, handles image
resizing, PDF extraction, voice transcription, and stream persistence
(even if frontend disconnects mid-stream).

All settings are read from config.yaml under `plugins.ws_proxy`.

REST API:
  GET    /health
  GET    /sessions
  POST   /sessions
  DELETE /sessions/{id}
  PUT    /sessions/{id}
  GET    /sessions/{id}/history
  WS     /ws/{id}
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from march.logging import get_logger
from march.logging.logger import MetricsLogger
from march.plugins._base import Plugin

if TYPE_CHECKING:
    from march.app import MarchApp
    from march.core.agent import Agent

# Subsystem loggers
logger = get_logger("march.ws_proxy", subsystem="ws_proxy")
stream_log = get_logger("march.stream", subsystem="stream")


# ── Default config values (overridden by config.yaml) ────────────────────────

DEFAULTS = {
    "port": 8101,
    "host": "0.0.0.0",
    "cors_origins": [],
    "db_path": "~/.march/march.db",
    "max_image_dimension": 512,
    "image_quality": 85,
    "message_buffer_seconds": 3.0,
    "max_message_size": 20 * 1024 * 1024,  # 20MB
    "stream_drain_timeout": 120,
    "max_queue_size": 100,
    "context_keep_recent": 20,      # Messages to keep in active context
    "compaction_threshold": 40,     # Compact when total messages exceed this
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resize_image(
    raw_bytes: bytes, mime_type: str, max_dim: int, quality: int
) -> tuple[bytes, str]:
    """Resize image for LLM: max_dim px on longest side, RGB, JPEG."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw_bytes))
        w, h = img.size

        # Always convert to RGB (no alpha)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if either dimension exceeds limit
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info("image resized", original_w=w, original_h=h, new_w=new_w, new_h=new_h)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        result = buf.getvalue()
        logger.info(
            "image output",
            width=img.width, height=img.height,
            size_kb=len(result) // 1024, quality=quality,
        )
        return result, "image/jpeg"
    except Exception as e:
        logger.warning("image resize failed", error=str(e))
        return raw_bytes, mime_type


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
        doc.close()
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not installed, cannot extract PDF text")
    except Exception as e:
        logger.warning("PDF text extraction failed", error=str(e))
    return ""


def _build_attachment_content(
    raw_bytes: bytes,
    filename: str,
    mime_type: str,
    tmp_path: str,
    max_image_dim: int,
    image_quality: int,
) -> str | list:
    """Build LLM-ready content from an attachment based on its type.

    DEPRECATED: Kept for reference. Use _process_attachment() instead.
    """
    raise NotImplementedError("Use _process_attachment() instead")


async def _summarize_with_llm(
    app: "MarchApp",
    content: str | list,
    prompt: str,
    max_tokens: int = 500,
) -> str:
    """Use the LLM to summarize content. Returns summary text or empty string on failure.

    For text content, sends a simple user message.
    For image content, sends a multimodal message (list of content blocks).
    """
    try:
        provider = await app.agent.llm.route()
        if isinstance(content, list):
            # Multimodal content (image + text prompt)
            messages = [{"role": "user", "content": content}]
        else:
            # Text content — combine with prompt
            messages = [{"role": "user", "content": f"{prompt}\n\n{content}"}]
        response = await provider.converse(
            messages=messages,
            system="You are a concise summarizer. Output only the summary, nothing else.",
            max_tokens=max_tokens,
        )
        result = response.content.strip() if response and response.content else ""
        logger.info(
            "llm.summarize",
            content_type="multimodal" if isinstance(content, list) else "text",
            result_length=len(result),
            result_preview=result[:200] if result else "(empty)",
        )
        return result
    except Exception as e:
        logger.warning("llm.summarize_failed", error=str(e))
        return ""


async def _process_attachment(
    raw_bytes: bytes,
    filename: str,
    mime_type: str,
    file_path: str,
    max_image_dim: int,
    image_quality: int,
    app: "MarchApp",
    summary_max_tokens: int = 500,
    summary_chunk_size: int = 4000,
) -> str | list:
    """Process an attachment: save to disk, return content for the agent.

    Images → multimodal content (image block + text label) sent directly to LLM.
    PDFs/text → summarized via LLM, agent gets text note + summary.
    Binary → agent gets text note with path.
    """
    size_kb = len(raw_bytes) // 1024

    # ── Images → inline multimodal (no separate vision call) ──────────────
    if mime_type.startswith("image/"):
        resized_bytes, out_mime = _resize_image(
            raw_bytes, mime_type, max_image_dim, image_quality
        )
        # Overwrite the tmp file with the resized version
        try:
            Path(file_path).write_bytes(resized_bytes)
        except Exception as e:
            logger.warning("attachment.resize_save_failed", error=str(e))

        resized_kb = len(resized_bytes) // 1024
        b64_str = base64.b64encode(resized_bytes).decode("ascii")

        # Return multimodal content — agent sees the image directly
        return [
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
                "text": f"[User attached image: {filename} ({resized_kb}KB, saved to {file_path})]",
            },
        ]

    # ── PDFs ──────────────────────────────────────────────────────────────
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        text = _extract_pdf_text(raw_bytes)
        # Count pages
        pages = 0
        try:
            import fitz
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = len(doc)
            doc.close()
        except Exception:
            pass

        if text:
            header = f"[media attached: {file_path} (PDF, {pages} pages, {size_kb}KB)]"
            summary = await _chunked_summarize(
                app, text, summary_max_tokens, summary_chunk_size
            )
            if summary:
                return f"{header}\n\nSummary: {summary}"
            return (
                f"{header}\n\n"
                "Could not generate summary. Use the pdf tool to analyze this file."
            )
        header = f"[media attached: {file_path} (PDF, {size_kb}KB)]"
        return (
            f"{header}\n\n"
            "Could not extract text. Use the pdf tool to analyze this file."
        )

    # ── Text files ────────────────────────────────────────────────────────
    text_extensions = {
        ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".csv",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
        ".sh", ".bash", ".zsh", ".fish", ".conf", ".cfg", ".ini",
        ".xml", ".svg", ".sql", ".rs", ".go", ".java", ".c", ".cpp",
        ".h", ".hpp", ".rb", ".lua", ".r", ".swift", ".kt",
        ".dockerfile", ".env", ".gitignore", ".log",
    }
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    is_text = (
        mime_type.startswith("text/")
        or mime_type in ("application/json", "application/xml", "application/yaml")
        or ext in text_extensions
    )

    if is_text:
        try:
            text_content = raw_bytes.decode("utf-8")
            header = f"[media attached: {file_path} ({mime_type}, {size_kb}KB)]"
            summary = await _chunked_summarize(
                app, text_content, summary_max_tokens, summary_chunk_size
            )
            if summary:
                return f"{header}\n\nSummary: {summary}"
            return (
                f"{header}\n\n"
                "Could not generate summary. Use the read tool to view the full content."
            )
        except UnicodeDecodeError:
            pass

    # ── Binary/unknown ────────────────────────────────────────────────────
    return (
        f"[media attached: {file_path} ({mime_type}, {size_kb}KB)]\n\n"
        "Binary file. Use exec to inspect if needed."
    )


async def _chunked_summarize(
    app: "MarchApp",
    text: str,
    max_tokens: int = 500,
    chunk_size: int = 4000,
) -> str:
    """Summarize text, chunking if necessary.

    If text <= chunk_size*2: single-pass summary.
    If text > chunk_size*2: chunk → summarize each → combine.
    """
    if len(text) <= chunk_size * 2:
        prompt = (
            "Summarize the following document concisely. "
            "Focus on key topics, main points, and important details. "
            "Keep it under 200 words.\n\nDocument:"
        )
        return await _summarize_with_llm(app, text, prompt, max_tokens=max_tokens)

    # Chunk the text
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize this section concisely in 2-3 sentences:"
        summary = await _summarize_with_llm(app, chunk, prompt, max_tokens=200)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    # Combine chunk summaries into final summary
    combined = "\n\n".join(chunk_summaries)
    prompt = (
        "Combine these section summaries into a coherent overall summary "
        "under 200 words:"
    )
    return await _summarize_with_llm(app, combined, prompt, max_tokens=max_tokens)


# ── Database Layer ────────────────────────────────────────────────────────────

class ChatDB:
    """Adapter that wraps the unified SessionStore for ws_proxy compatibility.

    Provides the same API that ws_proxy expects, but delegates to SessionStore.
    This allows ws_proxy to share the same DB/tables as all other channels.
    """

    def __init__(self, store: Any = None, db_path: Path | None = None) -> None:
        self._store = store  # SessionStore instance
        self.db_path = db_path or Path("~/.march/march.db")

    @classmethod
    async def from_session_store(cls, store: Any) -> "ChatDB":
        """Create a ChatDB adapter from an existing SessionStore."""
        adapter = cls(store=store)
        return adapter

    async def initialize(self) -> None:
        """Initialize — if no store provided, create our own SessionStore."""
        if self._store is None:
            from march.core.session import SessionStore
            self._store = SessionStore(db_path=self.db_path)
            await self._store.initialize()
        logger.info("database initialized", path=str(self._store.db_path))

    async def close(self) -> None:
        """Close — only if we own the store (not shared)."""
        pass  # Shared store is closed by MarchApp

    async def list_sessions(self) -> list[dict]:
        return await self._store.list_sessions()

    async def create_session(self, name: str, description: str = "") -> dict:
        session = await self._store.create_session(
            source_type="ws", source_id=name,
            name=name, metadata={"description": description},
        )
        return {
            "id": session.id,
            "name": name,
            "description": description,
            "created_at": session.created_at,
        }

    async def delete_session(self, session_id: str) -> bool:
        try:
            await self._store.delete_session(session_id)
            return True
        except Exception:
            return False

    async def rename_session(self, session_id: str, name: str) -> bool:
        session = await self._store.get_session(session_id)
        if not session:
            return False
        session.name = name
        await self._store.update_session(session)
        return True

    async def get_history(self, session_id: str) -> dict | None:
        session = await self._store.get_session(session_id)
        if not session:
            return None
        messages = await self._store.get_messages_raw(session_id)
        # Adapt to ws_proxy's expected format
        for m in messages:
            # Ensure tool_calls is parsed
            tc = m.get("tool_calls", "[]")
            if isinstance(tc, str):
                try:
                    m["tool_calls"] = json.loads(tc)
                except (json.JSONDecodeError, TypeError):
                    m["tool_calls"] = []
            # Map attachments to image_data for dashboard compatibility
            m["image_data"] = None
            att = m.get("attachments", "[]")
            if isinstance(att, str):
                try:
                    att_list = json.loads(att)
                    if att_list:
                        m["image_data"] = att_list  # Dashboard can handle refs
                except (json.JSONDecodeError, TypeError):
                    pass
        return {
            "session": {
                "id": session.id,
                "name": session.name,
                "description": session.metadata.get("description", ""),
                "rolling_summary": session.rolling_summary,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "is_active": True,
            },
            "messages": messages,
        }

    async def session_exists(self, session_id: str) -> bool:
        session = await self._store.get_session(session_id)
        return session is not None

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list | None = None,
        image_data: str | None = None,
        summary: str = "",
    ) -> str:
        from march.core.message import Message
        msg = Message(role=role, content=content)
        if tool_calls:
            from march.core.message import ToolCall
            msg.tool_calls = [
                ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
                for tc in tool_calls
            ]
        # Store image_data as attachment ref if provided
        attachments = []
        if image_data:
            attachments = [{"type": "image_data", "data": image_data}]
        msg.metadata = {"summary": summary} if summary else {}
        return await self._store.add_message(session_id, msg, attachments=attachments)

    async def save_draft(self, session_id: str, draft_id: str, content: str) -> None:
        await self._store.save_draft(session_id, draft_id, content)

    async def finalize_draft(self, draft_id: str, content: str,
                              tool_calls: list | None = None, summary: str = "") -> None:
        await self._store.finalize_draft(
            draft_id, content, tool_calls,
            metadata={"summary": summary} if summary else None,
        )

    async def clear_session_messages(self, session_id: str) -> None:
        await self._store.clear_session(session_id)

    async def get_rolling_summary(self, session_id: str) -> str:
        return await self._store.get_rolling_summary(session_id)

    async def update_rolling_summary(self, session_id: str, summary: str) -> None:
        await self._store.update_rolling_summary(session_id, summary)

    async def get_message_count(self, session_id: str) -> int:
        return await self._store.get_message_count(session_id)

    async def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        """Get the N most recent messages for LLM context."""
        messages = await self._store.get_messages_raw(session_id, limit=limit)
        # get_messages_raw returns chronological; we need most recent N
        # So get all and take last N
        return messages[-limit:] if len(messages) > limit else messages

    async def get_messages_before(self, session_id: str, offset: int, limit: int) -> list[dict]:
        """Get messages older than the recent N (for summarization)."""
        all_msgs = await self._store.get_messages_raw(session_id)
        # Skip the most recent `offset` messages, take `limit` before that
        if offset >= len(all_msgs):
            return []
        end = len(all_msgs) - offset
        start = max(0, end - limit)
        return all_msgs[start:end]


# ── Plugin ────────────────────────────────────────────────────────────────────

class WSProxyPlugin(Plugin):
    """Embedded HTTP/WS server for the frontend chat with DB persistence.

    Configuration (config.yaml → plugins.ws_proxy):
      port: 8101                    # HTTP/WS server port
      host: "0.0.0.0"              # Bind address
      cors_origins:                 # Allowed CORS origins
        - "https://your-host:3443"
      db_path: "~/.march/ws_proxy.db"  # SQLite DB path
      max_image_dimension: 512      # Max px on longest side
      image_quality: 85             # JPEG quality (1-100)
      message_buffer_seconds: 3.0   # Queue drain buffer
      max_message_size: 20971520    # Max WS message size (20MB)
      stream_drain_timeout: 120     # Seconds to wait for stream drain
    """

    name = "ws_proxy"
    version = "1.0.0"
    priority = 50

    def __init__(self) -> None:
        super().__init__()
        self._agent: Agent | None = None
        self._app_ref: MarchApp | None = None
        self._db: ChatDB | None = None
        self._site: Any = None
        self._runner: Any = None
        self._march_sessions: dict[str, Any] = {}
        self._stream_buffers: dict[str, _StreamBuffer] = {}  # session_id → buffer
        self._metrics = MetricsLogger.get()
        # Config values (populated in on_start from config.yaml)
        self._port: int = DEFAULTS["port"]
        self._host: str = DEFAULTS["host"]
        self._cors_origins: list[str] = DEFAULTS["cors_origins"]
        self._max_image_dim: int = DEFAULTS["max_image_dimension"]
        self._image_quality: int = DEFAULTS["image_quality"]
        self._buffer_seconds: float = DEFAULTS["message_buffer_seconds"]
        self._max_msg_size: int = DEFAULTS["max_message_size"]
        self._drain_timeout: int = DEFAULTS["stream_drain_timeout"]
        self._max_queue_size: int = DEFAULTS["max_queue_size"]
        self._keep_recent: int = DEFAULTS["context_keep_recent"]
        self._max_upload_bytes: int = 20 * 1024 * 1024
        self._active_connections: dict[str, Any] = {}  # session_id → WebSocketResponse
        self._summary_max_tokens: int = 500
        self._summary_chunk_size: int = 4000

    def _load_config(self, app: Any) -> None:
        """Load plugin config from app.config.plugins.ws_proxy."""
        cfg = None
        if hasattr(app, "config") and app.config:
            plugins_cfg = getattr(app.config, "plugins", None)
            if plugins_cfg:
                cfg = getattr(plugins_cfg, "ws_proxy", None)

        if cfg and hasattr(cfg, "port"):
            # Pydantic model — read attributes directly
            self._port = cfg.port
            self._host = cfg.host
            self._cors_origins = list(cfg.cors_origins)
            self._max_image_dim = cfg.max_image_dimension
            self._image_quality = cfg.image_quality
            self._buffer_seconds = cfg.message_buffer_seconds
            self._max_msg_size = cfg.max_message_size
            self._drain_timeout = cfg.stream_drain_timeout
            self._db_path = Path(cfg.db_path).expanduser()
            if hasattr(cfg, "max_upload_bytes"):
                self._max_upload_bytes = cfg.max_upload_bytes
            if hasattr(cfg, "summary_max_tokens"):
                self._summary_max_tokens = cfg.summary_max_tokens
            if hasattr(cfg, "summary_chunk_size"):
                self._summary_chunk_size = cfg.summary_chunk_size
        else:
            # Fallback to defaults
            self._db_path = Path(DEFAULTS["db_path"]).expanduser()

        logger.info(
            "config loaded",
            port=self._port, host=self._host, cors_origins=self._cors_origins,
            db_path=str(self._db_path), max_image_dim=self._max_image_dim,
            image_quality=self._image_quality,
        )

    async def on_start(self, app: Any) -> None:
        """Start the HTTP/WS server and initialize the DB."""
        import aiohttp.web as web

        self._app_ref = app
        self._agent = app.agent

        # Load config
        self._load_config(app)

        # Use the app's shared SessionStore via ChatDB adapter
        if app.session_store:
            self._db = ChatDB(store=app.session_store)
        else:
            # Fallback: create our own store
            self._db = ChatDB(db_path=self._db_path)
            await self._db.initialize()

        # Build aiohttp app
        webapp = web.Application()
        webapp["plugin"] = self
        webapp.router.add_get("/health", self._handle_health)
        webapp.router.add_get("/sessions", self._handle_list_sessions)
        webapp.router.add_post("/sessions", self._handle_create_session)
        webapp.router.add_delete("/sessions/{session_id}", self._handle_delete_session)
        webapp.router.add_put("/sessions/{session_id}", self._handle_rename_session)
        webapp.router.add_get(
            "/sessions/{session_id}/history", self._handle_get_history
        )
        webapp.router.add_get("/ws/{session_id}", self._handle_ws)

        # CORS middleware
        cors_origins = self._cors_origins
        webapp.middlewares.append(self._make_cors_middleware(cors_origins))

        # Start server
        self._runner = web.AppRunner(webapp)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "server started",
            host=self._host, port=self._port, cors_origins=cors_origins,
        )

    async def on_shutdown(self, app: Any) -> None:
        """Stop the HTTP/WS server, flush in-flight streams, and close the DB."""
        # Flush all in-flight streaming drafts to DB before shutdown
        await self._flush_all_streams()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._db:
            await self._db.close()
        logger.info("server stopped")

    async def _flush_all_streams(self) -> None:
        """Save all in-flight streaming content to DB as drafts.

        Called before shutdown/restart to prevent data loss.
        """
        flushed = 0
        for session_id, buf in self._stream_buffers.items():
            if buf.streaming and buf.collected and buf.draft_id:
                try:
                    await self._db.save_draft(session_id, buf.draft_id, buf.collected)
                    flushed += 1
                    stream_log.info(
                        "draft flushed on shutdown",
                        session_id=session_id,
                        content_length=len(buf.collected),
                        chunks=buf.next_id,
                    )
                except Exception as e:
                    stream_log.warning("draft flush failed",
                                       session_id=session_id, error=str(e),
                                       action="data may be lost")
        if flushed:
            stream_log.info("shutdown flush complete", drafts_saved=flushed)

    # ── CORS Middleware ───────────────────────────────────────────────────

    @staticmethod
    def _make_cors_middleware(allowed_origins: list[str]) -> Any:
        """Create an aiohttp CORS middleware with configurable origins."""
        import aiohttp.web as web

        origins_set = set(allowed_origins)

        @web.middleware
        async def cors_middleware(
            request: web.Request, handler: Any
        ) -> web.StreamResponse:
            origin = request.headers.get("Origin", "")

            if request.method == "OPTIONS":
                resp = web.Response(status=204)
            else:
                try:
                    resp = await handler(request)
                except web.HTTPException as e:
                    resp = e

            if origin in origins_set:
                resp.headers["Access-Control-Allow-Origin"] = origin
            elif allowed_origins:
                # Fallback to first origin for non-browser requests
                resp.headers["Access-Control-Allow-Origin"] = allowed_origins[0]

            resp.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            resp.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            resp.headers["Access-Control-Max-Age"] = "3600"
            return resp

        return cors_middleware

    # ── REST Handlers ─────────────────────────────────────────────────────

    async def _handle_health(self, request: Any) -> Any:
        import aiohttp.web as web

        return web.json_response({"status": "ok", "agent": self._agent is not None})

    async def _handle_list_sessions(self, request: Any) -> Any:
        import aiohttp.web as web

        sessions = await self._db.list_sessions()
        return web.json_response({"sessions": sessions})

    async def _handle_create_session(self, request: Any) -> Any:
        import aiohttp.web as web

        body = await request.json()
        name = body.get("name", "New Chat")
        description = body.get("description", "")
        result = await self._db.create_session(name, description)
        return web.json_response(result, status=201)

    async def _handle_delete_session(self, request: Any) -> Any:
        import aiohttp.web as web

        session_id = request.match_info["session_id"]
        deleted = await self._db.delete_session(session_id)
        if not deleted:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"deleted": True})

    async def _handle_rename_session(self, request: Any) -> Any:
        import aiohttp.web as web

        session_id = request.match_info["session_id"]
        body = await request.json()
        name = body.get("name", "")
        if not name:
            return web.json_response({"error": "Name required"}, status=400)
        renamed = await self._db.rename_session(session_id, name)
        if not renamed:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"id": session_id, "name": name})

    async def _handle_get_history(self, request: Any) -> Any:
        import aiohttp.web as web

        session_id = request.match_info["session_id"]
        result = await self._db.get_history(session_id)
        if result is None:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response(result)

    # ── WebSocket Handler ─────────────────────────────────────────────────

    async def _handle_ws(self, request: Any) -> Any:
        import aiohttp.web as web

        session_id = request.match_info["session_id"]

        if not await self._db.session_exists(session_id):
            return web.json_response({"error": "Session not found"}, status=404)

        # Enforce one active connection per session
        existing_ws = self._active_connections.get(session_id)
        if existing_ws is not None and not existing_ws.closed:
            logger.info("closing existing connection", session_id=session_id)
            await _try_send(existing_ws, {
                "type": "session.takeover",
                "message": "Another client connected to this session",
            })
            await existing_ws.close()

        ws = web.WebSocketResponse(max_msg_size=self._max_msg_size)
        await ws.prepare(request)
        self._active_connections[session_id] = ws
        logger.info("client connected", session_id=session_id)

        conn = _WSConn(ws=ws, session_id=session_id)

        # Notify client about current stream state on connect
        buf = self._get_stream_buffer(session_id)
        if buf.streaming:
            await _try_send(ws, {
                "type": "stream.active",
                "chunk_id": buf.next_id - 1 if buf.next_id > 0 else -1,
                "collected": buf.collected,
            })
        elif buf.done and buf.collected:
            await _try_send(ws, {
                "type": "stream.catchup",
                "content": buf.collected,
                "done": True,
                "chunk_id": buf.next_id - 1,
            })

        try:
            async for raw_msg in ws:
                if raw_msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(raw_msg.data)
                        await self._handle_ws_message(conn, data)
                    except json.JSONDecodeError:
                        await _try_send(ws, {"type": "error", "message": "Invalid JSON"})
                    except Exception as e:
                        logger.error("message handler error",
                                     session_id=conn.session_id, error=str(e),
                                     exc_info=True)
                        await _try_send(ws, {"type": "error", "message": str(e)})
                elif raw_msg.type in (
                    web.WSMsgType.ERROR,
                    web.WSMsgType.CLOSE,
                    web.WSMsgType.CLOSING,
                ):
                    break
        except Exception as e:
            logger.error("WebSocket connection error",
                         session_id=session_id, error=str(e))
        finally:
            # Only remove if this is still the active connection
            if self._active_connections.get(session_id) is ws:
                self._active_connections.pop(session_id, None)
            logger.info("client disconnected", session_id=session_id)

        return ws

    async def _handle_ws_message(self, conn: "_WSConn", data: dict) -> None:
        """Route an inbound WS message."""
        msg_type = data.get("type", "")

        if msg_type == "message":
            await self._ws_handle_message(conn, data)
        elif msg_type == "attachment":
            await self._ws_handle_attachment(conn, data)
        elif msg_type == "voice":
            await self._ws_handle_voice(conn, data)
        elif msg_type == "resume":
            await self._ws_handle_resume(conn, data)
        else:
            await _try_send(conn.ws, {
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            })

    async def _ws_handle_message(self, conn: "_WSConn", data: dict) -> None:
        """Handle a text message from the frontend."""
        content = (data.get("content") or data.get("text") or "").strip()
        if not content:
            return

        logger.info("message received", session_id=conn.session_id, content_length=len(content))
        self._metrics.message_received(session_id=conn.session_id, content_length=len(content))

        if content.lower() == "/reset":
            await self._handle_reset(conn)
            return

        if conn.busy:
            if len(conn.pending) >= self._max_queue_size:
                logger.warning("message queue full",
                               session_id=conn.session_id,
                               max_size=self._max_queue_size,
                               action="rejecting message")
                await _try_send(conn.ws, {
                    "type": "error",
                    "message": f"Queue full ({self._max_queue_size} messages). Wait for current response to finish.",
                })
                return
            conn.pending.append(content)
            logger.info("message queued", session_id=conn.session_id, queue_size=len(conn.pending))
            await _try_send(conn.ws, {
                "type": "message.queued",
                "count": len(conn.pending),
            })
            return

        await self._db.save_message(conn.session_id, "user", content)
        await self._run_agent(conn, content)

    async def _ws_handle_attachment(self, conn: "_WSConn", data: dict) -> None:
        """Handle an attachment from the frontend."""
        filename = data.get("filename", "attachment")
        mime_type = data.get("mime_type", "application/octet-stream")
        data_b64 = data.get("data", "")

        if not data_b64:
            await _try_send(conn.ws, {"type": "error", "message": "No attachment data"})
            return

        try:
            raw_bytes = base64.b64decode(data_b64)
        except Exception as e:
            await _try_send(conn.ws, {
                "type": "error",
                "message": f"Invalid base64 data: {e}",
            })
            return

        # Upload size validation
        if len(raw_bytes) > self._max_upload_bytes:
            await _try_send(conn.ws, {
                "type": "error",
                "message": (
                    f"File too large ({len(raw_bytes) // (1024*1024)}MB). "
                    f"Max: {self._max_upload_bytes // (1024*1024)}MB"
                ),
            })
            return

        suffix = Path(filename).suffix or ""
        attach_dir = Path.home() / ".march" / "attachments" / conn.session_id
        attach_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = attach_dir / f"{uuid.uuid4().hex[:8]}_{filename}"
        tmp_path.write_bytes(raw_bytes)

        logger.info(
            "attachment received",
            session_id=conn.session_id, filename=filename,
            mime_type=mime_type, size_bytes=len(raw_bytes), path=str(tmp_path),
        )

        # Process attachment: save to disk + generate LLM summary
        user_content = await _process_attachment(
            raw_bytes, filename, mime_type, str(tmp_path),
            self._max_image_dim, self._image_quality,
            self._app_ref,
            summary_max_tokens=self._summary_max_tokens,
            summary_chunk_size=self._summary_chunk_size,
        )

        display = f"📎 {filename}"

        # For images: send preview to frontend and store in DB for history
        # _process_attachment() already resized and saved to tmp_path — read that.
        image_data_uri = None
        if mime_type.startswith("image/"):
            resized_bytes = tmp_path.read_bytes()
            # _resize_image converts to JPEG on success; on failure it keeps original mime
            # Check if the file starts with JPEG magic bytes (FF D8 FF)
            out_mime = "image/jpeg" if resized_bytes[:3] == b'\xff\xd8\xff' else mime_type
            preview_b64 = base64.b64encode(resized_bytes).decode("ascii")
            image_data_uri = f"data:{out_mime};base64,{preview_b64}"
            await _try_send(conn.ws, {
                "type": "image.preview",
                "data": preview_b64,
                "mime_type": out_mime,
                "filename": filename,
            })

        await self._db.save_message(
            conn.session_id, "user", display, image_data=image_data_uri
        )

        if conn.busy:
            if len(conn.pending) >= self._max_queue_size:
                await _try_send(conn.ws, {
                    "type": "error",
                    "message": f"Queue full ({self._max_queue_size} messages). Wait for current response to finish.",
                })
                return
            conn.pending.append(user_content)
            await _try_send(conn.ws, {
                "type": "message.queued",
                "count": len(conn.pending),
            })
            return

        await self._run_agent(conn, user_content)

    async def _ws_handle_resume(self, conn: "_WSConn", data: dict) -> None:
        """Handle stream resume after reconnect.

        Client sends { type: "resume", last_chunk_id: N }
        Server replays missed chunks or sends catchup with full content.
        """
        last_chunk_id = data.get("last_chunk_id", -1)
        buf = self._get_stream_buffer(conn.session_id)

        if buf.done:
            # Stream already finished — send the full collected text as catchup
            await _try_send(conn.ws, {
                "type": "stream.catchup",
                "content": buf.collected,
                "done": True,
                "chunk_id": buf.next_id - 1,
            })
            stream_log.info(
                "resume catchup sent",
                session_id=conn.session_id, content_length=len(buf.collected),
            )
        elif buf.streaming:
            # Stream still in progress — replay missed chunks
            missed = buf.get_chunks_after(last_chunk_id)
            if missed:
                for chunk in missed:
                    await _try_send(conn.ws, chunk)
                stream_log.info(
                    "resume replay sent",
                    session_id=conn.session_id,
                    replayed_chunks=len(missed), from_chunk_id=last_chunk_id,
                )
            else:
                # No missed chunks — client is caught up, just continue
                await _try_send(conn.ws, {
                    "type": "stream.resumed",
                    "chunk_id": buf.next_id - 1 if buf.next_id > 0 else -1,
                })
        else:
            # No active stream — nothing to resume
            await _try_send(conn.ws, {
                "type": "stream.idle",
            })

    async def _ws_handle_voice(self, conn: "_WSConn", data: dict) -> None:
        """Handle a voice message: transcribe then process as text."""
        mime_type = data.get("mime_type", "audio/webm")
        data_b64 = data.get("data", "")

        if not data_b64:
            await _try_send(conn.ws, {"type": "error", "message": "No voice data"})
            return

        try:
            raw_bytes = base64.b64decode(data_b64)
        except Exception as e:
            await _try_send(conn.ws, {
                "type": "error",
                "message": f"Invalid base64 data: {e}",
            })
            return

        ext_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
        }
        ext = ext_map.get(mime_type, ".webm")
        attach_dir = Path.home() / ".march" / "attachments" / conn.session_id
        attach_dir.mkdir(parents=True, exist_ok=True)
        voice_path = attach_dir / f"voice_{uuid.uuid4().hex[:8]}{ext}"
        voice_path.write_bytes(raw_bytes)

        logger.info("voice message received", session_id=conn.session_id, mime_type=mime_type, size_bytes=len(raw_bytes))

        # Transcribe
        transcription = ""
        try:
            from march.core.message import ToolCall as MarchToolCall

            vtt_args: dict[str, Any] = {"path": str(voice_path)}
            if self._app_ref and self._app_ref.config:
                vtt_cfg = getattr(self._app_ref.config.tools, "voice_to_text", None)
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
                logger.error("voice transcription failed",
                             session_id=conn.session_id, error=result.error,
                             action="returning error to client")
            else:
                transcription = result.content.strip()
        except Exception as e:
            logger.error("voice transcription error",
                         session_id=conn.session_id, error=str(e),
                         action="returning error to client")

        if not transcription:
            await _try_send(conn.ws, {
                "type": "error",
                "message": "Could not transcribe voice message",
            })
            return

        logger.info("voice transcribed", session_id=conn.session_id, text_length=len(transcription))
        await _try_send(conn.ws, {"type": "voice.transcribed", "text": transcription})
        await self._db.save_message(conn.session_id, "user", transcription)

        if conn.busy:
            if len(conn.pending) >= self._max_queue_size:
                await _try_send(conn.ws, {
                    "type": "error",
                    "message": f"Queue full ({self._max_queue_size} messages). Wait for current response to finish.",
                })
                return
            conn.pending.append(transcription)
            await _try_send(conn.ws, {
                "type": "message.queued",
                "count": len(conn.pending),
            })
            return

        await self._run_agent(conn, transcription)

    # ── Agent Execution ───────────────────────────────────────────────────

    async def _run_agent(self, conn: "_WSConn", content: str | list) -> None:
        """Run the agent on a message, then drain any queued messages."""
        conn.busy = True
        _agent_t0 = time.monotonic()
        try:
            await self._stream_response(conn, content)
        finally:
            _agent_dur = (time.monotonic() - _agent_t0) * 1000
            logger.info("message complete",
                        session_id=conn.session_id,
                        duration_ms=round(_agent_dur, 1))
            self._metrics.message_complete(
                session_id=conn.session_id,
                duration_ms=_agent_dur,
            )
            conn.busy = False
            # Clear cached session so next turn rebuilds from DB
            self._march_sessions.pop(conn.session_id, None)
        await self._drain_queue(conn)

    async def _drain_queue(self, conn: "_WSConn") -> None:
        """Drain pending messages after agent finishes."""
        if not conn.pending:
            return

        await asyncio.sleep(self._buffer_seconds)

        queued = list(conn.pending)
        conn.pending.clear()
        if not queued:
            return

        combined = queued[0] if len(queued) == 1 else "\n\n".join(
            q if isinstance(q, str) else "[attachment]" for q in queued
        )
        logger.info("draining message queue", session_id=conn.session_id, queued_count=len(queued))

        if isinstance(combined, str):
            await self._db.save_message(conn.session_id, "user", combined)

        await self._run_agent(conn, combined)

    async def _stream_response(self, conn: "_WSConn", content: str | list) -> None:
        """Stream the agent response, persisting even if client disconnects.

        All chunks are buffered with sequential IDs for reconnect recovery.
        Draft is saved to SQLite every 10 chunks or 60s for crash resilience.
        """
        assert self._agent is not None

        session = await self._resolve_march_session(conn.session_id)
        buf = self._get_stream_buffer(conn.session_id)
        buf.reset()
        buf.streaming = True
        buf.draft_id = str(uuid.uuid4())
        buf.last_draft_save_time = time.monotonic()

        client_gone = False

        async def try_send(msg: dict) -> dict:
            """Buffer the chunk and send to client. Returns the buffered chunk."""
            nonlocal client_gone
            chunk = buf.add_chunk(msg)
            if client_gone:
                return chunk
            try:
                await conn.ws.send_json(chunk)
            except Exception:
                client_gone = True
            return chunk

        await try_send({"type": "stream.start"})

        collected = ""
        tool_calls_collected: list[dict] = []
        final_response = None

        try:
            from march.core.agent import AgentResponse

            async for item in self._agent.run_stream(content, session):
                if isinstance(item, AgentResponse):
                    final_response = item
                    break

                if hasattr(item, "tool_call_delta") and item.tool_call_delta:
                    td = item.tool_call_delta
                    if isinstance(td, dict):
                        name = td.get("name", "")
                        if name and td.get("status"):
                            tool_calls_collected.append({
                                "name": name,
                                "args": td.get("args", {}),
                            })
                        await try_send({
                            "type": "tool.start",
                            "name": name,
                            "args": td.get("args", {}),
                        })

                if hasattr(item, "delta_tool_call") and item.delta_tool_call:
                    dtc = item.delta_tool_call
                    if dtc.name:
                        await try_send({
                            "type": "tool.start",
                            "name": dtc.name,
                            "args": {},
                        })

                if hasattr(item, "delta") and item.delta:
                    collected += item.delta
                    buf.collected = collected
                    await try_send({
                        "type": "stream.delta",
                        "content": item.delta,
                    })
                    # Periodic draft save (every 10 chunks or 60s)
                    if buf.needs_draft_save():
                        try:
                            await self._db.save_draft(
                                conn.session_id, buf.draft_id, collected
                            )
                            buf.mark_draft_saved()
                        except Exception:
                            pass  # Non-critical, best-effort
        except Exception as e:
            stream_log.error("streaming error",
                             session_id=conn.session_id, error=str(e),
                             collected_chars=len(collected),
                             action="aborting stream",
                             exc_info=True)
            await try_send({"type": "error", "message": str(e)})
            buf.streaming = False
            buf.done = True
            if collected:
                error_content = collected + "\n\n⚠️ _Response interrupted by error_"
                if buf.last_draft_save_chunks > 0:
                    await self._db.finalize_draft(buf.draft_id, error_content, tool_calls_collected)
                else:
                    await self._db.save_message(
                        conn.session_id, "assistant", error_content, tool_calls_collected,
                    )
            return

        end_msg: dict[str, Any] = {"type": "stream.end"}
        if final_response:
            end_msg["usage"] = {
                "input_tokens": getattr(final_response, "total_tokens", 0),
                "output_tokens": getattr(final_response, "total_tokens", 0),
                "cost": getattr(final_response, "total_cost", 0),
            }
            turn_summary = getattr(final_response, "turn_summary", "")
            if turn_summary:
                end_msg["turn_summary"] = turn_summary
        await try_send(end_msg)

        buf.streaming = False
        buf.done = True
        buf.collected = collected

        # Store summary alongside the message if available
        turn_summary = ""
        if final_response:
            turn_summary = getattr(final_response, "turn_summary", "")

        if collected or tool_calls_collected:
            # If draft exists in DB, finalize it; otherwise save new
            if buf.last_draft_save_chunks > 0:
                await self._db.finalize_draft(
                    buf.draft_id, collected, tool_calls_collected, turn_summary
                )
            else:
                await self._db.save_message(
                    conn.session_id, "assistant", collected, tool_calls_collected,
                    summary=turn_summary,
                )

        # Compaction is handled by agent.py run_stream() — no DB-level compaction needed.

        if client_gone:
            stream_log.info(
                "client disconnected during stream",
                session_id=conn.session_id,
                content_length=len(collected), chunks_buffered=buf.next_id,
            )

    async def _handle_reset(self, conn: "_WSConn") -> None:
        """Handle /reset: clear DB messages, queue, attachments, and reset March session.
        
        Only cleans data belonging to this session — other sessions are untouched.
        """
        conn.pending.clear()
        await self._db.clear_session_messages(conn.session_id)

        # Clear stream buffer so reconnect doesn't replay old content
        if conn.session_id in self._stream_buffers:
            self._stream_buffers[conn.session_id].reset()

        # Clear cached March session
        self._march_sessions.pop(conn.session_id, None)

        # Clean up this session's attachment folder
        attach_dir = Path.home() / ".march" / "attachments" / conn.session_id
        if attach_dir.exists():
            import shutil
            shutil.rmtree(attach_dir, ignore_errors=True)
            logger.info("attachments cleaned", session_id=conn.session_id)

        session = await self._resolve_march_session(conn.session_id)
        try:
            result = await self._agent.memory.reset_session(session.id)
            session.clear()
            logger.info("session reset", session_id=conn.session_id, result=str(result))
        except Exception as e:
            logger.error("session reset failed",
                         session_id=conn.session_id, error=str(e),
                         action="session may be in inconsistent state")

        # Clear cached session again after reset
        self._march_sessions.pop(conn.session_id, None)

        await _try_send(conn.ws, {"type": "session.reset"})

    # ── Session Management ────────────────────────────────────────────────

    def _get_stream_buffer(self, session_id: str) -> _StreamBuffer:
        """Get or create a stream buffer for a session."""
        if session_id not in self._stream_buffers:
            self._stream_buffers[session_id] = _StreamBuffer()
        return self._stream_buffers[session_id]

    async def _resolve_march_session(self, session_id: str) -> Any:
        """Get or create a March Session for a chat session ID.

        Builds context from DB: last N messages.
        Compaction is handled by agent.py run_stream() when context grows too large.
        No in-memory history accumulation — DB is the source of truth.
        """
        from march.core.session import Session
        from march.core.message import Message

        # Always rebuild from DB (no caching — DB is source of truth)
        session = Session(
            id=session_id,
            source_type="ws_proxy",
            source_id=session_id,
        )

        # Load only recent messages for active context
        keep_recent = self._keep_recent
        recent = await self._db.get_recent_messages(session_id, limit=keep_recent)
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                session.add_message(Message(role=role, content=content))

        self._march_sessions[session_id] = session
        return session



# ── Internal Helpers ──────────────────────────────────────────────────────────

class _WSConn:
    """A single WebSocket connection."""

    def __init__(self, ws: Any, session_id: str) -> None:
        self.ws = ws
        self.session_id = session_id
        self.busy: bool = False
        self.pending: list[Any] = []


class _StreamBuffer:
    """Per-session buffer for stream chunks, enabling reconnect recovery.

    Stores chunks with sequential IDs. On reconnect, client sends
    last_chunk_id and server replays the gap.
    Also tracks draft_id for periodic SQLite saves.
    """

    def __init__(self, max_chunks: int = 2000) -> None:
        self.chunks: list[dict] = []  # [{id, type, ...}, ...]
        self.next_id: int = 0
        self.max_chunks = max_chunks
        self.streaming: bool = False
        self.collected: str = ""  # Full text accumulated so far
        self.done: bool = False  # True when stream finished
        self.draft_id: str = ""  # DB message ID for draft saves
        self.last_draft_save_chunks: int = 0  # Chunk count at last draft save
        self.last_draft_save_time: float = 0.0  # Timestamp of last draft save

    def add_chunk(self, msg: dict) -> dict:
        """Add a chunk to the buffer, returns the chunk with id."""
        chunk = {**msg, "chunk_id": self.next_id}
        self.chunks.append(chunk)
        self.next_id += 1
        # Trim old chunks if buffer is full
        if len(self.chunks) > self.max_chunks:
            self.chunks = self.chunks[-self.max_chunks:]
        return chunk

    def get_chunks_after(self, last_chunk_id: int) -> list[dict]:
        """Get all chunks after the given ID."""
        result = []
        for chunk in self.chunks:
            if chunk["chunk_id"] > last_chunk_id:
                result.append(chunk)
        return result

    def needs_draft_save(self, chunk_interval: int = 10, time_interval: float = 60.0) -> bool:
        """Check if we should save a draft to SQLite."""
        if not self.collected:
            return False
        chunks_since = self.next_id - self.last_draft_save_chunks
        time_since = time.monotonic() - self.last_draft_save_time
        return chunks_since >= chunk_interval or time_since >= time_interval

    def mark_draft_saved(self) -> None:
        """Mark that a draft save just happened."""
        self.last_draft_save_chunks = self.next_id
        self.last_draft_save_time = time.monotonic()

    def reset(self) -> None:
        """Reset buffer for a new stream."""
        self.chunks.clear()
        self.next_id = 0
        self.streaming = False
        self.collected = ""
        self.done = False
        self.draft_id = ""
        self.last_draft_save_chunks = 0
        self.last_draft_save_time = 0.0


async def _try_send(ws: Any, msg: dict) -> None:
    """Send JSON to a WebSocket, silently ignoring errors."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass
