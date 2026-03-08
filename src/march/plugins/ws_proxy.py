"""WSProxyPlugin — Embedded HTTP/WS server for the frontend chat.

Pure I/O adapter: accepts WebSocket/HTTP connections, converts user input
into Orchestrator calls, and maps OrchestratorEvents back to WebSocket JSON.

All agent execution, session management, message persistence, and draft
handling are delegated to the Orchestrator layer.

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
    from march.core.orchestrator import Orchestrator

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
    "context_keep_recent": 50,      # Messages to keep in active context
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
            source_type="ws", source_id="",
            name=name, metadata={"description": description},
        )
        # Re-fetch to get the DB-formatted created_at (ISO string)
        stored = await self._store.get_session(session.id)
        created_at = stored.created_at if stored else _now()
        return {
            "id": session.id,
            "name": name,
            "description": description,
            "created_at": created_at,
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

    async def update_message_image(self, message_id: str, image_data: str) -> None:
        """Update the attachments field of an existing message with image data."""
        attachments = json.dumps([{"type": "image_data", "data": image_data}])
        await self._store._db.execute(
            "UPDATE messages SET attachments = ? WHERE id = ?",
            (attachments, message_id),
        )
        await self._store._db.commit()

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
    """Embedded HTTP/WS server for the frontend chat — pure I/O adapter.

    All agent execution, session resolution, message persistence, and draft
    handling are delegated to the Orchestrator.  This plugin only handles:
    - WebSocket I/O (accept connections, send/receive JSON messages)
    - Converting user input → Orchestrator.handle_message()
    - Converting OrchestratorEvents → WebSocket JSON messages
    - File upload handling (save to disk, pass path to Orchestrator)
    - Voice transcription (transcribe, pass text to Orchestrator)
    - HTTP REST endpoints (delegating to ChatDB / Orchestrator)
    - Stream buffer management for reconnect recovery

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
        self._orchestrator: Orchestrator | None = None
        self._site: Any = None
        self._runner: Any = None
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
        """Start the HTTP/WS server, initialize DB, and create Orchestrator."""
        import aiohttp.web as web
        from march.core.orchestrator import Orchestrator

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

        # Create the Orchestrator — single control layer for agent execution
        self._orchestrator = Orchestrator(
            agent=app.agent,
            session_store=app.session_store,
        )

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
        """Stop the HTTP/WS server and close the DB."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._db:
            await self._db.close()
        logger.info("server stopped")

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

        # Run in thread pool to avoid event loop contention
        # (agent tool calls like glob can block the event loop)
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(None, self._list_sessions_sync)
        return web.json_response({"sessions": sessions})

    def _list_sessions_sync(self) -> list[dict]:
        """Synchronous fallback for listing sessions when event loop is busy."""
        import sqlite3 as _sqlite3

        if self._db is None or self._db._store is None:
            return []
        db_path = str(self._db._store.db_path)
        conn = _sqlite3.connect(db_path, timeout=1)
        conn.row_factory = _sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE is_active = 1 ORDER BY last_active DESC"
            ).fetchall()
            results = []
            for row in rows:
                s = dict(row)
                s["is_active"] = bool(s["is_active"])
                msg = conn.execute(
                    "SELECT content, role FROM messages WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                s["last_message"] = msg["content"][:100] if msg else None
                s["last_message_role"] = msg["role"] if msg else None
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (row["id"],),
                ).fetchone()
                s["message_count"] = cnt[0] if cnt else 0
                results.append(s)
            return results
        finally:
            conn.close()

    async def _handle_create_session(self, request: Any) -> Any:
        import aiohttp.web as web

        body = await request.json()
        name = body.get("name", "New Chat")
        description = body.get("description", "")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._create_session_sync, name, description
        )
        return web.json_response(result, status=201)

    def _create_session_sync(self, name: str, description: str = "") -> dict:
        """Synchronous fallback for creating sessions."""
        import sqlite3 as _sqlite3

        if self._db is None or self._db._store is None:
            raise ValueError("WSProxyPlugin DB not initialised")
        db_path = str(self._db._store.db_path)
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = _sqlite3.connect(db_path, timeout=1)
        try:
            conn.execute(
                """INSERT INTO sessions (id, source_type, source_id, name, rolling_summary,
                   compaction_summary, metadata, created_at, last_active, is_active)
                   VALUES (?, 'ws', '', ?, '', '', ?, ?, ?, 1)""",
                (session_id, name, json.dumps({"description": description}), now, now),
            )
            conn.commit()
            return {"id": session_id, "name": name, "description": description, "created_at": now}
        finally:
            conn.close()

    async def _handle_delete_session(self, request: Any) -> Any:
        import aiohttp.web as web

        session_id = request.match_info["session_id"]
        deleted = await self._db.delete_session(session_id)
        if not deleted:
            return web.json_response({"error": "Session not found"}, status=404)
        # Evict from Orchestrator cache
        if self._orchestrator:
            self._orchestrator.evict_session(session_id)
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

        ws = web.WebSocketResponse(
            max_msg_size=self._max_msg_size,
            heartbeat=30.0,  # Send ping every 30s to keep connection alive
        )
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
            # Check for stop/interrupt commands
            if content.lower().strip() in _WSConn.STOP_COMMANDS:
                logger.info("stop command received",
                            session_id=conn.session_id,
                            command=content.strip())
                # Signal cancellation to the running stream
                conn.cancel_event.set()
                # Clear any queued messages — user wants to stop everything
                conn.pending.clear()
                # Notify the client immediately
                await _try_send(conn.ws, {"type": "stream.cancelled"})
                return

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

        # User message persistence is handled by the Orchestrator
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

        display = f"📎 {filename}"

        # Process attachment: save to disk + generate LLM summary
        user_content = await _process_attachment(
            raw_bytes, filename, mime_type, str(tmp_path),
            self._max_image_dim, self._image_quality,
            self._app_ref,
            summary_max_tokens=self._summary_max_tokens,
            summary_chunk_size=self._summary_chunk_size,
        )

        # For images: send preview to frontend
        if mime_type.startswith("image/"):
            resized_bytes = tmp_path.read_bytes()
            out_mime = "image/jpeg" if resized_bytes[:3] == b'\xff\xd8\xff' else mime_type
            preview_b64 = base64.b64encode(resized_bytes).decode("ascii")
            await _try_send(conn.ws, {
                "type": "image.preview",
                "data": preview_b64,
                "mime_type": out_mime,
                "filename": filename,
            })

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

        # User message persistence is handled by the Orchestrator
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

    # ── Agent Execution (via Orchestrator) ────────────────────────────────

    async def _run_agent(self, conn: "_WSConn", content: str | list) -> None:
        """Run the agent on a message via Orchestrator, then drain queued messages."""
        conn.busy = True
        conn.cancel_event.clear()  # Reset cancellation flag for new run
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

        await self._run_agent(conn, combined)

    async def _stream_response(self, conn: "_WSConn", content: str | list) -> None:
        """Stream the agent response via Orchestrator, mapping events to WS JSON.

        All agent execution, session resolution, message persistence, and
        draft handling are delegated to the Orchestrator.  This method only:
        1. Calls orchestrator.handle_message() with the content and cancel_event
        2. Iterates OrchestratorEvents and maps them to WebSocket JSON messages
        3. Manages the stream buffer for reconnect recovery
        """
        if self._orchestrator is None:
            raise ValueError("WSProxyPlugin._stream_response called before Orchestrator was initialised")

        from march.core.orchestrator import (
            TextDelta, ToolProgress, FinalResponse, Cancelled, Error,
        )

        buf = self._get_stream_buffer(conn.session_id)
        buf.reset()
        buf.streaming = True

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

        try:
            async for event in self._orchestrator.handle_message(
                session_id=conn.session_id,
                content=content,
                source="ws",
                cancel_event=conn.cancel_event,
            ):
                if isinstance(event, TextDelta):
                    buf.collected += event.delta
                    await try_send({
                        "type": "stream.delta",
                        "content": event.delta,
                    })

                elif isinstance(event, ToolProgress):
                    await try_send({
                        "type": "tool.progress",
                        "name": event.name,
                        "status": event.status,
                        "summary": event.summary,
                        "duration_ms": event.duration_ms,
                    })

                elif isinstance(event, FinalResponse):
                    end_msg: dict[str, Any] = {"type": "stream.end"}
                    if event.total_tokens or event.total_cost:
                        end_msg["usage"] = {
                            "total_tokens": event.total_tokens,
                            "cost": event.total_cost,
                        }
                    if event.turn_summary:
                        end_msg["turn_summary"] = event.turn_summary
                    await try_send(end_msg)

                    buf.streaming = False
                    buf.done = True
                    buf.collected = event.content or buf.collected

                    if client_gone:
                        stream_log.info(
                            "client disconnected during stream",
                            session_id=conn.session_id,
                            content_length=len(buf.collected),
                            chunks_buffered=buf.next_id,
                        )
                    return

                elif isinstance(event, Cancelled):
                    buf.streaming = False
                    buf.done = True
                    await try_send({"type": "stream.end", "cancelled": True})
                    stream_log.info(
                        "stream cancelled — partial content saved by orchestrator",
                        session_id=conn.session_id,
                        collected_chars=len(event.partial_content),
                    )
                    return

                elif isinstance(event, Error):
                    buf.streaming = False
                    buf.done = True
                    await try_send({"type": "error", "message": event.message})
                    stream_log.error(
                        "orchestrator error",
                        session_id=conn.session_id,
                        error=event.message,
                    )
                    return

        except Exception as e:
            stream_log.error("streaming error",
                             session_id=conn.session_id, error=str(e),
                             collected_chars=len(buf.collected),
                             action="aborting stream",
                             exc_info=True)
            await try_send({"type": "error", "message": str(e)})
            buf.streaming = False
            buf.done = True
            return

        # If we exit the loop without a terminal event (shouldn't happen),
        # mark the stream as done to avoid leaving the client hanging.
        buf.streaming = False
        buf.done = True

    async def _handle_reset(self, conn: "_WSConn") -> None:
        """Handle /reset: delegate to Orchestrator for full session cleanup.

        Only cleans data belonging to this session — other sessions are untouched.
        """
        conn.pending.clear()

        # Clear stream buffer so reconnect doesn't replay old content
        if conn.session_id in self._stream_buffers:
            self._stream_buffers[conn.session_id].reset()

        # Delegate full reset to Orchestrator (clears cache + DB + memory + session files)
        if self._orchestrator:
            try:
                await self._orchestrator.reset_session(conn.session_id)
            except Exception as e:
                logger.error("orchestrator reset_session failed",
                             session_id=conn.session_id, error=str(e))
        else:
            # Fallback: manual DB clear if orchestrator not available
            await self._db.clear_session_messages(conn.session_id)

        # Clean up this session's attachment folder
        attach_dir = Path.home() / ".march" / "attachments" / conn.session_id
        if attach_dir.exists():
            import shutil
            shutil.rmtree(attach_dir, ignore_errors=True)
            logger.info("attachments cleaned", session_id=conn.session_id)

        await _try_send(conn.ws, {"type": "session.reset"})

    # ── Session Management ────────────────────────────────────────────────

    def _get_stream_buffer(self, session_id: str) -> "_StreamBuffer":
        """Get or create a stream buffer for a session."""
        if session_id not in self._stream_buffers:
            self._stream_buffers[session_id] = _StreamBuffer()
        return self._stream_buffers[session_id]


# ── Internal Helpers ──────────────────────────────────────────────────────────

class _WSConn:
    """A single WebSocket connection."""

    # Commands that trigger cancellation of the current agent run.
    STOP_COMMANDS = frozenset({"stop", "停止", "/stop"})

    def __init__(self, ws: Any, session_id: str) -> None:
        self.ws = ws
        self.session_id = session_id
        self.busy: bool = False
        self.pending: list[Any] = []
        self.cancel_event: asyncio.Event = asyncio.Event()


class _StreamBuffer:
    """Per-session buffer for stream chunks, enabling reconnect recovery.

    Stores chunks with sequential IDs. On reconnect, client sends
    last_chunk_id and server replays the gap.
    """

    def __init__(self, max_chunks: int = 2000) -> None:
        self.chunks: list[dict] = []  # [{id, type, ...}, ...]
        self.next_id: int = 0
        self.max_chunks = max_chunks
        self.streaming: bool = False
        self.collected: str = ""  # Full text accumulated so far
        self.done: bool = False  # True when stream finished

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

    def reset(self) -> None:
        """Reset buffer for a new stream."""
        self.chunks.clear()
        self.next_id = 0
        self.streaming = False
        self.collected = ""
        self.done = False


async def _try_send(ws: Any, msg: dict) -> None:
    """Send JSON to a WebSocket, silently ignoring errors."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass
