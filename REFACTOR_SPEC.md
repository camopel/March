# March: Unified Session & Message Storage Refactor

## Goal
All channels (terminal, ACP, ws_proxy, future matrix) share ONE session/message storage service.
No channel owns DB logic. Every channel calls `SessionStore`.

## Current State
- `ws_proxy.py` has its own SQLite DB code (~200 lines of DB ops, inline schema)
- `core/session.py` has `SessionStore` class but it's disabled (wired to None)
- `core/agent.py` calls `session_store.save_session()` after each exchange but store is None
- Images stored as base64 TEXT blobs in ws_proxy's `messages.image_data` column
- Attachments saved to disk via `AttachmentManager` but not referenced in DB
- ACP channel uses `agent.run()` (non-streaming) — needs `run_stream()`

## Tasks

### 1. Unified Schema in SessionStore (`src/march/core/session.py`)

Replace the current disabled schema with:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,      -- 'terminal', 'acp', 'ws', 'matrix'
    source_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    rolling_summary TEXT DEFAULT '',
    compaction_summary TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',     -- JSON: channel-specific metadata
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_sessions_source
    ON sessions(source_type, source_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool', 'system')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT DEFAULT '[]',       -- JSON array of tool call dicts
    tool_results TEXT DEFAULT '[]',     -- JSON array of tool result dicts
    attachments TEXT DEFAULT '[]',      -- JSON array of AttachmentRef dicts (paths, NOT base64)
    metadata TEXT DEFAULT '{}',         -- JSON: tokens, cost, timing, etc.
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
```

Key changes from ws_proxy's current schema:
- `image_data TEXT` → `attachments TEXT` (JSON array of AttachmentRef dicts with file paths)
- Added `source_type`, `source_id` to sessions
- Added `tool_results`, `compaction_summary`, `metadata`
- `role` allows 'tool' and 'system' (not just user/assistant)

### 2. SessionStore API

The `SessionStore` class should provide these methods (update existing ones):

```python
class SessionStore:
    async def initialize(self) -> None  # Create tables, run migrations
    async def close(self) -> None

    # Session CRUD
    async def create_session(self, session: Session) -> None
    async def get_session(self, session_id: str) -> Session | None
    async def get_or_create_session(self, source_type: str, source_id: str, name: str = "") -> Session
    async def list_sessions(self, source_type: str | None = None, active_only: bool = True) -> list[dict]
    async def update_session(self, session: Session) -> None
    async def delete_session(self, session_id: str) -> None  # soft delete (is_active=0)

    # Message CRUD
    async def add_message(self, session_id: str, message: Message) -> str  # returns message_id
    async def get_messages(self, session_id: str, limit: int | None = None, offset: int = 0) -> list[Message]
    async def get_message_count(self, session_id: str) -> int
    async def clear_messages(self, session_id: str) -> None

    # Compaction
    async def get_rolling_summary(self, session_id: str) -> str
    async def update_rolling_summary(self, session_id: str, summary: str) -> None
    async def update_compaction_summary(self, session_id: str, summary: str) -> None

    # Streaming support (for ws_proxy dashboard)
    async def save_draft(self, session_id: str, draft_id: str, content: str) -> None
    async def finalize_draft(self, draft_id: str, content: str, tool_calls: list | None = None) -> None
```

Session identity: `get_or_create_session(source_type, source_id)` uses `deterministic_session_id()` (already exists in session.py) to generate consistent IDs. Same source always gets same session.

### 3. Wire SessionStore into MarchApp (`src/march/app.py`)

In `MarchApp.initialize()`:
```python
self.session_store = SessionStore(db_path=self.config.storage.db_path or "~/.march/march.db")
await self.session_store.initialize()
self.agent.session_store = self.session_store
```

Pass `session_store` to all channels on start.

### 4. Refactor ws_proxy (`src/march/plugins/ws_proxy.py`)

Remove ALL DB code from ws_proxy. It should:
- Get `session_store` from the app (via `on_start` hook or constructor)
- Call `session_store.create_session()`, `session_store.add_message()`, etc.
- Remove: `_init_db()`, `save_message()`, `save_draft()`, `finalize_draft()`, `list_sessions()`, `get_session_messages()`, `clear_session_messages()`, `get_rolling_summary()`, `update_rolling_summary()`, `get_message_count()`, `session_exists()`
- Keep: WebSocket handling, image processing/resizing, the HTTP routes (but they call SessionStore)
- The `_process_attachment()` function stays but returns `AttachmentRef` dicts instead of base64

For images: save to disk via AttachmentManager (already exists), store the ref path in `messages.attachments`. When sending to the dashboard, load from disk on demand (or serve via HTTP endpoint).

### 5. Update ACP Channel (`src/march/channels/acp.py`)

- Wire up `run_stream` instead of `run`:
  - Call `agent.run_stream(text, session)` 
  - Iterate the async iterator, send `session/update` notifications for each chunk
  - The final item is an `AgentResponse` — send the prompt response with stopReason
- Use `session_store.get_or_create_session("acp", workspace_path)` for session management
- Session persists across ACP reconnections to the same workspace

### 6. Update Terminal Channel

- Use `session_store.get_or_create_session("terminal", terminal_id)` 
- Terminal ID can be a hash of the TTY or a fixed "default" for single-user

### 7. Agent Core (`src/march/core/agent.py`)

The agent already calls `session_store.save_session()` after exchanges. Update to:
- After each exchange: `await session_store.add_message(session.id, user_msg)` + `await session_store.add_message(session.id, assistant_msg)`
- After compaction: `await session_store.update_compaction_summary(session.id, summary)`
- The agent should NOT do bulk save_session anymore — messages are saved incrementally

### 8. Attachment Handling

Consistent across all channels:
- Images/files → `AttachmentManager.save()` → disk file → `AttachmentRef` dict
- `AttachmentRef` stored in `messages.attachments` as JSON array (paths, NOT base64)
- **In-memory**: current turn keeps full base64 in Message.content (multimodal list)
- **In DB**: only AttachmentRef paths stored, never base64 blobs
- **On reload from DB**: rehydrate recent messages' images from disk (load bytes → base64)
- **Older messages**: get text placeholder `[Image: filename.jpg (saved to /path)]` — same as current strip behavior
- `strip_attachments_from_messages()` in `attachments.py` already handles this pattern
- New: `rehydrate_attachments(messages, keep_recent=2)` loads disk images for last N messages
- This ensures LLM can see recent images but context doesn't explode with old ones

### 9. Migration

Since there are zero ws_proxy sessions (user confirmed), no migration needed. Just:
- Drop old tables if they exist (the ws_proxy ones)
- Create new schema
- Done

## Files to Modify
1. `src/march/core/session.py` — Rewrite SessionStore with unified schema + API
2. `src/march/plugins/ws_proxy.py` — Remove all DB code, use SessionStore
3. `src/march/channels/acp.py` — Wire run_stream, use SessionStore
4. `src/march/channels/terminal.py` — Use SessionStore (if it exists, check first)
5. `src/march/app.py` — Initialize SessionStore, pass to channels
6. `src/march/core/agent.py` — Switch from bulk save to incremental message saves

## Constraints
- DB path: `~/.march/march.db` (same as current)
- Don't break the ws_proxy WebSocket protocol (dashboard still needs to work)
- Don't break the ACP JSON-RPC protocol
- Keep `AttachmentManager` as-is — it already works well
- All existing tests must still pass
- The `run_stream` method already exists in agent.py — just wire it up in ACP

## Testing
After changes:
1. `python tests/test_acp_integration.py` — must pass (16 tests)
2. `python tests/test_cli.py` — must pass (17 tests)
3. Manual: start march, create session via ws, send message, restart, session persists
4. Manual: ACP init → session/new → prompt → see streaming updates
