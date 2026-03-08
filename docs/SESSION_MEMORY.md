# Session Memory

How March persists important information within a session so it survives context compaction.

---

## Overview

Session memory is a **per-session, file-based** storage system that lets the LLM save important facts, plans, and progress to disk. Unlike conversation messages (which get compacted and compressed), session memory files persist in full and are folded into every compaction cycle.

**Purpose:** Ensure critical information — decisions, requirements, extracted document content, task progress — is never lost, even across multiple compaction cycles.

---

## Storage

Session memory lives at:

```
~/.march/memory/{session_id}/
├── facts.md     ← facts, requirements, extracted content
└── plan.md      ← plans, checkpoints, progress updates
```

Each file is append-only (from the tool's perspective) and auto-timestamped.

---

## The `session_memory` Tool

The LLM saves to session memory by calling the `session_memory` tool with two parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `type` | `facts`, `plan`, `checkpoint`, `progress` | What kind of information |
| `content` | Markdown text | The content to save |

### Type → File Mapping

| Type | File | Use Case |
|------|------|----------|
| `facts` | `facts.md` | Key decisions, requirements, config values, extracted document content |
| `plan` | `plan.md` | Task plans, roadmaps, multi-step goals |
| `checkpoint` | `plan.md` | Milestone snapshots — what was decided, accomplished, and what's next |
| `progress` | `plan.md` | Step-by-step execution tracking (✅ done, 🔄 in progress, ❌ blocked) |

### Auto-Timestamping

Every entry is automatically timestamped on save:

```markdown
[2026-03-07 14:30 UTC]
- User's project uses Python 3.12
- Deploy target is AWS ECS

[2026-03-07 15:00 UTC]
- [UPDATE] Deploy target changed from ECS to Lambda
```

Timestamps enable the dedup process to identify which version of a fact is the latest.

### Session ID Resolution

The tool resolves `session_id` automatically from a Python `contextvars.ContextVar` set by the agent loop before tool execution. The LLM never needs to know or pass the session ID.

---

## When the LLM Should Save

The agent is instructed (via `AGENT.md`) to call `session_memory` proactively:

### Facts
- Key decisions, requirements, or constraints from the user
- Technical details: config values, architecture choices, names, IDs, URLs
- Full content extracted from uploaded files (PDFs, documents, images)
- Updates to previous facts (prefixed with `[UPDATE]`)

### Plans
- Task lists, roadmaps, or multi-step goals
- Decomposed task breakdowns (after user approval)
- Priority or deadline changes

### Checkpoints (auto-triggered)
- After completing a major phase or deliverable
- Before risky or destructive operations
- After long conversations (>10 turns) with no checkpoint
- After important direction-changing decisions

### Progress (auto-triggered)
- After completing a step in a multi-step plan
- When a step fails or is blocked
- When switching between steps

---

## Lifecycle During Compaction

Session memory plays a key role during context compaction:

### 1. Load

`_load_session_memory(session_id)` reads all files from the session memory directory:
- `facts.md` → `memory_dict["facts"]`
- `plan.md` → `memory_dict["plan"]`
- Any other `.md` or `.txt` files → appended to `facts`

### 2. Dedup

`dedup_session_memory()` asks the LLM to clean up accumulated entries:

- Remove true duplicates (same fact saved multiple times)
- Keep only the latest version of evolved facts (by timestamp)
- Merge overlapping information
- **Do NOT compress or summarize** — maintain full detail level
- Size target: `min(current_size, 30% × context_window)`

The cleaned result is written back to disk, replacing the old files.

### 3. Fold into Rolling Summary

After dedup, session memory is appended to the compaction summary:

```markdown
[Conversation Summary]
...compressed conversation...

---

**Session Memory:**
**Facts:**
- DB is PostgreSQL
- Deploy target is Lambda

**Plan:**
1. Build API layer
2. Write integration tests
```

This ensures the LLM sees session memory in the compacted context, even though the original messages that produced those facts have been compressed away.

### 4. Persist Across Compactions

**Session memory files are never cleared by compaction.** They accumulate permanently until:
- The user runs `/reset` (which calls `delete_session_memory()`)
- The session is explicitly deleted

This means every compaction cycle sees the full history of facts and plans, and the dedup step prevents unbounded growth.

---

## Flow Diagram

```
 Agent Turn                    Compaction Trigger
     │                               │
     │  LLM calls session_memory     │  1. Load facts.md, plan.md
     │  tool with facts/plans        │
     │         │                     │  2. Dedup (LLM removes duplicates)
     │         ▼                     │         │
     │  Append to facts.md           │         ▼
     │  or plan.md                   │  3. Write cleaned files back
     │  (timestamped)                │
     │                               │  4. Fold into rolling summary
     │                               │
     │  ◄────────────────────────────│  5. LLM sees memory in
     │  Next turn: memory visible    │     compacted context
     │  via rolling summary          │
```

---

## Reset Behavior

`/reset` performs a full cleanup:

1. `session.clear()` — clears messages and rolling summary
2. `session_store.clear_session()` — deletes messages from SQLite
3. `memory.reset_session()` — clears SQLite-based memory entries
4. `delete_session_memory()` — removes the entire `~/.march/memory/{session_id}/` directory

After reset, the session starts fresh with no memory of previous conversations.

---

## Key Design Decisions

1. **Append-only from the tool** — The tool only appends; it never reads or modifies existing entries. Dedup is handled separately during compaction.
2. **Two files, not four** — `checkpoint` and `progress` both write to `plan.md` to keep the file structure simple. The timestamps and content make the type clear.
3. **Files, not database** — Session memory uses plain markdown files, not SQLite. This makes it easy to inspect, debug, and manually edit.
4. **Dedup, not compress** — During compaction, session memory is deduplicated (removing true duplicates) but never summarized or compressed. Every unique detail is preserved.
5. **Proactive saving** — The LLM is instructed to save early and often, rather than waiting for compaction. By the time compaction runs, it's too late to extract information from messages that are about to be compressed.
