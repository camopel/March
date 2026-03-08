# Rolling Context

How March manages conversation context within a finite LLM context window.

---

## Overview

March uses a **sliding window** model for conversation context. Instead of keeping the entire conversation history in memory, it maintains:

1. **Rolling summary** — a compressed carry-over from previous compaction cycles
2. **Recent messages** — the most recent user/assistant exchanges since the last compaction

Together, these form the **rolling context** that the LLM sees on each turn.

---

## How It Works

### Session Structure

Each `Session` holds:

| Field | Purpose |
|-------|---------|
| `rolling_summary` | Carry-over text from the last compaction (may be empty on first turn) |
| `messages` | Recent `Message` objects since the last compaction |
| `last_processed_seq` | Sequence number at the time of last compaction (used for DB restore) |

### What the LLM Sees

On each turn, `session.get_messages_for_llm()` assembles:

```
[rolling_summary as user message]   ← if non-empty
[recent message 1]
[recent message 2]
...
[new user message]                   ← current turn input
```

The rolling summary is prepended as a `user` message with a `[Context Summary]` prefix. This gives the LLM full conversational continuity without sending the entire history.

### First Turn vs Subsequent Turns

**First turn** (empty `rolling_summary`):
- Agent reads all `.md` files: `SYSTEM.md`, `AGENT.md`, `TOOLS.md`, `MEMORY.md`
- These are assembled into a multi-section system prompt via `Context.build_system_prompt()`

**Subsequent turns** (after compaction):
- The `rolling_summary` already contains a deduped version of the `.md` files + conversation context
- Agent skips reading `.md` files — uses the rolling summary directly as the system prompt
- This avoids redundant I/O and reduces token waste from repeated static content

---

## Compaction

### When It Triggers

Compaction is checked at the start of every agent turn (both `run()` and `run_stream()`).

```python
needs_compaction(messages, context_window, system_tokens, threshold)
```

It triggers when **both** conditions are met:
1. Estimated message tokens exceed `threshold × context_window` (default: 95%)
2. There are more than `MIN_RECENT_KEEP` (10) messages

### How It Splits

`split_for_compaction()` walks backward from the newest message, keeping messages until the budget is exhausted:

```
[old messages → summarized]  |  [recent messages → kept as-is]
```

- **Keep budget** = 80% of context window − system tokens − summary budget
- **Summary budget** = 15% of context window (configurable)
- At least `MIN_RECENT_KEEP` (10) messages are always preserved

### Two-Step Compaction

March uses a **two-step compaction** process (in `Agent._two_step_compaction()`):

**Step 1 — Summarize:** Compress the rolling summary + all messages into a conversation summary. The LLM is instructed to be ruthless about brevity (≤30% of input size), preserving decisions, identifiers, code snippets, and action items.

**Step 2 — Dedup:** Compare the summary against all `.md` files (system rules, agent profile, tools, long-term memory) and session memory (facts, plan). Remove anything from the summary that duplicates the static files. The result is a self-contained rolling context document.

```
                    ┌─────────────────┐
                    │  Old Messages   │
                    │  + Rolling Sum  │
                    └────────┬────────┘
                             │ Step 1: Summarize
                             ▼
                    ┌─────────────────┐
                    │   Conversation  │
                    │    Summary      │
                    └────────┬────────┘
                             │ Step 2: Dedup against .md files
                             ▼
                    ┌─────────────────┐
                    │  New Rolling    │
                    │  Summary        │
                    └─────────────────┘
```

After compaction, `session.compact(new_rolling_summary)` clears all messages and stores the new summary. Fresh messages accumulate from this point until the next compaction.

### Session Memory During Compaction

During compaction, session memory files (`facts.md`, `plan.md`) are:
1. **Loaded** from `~/.march/memory/{session_id}/`
2. **Deduplicated** by the LLM (removes true duplicates and superseded entries)
3. **Written back** to disk (cleaned versions)
4. **Folded into** the rolling summary so the LLM sees them in the compacted context

Session memory files are **never cleared** by compaction — they accumulate across cycles so every compaction sees the full facts and plans.

---

## Persistence & Cold Start

### SQLite Storage

Sessions are persisted in SQLite (`~/.march/march.db`):
- `sessions` table: stores `rolling_summary`, `last_processed_seq`, metadata
- `messages` table: stores all messages with `seq` numbers

### Cold Start Restore

When a session is loaded from DB after a restart:

1. Load session metadata (including `rolling_summary`)
2. Query messages where `seq > last_processed_seq`
3. Rebuild `session.messages` from those messages
4. Restore `_seq_counter` from the highest `seq` found

This means only messages **after** the last compaction are loaded — older messages were already folded into the rolling summary.

### Dirty Message Flushing

Messages are batched before writing to DB. A flush triggers when:
- 10+ dirty (unflushed) messages accumulate, OR
- 10+ seconds have passed since the last flush

---

## Configuration

In `config.yaml` under `memory.compaction`:

```yaml
memory:
  compaction:
    threshold: 0.95           # Trigger at 95% of context window
    summary_budget_ratio: 0.15 # Reserve 15% for the summary
    dedup_max_ratio: 0.30      # Max 30% of context window for dedup target
```

---

## Key Design Decisions

1. **Clean history** — `session.messages` only contains `[user, assistant, user, assistant, ...]`. Tool call intermediates are ephemeral and never stored.
2. **Deterministic sessions** — Same source (e.g., Matrix room ID) always maps to the same session UUID, enabling seamless resume across restarts.
3. **No draft persistence** — If a turn crashes mid-way, it's considered failed. The user message is already saved; the user can retry.
4. **Static content dedup** — The two-step compaction ensures `.md` file content isn't duplicated in the rolling summary, saving tokens.
