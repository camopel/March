# March

**Most AI agents get dumber the longer you use them. March doesn't.**

Agent frameworks love to dump everything into the LLM context window — your entire conversation history, vector DB search results from six months ago, every sub-agent's internal monologue. You pay for every token, and the LLM drowns in noise.

March takes a different approach: **structured compression, isolated memory, and selective recall.** The result is an agent that stays sharp after hundreds of turns, not one that slowly forgets what you told it ten minutes ago.

```bash
pip install march-ai
march start
```

Two commands. You're running.

---

## Why March?

### 🧠 Memory That Actually Works

Most frameworks either truncate history (losing critical decisions) or shove everything into a vector DB (retrieving outdated garbage). March uses **rolling context compaction** — a two-step process that compresses conversation history while preserving every decision, requirement, and code snippet that matters.

**How it works:**
1. When context fills up, March summarizes the conversation (keeping facts, dropping filler)
2. Then deduplicates against your static files — no redundant information in context
3. Session memory (`facts.md`, `plan.md`) is folded in and survives every compaction cycle

The result: after 100 turns, March still knows your project uses PostgreSQL, deploys to Lambda, and has a March 15 deadline. Other frameworks forgot that at turn 30.

### 🔒 Sub-Agent Isolation

When March spawns a sub-agent, it gets its own process, its own memory, its own LLM connection. A sub-agent can crash, OOM, or go rogue — the parent agent is unaffected.

| | mtAgent (lightweight) | mpAgent (isolated) |
|---|---|---|
| Runs as | asyncio task | Separate OS process |
| Memory | Shared with parent | Fully independent |
| Best for | I/O-bound work, API calls | GPU compute, simulations, risky ops |
| Crash impact | Could affect parent | Contained — parent continues |

Sub-agents communicate via IPC (Unix socketpair + msgpack). Results are pushed to the parent — no polling, no shared state corruption.

### 📝 Selective Memory (`/rmb`)

Vector databases remember everything. That sounds good until your agent retrieves a "relevant" decision from three months ago that was superseded last week.

March uses **explicit, human-like memory:**
- `/rmb` saves what you tell it to save — decisions, preferences, key facts
- Session memory auto-captures facts and plans during work
- Compaction deduplicates and keeps only the latest version of evolved facts
- Nothing stale sneaks back in through similarity search

### 🔌 Plugin Pipeline

Before/after hooks on every agent step. Write a plugin in 10 lines:

```python
from march.plugins import hook

@hook("before_llm_call")
async def log_tokens(context):
    print(f"Sending {context.token_count} tokens")
```

### 📡 Multi-Channel, Single Codebase

One `march start` gives you:

- **Terminal** — interactive or one-shot mode
- **Matrix** — encrypted chat with E2EE support
- **WebSocket** — connect [March Deck](https://github.com/camopel/MarchDeck) or any custom frontend
- **ACP** — IDE integration (VS Code, Cursor, any editor with agent protocol support)

All channels share the same agent, same memory, same session state.

---

## How Rolling Context Works

```
Turn 1-30: Full messages in context
           ↓ context window fills up
Turn 31:   Compaction triggers
           ├─ Step 1: Summarize (keep decisions, drop filler)
           ├─ Step 2: Dedup against static files
           └─ Session memory folded in
           ↓
Turn 31+:  [Compact rolling summary] + [Recent messages]
           ↓ context fills again
Turn 60:   Compaction triggers again
           └─ Builds on previous summary (accumulative)
           ↓
Turn 100:  Still knows your project constraints from Turn 3
```

**Key properties:**
- **Accumulative** — each compaction builds on the last, so early context is never fully lost
- **Lossless for decisions** — identifiers, code, URLs, action items always preserved
- **Self-contained** — after compaction, the rolling summary contains everything the LLM needs
- **Configurable** — tune compaction threshold, summary budget, and dedup ratio

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Channels (Pure I/O)                  │
│  Terminal  ·  Matrix  ·  WebSocket  ·  ACP       │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│              Orchestrator                         │
│  LLM calls · Tool dispatch · Cancel/redirect     │
│  Ephemeral turn state (never persisted)          │
└──────────┬────────────────────┬──────────────────┘
           │                    │
           ▼                    ▼
┌──────────────────┐  ┌────────────────────────────┐
│   Sub-Agents     │  │     Memory System          │
│  mtAgent (async) │  │  FileMemory (md files)     │
│  mpAgent (proc)  │  │  SessionMemory (per-task)  │
│  Nested (depth>1)│  │  Rolling Context           │
│  IPC + heartbeat │  │  SQLite persistence        │
└──────────────────┘  └────────────────────────────┘
```

**23,000 lines of Python.** No JavaScript. 6 native LLM providers.

---

## March Deck — PWA App Platform

Want a mobile-friendly UI? **[March Deck](https://github.com/camopel/MarchDeck)** is a PWA platform that turns your agent into a collection of mini-apps you can open on any device.

| App | What it does |
|-----|-------------|
| 🤖 **March** | Chat + dashboard (sessions, cost, providers, logs) |
| 📰 **Finviz** | Financial news with 24h AI summaries |
| 📄 **ArXiv** | Research paper semantic search |
| 📊 **System** | Server monitoring (CPU, RAM, GPU, services) |
| 📁 **Files** | File browser |
| 📝 **Notes** | Markdown notes |
| 📺 **Cast** | Chromecast streaming |
| 🦞 **OpenClaw** | OpenClaw agent management |

No app store. Add to home screen and go. Works with any WebSocket-compatible agent.

---

## Quick Start

```bash
# Install
pip install march-ai

# Or with extras
pip install "march-ai[anthropic]"     # Claude support
pip install "march-ai[bedrock]"       # AWS Bedrock
pip install "march-ai[matrix]"        # Matrix + E2EE
pip install "march-ai[browser]"       # Playwright browser tools
pip install "march-ai[voice]"         # Speech-to-text
pip install "march-ai[all]"           # Everything

# Run
march start                           # Start agent
march chat                            # Interactive terminal
march chat "summarize this repo"      # One-shot mode
```

### Configuration

`~/.march/config.yaml` — created on first run:

```yaml
llm:
  default: "openai"
  providers:
    openai:
      model: "${MARCH_MODEL:gpt-4o}"
      api_key: "${OPENAI_API_KEY:}"

channels:
  terminal:
    enabled: true

memory:
  compaction:
    threshold: 0.95
    summary_budget_ratio: 0.15
```

Supports `${VAR:default}` environment variable interpolation.

---

## CLI

```
march start                  Start agent
march stop                   Stop all services
march restart                Restart
march enable / disable       Systemd service (auto-start on boot)

march chat                   Interactive terminal session
march chat "prompt"          One-shot mode
march status                 Health, version, model, plugins, channels
march log                    Follow log stream

march config show            Show config path
march agent list / show      Sub-agent management
march plugin list / enable   Plugin management
```

---

## Built-in Tools

March ships with 24 tools out of the box:

| Category | Tools |
|----------|-------|
| **Files** | read, write, edit, glob, diff, apply_patch |
| **Code** | exec, process (background jobs) |
| **Web** | web_search, web_fetch, browser (Playwright) |
| **Memory** | session_memory (facts/plans/checkpoints) |
| **Media** | screenshot, pdf, voice-to-text, tts |
| **Integration** | github, huggingface, clipboard, translate |
| **Agent** | sessions (sub-agent spawn/manage), message |

All tools are registered via `@tool` decorator. Add custom tools by dropping a Python file in your plugin directory.

---

## Comparison

### March vs OpenClaw vs LangChain vs CrewAI

| Feature | March | OpenClaw | LangChain | CrewAI |
|---------|:-----:|:--------:|:---------:|:------:|
| Rolling context compaction | ✅ Structured 2-step | Basic | ❌ | ❌ |
| Session memory (survives compaction) | ✅ facts/plans/checkpoints | ❌ | ❌ | ❌ |
| Process-isolated sub-agents | ✅ mpAgent | ❌ In-process | ❌ | ❌ |
| Selective memory (`/rmb`) | ✅ | ❌ | ❌ | ❌ |
| Native LLM providers | ✅ 6 providers | 1 (via LiteLLM proxy) | Via wrappers | Via wrappers |
| Multi-channel | 4 (Matrix, Terminal, WS, ACP) | 10+ (Telegram, WhatsApp, Discord, Signal…) | ❌ | ❌ |
| Plugin hooks | ✅ Lifecycle hooks | Skill-based | Via callbacks | ❌ |
| Cost tracking | ✅ Built-in per-turn | Via LiteLLM | ❌ | ❌ |
| Browser automation | ✅ Playwright (headless) | ✅ Playwright (multi-tab, profiles) | ❌ | ❌ |
| Mobile device integration | ❌ | ✅ Node pairing | ❌ | ❌ |
| Community ecosystem | New | ✅ Active + ClawHub | ✅ Large | Growing |
| Codebase | **~23K lines** Python | ~145K lines TypeScript | ~300K+ | ~50K+ |

### Where March wins

- **Memory that doesn't degrade** — Rolling context + session memory means your agent remembers decisions from turn 3 at turn 100. OpenClaw and LangChain lose this during compaction.
- **Fault isolation** — mpAgent sub-agents run in separate OS processes. A crash stays contained. Every other framework runs sub-agents in-process.
- **No proxy dependency** — March talks directly to Bedrock, Anthropic, OpenAI, Ollama, OpenRouter. OpenClaw requires a LiteLLM proxy for all LLM calls.
- **Auditable** — 23K lines. You can read the entire codebase in an afternoon.

### Where OpenClaw wins

- **Channel breadth** — 10+ messaging platforms vs March's 4. If you need Telegram, WhatsApp, or Discord, OpenClaw has it.
- **Browser & device integration** — Multi-tab browser profiles, Chrome extension relay, mobile camera/location/notifications.
- **Maturity** — Production-tested with an active community and skill marketplace.

---

## Supported LLM Providers

| Provider | Config key | Notes |
|----------|-----------|-------|
| OpenAI | `openai` | GPT-4o, GPT-4, etc. |
| Anthropic | `anthropic` | Claude Opus, Sonnet, Haiku |
| AWS Bedrock | `bedrock` | Claude, Llama, Mistral via AWS |
| Ollama | `ollama` | Local models, no API key |
| OpenRouter | `openrouter` | Multi-provider gateway |
| LiteLLM | `litellm` | Universal proxy |

Switch providers per-session or per-sub-agent. No code changes needed.

---

## Development

```bash
git clone https://github.com/camopel/March.git
cd March
pip install -e ".[dev]"
pytest                        # 785 tests
```

---

## Design Philosophy

1. **Memory is curation, not accumulation.** The value of memory isn't how much you store — it's how well you filter. March compresses, deduplicates, and preserves only what matters.

2. **Isolation prevents corruption.** Sub-agents run in separate processes with their own memory. A rogue sub-agent can't pollute the parent's context or crash the main loop.

3. **Explicit beats implicit.** `/rmb` lets you decide what's worth remembering. No black-box embeddings, no stale vector search results sneaking into your context.

4. **Small is fast.** 23K lines means you can read the entire codebase in an afternoon. Every abstraction earns its place.

---

## License

[MIT](LICENSE)
