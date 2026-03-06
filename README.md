# March

**The agent framework that doesn't waste your tokens.**

[![PyPI version](https://img.shields.io/pypi/v/march.svg)](https://pypi.org/project/march/) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

March is a Python framework for building AI agents that are cheap to run, easy to extend, and hard to break. It solves the problems that make existing agent platforms expensive, fragile, and frustrating: bloated context windows, polluted memory, no crash recovery, and monolithic architectures you can't customize without forking.

If you've ever watched an agent burn $2 on a task that should cost $0.10 — or had a sub-agent corrupt your main agent's memory — or lost a running task because the framework crashed with no rollback — March was built for you.

## Why March?

### The Problem with Current Agent Frameworks

Most agent platforms share the same fundamental issues:

**Context pollution and token waste.** Every tool call, every sub-agent result, every memory lookup gets dumped into the LLM context. A 200k token window fills up fast when your framework treats it like a landfill. You pay for every token — and your agent gets dumber as the context grows.

**Memory that degrades over time.** Long-term memory becomes a liability when every session writes to it indiscriminately. Sub-agents pollute the shared memory space. Linear conversation history means old, irrelevant context keeps getting loaded. Eventually your agent "remembers" things that actively hurt its performance.

**No lifecycle hooks.** Want to inject context before an LLM call? Log tool results? Enforce budgets? Rate-limit API calls? Most frameworks give you a config file and a prayer. There's no clean way to intercept, modify, or extend the agent's behavior at each step.

**No crash recovery.** Agent process dies mid-task? Config change breaks startup? Most frameworks just... stop. No rollback, no notification, no recovery. You find out when you check on it hours later.

**Kitchen-sink architecture.** Dozens of features you don't need, bundled into a single install. Corporate policies block half of them. You can't deploy it on a work laptop because it ships with browser automation, screen capture, and system-level access you never asked for.

**Fragile updates.** Upgrades break configs silently. No validation, no migration path, no automatic rollback. One bad update and your agent is down until you figure out what changed.

**No cost controls.** Sub-agents spawn without limits. LLM calls happen without budgets. You get the bill at the end of the month.

### How March is Different

March is a **framework**, not a product. Like FastAPI gave you a clean way to build web APIs, March gives you a clean way to build agents.

```
pip install march && march init && march chat
```

**Plugin pipeline with lifecycle hooks.** Every step in the agent loop — LLM calls, tool execution, memory operations, sub-agent spawns — has `before_` and `after_` hooks. Write a plugin in 10 lines. No monkey-patching, no forks.

```python
@app.plugin(name="my-guard", priority=1)
class MyGuard(Plugin):
    async def before_tool(self, tool_name, args):
        if tool_name == "exec" and "rm -rf" in args.get("command", ""):
            raise ToolBlocked("Nice try.")
        return tool_name, args
```

**2-tier memory with isolation.** Session memory stays in the session. System rules load from markdown files you control. Sub-agents get their own memory scope — they can't pollute the parent.

**Guardian process.** A separate watchdog that survives agent restarts. Detects crashes, rolls back bad configs, notifies you, and can revive dead tasks. Your agent doesn't just fail silently anymore.

**Cost tracking built in.** Per-session and per-day budgets. Token counting on every call. Alert thresholds. The `cost` plugin ships out of the box — just set your limits.

**Modular by default.** Only install what you need. Core framework is lightweight. Matrix support? `pip install march[matrix]`. Browser tools? `pip install march[browser]`. Nothing gets pulled in unless you ask for it.

**Config validation at startup.** Pydantic-validated configuration with environment variable interpolation. Bad config? Clear error message before anything starts. Not a silent failure at 3 AM.

**Pure Python.** No Node.js runtime. No Electron. No system dependencies beyond Python 3.12+. If you can `pip install`, you can run March. Works on your laptop, your server, your Raspberry Pi.

## Quick Start

```bash
pip install march
march init
march chat
```

Three commands. You have a working agent with 24 built-in tools, a plugin pipeline, 2-tier memory, and a guardian process.

### Customize Your Agent

```bash
# Copy system rule templates to ~/.march/ for editing
march init-templates

# Then edit:
# ~/.march/SYSTEM.md — persona and behavior rules
# ~/.march/AGENT.md  — role and specialization
# ~/.march/TOOLS.md  — tool usage guidance
```

### Add Custom Tools

```python
from march import MarchApp, tool

app = MarchApp()

@app.tool(name="lookup_user", description="Look up a user by email")
async def lookup_user(email: str) -> str:
    result = await my_db.query(email)
    return f"User: {result.name}, Role: {result.role}"
```

### Add Plugins

```python
from march import Plugin

@app.plugin(name="audit", priority=5)
class AuditPlugin(Plugin):
    async def after_tool(self, tool_name, args, result):
        await log_to_splunk(tool_name, args, len(str(result)))
        return result
```

### Run Anywhere

```bash
march chat                   # Terminal (Rich TUI)
march serve                  # HomeHub WebSocket server
march serve --all            # All enabled channels
```

### IDE Integration

March supports IDE integration through two protocols:

**WebSocket (VS Code)**

The `ws_proxy` plugin runs an embedded HTTP/WebSocket server. VS Code connects as a client:

```bash
march serve                  # Starts WS server on port 8100
```

The WebSocket protocol supports:
- `message` — send user messages with editor context (file, selection, language)
- `stream.start` / `stream.delta` / `stream.end` — streaming responses
- `session.create` / `session.reset` — session management
- REST API: `GET /sessions`, `POST /sessions`, `GET /sessions/{id}/history`

**ACP (IntelliJ, Zed, VS Code)**

[Agent Client Protocol](https://github.com/nicepkg/agent-client-protocol) — JSON-RPC over stdio:

```bash
march acp                    # Launched by IDE, not manually
```

ACP supports:
- `initialize` — capability negotiation
- `agent/message` — user message with editor context
- `agent/stream` — streaming response deltas
- `agent/edit` — apply file edits via IDE
- `agent/terminal` — run commands via IDE terminal
- `shutdown` — clean disconnect

### Dashboard

```bash
march dashboard              # Opens http://localhost:8200
march dashboard --port 9000  # Custom port
```

The dashboard is a local web UI showing sessions, cost tracking, provider health, and a live log stream. Auto-refreshes every 3 seconds via `/api/state`.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       MARCH RUNTIME                          │
│                                                              │
│  CHANNELS                                                    │
│  Terminal │ HomeHub (WS) │ IDE (ACP) │ VS Code │ Matrix      │
│                          │                                   │
│  AGENT MANAGER                                               │
│  Session routing │ Sub-agent orchestration │ Task queue       │
│                          │                                   │
│  PLUGIN PIPELINE                                             │
│  before/after hooks: LLM, tools, memory, sessions, errors    │
│                          │                                   │
│  ┌────────┬────────┬─────┴────┬──────────┬────────────────┐ │
│  │  LLM   │ Tools  │ Memory   │ Config   │ Logging        │ │
│  │ Router │ Engine │ (2-tier) │ (cascade)│ (structured)   │ │
│  └────────┴────────┴──────────┴──────────┴────────────────┘ │
│                                                              │
│  GUARDIAN (separate process — survives crashes)               │
└──────────────────────────────────────────────────────────────┘
```

### Core Loop

```
Message → Context → [before_llm] → LLM → [after_llm]
  → Tool call? → [before_tool] → Execute → [after_tool] → Loop
  → Done? → [on_response] → Save session → Reply
```

Every bracket is a plugin hook. Every step is interceptable.

## Features

**Channels** — Terminal (Rich TUI), WebSocket (HomeHub / VS Code), ACP (IntelliJ, Zed, VS Code), Matrix

**24 Built-in Tools** — Files, exec, web search, browser, PDF, TTS, speech-to-text, memory, sessions, and more

**7 LLM Providers** — OpenAI, Anthropic, Bedrock, Ollama, OpenRouter, LiteLLM — with fallback chains

**Plugin Pipeline** — Before/after hooks on every step. Ships with: safety, cost tracking, structured logging, git context injection

**2-Tier Memory** — Session (auto-saved conversation history), System rules (SYSTEM.md, AGENT.md, TOOLS.md)

**Sub-Agents** — Parallel workers with isolated sessions. Lane-based task queue. Push-based completion notifications

**Skill System** — Self-contained packages (SKILL.md + tools + config). MCP-compatible

**Guardian** — Crash recovery, config rollback, restart protection, stale task detection

**Dashboard** — Local web UI for monitoring sessions, costs, and agent status

### How Sub-Agents Work

The LLM decides when to delegate. When a task is complex or long-running, the agent calls `sessions_spawn`:

```
User: "Refactor the auth module and update all tests"

Agent: This is a multi-step task. I'll spawn a sub-agent.
       → calls sessions_spawn(task="Refactor auth module...")

Agent: Sub-agent spawned: subagent_a3f2b1. Working on it.

       ... (sub-agent runs in background with isolated session) ...

Agent: ✅ Subagent subagent_a3f2b1 finished
       Refactored auth module: extracted JWT logic into auth/jwt.py,
       updated 12 test files. All tests pass.
```

Sub-agents have their own session memory (can't pollute the parent), configurable depth limits, and push-based completion — the parent never polls.

## Plugin Lifecycle Hooks

Every step in the agent loop is interceptable. Override only what you need — defaults are no-ops.

| Category | Hooks |
|----------|-------|
| **App lifecycle** | `on_start`, `on_shutdown` |
| **Session** | `on_session_connect`, `on_session_reset` |
| **LLM pipeline** | `before_llm`, `after_llm`, `on_llm_error`, `on_stream_chunk` |
| **Tool pipeline** | `before_tool`, `after_tool`, `on_tool_error` |
| **Response** | `on_response` |
| **Error** | `on_error` |

`before_llm` can short-circuit the LLM entirely by returning a response. `before_tool` can block tool execution by returning `None`. `on_response` fires once after all tool loops complete — use it to modify the final output. All hooks run in priority order (lower = first).

## Configuration

March uses cascading config: defaults → `config.yaml` → environment variables.

```yaml
# config.yaml
llm:
  default: "openai"
  providers:
    openai:
      model: "${MARCH_MODEL:gpt-4o}"
      url: "${MARCH_LLM_URL:https://api.openai.com/v1}"
      api_key: "${OPENAI_API_KEY:}"
      max_tokens: 16384
      context_window: 128000
      temperature: 0.7
      timeout: 300
      reasoning: false

    # anthropic:
    #   model: "claude-sonnet-4-20250514"
    #   api_key: "${ANTHROPIC_API_KEY:}"

    # openrouter:
    #   model: "anthropic/claude-sonnet-4-20250514"
    #   api_key: "${OPENROUTER_API_KEY:}"

    # litellm:
    #   model: "claude-sonnet-4-20250514"
    #   url: "http://localhost:4000"

    # ollama:
    #   model: "llama3.1"
    #   url: "http://localhost:11434"

    # bedrock:
    #   model: "anthropic.claude-sonnet-4-20250514-v1:0"
    #   region: "us-west-2"
    #   profile: "default"

plugins:
  enabled: ["safety", "cost", "logger"]
  cost:
    budget_per_session: 5.00
    budget_per_day: 20.00

channels:
  terminal:
    enabled: true
    streaming: true
```

All environment-specific values (API URLs, tokens, model names) support `${VAR:default}` interpolation. Sensitive config goes in `.march/` — which is gitignored by default.

## CLI

```
march init                   Initialize a new project
march init-templates         Copy system rule templates to ~/.march/
march chat                   Interactive terminal session
march chat --new             Start a fresh session
march serve                  Start server (HomeHub + API)
march dashboard              Open local dashboard

march config show            Show current config (JSON)
march config set KEY VALUE   Set a config value
march config edit            Open config.yaml in $EDITOR
march config validate        Validate config.yaml

march agent list             List active sub-agents
march agent kill ID          Kill a sub-agent
march agent send ID MSG      Send a message to a sub-agent
march agent logs ID          View sub-agent logs

march skill list             List installed skills
march skill create NAME      Scaffold a new skill
march skill install PATH     Install a skill from path
march skill info NAME        Show skill details

march plugin list            List active plugins
march plugin enable NAME     Enable a plugin
march plugin disable NAME    Disable a plugin
march plugin create NAME     Scaffold a new plugin

march memory show            Show memory statistics
march memory clear           Clear MEMORY.md

march log                    Show recent logs
march log tail               Follow logs in real-time
march log cost               Show token/cost usage
march log audit              Show audit trail

march status                 Full health report
march version                Show version
```

## Optional Dependencies

```bash
pip install march              # Core (OpenAI, Ollama, OpenRouter, LiteLLM)
pip install march[anthropic]   # + Anthropic Claude
pip install march[bedrock]     # + AWS Bedrock
pip install march[matrix]      # + Matrix channel
pip install march[browser]     # + Playwright browser tools
pip install march[voice]       # + Speech-to-text (faster-whisper)
pip install march[fast]        # + uvloop async performance
pip install march[all]         # Everything
pip install march[dev]         # Development tools
```

## Project Structure

After `march init`:

```
~/.march/                    # All March data (created by init)
├── config.yaml              # Agent configuration
├── march.db                 # SQLite DB (sessions, messages, memory)
├── MEMORY.md                # Long-term curated memory
└── logs/                    # Structured logs
```

System rules (SYSTEM.md, AGENT.md, TOOLS.md) are loaded from package templates by default. Run `march init-templates` to copy them to `~/.march/` for customization.

Optionally, create `plugins/` and `skills/` directories in your working directory for custom extensions.

## Development

```bash
git clone https://github.com/march/march.git
cd march
make dev         # Install with dev dependencies
make test        # Run tests
make lint        # Lint with ruff
make format      # Format with ruff
```

## License

[MIT](LICENSE)
