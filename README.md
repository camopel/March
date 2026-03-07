# March

**An agent runtime that doesn't waste your tokens.**

Most agent frameworks dump everything into the LLM context — tool results, sub-agent outputs, stale memory — and you pay for every token. Sub-agents pollute shared memory. Crashes lose running tasks with no recovery. Config changes break things silently at 3 AM.

March fixes this. Plugin hooks at every step so you control what enters context. Isolated 2-tier memory so sub-agents can't corrupt the parent. A guardian process that survives crashes, rolls back bad configs, and revives dead tasks. Structured cost tracking on every LLM call. Multi-channel (terminal, Matrix, WebSocket, ACP) from a single `march start`. One unified session store — conversations persist across restarts and channels share the same DB.

---

## Quick Start

```bash
pip install git+https://github.com/camopel/March.git
march start
```

Two commands. Agent + guardian + dashboard running.

```bash
march enable                 # Install as systemd service (auto-start on boot)
```

## Why March?

- **Plugin pipeline** — before/after hooks on every step (LLM calls, tools, sessions). Write a plugin in 10 lines.
- **2-tier memory** — session memory stays isolated. Sub-agents can't pollute the parent.
- **Guardian process** — crash recovery, config rollback, restart protection.
- **Cost tracking** — token counting on every LLM call, structured logging.
- **Unified storage** — one SQLite DB for all channels. Sessions resume on reconnect.
- **IDE integration** — ACP protocol for VS Code, IntelliJ, Zed. Real-time streaming.
- **Modular** — only install what you need. Core is lightweight.
- **Pure Python** — no Node.js, no Electron. Python 3.12+ and you're done.

## CLI

```
march start                  Init + start agent + guardian + dashboard
march start --channel matrix Start with Matrix channel
march start --all            Start all enabled channels
march stop                   Stop all services
march restart                Stop + start
march enable                 Install as systemd service
march disable                Remove systemd service

march chat                   Interactive terminal session
march chat "prompt"          One-shot mode
march status                 Health, version, model, plugins, skills
march log                    Follow log stream
march log -n 100             Last 100 lines + follow

march config show            Show config file path
march agent list             List active sub-agents
march agent show             Agent details (db, logs, memory)

march skill list             List installed skills
march skill install PATH     Install a skill
march skill show NAME        Skill details

march plugin list             List plugins
march plugin enable NAME      Enable a plugin
march plugin disable NAME     Disable a plugin

march guardian start         Start guardian (background)
march guardian stop          Stop guardian
march guardian status        Show watched entries
```

## Configuration

All config lives in `~/.march/config.yaml`. Created automatically on first `march start`.

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

plugins:
  enabled: []
```

Supports `${VAR:default}` interpolation. Edit the file directly — no CLI needed.

## Optional Dependencies

```bash
pip install march[anthropic]   # Anthropic Claude
pip install march[bedrock]     # AWS Bedrock
pip install march[matrix]      # Matrix channel + E2EE
pip install march[browser]     # Playwright browser tools
pip install march[voice]       # Speech-to-text
pip install march[all]         # Everything
```

## Project Structure

```
~/.march/
├── config.yaml              # Configuration
├── march.db                 # SQLite (sessions + messages, all channels)
├── attachments/             # Saved images, PDFs, audio (by date)
├── MEMORY.md                # Long-term memory
├── SYSTEM.md                # System rules
├── AGENT.md                 # Agent profile
├── TOOLS.md                 # Tool guidance
└── logs/                    # Structured logs
```

## Development

```bash
git clone https://github.com/camopel/March.git
cd March
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
