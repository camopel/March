# March

**An agent runtime that doesn't waste your tokens.**

Most agent frameworks dump everything into the LLM context and you pay for every token. Sub-agents pollute shared memory. Crashes lose running tasks. Config changes break things at 3 AM.

March fixes this. Plugin hooks control what enters context. Isolated memory so sub-agents can't corrupt the parent. Multi-channel from a single `march start`.

---

## Quick Start

```bash
pip install git+https://github.com/camopel/March.git
march start
```

Two commands. Agent + dashboard running.

```bash
march enable                 # Install as systemd service (auto-start on boot)
```

## Why March?

- **Plugin pipeline** — before/after hooks on every step. Write a plugin in 10 lines.
- **Isolated memory** — sub-agents can't pollute the parent session.
- **Multi-channel** — terminal, Matrix, WebSocket, IDE (ACP). One codebase.
- **Cost tracking** — token counting on every LLM call.
- **Pure Python** — Python 3.12+, no Node.js.

## CLI

```
march start                  Start agent + dashboard
march stop                   Stop all services
march restart                Stop + start
march enable / disable       Systemd service management

march chat                   Interactive terminal session
march chat "prompt"          One-shot mode
march status                 Health, version, model, plugins, skills
march log                    Follow log stream

march config show            Show config path
march agent list / show      Sub-agent management
march skill list / install   Skill management
march plugin list / enable   Plugin management
```

## Configuration

`~/.march/config.yaml` — created on first run.

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

Supports `${VAR:default}` interpolation.

## Optional Dependencies

```bash
pip install march[anthropic]   # Anthropic Claude
pip install march[bedrock]     # AWS Bedrock
pip install march[matrix]      # Matrix channel + E2EE
pip install march[browser]     # Playwright browser tools
pip install march[voice]       # Speech-to-text
pip install march[all]         # Everything
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
