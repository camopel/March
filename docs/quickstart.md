# How to use March

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

## Configuration

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

## Multi-Channel

One `march start` gives you:

- **Terminal** — interactive or one-shot mode
- **Matrix** — encrypted chat with E2EE support
- **WebSocket** — connect [March Deck](https://github.com/camopel/march-deck) or any custom frontend
- **ACP** — IDE integration (VS Code, Cursor, any editor with agent protocol support)

All channels share the same agent, same memory, same session state.

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

## Plugin Pipeline

Before/after hooks on every agent step. Write a plugin in 10 lines:

```python
from march.plugins import hook

@hook("before_llm_call")
async def log_tokens(context):
    print(f"Sending {context.token_count} tokens")
```

## Development

```bash
git clone https://github.com/camopel/March.git
cd March
pip install -e ".[dev]"
pytest                        # 785 tests
```
