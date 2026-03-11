# March – An agent framework where memory gets smarter, not bigger

I've been building AI agents for the past year, and the biggest problem I kept hitting wasn't context windows or tool calling — it was **memory degradation**.

Every agent framework I tried had the same failure mode: the longer you use them, the worse they get. Vector DBs retrieve stale results. Truncation drops critical context. Full-context works until it doesn't.

So I built March — an agent framework with **rolling context compaction**, **session memory**, **selective long-term memory**, and **process-isolated sub-agents**. After 100 turns, March still knows your project details. Other frameworks forgot them at turn 30.

## March Deck — the app layer

Most agent frameworks are headless — you get a CLI and that's it. I also built **March Deck**, a PWA platform where every feature is a mini-app:

- **March** — chat with your agent + dashboard (sessions, cost, providers)
- **Finviz** — financial news with 24h AI summaries
- **ArXiv** — research paper semantic search
- **System** — server monitoring (CPU, RAM, GPU, services)
- **Files** — file browser
- **Notes** — markdown notes
- **Cast** — media casting and control
- **OpenClaw** — OpenClaw agent management

Open it on your phone, add to home screen, and you have a native-like AI assistant. No app store, no account. Works with any agent that exposes a WebSocket endpoint.

https://github.com/camopel/march-deck

## Try it

```bash
pip install march-ai
march start
march chat
```

- GitHub: https://github.com/camopel/March
- PyPI: https://pypi.org/project/march-ai/

---

## Documentation

| Doc | Description |
|-----|-------------|
| **[Quickstart](docs/quickstart.md)** | Configuration, CLI, providers, tools, plugins |
| **[Rolling context](docs/rolling_context.md)** | Deep dive into the compaction algorithm |
| **[Rolling summary](docs/rolling_summary.md)** | How summaries accumulate across cycles |
| **[Session memory](docs/session_memory.md)** | Facts, plans, and checkpoints |
| **[Sub-agents](docs/subagents.md)** | mtAgent, mpAgent, isolation, IPC |
| **[Orchestrator](docs/orchestration.md)** | Channel → orchestrator → agent architecture |

## License

[MIT](LICENSE)
