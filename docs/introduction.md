# March – An agent framework where memory gets smarter, not bigger

I've been building AI agents for the past year, and the biggest problem I kept hitting wasn't context windows or tool calling — it was **memory degradation**.

Every agent framework I tried had the same failure mode: the longer you use them, the worse they get.

- **Vector DB memory** retrieves "relevant" results from months ago that were superseded last week. The agent confidently acts on stale information.
- **Simple truncation** drops the oldest messages. Your agent forgets the project requirements you discussed 20 minutes ago.
- **Full context** works until you hit the window limit, then everything breaks at once.

So I built March.

## What March does differently

**Rolling context compaction.** When the context window fills up, March runs a two-step compression:

1. Summarize the conversation — keep decisions, code, identifiers, action items. Drop greetings, failed attempts, verbose tool output.
2. Deduplicate against your static files (system prompt, agent profile, tools, long-term memory). No redundant information in context.

The result is a self-contained rolling summary that accumulates across compaction cycles. After 100 turns, March still knows your project uses PostgreSQL, deploys to Lambda, and has a March 15 deadline. Other frameworks forgot that at turn 30.

**Session memory.** During work, March automatically saves facts, plans, checkpoints, and progress to per-session markdown files. These survive every compaction cycle — they're folded in, deduplicated, but never lost. It's like a human taking notes during a meeting.

**Selective long-term memory.** Instead of dumping everything into a vector DB, March uses `/rmb` — you explicitly tell it what to remember. Decisions, preferences, key facts. Like how humans actually work: you don't remember every sentence from every conversation, but you remember "I'm allergic to peanuts."

**Process-isolated sub-agents.** When March spawns a sub-agent, it gets its own OS process, its own memory, its own LLM connection. If it crashes, OOMs, or goes rogue, the parent agent is completely unaffected. Every other framework I've seen runs sub-agents in-process — one crash kills everything.

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
- March Deck: https://github.com/camopel/march-deck

## Feedback welcome

1. Is the rolling context approach something you'd actually want? Or is vector DB memory "good enough" for your use cases?
2. The session memory model (facts/plans/checkpoints) — too opinionated, or does the structure help?
3. March Deck as a concept — do you want a PWA app layer for your agent, or is CLI/API enough?
