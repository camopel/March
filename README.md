# March

**March remembers so you don't have to.**

```bash
pip install march-ai
march start
march chat
```

---

## The problem

Every agent framework has the same failure mode: the longer you use them, the worse they get.

- **Vector DB memory** retrieves "relevant" results from months ago that were superseded last week. The agent confidently acts on stale information.
- **Simple truncation** drops the oldest messages. Your agent forgets the project requirements you discussed 20 minutes ago.
- **Full context** works until you hit the window limit, then everything breaks at once.

## What March does differently

**Rolling context compaction.** When the context window fills up, March runs a two-step compression:

1. Summarize the conversation — keep decisions, code, identifiers, action items. Drop greetings, failed attempts, verbose tool output.
2. Deduplicate against your static files (system prompt, agent profile, tools, long-term memory). No redundant information in context.

The result is a self-contained rolling summary that accumulates across compaction cycles. After 100 turns, March still knows your project uses PostgreSQL, deploys to Lambda, and has a March 15 deadline. Other frameworks forgot that at turn 30.

**Session memory.** During work, March automatically saves facts, plans, checkpoints, and progress to per-session markdown files. These survive every compaction cycle — they're folded in, deduplicated, but never lost. It's like a human taking notes during a meeting.

**Selective long-term memory.** Instead of dumping everything into a vector DB, March uses `/rmb` — you explicitly tell it what to remember. Decisions, preferences, key facts. Like how humans actually work: you don't remember every sentence from every conversation, but you remember "I'm allergic to peanuts."

**Process-isolated sub-agents.** When March spawns a sub-agent, it gets its own OS process, its own memory, its own LLM connection. If it crashes, OOMs, or goes rogue, the parent agent is completely unaffected. Every other framework I've seen runs sub-agents in-process — one crash kills everything.

## Design Philosophy

1. **Memory is curation, not accumulation.** The value of memory isn't how much you store — it's how well you filter.
2. **Isolation prevents corruption.** Sub-agents run in separate processes. A rogue sub-agent can't pollute the parent's context.
3. **Explicit beats implicit.** `/rmb` lets you decide what's worth remembering. No black-box embeddings sneaking stale results into context.
4. **Small is fast.** 23K lines means you can read the entire codebase in an afternoon.

## Documentation

| Doc | Description |
|-----|-------------|
| **[How to use March](docs/how_to.md)** | Installation, configuration, CLI, providers, tools, plugins |
| **[Introduction](docs/introduction.md)** | The story behind March — why it exists and what it solves |
| **[March Deck — Apps](docs/march_deck.md)** | PWA app platform — 8 mini-apps for your agent |
| **[Rolling context](docs/rolling_context.md)** | Deep dive into the compaction algorithm |
| **[Rolling summary](docs/rolling_summary.md)** | How summaries accumulate across cycles |
| **[Session memory](docs/session_memory.md)** | Facts, plans, and checkpoints |
| **[Sub-agents](docs/subagents.md)** | mtAgent, mpAgent, isolation, IPC |
| **[Orchestrator](docs/orchestration.md)** | Channel → orchestrator → agent architecture |

## License

[MIT](LICENSE)
