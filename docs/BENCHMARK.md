# March vs OpenClaw — Agent Runtime Benchmark

> **March v0.1.0** (config: `1fe09dd`) vs **OpenClaw 2026.3.7** (`42a1394`)
> Model: Claude Opus 4.6 · Machine: AMD Ryzen 9 9950X3D, 192GB RAM, Ubuntu 24.04
> Date: March 8, 2026

## TL;DR

March delivers **equivalent task completion** to OpenClaw with a **6.2× smaller codebase** and **native multi-provider LLM support** — no proxy dependency. Both achieve 100% goal retention across all benchmark scenarios. March's system prompt is ~20% larger per turn due to richer agent context, but this is offset by architectural advantages in multi-turn conversations (persistent session memory, cost tracking, fault-isolated sub-agents).

---

## Codebase Comparison

| Metric | March | OpenClaw | Ratio |
|--------|------:|--------:|:-----:|
| **Source lines** | 23,546 | 145,013 | **6.2×** smaller |
| **Source files** | 101 | 2,816 | **28×** fewer |
| **Language** | Python 3.12 | TypeScript (Node.js) | — |
| **Tests** | 785 | — | — |
| **LLM providers** | 6 native | 1 (via LiteLLM proxy) | — |
| **Channels** | 4 (Matrix, Terminal, WS, ACP) | 10+ | — |

### What "6.2× smaller" means

March implements a full agent runtime — orchestrator, multi-process sub-agents, session memory, context compaction, MCP support, plugin system, 24 built-in tools, and 6 native LLM providers — in **23,546 lines of Python**. OpenClaw requires **145,013 lines of TypeScript** for comparable core agent functionality (plus additional channel integrations).

---

## Token Efficiency — Single-Turn Scenarios

Adapted from the [SER benchmark suite](https://github.com/camopel/SemanticExecutionRuntime) (4 real-world scenarios testing task decomposition, debug isolation, long-form generation, and constraint retention).

| Scenario | March | OpenClaw* | Goal Retention |
|----------|------:|--------:|:-:|
| **Multi-Step Planning** (8 subtasks) | 18,675 | ~22,104 | Both ✅ 100% |
| **Debug Loop** (3 bugs) | 11,862 | ~9,601 | Both ✅ 100% |
| **Long Workflow** (6-section report) | 15,937 | ~14,129 | Both ✅ 100% |
| **Totals** | **46,474** | **~45,834** | **Both 4/4** |

*\*OpenClaw token counts estimated from content length (streaming mode doesn't report usage).*

**Analysis:** Single-turn performance is nearly identical. The difference is output verbosity, not architectural efficiency. Both agents achieve **100% goal retention** — all subtasks completed, all constraints addressed.

### System Prompt Overhead

| Component | March | OpenClaw |
|-----------|------:|--------:|
| System prompt per turn | ~9,300 tokens | ~7,500 tokens |
| Workspace files | SYSTEM + AGENT + TOOLS + MEMORY | AGENTS + SOUL + TOOLS + IDENTITY + USER + HEARTBEAT |
| Tool definitions | ~1,500 tokens | ~1,500 tokens |

March's system prompt is ~1,800 tokens larger because it includes richer agent context (session memory instructions, execution mode selection, task decomposition guidelines). This is a deliberate trade-off: more upfront context → better autonomous behavior.

---

## Token Efficiency — Multi-Turn Conversation (12 turns)

The **context pollution** scenario: 12 sequential messages about a product launch, with exact constraints ($50,000 budget, March 15, 2026 deadline) that must be retained across all turns.

| Metric | March (actual) | OpenClaw (actual*) |
|--------|------:|--------:|
| **Total tokens (12 turns)** | 146,363 | ~116,942 |
| **Total cost** | $2.58 | ~$2.00 |
| **Wall time** | 153s | 145s |
| **Turn 1 input** | 9,307 | ~7,700 |
| **Turn 12 input** | 15,685 | ~14,078 |
| **Context growth** | +67% over 12 turns | +83% over 12 turns |
| **Constraint recall (turn 7)** | ✅ 100% | ✅ 100% |
| **Constraint recall (turn 12)** | ✅ 100% | ✅ 100% |

*\*OpenClaw token counts estimated from content length — streaming mode doesn't report usage. True 12-turn multi-turn via `openclaw agent --session-id`.*

**Both agents retain constraints perfectly** across all 12 turns. March uses ~25% more tokens due to its larger system prompt (~9.3K vs ~7.5K per turn), but this includes richer agent context (session memory instructions, execution mode selection, task decomposition guidelines).

### March's Advantage: session_memory

On turn 1, March automatically called `session_memory(type="facts")` to persist the budget and deadline constraints to disk. This means:
- Constraints survive **context compaction** (when conversation exceeds the context window)
- Constraints survive **session restarts**
- In a 50+ turn conversation where compaction triggers, March would retain constraints while a standard linear approach would lose them

OpenClaw has no equivalent persistent memory tool for mid-conversation facts.

---

## Feature Comparison

### ✅ March advantages over OpenClaw

| Feature | March | OpenClaw |
|---------|-------|----------|
| **LLM providers** | 6 native (Bedrock, Anthropic, OpenAI, Ollama, OpenRouter, LiteLLM) | 1 (LiteLLM proxy required) |
| **Sub-agent isolation** | Multi-process (mpAgent) with fault isolation — child crash doesn't kill parent | In-process only — crash kills session |
| **Session memory** | Persistent facts/plans/checkpoints/progress — survives compaction | No equivalent |
| **Cost tracking** | Built-in per-turn cost plugin with JSONL logging | No native cost tracking (LiteLLM DB required) |
| **Context compaction** | Configurable threshold, dedup ratio, summary budget | Basic compaction |
| **Plugin system** | Lifecycle hooks (before/after LLM, safety, git context, cost) | Skill-based (file-level) |
| **MCP support** | Native stdio JSON-RPC client with auto-discovery | Via external skill (mcporter) |
| **Turn logging** | Structured JSONL with per-turn token/cost/duration metrics | Session JSONL (no usage data in streaming mode) |
| **Git context** | Auto-injects repo status into agent context | Manual via skill |
| **Safety plugin** | Configurable guardrails with deny patterns | Rule-based in system prompt |
| **Codebase** | 23.5K lines Python — auditable, hackable | 145K lines TypeScript — complex, harder to modify |

### ✅ OpenClaw advantages over March

| Feature | OpenClaw | March |
|---------|----------|-------|
| **Channels** | 10+ (Telegram, WhatsApp, Discord, Signal, Matrix, Slack, IRC, iMessage, Line, Google Chat) | 4 (Matrix, Terminal, WebSocket, ACP) |
| **Browser automation** | Full Playwright-based browser control | Basic browser tool |
| **Node pairing** | Mobile device integration (camera, location, notifications) | Not implemented |
| **Canvas** | UI rendering and presentation | Not implemented |
| **Cron/scheduling** | Built-in cron with channel delivery | Not implemented |
| **Semantic memory** | SQLite + FAISS vector search over workspace files | File-based memory (no vector search) |
| **Media understanding** | Image/PDF/audio analysis pipeline | Basic PDF + image tools |
| **Community** | Open-source with active community, skill marketplace (ClawHub) | Private, single-developer |
| **Maturity** | Production-tested across thousands of users | Pre-release (v0.1.0) |

### Feature parity

Both support: E2EE Matrix, sub-agent spawning, context compaction, tool execution, web search/fetch, file operations, GitHub integration, TTS, voice-to-text, ACP (IDE integration).

---

## Architectural Differences

```
March Architecture (Lean)              OpenClaw Architecture (Full-Stack)
┌─────────────────────┐                ┌──────────────────────────┐
│   Channels (4)      │                │   Channels (10+)         │
│   Matrix│Term│WS│ACP│                │   TG│WA│DC│Sig│Mx│Slack… │
└────────┬────────────┘                └────────┬─────────────────┘
         │                                      │
┌────────▼────────────┐                ┌────────▼─────────────────┐
│   Orchestrator      │                │   Gateway (Node.js)      │
│   Plugin Pipeline   │                │   Session Manager        │
│   Context Builder   │                │   Context Engine         │
└────────┬────────────┘                └────────┬─────────────────┘
         │                                      │
┌────────▼────────────┐                ┌────────▼─────────────────┐
│   Agent Core        │                │   Pi Embedded Runner     │
│   Session Memory    │                │   (Agent Loop)           │
│   Compaction        │                │                          │
│   Turn Log          │                │                          │
└────────┬────────────┘                └────────┬─────────────────┘
         │                                      │
┌────────▼────────────┐                ┌────────▼─────────────────┐
│   LLM Router        │                │   LiteLLM Proxy          │
│   6 Native Providers│                │   (External Dependency)  │
│   Bedrock│Anthropic │                │                          │
│   OpenAI│Ollama│OR  │                │                          │
└────────┬────────────┘                └──────────────────────────┘
         │
┌────────▼────────────┐
│   Sub-Agents        │
│   mtAgent (async)   │
│   mpAgent (process) │
│   IPC + Heartbeats  │
└─────────────────────┘
```

**Key architectural difference:** March talks directly to LLM providers (Bedrock, Anthropic, OpenAI, Ollama, OpenRouter) with native implementations. OpenClaw routes all LLM calls through a LiteLLM proxy — an additional network hop and external dependency.

---

## Conclusions

1. **Token efficiency is comparable.** Both agents use similar total tokens for equivalent tasks. March's system prompt is ~20% larger but includes richer autonomous behavior context.

2. **Goal retention is identical.** Both achieve 100% across all 4 benchmark scenarios — the metric that actually matters.

3. **March is 6.2× leaner.** 23.5K lines vs 145K lines for comparable core functionality. Smaller codebase = easier to audit, modify, and deploy.

4. **March has architectural advantages for production agents:** fault-isolated sub-agents (mpAgent), persistent session memory, native multi-provider LLM support, built-in cost tracking, and a plugin system with lifecycle hooks.

5. **OpenClaw has broader integration.** 10+ messaging channels, browser automation, mobile device pairing, and a community ecosystem. These are valuable for consumer-facing deployments.

6. **March is purpose-built for autonomous agent workloads.** OpenClaw is a general-purpose AI assistant platform. March trades breadth for depth — fewer channels, but deeper agent capabilities (memory persistence, fault isolation, cost awareness).

---

*Benchmark data: `benchmarks/results/` · Methodology: [token-benchmark-plan.md](benchmarks/token-benchmark-plan.md)*
