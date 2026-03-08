# Available Tools

## File Operations
- **read** — Read file contents (text or images)
- **write** — Create or overwrite files
- **edit** — Replace exact text in a file
- **apply_patch** — Apply unified diff patches
- **glob** — Find files matching a pattern
- **diff** — Compare files or strings

## Execution
- **exec** — Run shell commands (install CLIs via pip/brew/apt as needed)
- **process** — Manage background exec sessions (list, poll, log, write, kill)

## Web & Search
- **web_search** — Search the web (DuckDuckGo, multi-backend, no API key)
- **web_fetch** — Fetch URL content as markdown
- **browser** — Browser automation (navigate, click, type, screenshot, evaluate)

## Media & Analysis
- **pdf** — Analyze PDF documents
- **screenshot** — Capture screen
- **clipboard** — Read clipboard content
- **voice_to_text** — Transcribe audio (Whisper, local)
- **tts** — Text-to-speech
- **translate** — LLM-powered translation

## Messaging & Sessions
- **message** — Send messages across channels
- **session_memory** — Save facts or plans to session memory. Types: facts, plan, progress, checkpoint. Facts go to facts.md; plan/progress/checkpoint all go to plan.md. Auto-timestamped, append-only.
- **cron** — Scheduled jobs (create, list, delete, enable/disable)
- **sessions_list** — List sessions
- **sessions_history** — Get session history
- **sessions_send** — Send message to another session
- **sessions_spawn** — Spawn a sub-agent
- **subagents** — Manage sub-agents (list, steer, kill)
- **session_status** — Current session stats

## CLI Tools (use via exec)
- Install any CLI tool as needed via `pip`, `brew`, or `apt`
- Check TOOLS.md in ~/.march/ for environment-specific notes
