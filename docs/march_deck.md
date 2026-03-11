# March Deck — PWA app platform

Want a mobile-friendly UI? **[March Deck](https://github.com/camopel/march-deck)** is a PWA platform that turns your agent into a collection of mini-apps you can open on any device.

No app store. No account. Add to home screen and go. Works with any agent that exposes a WebSocket endpoint.

## Apps

- 🤖 **March** — chat with your agent + dashboard (sessions, cost, providers, logs)
- 📰 **Finviz** — financial news with 24h AI summaries
- 📄 **ArXiv** — research paper semantic search
- 📊 **System** — server monitoring (CPU, RAM, GPU, services)
- 📁 **Files** — file browser
- 📝 **Notes** — markdown notes
- 📺 **Cast** — media casting and control
- 🦞 **OpenClaw** — OpenClaw agent management

## How it works

Each app is a self-contained mini-app served by a single Python backend. The March app connects to your agent via WebSocket — the rest are standalone tools that run alongside your agent.

Data is stored per-app at `~/.march-deck/{app name}/`.

## Build your own

March Deck apps are just HTML + CSS + JS served by a Python backend. Drop a new app directory in, register it, and it shows up in the deck. See the [March Deck repo](https://github.com/camopel/march-deck) for the full guide.
