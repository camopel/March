"""Dashboard HTTP server for March.

Serves a single HTML page with fetch-based updates showing:
- Active sessions
- Sub-agent status
- Cost summary
- Memory stats
- Provider health
- Log stream
"""

from __future__ import annotations

import asyncio
import json
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any

from march.logging import get_logger

logger = get_logger("march.dashboard")

# Shared state for the dashboard API
_dashboard_state: dict[str, Any] = {
    "sessions": [],
    "subagents": [],
    "cost": {"session_cost": 0.0, "daily_cost": 0.0, "session_tokens": 0, "daily_tokens": 0},
    "memory": {"file": "MEMORY.md", "size_bytes": 0},
    "providers": {},
    "logs": [],
    "started_at": time.time(),
}


def update_state(key: str, value: Any) -> None:
    """Update a dashboard state key."""
    _dashboard_state[key] = value


def get_state() -> dict[str, Any]:
    """Get the current dashboard state."""
    return dict(_dashboard_state)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>March Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f6f8fa; color: #1f2328; padding: 20px; }
h1 { color: #0969da; margin-bottom: 20px; font-size: 1.6em; }
h2 { color: #656d76; font-size: 1.1em; margin-bottom: 10px; text-transform: uppercase;
     letter-spacing: 0.05em; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 16px; margin-bottom: 20px; }
.card { background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px;
        padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.stat { font-size: 2em; color: #0969da; font-weight: bold; }
.stat-label { color: #656d76; font-size: 0.85em; }
.stat-row { display: flex; gap: 30px; margin-top: 8px; }
.stat-item { text-align: center; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #d8dee4; }
th { color: #656d76; font-weight: 500; }
.status-ok { color: #1a7f37; }
.status-warn { color: #9a6700; }
.status-err { color: #cf222e; }
#log-stream { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8em;
              background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px;
              padding: 10px; height: 300px; overflow-y: auto; white-space: pre-wrap;
              word-wrap: break-word; }
.log-entry { margin-bottom: 2px; line-height: 1.4; }
.log-time { color: #8b949e; }
.log-info { color: #1a7f37; }
.log-warn { color: #9a6700; }
.log-error { color: #cf222e; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 0.75em; font-weight: 600; }
.badge-green { background: #dafbe1; color: #1a7f37; }
.badge-yellow { background: #fff8c5; color: #9a6700; }
.badge-red { background: #ffebe9; color: #cf222e; }
footer { text-align: center; color: #8b949e; margin-top: 30px; font-size: 0.8em; }
</style>
</head>
<body>
<h1>Dashboard</h1>

<div class="grid">
  <div class="card">
    <h2>💰 Cost</h2>
    <div class="stat" id="session-cost">$0.00</div>
    <div class="stat-label">Session Cost</div>
    <div class="stat-row">
      <div class="stat-item">
        <div id="daily-cost" style="font-size:1.2em;color:#1f2328;">$0.00</div>
        <div class="stat-label">Daily</div>
      </div>
      <div class="stat-item">
        <div id="session-tokens" style="font-size:1.2em;color:#1f2328;">0</div>
        <div class="stat-label">Session Tokens</div>
      </div>
      <div class="stat-item">
        <div id="daily-tokens" style="font-size:1.2em;color:#1f2328;">0</div>
        <div class="stat-label">Daily Tokens</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🧠 Memory</h2>
    <div class="stat-row">
      <div class="stat-item">
        <div id="mem-file" style="font-size:1.2em;color:#1f2328;">MEMORY.md</div>
        <div class="stat-label">File</div>
      </div>
      <div class="stat-item">
        <div id="mem-size" style="font-size:1.2em;color:#1f2328;">0 B</div>
        <div class="stat-label">Size</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>⏱ Uptime</h2>
    <div class="stat" id="uptime">0s</div>
    <div class="stat-label">Since start</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>📡 Sessions</h2>
    <table>
      <thead><tr><th>ID</th><th>Channel</th><th>Status</th></tr></thead>
      <tbody id="sessions-table"><tr><td colspan="3" style="color:#8b949e;">No active sessions</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Sub-Agents</h2>
    <table>
      <thead><tr><th>ID</th><th>Task</th><th>Status</th></tr></thead>
      <tbody id="subagents-table"><tr><td colspan="3" style="color:#8b949e;">No sub-agents</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2>🔌 Providers</h2>
    <table>
      <thead><tr><th>Name</th><th>Model</th><th>Health</th></tr></thead>
      <tbody id="providers-table"><tr><td colspan="3" style="color:#8b949e;">No providers configured</td></tr></tbody>
    </table>
  </div>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>📝 Log Stream</h2>
  <div id="log-stream"></div>
</div>

<footer>Dashboard</footer>

<script>
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}
function formatUptime(s) {
  if (s < 60) return Math.floor(s) + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + Math.floor(s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}
function statusBadge(s) {
  const cls = s === 'healthy' ? 'badge-green' : s === 'degraded' ? 'badge-yellow' : 'badge-red';
  return '<span class="badge ' + cls + '">' + s + '</span>';
}

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();

    document.getElementById('session-cost').textContent = '$' + (d.cost.session_cost || 0).toFixed(4);
    document.getElementById('daily-cost').textContent = '$' + (d.cost.daily_cost || 0).toFixed(4);
    document.getElementById('session-tokens').textContent = (d.cost.session_tokens || 0).toLocaleString();
    document.getElementById('daily-tokens').textContent = (d.cost.daily_tokens || 0).toLocaleString();

    document.getElementById('mem-file').textContent = d.memory.file || 'MEMORY.md';
    document.getElementById('mem-size').textContent = formatBytes(d.memory.size_bytes || 0);

    const uptime = Date.now() / 1000 - (d.started_at || Date.now() / 1000);
    document.getElementById('uptime').textContent = formatUptime(uptime);

    // Sessions
    const st = document.getElementById('sessions-table');
    if (d.sessions && d.sessions.length > 0) {
      st.innerHTML = d.sessions.map(s =>
        '<tr><td>' + s.id + '</td><td>' + s.channel + '</td><td>' + statusBadge(s.status) + '</td></tr>'
      ).join('');
    } else {
      st.innerHTML = '<tr><td colspan="3" style="color:#8b949e;">No active sessions</td></tr>';
    }

    // Sub-agents
    const sa = document.getElementById('subagents-table');
    if (d.subagents && d.subagents.length > 0) {
      sa.innerHTML = d.subagents.map(a =>
        '<tr><td>' + a.id + '</td><td>' + (a.task || '-') + '</td><td>' + statusBadge(a.status) + '</td></tr>'
      ).join('');
    } else {
      sa.innerHTML = '<tr><td colspan="3" style="color:#8b949e;">No sub-agents</td></tr>';
    }

    // Providers
    const pt = document.getElementById('providers-table');
    const providers = Object.entries(d.providers || {});
    if (providers.length > 0) {
      pt.innerHTML = providers.map(([name, p]) =>
        '<tr><td>' + name + '</td><td>' + (p.model || '-') + '</td><td>' + statusBadge(p.health || 'unknown') + '</td></tr>'
      ).join('');
    } else {
      pt.innerHTML = '<tr><td colspan="3" style="color:#8b949e;">No providers configured</td></tr>';
    }

    // Logs
    const ls = document.getElementById('log-stream');
    if (d.logs && d.logs.length > 0) {
      const newLogs = d.logs.slice(-100);
      ls.innerHTML = newLogs.map(l => {
        const cls = l.level === 'ERROR' ? 'log-error' : l.level === 'WARNING' ? 'log-warn' : 'log-info';
        return '<div class="log-entry"><span class="log-time">' + (l.time || '') + '</span> <span class="' + cls + '">[' + (l.level || 'INFO') + ']</span> ' + (l.message || '') + '</div>';
      }).join('');
      ls.scrollTop = ls.scrollHeight;
    }
  } catch (e) {
    console.error('Dashboard refresh failed:', e);
  }
}

setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>"""


class _DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/state":
            self._serve_json(get_state())
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        content = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_json(self, data: dict[str, Any]) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default access log output."""
        pass


class DashboardServer:
    """Simple embedded HTTP server for the March dashboard.

    Usage:
        server = DashboardServer(port=8200)
        server.start()     # Starts in background thread
        server.stop()

        # Or as context manager:
        with DashboardServer(port=8200) as server:
            ...
    """

    def __init__(self, port: int = 8200, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    def start(self, open_browser: bool = False) -> None:
        """Start the dashboard server in a background thread."""
        self._server = HTTPServer((self.host, self.port), _DashboardHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://{self.host}:{self.port}"
        logger.info("dashboard.started url=%s", url)

        if open_browser:
            webbrowser.open(url)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("dashboard.stopped")

    @property
    def url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "DashboardServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
