"""Default configuration values and default config.yaml content."""

from __future__ import annotations

DEFAULT_CONFIG_YAML = """\
# March Agent Framework Configuration
# ~/.march/config.yaml
#
# Environment variable interpolation: ${VAR} or ${VAR:default}
# Pydantic-validated at startup — bad config fails fast with clear errors.

# ─── Agent Identity ───
agent:
  name: "march"
  emoji: ""
  version: "0.1.0"

# ─── LLM Providers ───
# Configure at least one provider in config.yaml.
# March will fail fast with a clear error if no provider is configured.
llm:
  default: ""
  fallback_chain: []
  providers: {}

# ─── Tools ───
tools:
  builtin:
    - read
    - write
    - edit
    - apply_patch
    - exec
    - process
    - web_search
    - web_fetch
    - pdf
    - browser
    - screenshot
    - clipboard
    - message
    - diff
    - glob
    - voice_to_text
    - tts
    - translate
    - sessions_list
    - sessions_history
    - sessions_send
    - sessions_spawn
    - subagents
    - session_status

  deny: []
  default_profile: "full"

  exec:
    sandbox: true
    timeout: 30
    pty: false

  web_search:
    engine: "ddgs"
    max_results: 10
    backends:
      - "google"

  browser:
    backend: "playwright"

  voice_to_text:
    model: "large-v3"
    device: "auto"
    language: "auto"

  tts:
    backend: "system"
    voice: "default"

  mcp_servers: {}

# ─── Memory ───
memory:
  system_rules: "SYSTEM.md"
  agent_profile: "AGENT.md"
  tool_rules: "TOOLS.md"

  session:
    auto_save: true

  global_memory: {}

# ─── Channels ───
channels:
  terminal:
    enabled: true
    theme: "dark"
    streaming: true

  acp:
    enabled: false
    auto_register: true

  vscode:
    enabled: false
    use_websocket: true
    use_acp: true

  matrix:
    enabled: false
    homeserver: "auto"
    user: "@march:localhost"
    password: "auto"
    auto_setup: true
    rooms:
      - "#agents:localhost"
    e2ee: true

# ─── Plugins ───
plugins:
  enabled:
    - "safety"
    - "cost"
    - "logger"
    - "git_context"
  directory: "plugins"

  safety:
    require_confirmation:
      - "exec"

  cost:
    budget_per_session: 5.00
    budget_per_day: 20.00
    alert_threshold: 0.8

  logger:
    log_tool_results: true
    log_llm_calls: true

  git_context:
    auto_detect: true
    inject_branch: true
    inject_status: true
    inject_diff: false

# ─── Sub-Agents ───
agents:
  identity:
    name: "march"
    emoji: ""
    version: "0.1.0"
  max_concurrent: 4
  subagents:
    max_concurrent: 8
    max_spawn_depth: 1
    max_children_per_agent: 5
    run_timeout_seconds: 0
    archive_after_minutes: 60
    announce_timeout_seconds: 60

# ─── Guardian ───
guardian:
  enabled: true
  log_stale_threshold: 300
  config_backup_count: 5
  default_channel: "matrix"
  notification:
    type: "stdout"
    url: ""
    room: ""

# ─── Logging ───
logging:
  level: "INFO"
  format: "both"
  file: ".march/logs/session.log"
  retention: 7
  audit_trail: true

# ─── Dashboard ───
dashboard:
  enabled: true
  port: "auto"
  open_browser: false

# ─── Language ───
i18n:
  locale: "auto"
"""
