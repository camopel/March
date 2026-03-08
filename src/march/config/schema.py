"""Configuration schema — Pydantic v2 models for all March config sections."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── LLM Provider Config ───


class LLMCostConfig(BaseModel):
    """Per-million-token cost in USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider.

    Supported parameters vary by provider:
      - All: model, url, api_key, max_tokens, temperature, timeout, cost
      - Bedrock: region, profile (AWS)
      - OpenRouter: app_name, app_url
      - OpenAI: organization
    """

    model: str = ""
    name: str = ""
    url: str = ""
    api_key: str = ""
    region: str = ""           # AWS region (Bedrock only)
    profile: str = ""          # AWS profile (Bedrock only)
    organization: str = ""     # OpenAI organization
    max_tokens: int = 16384
    context_window: int = 128000
    temperature: float = 0.3
    timeout: int = 300
    reasoning: bool = False
    streaming: bool = True
    input_types: list[str] = Field(default_factory=lambda: ["text"])
    cost: LLMCostConfig = Field(default_factory=LLMCostConfig)


class LLMConfig(BaseModel):
    """LLM routing and provider configuration."""

    default: str = ""
    fallback_chain: list[str] = Field(default_factory=list)
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)


# ─── Tool Configs ───


class ExecToolConfig(BaseModel):
    """Exec tool configuration."""

    sandbox: bool = True
    timeout: int = 30
    pty: bool = False


class WebSearchToolConfig(BaseModel):
    """Web search tool configuration."""

    engine: str = "ddgs"
    max_results: int = 10
    backends: list[str] = Field(default_factory=lambda: ["google"])


class BrowserToolConfig(BaseModel):
    """Browser tool configuration."""

    backend: str = "playwright"


class VoiceToTextToolConfig(BaseModel):
    """Voice-to-text tool configuration."""

    model: str = "large-v3"
    device: str = "auto"
    language: str = "auto"


class TTSToolConfig(BaseModel):
    """Text-to-speech tool configuration."""

    backend: str = "system"
    voice: str = "default"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    """Tool system configuration."""

    builtin: list[str] = Field(
        default_factory=lambda: [
            "read", "write", "edit", "apply_patch", "exec", "process",
            "web_search", "web_fetch",
            "pdf", "browser", "screenshot",
            "clipboard", "message", "diff", "glob",
            "voice_to_text", "tts", "translate",
            "sessions_list", "sessions_history", "sessions_send",
            "sessions_spawn", "subagents", "session_status",
        ]
    )
    deny: list[str] = Field(default_factory=list)
    default_profile: str = "full"
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    browser: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    voice_to_text: VoiceToTextToolConfig = Field(default_factory=VoiceToTextToolConfig)
    tts: TTSToolConfig = Field(default_factory=TTSToolConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


# ─── Memory Config ───


class SessionMemoryConfig(BaseModel):
    """Session memory configuration."""

    auto_save: bool = True


class GlobalMemoryConfig(BaseModel):
    """Global memory configuration — written only on /rmb command."""

    pass


class CompactionConfig(BaseModel):
    """Compaction configuration for context window management.

    The dedup_session_memory() function handles deduplication with a size
    target of min(current_size, context_window * 30%).  No separate
    facts/plan budget ratios are needed.
    """

    threshold: float = 0.95  # Trigger compaction at this % of context window
    summary_budget_ratio: float = 0.15  # Fraction of context window reserved for summary


class MemoryConfig(BaseModel):
    """Two-tier memory system configuration (FileMemory + SQLiteStore)."""

    system_rules: str = "SYSTEM.md"
    agent_profile: str = "AGENT.md"
    tool_rules: str = "TOOLS.md"
    session: SessionMemoryConfig = Field(default_factory=SessionMemoryConfig)
    global_memory: GlobalMemoryConfig = Field(default_factory=GlobalMemoryConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)


# ─── Channel Configs ───


class TerminalChannelConfig(BaseModel):
    """Terminal channel configuration."""

    enabled: bool = True
    theme: Literal["dark", "light"] = "dark"


class ACPChannelConfig(BaseModel):
    """ACP (Agent Client Protocol) channel configuration."""

    enabled: bool = True
    auto_register: bool = True


class MatrixChannelConfig(BaseModel):
    """Matrix channel configuration."""

    enabled: bool = False
    homeserver: str = "auto"
    user: str = "@march:localhost"
    password: str = ""
    access_token: str = ""
    auto_setup: bool = True
    rooms: list[str] = Field(default_factory=list)
    e2ee: bool = True


class ChannelsConfig(BaseModel):
    """All channel configurations."""

    terminal: TerminalChannelConfig = Field(default_factory=TerminalChannelConfig)
    acp: ACPChannelConfig = Field(default_factory=ACPChannelConfig)
    matrix: MatrixChannelConfig = Field(default_factory=MatrixChannelConfig)


# ─── Plugin Configs ───


class WSProxyPluginConfig(BaseModel):
    """WS Proxy plugin configuration.

    Runs an embedded HTTP/WS server for the frontend chat app.
    """

    port: int = 8101
    host: str = "0.0.0.0"
    cors_origins: list[str] = Field(default_factory=list)
    db_path: str = "~/.march/ws_proxy.db"
    max_image_dimension: int = 1200
    image_quality: int = 85
    message_buffer_seconds: float = 3.0
    max_message_size: int = 20 * 1024 * 1024
    stream_drain_timeout: int = 120
    max_upload_bytes: int = 20 * 1024 * 1024  # 20MB max upload
    summary_max_tokens: int = 500  # Max tokens for attachment summaries
    summary_chunk_size: int = 4000  # Chars per chunk for large file summarization


class PluginsConfig(BaseModel):
    """Plugin system configuration."""

    enabled: list[str] = Field(default_factory=list)
    directory: str = "plugins"
    ws_proxy: WSProxyPluginConfig = Field(default_factory=WSProxyPluginConfig)


# ─── Agent / Sub-Agent Config ───


class SubagentConfig(BaseModel):
    """Sub-agent configuration."""

    max_concurrent: int = 8
    max_spawn_depth: int = 1
    max_children_per_agent: int = 5
    run_timeout_seconds: int = 0
    archive_after_minutes: int = 60
    announce_timeout_seconds: int = 60


class AgentIdentityConfig(BaseModel):
    """Agent identity configuration."""

    name: str = "march"
    version: str = "0.1.0"


class AgentsConfig(BaseModel):
    """Agent manager configuration."""

    identity: AgentIdentityConfig = Field(default_factory=AgentIdentityConfig)
    max_concurrent: int = 4
    subagents: SubagentConfig = Field(default_factory=SubagentConfig)


# ─── Dashboard Config ───


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    enabled: bool = True
    port: str = "auto"
    open_browser: bool = False


# ─── i18n Config ───


class I18nConfig(BaseModel):
    """Internationalization configuration."""

    locale: str = "auto"


# ─── Root Config ───


class MarchConfig(BaseModel):
    """Root configuration model for March agent framework.

    All sections are optional with sensible defaults. Config is loaded from
    ~/.march/config.yaml and validated at startup.
    """

    agent: AgentIdentityConfig = Field(default_factory=AgentIdentityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)

    model_config = {"extra": "ignore"}
