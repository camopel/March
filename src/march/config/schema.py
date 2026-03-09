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

    timeout: int = 30


class WebSearchToolConfig(BaseModel):
    """Web search tool configuration."""

    engine: str = "ddgs"
    max_results: int = 10
    backends: list[str] = Field(default_factory=lambda: ["google"])


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

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    voice_to_text: VoiceToTextToolConfig = Field(default_factory=VoiceToTextToolConfig)
    tts: TTSToolConfig = Field(default_factory=TTSToolConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


# ─── Memory Config ───


class SessionMemoryConfig(BaseModel):
    """Session memory configuration."""

    auto_save: bool = True


class CompactionConfig(BaseModel):
    """Compaction configuration for context window management.

    The dedup_session_memory() function handles deduplication with a size
    target of min(current_size, context_window * 30%).  No separate
    facts/plan budget ratios are needed.
    """

    threshold: float = 0.95  # Trigger compaction at this % of context window
    summary_budget_ratio: float = 0.15  # Fraction of context window reserved for summary
    dedup_max_ratio: float = 0.30  # Max ratio of context window for dedup target


class MemoryConfig(BaseModel):
    """Two-tier memory system configuration (FileMemory + SQLiteStore)."""

    system_rules: str = "SYSTEM.md"
    agent_profile: str = "AGENT.md"
    tool_rules: str = "TOOLS.md"
    memory_path: str = "MEMORY.md"
    session: SessionMemoryConfig = Field(default_factory=SessionMemoryConfig)
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


class WSProxyChannelConfig(BaseModel):
    """WS Proxy channel configuration.

    Runs an embedded HTTP/WS server for the frontend chat app.
    """

    enabled: bool = True
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


# Deprecated alias — use WSProxyChannelConfig instead
WSProxyPluginConfig = WSProxyChannelConfig


class ChannelsConfig(BaseModel):
    """All channel configurations."""

    terminal: TerminalChannelConfig = Field(default_factory=TerminalChannelConfig)
    acp: ACPChannelConfig = Field(default_factory=ACPChannelConfig)
    matrix: MatrixChannelConfig = Field(default_factory=MatrixChannelConfig)
    ws_proxy: WSProxyChannelConfig = Field(default_factory=WSProxyChannelConfig)


# ─── Plugin Configs ───


class PluginsConfig(BaseModel):
    """Plugin system configuration."""

    enabled: list[str] = Field(default_factory=list)
    directory: str = "plugins"


# ─── Agent / Sub-Agent Config ───


class MtConfig(BaseModel):
    """mtAgent (multi-thread, asyncio) configuration."""

    max_concurrent: int = 8


class MpConfig(BaseModel):
    """mpAgent (multi-process, isolated) configuration."""

    max_concurrent: int = 8
    heartbeat_interval_seconds: int = 60
    heartbeat_timeout_seconds: int = 300
    kill_grace_seconds: int = 10
    spawn_method: str = "spawn"  # "spawn" | "forkserver"


class SubagentsCommonConfig(BaseModel):
    """Common sub-agent configuration shared by mt and mp."""

    max_spawn_depth: int = 1


class AgentsConfig(BaseModel):
    """Agent manager configuration."""

    max_concurrent: int = 4
    mt: MtConfig = Field(default_factory=MtConfig)
    mp: MpConfig = Field(default_factory=MpConfig)
    subagents: SubagentsCommonConfig = Field(default_factory=SubagentsCommonConfig)


# ─── Dashboard Config ───


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    enabled: bool = True
    port: int | str = "auto"


# ─── Root Config ───


class MarchConfig(BaseModel):
    """Root configuration model for March agent framework.

    All sections are optional with sensible defaults. Config is loaded from
    ~/.march/config.yaml and validated at startup.
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    model_config = {"extra": "ignore"}
