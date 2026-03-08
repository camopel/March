"""Tests for the March config system."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from march.config.interpolation import interpolate_config, interpolate_value
from march.config.loader import (
    ensure_config_exists,
    load_config,
    load_raw_yaml,
    reset_cache,
    ConfigNotFoundError,
)
from march.config.schema import (
    AgentIdentityConfig,
    AgentsConfig,
    ChannelsConfig,
    CompactionConfig,
    DashboardConfig,
    I18nConfig,
    LLMConfig,
    LLMProviderConfig,
    MarchConfig,
    MemoryConfig,
    PluginsConfig,
    SubagentConfig,
    ToolsConfig,
    WSProxyChannelConfig,
    WSProxyPluginConfig,
)


# ─── Fixtures ───


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset config cache before each test."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a temporary config file with minimal valid config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n"
        '  default: "openai"\n'
        "  providers:\n"
        "    openai:\n"
        '      model: "gpt-4o"\n'
        '      api_key: "test"\n',
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Create a minimal valid config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent("""\
            agent:
              name: "test-agent"
            llm:
              default: ""
        """),
        encoding="utf-8",
    )
    return config_path


# ─── Schema Tests ───


class TestSchemaDefaults:
    """Test that schema defaults are sensible."""

    def test_march_config_defaults(self):
        """MarchConfig with no input should produce valid defaults."""
        config = MarchConfig()
        assert config.agent.name == "march"
        assert config.llm.default == ""
        assert config.i18n.locale == "auto"

    def test_llm_config_defaults(self):
        config = LLMConfig()
        assert config.default == ""
        assert config.fallback_chain == []
        assert config.providers == {}

    def test_llm_provider_config(self):
        provider = LLMProviderConfig(model="gpt-4o", name="GPT-4o")
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.3
        assert provider.cost.input == 0.0

    def test_tools_config_defaults(self):
        config = ToolsConfig()
        assert "read" in config.builtin
        assert "write" in config.builtin
        assert "exec" in config.builtin
        assert config.deny == []
        assert config.default_profile == "full"
        assert config.exec.sandbox is True
        assert config.exec.timeout == 30

    def test_memory_config_defaults(self):
        config = MemoryConfig()
        assert config.system_rules == "SYSTEM.md"
        assert config.agent_profile == "AGENT.md"
        assert config.tool_rules == "TOOLS.md"

    def test_compaction_config_defaults(self):
        config = CompactionConfig()
        assert config.threshold == 0.95
        assert config.summary_budget_ratio == 0.15
        assert config.dedup_max_ratio == 0.30

    def test_compaction_config_override(self):
        config = CompactionConfig(threshold=0.80, summary_budget_ratio=0.20, dedup_max_ratio=0.40)
        assert config.threshold == 0.80
        assert config.summary_budget_ratio == 0.20
        assert config.dedup_max_ratio == 0.40

    def test_memory_config_has_compaction(self):
        config = MemoryConfig()
        assert hasattr(config, "compaction")
        assert isinstance(config.compaction, CompactionConfig)
        assert config.compaction.threshold == 0.95

    def test_march_config_compaction_access(self):
        config = MarchConfig()
        assert config.memory.compaction.threshold == 0.95
        assert config.memory.compaction.summary_budget_ratio == 0.15

    def test_channels_config_defaults(self):
        config = ChannelsConfig()
        assert config.terminal.enabled is True
        assert config.matrix.enabled is False
        assert config.matrix.e2ee is True

    def test_plugins_config_defaults(self):
        config = PluginsConfig()
        assert config.enabled == []

    def test_ws_proxy_channel_config_defaults(self):
        config = WSProxyChannelConfig()
        assert config.port == 8101
        assert config.host == "0.0.0.0"
        assert config.cors_origins == []
        assert config.max_image_dimension == 1200

    def test_ws_proxy_in_channels(self):
        config = ChannelsConfig()
        assert hasattr(config, "ws_proxy")
        assert isinstance(config.ws_proxy, WSProxyChannelConfig)
        assert config.ws_proxy.port == 8101

    def test_ws_proxy_deprecated_alias(self):
        """WSProxyPluginConfig should be an alias for WSProxyChannelConfig."""
        assert WSProxyPluginConfig is WSProxyChannelConfig
        cfg = WSProxyPluginConfig(port=9999)
        assert isinstance(cfg, WSProxyChannelConfig)
        assert cfg.port == 9999

    def test_agents_config_defaults(self):
        config = AgentsConfig()
        assert config.max_concurrent == 4
        assert config.subagents.max_concurrent == 8
        assert config.subagents.max_spawn_depth == 1
        assert config.subagents.reset_after_complete_minutes == 60
        assert config.subagents.announce_timeout_seconds == 60

    def test_dashboard_config_defaults(self):
        config = DashboardConfig()
        assert config.enabled is True
        assert config.port == "auto"
        assert config.open_browser is False

    def test_i18n_config_defaults(self):
        config = I18nConfig()
        assert config.locale == "auto"


class TestSchemaValidation:
    """Test schema validation catches bad input."""

    def test_invalid_terminal_theme(self):
        from march.config.schema import TerminalChannelConfig

        with pytest.raises(ValidationError):
            TerminalChannelConfig(theme="blue")  # type: ignore[arg-type]

    def test_extra_fields_ignored(self):
        """MarchConfig ignores extra fields at root."""
        config = MarchConfig(unknown_field="value")  # type: ignore[call-arg]
        assert config.agent.name == "march"

    def test_full_config_from_yaml(self):
        """Parse a full YAML config and validate."""
        raw = yaml.safe_load(
            "llm:\n"
            '  default: "openai"\n'
            "  providers:\n"
            "    openai:\n"
            '      model: "gpt-4o"\n'
            '      api_key: "test"\n'
        )
        config = MarchConfig.model_validate(raw)
        assert config.llm.default == "openai"
        assert len(config.tools.builtin) > 20


# ─── Interpolation Tests ───


class TestInterpolation:
    """Test environment variable interpolation."""

    def test_simple_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert interpolate_value("${TEST_VAR}") == "hello"

    def test_var_with_default(self):
        # Ensure the var doesn't exist
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        assert interpolate_value("${NONEXISTENT_VAR_XYZ:fallback}") == "fallback"

    def test_var_with_empty_default(self):
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        assert interpolate_value("${NONEXISTENT_VAR_XYZ:}") == ""

    def test_missing_var_no_default_raises(self):
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        with pytest.raises(ValueError, match="not set and has no default"):
            interpolate_value("${NONEXISTENT_VAR_XYZ}")

    def test_multiple_vars_in_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = interpolate_value("http://${HOST}:${PORT}/api")
        assert result == "http://localhost:8080/api"

    def test_no_vars(self):
        assert interpolate_value("plain text") == "plain text"

    def test_interpolate_config_dict(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("API_KEY", "secret123")
        data = {
            "provider": {
                "api_key": "${API_KEY}",
                "url": "http://localhost:${PORT:4000}",
            },
            "count": 42,
            "enabled": True,
            "items": ["${API_KEY}", "literal"],
        }
        result = interpolate_config(data)
        assert result["provider"]["api_key"] == "secret123"
        assert result["provider"]["url"] == "http://localhost:4000"
        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["items"] == ["secret123", "literal"]

    def test_interpolate_preserves_non_strings(self):
        data = {"a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2, 3]}
        assert interpolate_config(data) == data

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_VAR", "from_env")
        assert interpolate_value("${MY_VAR:from_default}") == "from_env"


# ─── Loader Tests ───


class TestLoader:
    """Test config loading."""

    def test_ensure_config_raises_if_missing(self, tmp_path: Path):
        config_path = tmp_path / "subdir" / "config.yaml"
        assert not config_path.exists()
        with pytest.raises(ConfigNotFoundError, match="march init"):
            ensure_config_exists(config_path)

    def test_ensure_config_doesnt_overwrite(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("custom: true\n")
        result = ensure_config_exists(config_path)
        assert result == config_path
        assert config_path.read_text() == "custom: true\n"

    def test_load_raw_yaml(self, tmp_config: Path):
        data = load_raw_yaml(tmp_config)
        assert isinstance(data, dict)
        assert "llm" in data

    def test_load_raw_yaml_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        data = load_raw_yaml(empty)
        assert data == {}

    def test_load_raw_yaml_nonexistent(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_raw_yaml(tmp_path / "nope.yaml")

    def test_load_config_full_defaults(self, tmp_config: Path):
        config = load_config(tmp_config, use_cache=False)
        assert isinstance(config, MarchConfig)
        assert config.agent.name == "march"
        assert config.llm.default == "openai"
        assert config.channels.terminal.enabled is True

    def test_load_config_minimal(self, minimal_config: Path):
        config = load_config(minimal_config, use_cache=False)
        assert config.agent.name == "test-agent"
        assert config.llm.default == ""
        # Everything else should be defaults

    def test_load_config_caching(self, tmp_config: Path):
        config1 = load_config(tmp_config, use_cache=True)
        config2 = load_config(tmp_config, use_cache=True)
        assert config1 is config2

    def test_load_config_no_cache(self, tmp_config: Path):
        config1 = load_config(tmp_config, use_cache=False)
        config2 = load_config(tmp_config, use_cache=False)
        assert config1 is not config2
        assert config1.agent.name == config2.agent.name

    def test_load_config_with_env_interpolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_MODEL", "gpt-4o")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent("""\
                llm:
                  default: "openai"
                  providers:
                    openai:
                      model: "${MY_MODEL}"
                      name: "GPT-4o"
            """),
            encoding="utf-8",
        )
        config = load_config(config_path, use_cache=False)
        assert config.llm.providers["openai"].model == "gpt-4o"

    def test_load_config_interpolation_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_MODEL", "gpt-4o")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent("""\
                llm:
                  default: "openai"
                  providers:
                    openai:
                      model: "${MY_MODEL}"
                      name: "GPT-4o"
            """),
            encoding="utf-8",
        )
        config = load_config(config_path, use_cache=False, interpolate=False)
        assert config.llm.providers["openai"].model == "${MY_MODEL}"

    def test_load_config_invalid_yaml(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("  bad:\n yaml: [unclosed", encoding="utf-8")
        with pytest.raises(Exception):
            load_config(bad, use_cache=False)

    def test_load_config_invalid_schema(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            textwrap.dedent("""\
                channels:
                  terminal:
                    theme: "neon"
            """),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(bad, use_cache=False)


class TestConfigRoundTrip:
    """Test that default YAML parses correctly and produces valid config."""

    def test_default_yaml_is_valid(self):
        """An empty config should produce valid MarchConfig via defaults."""
        config = MarchConfig()
        assert config.agent.name == "march"

    def test_config_serialization_roundtrip(self):
        """Config should survive dump→load cycle."""
        config = MarchConfig()
        dumped = config.model_dump()
        restored = MarchConfig.model_validate(dumped)
        assert restored.agent.name == config.agent.name
        assert restored.llm.default == config.llm.default

    def test_all_tool_names_present(self):
        """Verify all expected tools are in the default builtin list."""
        config = ToolsConfig()
        expected = {"read", "write", "edit", "exec", "web_search", "web_fetch", "browser"}
        assert expected.issubset(set(config.builtin))
