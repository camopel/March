"""Tests for the March config system."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from march.config.defaults import DEFAULT_CONFIG_YAML
from march.config.interpolation import interpolate_config, interpolate_value
from march.config.loader import (
    ensure_config_exists,
    load_config,
    load_raw_yaml,
    reset_cache,
)
from march.config.schema import (
    AgentIdentityConfig,
    AgentsConfig,
    ChannelsConfig,
    DashboardConfig,
    GuardianConfig,
    I18nConfig,
    LLMConfig,
    LLMProviderConfig,
    LoggingConfig,
    MarchConfig,
    MemoryConfig,
    PluginsConfig,
    SubagentConfig,
    ToolsConfig,
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
    """Create a temporary config file with defaults."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
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
        assert config.agent.emoji == "𝗠𝗔𝗥"
        assert config.llm.default == ""
        assert config.logging.level == "INFO"
        assert config.logging.format == "json"
        assert config.i18n.locale == "auto"

    def test_llm_config_defaults(self):
        config = LLMConfig()
        assert config.default == ""
        assert config.fallback_chain == [""]
        assert config.providers == {}

    def test_llm_provider_config(self):
        provider = LLMProviderConfig(model="gpt-4o", name="GPT-4o")
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.7
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
        assert config.vector_store.backend == "faiss"
        assert config.vector_store.dimension == 1024
        assert config.embeddings.model == "qwen3-embedding:0.6b"

    def test_channels_config_defaults(self):
        config = ChannelsConfig()
        assert config.terminal.enabled is True
        assert config.terminal.streaming is True
        assert config.homehub.enabled is False
        assert config.matrix.enabled is False
        assert config.matrix.e2ee is True

    def test_plugins_config_defaults(self):
        config = PluginsConfig()
        assert "safety" in config.enabled
        assert "cost" in config.enabled
        assert config.cost.budget_per_session == 5.00
        assert config.cost.alert_threshold == 0.8

    def test_agents_config_defaults(self):
        config = AgentsConfig()
        assert config.max_concurrent == 4
        assert config.subagents.max_concurrent == 8
        assert config.subagents.max_spawn_depth == 1
        assert config.subagents.max_children_per_agent == 5

    def test_guardian_config_defaults(self):
        config = GuardianConfig()
        assert config.enabled is True
        assert config.log_stale_threshold == 300
        assert config.config_backup_count == 5
        assert config.notification.type == "stdout"

    def test_logging_config_defaults(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.retention == 7
        assert config.audit_trail is True

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

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")  # type: ignore[arg-type]

    def test_invalid_log_format(self):
        with pytest.raises(ValidationError):
            LoggingConfig(format="xml")  # type: ignore[arg-type]

    def test_invalid_search_strategy(self):
        from march.config.schema import MemorySearchConfig

        with pytest.raises(ValidationError):
            MemorySearchConfig(strategy="neural")  # type: ignore[arg-type]

    def test_invalid_terminal_theme(self):
        from march.config.schema import TerminalChannelConfig

        with pytest.raises(ValidationError):
            TerminalChannelConfig(theme="blue")  # type: ignore[arg-type]

    def test_invalid_guardian_notification_type(self):
        from march.config.schema import GuardianNotificationConfig

        with pytest.raises(ValidationError):
            GuardianNotificationConfig(type="sms")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self):
        """MarchConfig forbids extra fields at root."""
        with pytest.raises(ValidationError):
            MarchConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_full_config_from_yaml(self):
        """Parse the full default YAML and validate."""
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)
        config = MarchConfig.model_validate(raw)
        assert config.agent.name == "march"
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

    def test_ensure_config_creates_file(self, tmp_path: Path):
        config_path = tmp_path / "subdir" / "config.yaml"
        assert not config_path.exists()
        result = ensure_config_exists(config_path)
        assert result == config_path
        assert config_path.exists()
        content = config_path.read_text()
        assert "march" in content

    def test_ensure_config_doesnt_overwrite(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("custom: true\n")
        ensure_config_exists(config_path)
        assert config_path.read_text() == "custom: true\n"

    def test_load_raw_yaml(self, tmp_config: Path):
        data = load_raw_yaml(tmp_config)
        assert isinstance(data, dict)
        assert "agent" in data
        assert data["agent"]["name"] == "march"

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
        assert config.llm.default == ""
        assert config.channels.terminal.enabled is True

    def test_load_config_minimal(self, minimal_config: Path):
        config = load_config(minimal_config, use_cache=False)
        assert config.agent.name == "test-agent"
        assert config.llm.default == ""
        # Everything else should be defaults
        assert config.logging.level == "INFO"

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
                logging:
                  level: "VERBOSE"
            """),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(bad, use_cache=False)


class TestConfigRoundTrip:
    """Test that default YAML parses correctly and produces valid config."""

    def test_default_yaml_is_valid(self):
        """The DEFAULT_CONFIG_YAML should always parse and validate."""
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)
        config = MarchConfig.model_validate(raw)
        assert config.agent.name == "march"

    def test_config_serialization_roundtrip(self):
        """Config should survive dump→load cycle."""
        config = MarchConfig()
        dumped = config.model_dump()
        restored = MarchConfig.model_validate(dumped)
        assert restored.agent.name == config.agent.name
        assert restored.llm.default == config.llm.default
        assert restored.logging.level == config.logging.level

    def test_all_tool_names_present(self):
        """Verify all expected tools are in the default builtin list."""
        config = ToolsConfig()
        expected = {"read", "write", "edit", "exec", "web_search", "memory_search"}
        assert expected.issubset(set(config.builtin))
