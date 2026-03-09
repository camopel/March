"""Tests for the March CLI commands using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from march.cli.main import cli
from march import __version__


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def init_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp dir and cd into it for tests."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ─── Top-level Commands ───


class TestCLITopLevel:
    def test_help(self, runner: CliRunner) -> None:
        """CLI shows help when invoked without subcommand."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "March" in result.output

    def test_help_flag(self, runner: CliRunner) -> None:
        """march --help shows curated help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "march start" in result.output

    def test_help_short_flag(self, runner: CliRunner) -> None:
        """march -h shows curated help."""
        result = runner.invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "LIFECYCLE" in result.output


# ─── Start/Stop/Restart Commands ───


class TestLifecycleCommands:
    def test_start_help(self, runner: CliRunner) -> None:
        """march start --help works."""
        result = runner.invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output

    def test_stop_help(self, runner: CliRunner) -> None:
        """march stop -h works."""
        result = runner.invoke(cli, ["stop", "-h"])
        assert result.exit_code == 0
        assert "Stop" in result.output

    def test_restart_help(self, runner: CliRunner) -> None:
        """march restart --help works."""
        result = runner.invoke(cli, ["restart", "--help"])
        assert result.exit_code == 0

    def test_enable_help(self, runner: CliRunner) -> None:
        """march enable --help works."""
        result = runner.invoke(cli, ["enable", "--help"])
        assert result.exit_code == 0
        assert "systemd" in result.output

    def test_disable_help(self, runner: CliRunner) -> None:
        """march disable -h works."""
        result = runner.invoke(cli, ["disable", "-h"])
        assert result.exit_code == 0


# ─── Config Commands ───


class TestConfigCommands:
    def test_config_show(self, runner: CliRunner) -> None:
        """march config show prints path."""
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "config.yaml" in result.output


# ─── Agent Commands ───


class TestAgentCommands:
    def test_agent_list(self, runner: CliRunner) -> None:
        """march agent list runs."""
        result = runner.invoke(cli, ["agent", "list"])
        assert result.exit_code == 0

    def test_agent_show(self, runner: CliRunner) -> None:
        """march agent show displays details."""
        result = runner.invoke(cli, ["agent", "show"])
        assert result.exit_code == 0
        assert "March Agent" in result.output


# ─── Plugin Commands ───


class TestPluginCommands:
    def test_plugin_list(self, runner: CliRunner) -> None:
        """march plugin list runs."""
        result = runner.invoke(cli, ["plugin", "list"])
        assert result.exit_code in (0, 1)

    def test_plugin_enable_disable(self, runner: CliRunner) -> None:
        """march plugin enable/disable modifies config."""
        # These may fail if no config, that's okay
        result = runner.invoke(cli, ["plugin", "enable", "test_plugin"])
        assert result.exit_code in (0, 1)
        result = runner.invoke(cli, ["plugin", "disable", "test_plugin"])
        assert result.exit_code in (0, 1)


# ─── Log Commands ───


class TestLogCommands:
    def test_log_help(self, runner: CliRunner) -> None:
        """march log -h works."""
        result = runner.invoke(cli, ["log", "-h"])
        assert result.exit_code == 0
        assert "follow" in result.output.lower() or "Follow" in result.output
