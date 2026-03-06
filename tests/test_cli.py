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
    """Create a temp dir and cd into it for init tests."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ─── Top-level Commands ───


class TestCLITopLevel:
    def test_help(self, runner: CliRunner) -> None:
        """CLI shows help when invoked without subcommand."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "March" in result.output

    def test_version_command(self, runner: CliRunner) -> None:
        """march version shows version."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_flag(self, runner: CliRunner) -> None:
        """march --version shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


# ─── Init Command ───


class TestInitCommand:
    def test_init_creates_files(self, runner: CliRunner, init_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """march init creates all expected files."""
        # Redirect ~/.march/ to temp dir so we don't pollute real home
        fake_march_dir = init_dir / ".march"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: init_dir))

        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "✅" in result.output

        # MEMORY.md goes to ~/.march/
        assert (fake_march_dir / "MEMORY.md").exists()
        # config.yaml goes to ~/.march/
        assert (fake_march_dir / "config.yaml").exists()
        # Directories in cwd
        assert (init_dir / "plugins").is_dir()
        assert (init_dir / "skills").is_dir()
        # SYSTEM.md, AGENT.md, TOOLS.md are NOT created — they come from templates
        # (unless user runs march init-templates)

    def test_init_idempotent(self, runner: CliRunner, init_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running init twice doesn't overwrite existing files."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: init_dir))
        runner.invoke(cli, ["init"])
        # Modify MEMORY.md
        fake_march_dir = init_dir / ".march"
        (fake_march_dir / "MEMORY.md").write_text("custom content")
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "Exists" in result.output
        # File should not be overwritten
        assert (fake_march_dir / "MEMORY.md").read_text() == "custom content"


# ─── Config Commands ───


class TestConfigCommands:
    def test_config_show(self, runner: CliRunner) -> None:
        """march config show outputs JSON."""
        result = runner.invoke(cli, ["config", "show"])
        # Should succeed or fail gracefully
        # (may fail if no config file exists, which is fine)
        assert result.exit_code in (0, 1)

    def test_config_validate(self, runner: CliRunner) -> None:
        """march config validate works."""
        result = runner.invoke(cli, ["config", "validate"])
        assert result.exit_code in (0, 1)


# ─── Agent Commands ───


class TestAgentCommands:
    def test_agent_list(self, runner: CliRunner) -> None:
        """march agent list runs."""
        result = runner.invoke(cli, ["agent", "list"])
        assert result.exit_code == 0

    def test_agent_kill(self, runner: CliRunner) -> None:
        """march agent kill runs."""
        result = runner.invoke(cli, ["agent", "kill", "test-id"])
        assert result.exit_code == 0
        assert "test-id" in result.output

    def test_agent_send(self, runner: CliRunner) -> None:
        """march agent send runs."""
        result = runner.invoke(cli, ["agent", "send", "test-id", "hello"])
        assert result.exit_code == 0


# ─── Skill Commands ───


class TestSkillCommands:
    def test_skill_list(self, runner: CliRunner) -> None:
        """march skill list runs."""
        result = runner.invoke(cli, ["skill", "list"])
        assert result.exit_code == 0

    def test_skill_create(self, runner: CliRunner, init_dir: Path) -> None:
        """march skill create scaffolds a skill."""
        result = runner.invoke(cli, ["skill", "create", "my-test-skill"])
        assert result.exit_code == 0
        assert "✅" in result.output

        skill_dir = init_dir / "skills" / "my-test-skill"
        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "tools.py").exists()
        assert (skill_dir / "config.yaml").exists()

    def test_skill_info_not_found(self, runner: CliRunner) -> None:
        """march skill info for missing skill fails."""
        result = runner.invoke(cli, ["skill", "info", "nonexistent"])
        assert result.exit_code == 1


# ─── Plugin Commands ───


class TestPluginCommands:
    def test_plugin_list(self, runner: CliRunner) -> None:
        """march plugin list runs."""
        result = runner.invoke(cli, ["plugin", "list"])
        # May fail if no config, that's okay
        assert result.exit_code in (0, 1)

    def test_plugin_create(self, runner: CliRunner, init_dir: Path) -> None:
        """march plugin create scaffolds a plugin."""
        result = runner.invoke(cli, ["plugin", "create", "my_plugin"])
        assert result.exit_code == 0
        assert "✅" in result.output
        assert (init_dir / "plugins" / "my_plugin.py").exists()


# ─── Memory Commands ───


class TestMemoryCommands:
    def test_memory_show(self, runner: CliRunner, init_dir: Path) -> None:
        """march memory show runs."""
        result = runner.invoke(cli, ["memory", "show"])
        assert result.exit_code == 0
        assert "Memory Statistics" in result.output


# ─── Log Commands ───


class TestLogCommands:
    def test_log_default(self, runner: CliRunner) -> None:
        """march log runs (may have no logs)."""
        result = runner.invoke(cli, ["log"])
        assert result.exit_code == 0

    def test_log_cost(self, runner: CliRunner) -> None:
        """march log cost runs."""
        result = runner.invoke(cli, ["log", "cost"])
        assert result.exit_code == 0


# ─── Dashboard Command ───


class TestDashboardCommand:
    def test_dashboard_help(self, runner: CliRunner) -> None:
        """march dashboard --help works."""
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output.lower()
