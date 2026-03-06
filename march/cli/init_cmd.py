"""march init — Initialize March in the current directory."""

from __future__ import annotations

import click


@click.command("init")
def init() -> None:
    """Initialize March for the current environment.

    Sets up:
    - ~/.march/config.yaml — March configuration
    - ~/.march/march.db   — Single SQLite DB (sessions, messages, memory, cron)
    - ~/.march/MEMORY.md  — Long-term curated memory
    - plugins/            — Custom plugins directory (in cwd)
    - skills/             — Custom skills directory (in cwd)

    SYSTEM.md, AGENT.md, TOOLS.md are loaded from package templates by default.
    To customize, run: march init-templates
    """
    from pathlib import Path

    config_dir = Path.home() / ".march"
    config_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Initializing March...\n")

    created = 0
    skipped = 0

    # Create MEMORY.md in ~/.march/ (mutable, per-environment)
    memory_path = config_dir / "MEMORY.md"
    if not memory_path.exists():
        memory_path.write_text(
            "# Memory\n\n"
            "Long-term curated memory. Updated by the agent over time.\n",
            encoding="utf-8",
        )
        click.echo(f"  ✅ Created ~/.march/MEMORY.md")
        created += 1
    else:
        click.echo(f"  ⏭️  Exists  ~/.march/MEMORY.md")
        skipped += 1

    # Create config.yaml in ~/.march/ if not exists
    config_yaml = config_dir / "config.yaml"
    if not config_yaml.exists():
        config_content = (
            "# March Agent Configuration\n"
            "# See: https://github.com/march/march\n"
            "#\n"
            "# Environment variable interpolation: ${VAR} or ${VAR:default}\n\n"
            "agent:\n"
            '  name: "my-agent"\n\n'
            "llm:\n"
            '  default: "openai"\n'
            "  providers:\n"
            "    openai:\n"
            '      model: "${MARCH_MODEL:gpt-4o}"\n'
            '      url: "${MARCH_LLM_URL:https://api.openai.com/v1}"\n'
            '      api_key: "${OPENAI_API_KEY:}"\n'
            "      max_tokens: 16384\n"
            "      context_window: 128000\n"
            "      temperature: 0.7\n"
            "      timeout: 300\n"
            "      reasoning: false\n\n"
            "    # anthropic:\n"
            '    #   model: "claude-sonnet-4-20250514"\n'
            '    #   api_key: "${ANTHROPIC_API_KEY:}"\n\n'
            "    # openrouter:\n"
            '    #   model: "anthropic/claude-sonnet-4-20250514"\n'
            '    #   api_key: "${OPENROUTER_API_KEY:}"\n\n'
            "    # litellm:\n"
            '    #   model: "claude-sonnet-4-20250514"\n'
            '    #   url: "http://localhost:4000"\n\n'
            "    # ollama:\n"
            '    #   model: "llama3.1"\n'
            '    #   url: "http://localhost:11434"\n\n'
            "plugins:\n"
            "  enabled:\n"
            "    - safety\n"
            "    - cost\n"
            "    - logger\n\n"
            "channels:\n"
            "  terminal:\n"
            "    enabled: true\n"
            "    streaming: true\n"
        )
        config_yaml.write_text(config_content, encoding="utf-8")
        click.echo("  ✅ Created ~/.march/config.yaml")
        created += 1
    else:
        click.echo("  ⏭️  Exists  ~/.march/config.yaml")
        skipped += 1

    # Create directories in cwd
    dirs = ["plugins", "skills"]
    for dirname in dirs:
        path = Path(dirname)
        path.mkdir(exist_ok=True)
        click.echo(f"  📁 {dirname}/")

    click.echo(f"\n✅ March initialized. ({created} created, {skipped} existing)")
    click.echo("")
    click.echo("  Config:     ~/.march/config.yaml")
    click.echo("  Memory:     ~/.march/MEMORY.md")
    click.echo("  Templates:  package defaults (run 'march init-templates' to customize)")
    click.echo("")
    click.echo("  Next steps:")
    click.echo("    1. Edit ~/.march/config.yaml to set your LLM provider")
    click.echo("    2. Run 'march chat' to start chatting")
    click.echo("")


@click.command("init-templates")
def init_templates() -> None:
    """Copy package templates to ~/.march/ for customization.

    Copies SYSTEM.md, AGENT.md, TOOLS.md from the package to ~/.march/
    so you can edit them. These will override the built-in defaults.
    """
    from importlib.resources import files as pkg_files
    from pathlib import Path

    config_dir = Path.home() / ".march"
    config_dir.mkdir(parents=True, exist_ok=True)
    templates_pkg = pkg_files("march.templates")

    template_files = ["SYSTEM.md", "AGENT.md", "TOOLS.md"]

    click.echo("Copying templates to ~/.march/ for customization...\n")

    for name in template_files:
        dest = config_dir / name
        if dest.exists():
            click.echo(f"  ⏭️  Exists  ~/.march/{name}")
            continue
        try:
            content = (templates_pkg / name).read_text(encoding="utf-8")
            dest.write_text(content, encoding="utf-8")
            click.echo(f"  ✅ Copied  ~/.march/{name}")
        except Exception as e:
            click.echo(f"  ⚠️  Failed  {name}: {e}")

    click.echo("\nEdit these files to customize your agent's behavior.")
    click.echo("They will override the built-in package templates.")
