"""MarchApp — The main entry point and framework API.

Wires together all components: config, logging, LLM router, tool registry,
plugin manager, memory store, skill loader, agent, and channels.

Provides the decorator-based framework API (@app.plugin, @app.tool, app.run()).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine, Type

from march.channels.base import Channel
from march.channels.terminal import TerminalChannel
from march.config.schema import MarchConfig
from march.config.loader import load_config
from march.core.agent import Agent
from march.core.session import Session
from march.llm.base import LLMProvider
from march.llm.router import LLMRouter, RouterConfig
from march.logging import get_logger
from march.memory.store import MemoryStore
from march.plugins._base import Plugin
from march.plugins._manager import PluginManager
from march.tools.base import Tool, ToolMeta, tool as tool_decorator
from march.tools.registry import ToolRegistry
from march.tools.skills.loader import SkillLoader
from march.tools.skills.base import Skill

logger = get_logger("march.app")


class MarchApp:
    """The March framework application.

    Wires all components together and provides the user-facing API.

    Usage:
        app = MarchApp(config="config.yaml")

        @app.tool(name="my_tool", description="Does something")
        async def my_tool(param: str) -> str:
            return "result"

        @app.plugin(name="my-plugin", priority=5)
        class MyPlugin(Plugin):
            async def before_llm(self, context, message):
                return context, message

        app.load_skill("./skills/my-skill/")
        app.run(channels=["terminal"])
    """

    def __init__(
        self,
        config: str | Path | MarchConfig | None = None,
    ):
        # Load config
        if isinstance(config, MarchConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = load_config(Path(config), use_cache=False)
        else:
            # Use defaults — don't hit the filesystem
            self.config = MarchConfig()

        # Initialize components
        router_config = RouterConfig(
            fallback_chain=list(self.config.llm.fallback_chain),
            default_provider=self.config.llm.default,
        )
        self.llm_router = LLMRouter(config=router_config, providers={})
        self.tool_registry = ToolRegistry()
        self.plugin_manager = PluginManager()
        self.skill_loader = SkillLoader()
        self.memory_store = MemoryStore(
            workspace=Path.cwd(),
            config_dir=Path.home() / ".march",
            system_rules_path=self.config.memory.system_rules,
            agent_profile_path=self.config.memory.agent_profile,
            tool_rules_path=self.config.memory.tool_rules,
        )
        # NOTE: ws_proxy plugin owns all persistence (sessions/messages in march.db)

        # Build agent
        self.agent = Agent(
            llm_router=self.llm_router,
            tool_registry=self.tool_registry,
            plugin_manager=self.plugin_manager,
            memory_store=self.memory_store,
            config=self.config,
        )

        # Available channels
        self._channels: dict[str, Channel] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components. Called automatically before first use.

        Order: config → logging → LLM providers → memory → plugins (builtins + directory) → tools → agent manager → channels.
        """
        if self._initialized:
            return

        # Configure logging first
        from march.logging.logger import configure_logging
        configure_logging(self.config.logging)

        # Create LLM providers from config
        self._create_providers()

        # Initialize memory (file-based: SYSTEM.md, AGENT.md, TOOLS.md, MEMORY.md)
        await self.memory_store.initialize()

        # Load built-in plugins
        self._load_builtin_plugins()

        # Load plugins from directory
        plugin_dir = Path.cwd() / self.config.plugins.directory
        if plugin_dir.is_dir():
            enabled = self.config.plugins.enabled or None
            self.plugin_manager.load_directory(plugin_dir, enabled=enabled)

        # Register all builtin tools
        from march.tools.builtin import register_all_builtin_tools
        register_all_builtin_tools(self.tool_registry)

        # Load skills from directory
        skills_dir = Path.cwd() / "skills"
        if skills_dir.is_dir():
            self.skill_loader.load_directory(skills_dir, registry=self.tool_registry)

        # Initialize agent manager with task queue
        from march.agents.manager import AgentManager, AgentManagerConfig
        from march.agents.task_queue import TaskQueue

        agent_mgr_config = AgentManagerConfig(
            max_spawn_depth=self.config.agents.subagents.max_spawn_depth,
            max_children_per_agent=self.config.agents.subagents.max_children_per_agent,
            max_concurrent_subagents=self.config.agents.subagents.max_concurrent,
            run_timeout_seconds=self.config.agents.subagents.run_timeout_seconds,
            archive_after_minutes=self.config.agents.subagents.archive_after_minutes,
            announce_timeout_seconds=self.config.agents.subagents.announce_timeout_seconds,
        )
        self.task_queue = TaskQueue()
        self.agent_manager = AgentManager(
            config=agent_mgr_config,
            task_queue=self.task_queue,
            session_store=None,
        )
        await self.agent_manager.initialize()

        # Expose agent manager to the agent for sub-agent spawning
        self.agent.agent_manager = self.agent_manager
        self.agent.session_store = None

        # Fire on_start hook
        await self.plugin_manager.dispatch_simple("on_start", self)

        self._initialized = True
        logger.info("March initialized")

    def _load_builtin_plugins(self) -> None:
        """Load plugins from the march/plugins/ package directory.

        All plugins are equal — discovered by scanning the plugins directory
        for Python files containing Plugin subclasses. Only plugins listed
        in config.plugins.enabled are loaded.
        """
        plugins_pkg_dir = Path(__file__).parent / "plugins"
        enabled = self.config.plugins.enabled or None
        count = self.plugin_manager.load_directory(plugins_pkg_dir, enabled=enabled)
        logger.info("Loaded %d plugins from %s", count, plugins_pkg_dir)

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        await self.plugin_manager.dispatch_simple("on_shutdown", self)
        await self.memory_store.close()
        self._initialized = False

    def _create_providers(self) -> None:
        """Instantiate LLM providers from config and register them with the router."""
        provider_map = {
            "litellm": "march.llm.litellm_provider.LiteLLMProvider",
            "ollama": "march.llm.ollama.OllamaProvider",
            "openai": "march.llm.openai_provider.OpenAIProvider",
            "anthropic": "march.llm.anthropic_provider.AnthropicProvider",
            "bedrock": "march.llm.bedrock.BedrockProvider",
            "openrouter": "march.llm.openrouter.OpenRouterProvider",
        }

        for name, pcfg in self.config.llm.providers.items():
            provider_cls_path = provider_map.get(name)
            if not provider_cls_path:
                logger.warning("Unknown LLM provider: %s — skipping", name)
                continue

            try:
                module_path, cls_name = provider_cls_path.rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_path)
                cls = getattr(module, cls_name)

                # Build kwargs based on provider type
                if name == "bedrock":
                    kwargs: dict = {
                        "model": pcfg.model,
                        "region": pcfg.region or "us-west-2",
                        "max_tokens": pcfg.max_tokens,
                        "temperature": pcfg.temperature,
                        "timeout": float(pcfg.timeout),
                        "input_price": pcfg.cost.input,
                        "output_price": pcfg.cost.output,
                    }
                    if pcfg.profile:
                        kwargs["profile"] = pcfg.profile
                else:
                    kwargs = {
                        "model": pcfg.model,
                        "url": pcfg.url,
                        "max_tokens": pcfg.max_tokens,
                        "temperature": pcfg.temperature,
                        "timeout": float(pcfg.timeout),
                        "input_price": pcfg.cost.input,
                        "output_price": pcfg.cost.output,
                    }
                    if pcfg.api_key:
                        kwargs["api_key"] = pcfg.api_key

                provider = cls(**kwargs)
                self.llm_router.providers[name] = provider
                # Also register health tracking for this provider
                from march.llm.router import ProviderHealth
                self.llm_router._health[name] = ProviderHealth(name=name)
                logger.info("Registered LLM provider: %s (model=%s)", name, pcfg.model)
            except Exception as e:
                logger.error("Failed to create provider %s: %s", name, e)
        logger.info("March shut down")

    # ── Framework API: Decorators ──

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        source: str = "custom",
    ) -> Callable[..., Any]:
        """Decorator to register a custom tool.

        Usage:
            @app.tool(name="my_tool", description="Does something")
            async def my_tool(param: str) -> str:
                return "result"
        """

        def decorator(fn: Callable[..., Coroutine[Any, Any, str]]) -> Callable[..., Any]:
            # Apply the @tool decorator to extract metadata
            decorated = tool_decorator(name=name, description=description)(fn)
            # Register with the tool registry
            self.tool_registry.register_function(
                decorated,
                name=name,
                description=description,
                source=source,
            )
            return decorated

        return decorator

    def plugin(
        self,
        name: str | None = None,
        priority: int = 100,
    ) -> Callable[..., Any]:
        """Decorator to register a custom plugin class.

        Usage:
            @app.plugin(name="my-plugin", priority=5)
            class MyPlugin(Plugin):
                async def before_llm(self, context, message):
                    return context, message
        """

        def decorator(cls: Type[Plugin]) -> Type[Plugin]:
            if name:
                cls.name = name
            cls.priority = priority
            instance = cls()
            self.plugin_manager.register(instance)
            return cls

        return decorator

    # ── Framework API: Registration Methods ──

    def register_tool(self, fn: Callable[..., Any], **kwargs: Any) -> None:
        """Register a tool function directly (without decorator)."""
        self.tool_registry.register_function(fn, **kwargs)

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin instance directly."""
        self.plugin_manager.register(plugin)

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register an LLM provider."""
        self.llm_router.providers[name] = provider
        if name not in self.llm_router.config.fallback_chain:
            self.llm_router.config.fallback_chain.append(name)

    def load_skill(self, path: str | Path) -> Skill | None:
        """Load a skill from a directory.

        Args:
            path: Path to the skill directory.

        Returns:
            The loaded Skill, or None if loading failed.
        """
        skill_path = Path(path)
        if not skill_path.is_dir():
            logger.warning("Skill directory not found: %s", skill_path)
            return None

        try:
            skill = self.skill_loader.load(skill_path, registry=self.tool_registry)
            return skill
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", skill_path, e)
            return None

    # ── Framework API: Run ──

    def run(
        self,
        channels: list[str] | None = None,
        session: Session | None = None,
    ) -> None:
        """Run the March app with the specified channels.

        This is the main entry point. Blocks until all channels are stopped.

        Args:
            channels: List of channel names to start (default: ["terminal"]).
            session: Optional pre-built session to use.
        """
        channel_names = channels or ["terminal"]
        asyncio.run(self._run_async(channel_names, session))

    async def _run_headless(self) -> None:
        """Run in headless mode: initialize plugins and wait forever.

        Used when the server is provided by a plugin (e.g. ws_proxy)
        rather than a built-in channel.
        """
        await self.initialize()
        # on_start is already dispatched by initialize()

        logger.info("March running in headless mode (plugins active, no channels)")

        try:
            # Wait forever — plugins provide the actual server
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutting down headless mode")
        finally:
            await self.plugin_manager.dispatch_simple("on_shutdown", self)

    async def _run_async(
        self,
        channel_names: list[str],
        session: Session | None = None,
    ) -> None:
        """Async implementation of run()."""
        await self.initialize()

        try:
            # Create channels
            tasks: list[asyncio.Task[None]] = []
            for name in channel_names:
                channel = self._create_channel(name)
                if channel:
                    self._channels[name] = channel
                    task = asyncio.create_task(
                        channel.start(self.agent, session=session, session_store=None)
                    )
                    tasks.append(task)
                else:
                    logger.warning("Unknown channel: %s", name)

            if not tasks:
                logger.error("No channels to start")
                return

            # Wait for all channels to complete
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            # Stop all channels
            for channel in self._channels.values():
                try:
                    await channel.stop()
                except Exception as e:
                    logger.error("Error stopping channel %s: %s", channel.name, e)
            await self.shutdown()

    def _create_channel(self, name: str) -> Channel | None:
        """Create a channel instance by name."""
        if name == "terminal":
            terminal_config = self.config.channels.terminal
            return TerminalChannel(
                streaming=terminal_config.streaming,
                theme=terminal_config.theme,
            )
        elif name == "acp":
            from march.channels.acp import ACPChannel
            return ACPChannel()
        elif name == "matrix":
            from march.channels.matrix_channel import MatrixChannel
            matrix_config = self.config.channels.matrix
            return MatrixChannel(
                homeserver=matrix_config.homeserver,
                user_id=matrix_config.user,
                password=matrix_config.password,
                access_token=matrix_config.access_token,
                rooms=list(matrix_config.rooms),
                e2ee=matrix_config.e2ee,
                auto_setup=matrix_config.auto_setup,
            )
        elif name == "vscode":
            from march.channels.vscode import VSCodeChannel
            vscode_config = self.config.channels.vscode
            # VSCode channel connects to the ws_proxy plugin's WebSocket
            proxy_cfg = self.config.plugins.ws_proxy
            ws_url = f"ws://{proxy_cfg.host}:{proxy_cfg.port}"
            return VSCodeChannel(
                ws_url=ws_url,
            )
        return None
