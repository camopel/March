"""Auto-discover MCP servers from config and register their tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from march.logging import get_logger
from march.tools.mcp.client import MCPClient, MCPToolDef, MCPError

logger = get_logger("march.tools.mcp.discovery")


# Default paths to search for MCP config
_CONFIG_PATHS = [
    "~/.march/mcp.json",
    "~/.march/mcp.yaml",
    ".march/mcp.json",
    "mcp.json",
]


def _load_mcp_config(config_path: str | None = None) -> dict[str, Any]:
    """Load MCP server configuration.

    Config format (JSON):
    {
        "mcpServers": {
            "server-name": {
                "command": "npx",
                "args": ["-y", "@some/mcp-server"],
                "env": {"KEY": "value"}
            }
        }
    }
    """
    import json

    if config_path:
        paths = [config_path]
    else:
        paths = _CONFIG_PATHS

    for p in paths:
        path = Path(p).expanduser()
        if path.exists():
            try:
                with open(path) as f:
                    if path.suffix == ".yaml" or path.suffix == ".yml":
                        try:
                            import yaml
                            return yaml.safe_load(f) or {}
                        except ImportError:
                            logger.warning("PyYAML not installed, skipping YAML config")
                            continue
                    else:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading MCP config {path}: {e}")
                continue

    return {}


async def discover_and_register(
    registry: Any,
    config_path: str | None = None,
) -> dict[str, list[str]]:
    """Discover MCP servers from config and register their tools.

    Args:
        registry: ToolRegistry instance.
        config_path: Optional explicit config path.

    Returns:
        Dict mapping server names to lists of registered tool names.
    """
    config = _load_mcp_config(config_path)
    servers = config.get("mcpServers", config.get("servers", {}))

    if not servers:
        logger.debug("No MCP servers configured")
        return {}

    registered: dict[str, list[str]] = {}
    clients: dict[str, MCPClient] = {}

    for server_name, server_config in servers.items():
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        disabled = server_config.get("disabled", False)

        if disabled:
            logger.debug(f"MCP server '{server_name}' is disabled, skipping")
            continue

        if not command:
            logger.warning(f"MCP server '{server_name}' has no command, skipping")
            continue

        cmd = [command] + args

        try:
            client = MCPClient()
            await client.connect(cmd, env=env)
            tools = await client.list_tools()

            tool_names = []
            for tool_def in tools:
                # Create a wrapper function for each MCP tool
                _register_mcp_tool(registry, server_name, client, tool_def)
                tool_names.append(tool_def.name)

            clients[server_name] = client
            registered[server_name] = tool_names
            logger.info(f"MCP server '{server_name}': registered {len(tool_names)} tools")

        except MCPError as e:
            logger.error(f"MCP server '{server_name}' failed to connect: {e}")
        except Exception as e:
            logger.error(f"MCP server '{server_name}' error: {e}")

    return registered


def _register_mcp_tool(
    registry: Any,
    server_name: str,
    client: MCPClient,
    tool_def: MCPToolDef,
) -> None:
    """Register a single MCP tool with the registry."""
    from march.tools.base import Tool

    async def mcp_wrapper(**kwargs: Any) -> str:
        try:
            return await client.call_tool(tool_def.name, kwargs)
        except MCPError as e:
            return f"MCP error: {e}"
        except Exception as e:
            return f"Error calling MCP tool {tool_def.name}: {e}"

    tool = Tool(
        name=tool_def.name,
        description=tool_def.description,
        parameters=tool_def.input_schema,
        fn=mcp_wrapper,
        source=f"mcp:{server_name}",
    )
    registry.register(tool)
