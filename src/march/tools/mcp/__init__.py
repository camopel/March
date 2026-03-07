"""MCP (Model Context Protocol) integration for March."""

from march.tools.mcp.client import MCPClient, MCPError, MCPToolDef
from march.tools.mcp.discovery import discover_and_register

__all__ = ["MCPClient", "MCPError", "MCPToolDef", "discover_and_register"]
