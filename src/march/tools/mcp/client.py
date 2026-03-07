"""MCP (Model Context Protocol) stdio JSON-RPC client.

Communicates with MCP servers via stdin/stdout using JSON-RPC 2.0.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from march.logging import get_logger

logger = get_logger("march.tools.mcp.client")


@dataclass
class MCPToolDef:
    """Tool definition from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MCPError(Exception):
    """MCP protocol or server error."""

    pass


class MCPClient:
    """Communicate with MCP servers via stdio JSON-RPC.

    Usage:
        client = MCPClient()
        await client.connect(["npx", "-y", "@some/mcp-server"])
        tools = await client.list_tools()
        result = await client.call_tool("tool_name", {"arg": "value"})
        await client.disconnect()
    """

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._server_info: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def connect(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Connect to an MCP server.

        Args:
            command: Command to start the MCP server (e.g. ["npx", "-y", "@server"]).
            env: Additional environment variables.
            timeout: Connection timeout in seconds.

        Returns:
            Server capabilities from the initialize response.
        """
        import os

        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        # Start reading responses
        self._reader_task = asyncio.create_task(self._read_loop())

        # Send initialize request
        try:
            result = await asyncio.wait_for(
                self._request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "march", "version": "0.1.0"},
                }),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            await self.disconnect()
            raise MCPError("Timeout waiting for MCP server initialization")

        self._server_info = result

        # Send initialized notification
        await self._notify("notifications/initialized", {})

        logger.info(f"MCP connected: {' '.join(command)}")
        return result

    async def list_tools(self) -> list[MCPToolDef]:
        """List tools available from the MCP server."""
        if not self.connected:
            raise MCPError("Not connected")

        result = await self._request("tools/list", {})
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPToolDef(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            ))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            The text content from the tool result.
        """
        if not self.connected:
            raise MCPError("Not connected")

        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        # Extract text content
        content_parts = result.get("content", [])
        texts = []
        for part in content_parts:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image":
                texts.append(f"[Image: {part.get('mimeType', 'image/unknown')}]")
            else:
                texts.append(str(part))

        return "\n".join(texts)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None

        # Fail all pending requests
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError("Disconnected"))
        self._pending.clear()

        logger.info("MCP disconnected")

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        req_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        await self._send(message)

        try:
            result = await asyncio.wait_for(future, timeout=60)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPError(f"Timeout waiting for response to {method}")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(message)

    async def _send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the server."""
        if not self._process or not self._process.stdin:
            raise MCPError("Not connected")

        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from the server."""
        if not self._process or not self._process.stdout:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError:
                    logger.warning(f"MCP: Invalid JSON: {line_str[:100]}")
                    continue

                req_id = message.get("id")
                if req_id is not None and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if "error" in message:
                        err = message["error"]
                        future.set_exception(
                            MCPError(f"MCP error {err.get('code', -1)}: {err.get('message', 'Unknown')}")
                        )
                    else:
                        future.set_result(message.get("result", {}))
                elif "method" in message:
                    # Server-initiated notification
                    logger.debug(f"MCP notification: {message['method']}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MCP read loop error: {e}")
            # Fail all pending
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(MCPError(f"Read error: {e}"))
            self._pending.clear()
