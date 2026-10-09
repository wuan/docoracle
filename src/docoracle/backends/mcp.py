"""MCP toolset construction and long-lived lifecycle management.

This module turns the ``docoracle.agent.mcp_servers`` configuration into
pydantic-ai MCP toolsets and owns their connections. Toolsets are context
managed at the process level (the FastAPI lifespan) or invocation level (the
CLI), never per request.

Only two transports are supported: ``stdio`` and ``streamable-http``. The
stdio runtime (e.g. ``uvx`` and the server package) must be provided by the
host; DocOracle does not install it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset

from ..core.config import (
    MCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)

logger = logging.getLogger("backends.mcp")


class MCPConnectionError(RuntimeError):
    """Raised when an enabled MCP server cannot be connected at startup."""

    def __init__(self, server_name: str, cause: BaseException) -> None:
        super().__init__(f"MCP server {server_name!r} failed to connect: {cause}")
        self.server_name = server_name
        self.cause = cause


def build_mcp_toolset(config: MCPServerConfig) -> MCPToolset[Any]:
    """Build a pydantic-ai MCP toolset for a single enabled server entry."""
    if isinstance(config, StdioMCPServerConfig):
        transport = StdioTransport(config.command, config.args, env=config.env)
    else:
        # The union on ``transport`` is exhaustive: non-stdio entries are
        # streamable-http.
        http_config: StreamableHTTPMCPServerConfig = config
        transport = StreamableHttpTransport(http_config.url, headers=http_config.headers)
    return MCPToolset(transport)


def build_mcp_toolsets(servers: Sequence[MCPServerConfig]) -> list[tuple[str, MCPToolset[Any]]]:
    """Build a toolset for every *enabled* server entry, skipping disabled ones.

    Returns ``(name, toolset)`` pairs in declaration order. Disabled servers are
    never built, so they are never contacted.
    """
    toolsets: list[tuple[str, MCPToolset[Any]]] = []
    for server in servers:
        if not server.enabled:
            logger.info("MCP server %r disabled; skipping", server.name)
            continue
        toolsets.append((server.name, build_mcp_toolset(server)))
    return toolsets


class MCPToolsetManager:
    """Long-lived owner of a set of MCP toolsets.

    Enter the manager once (via ``async with``) to connect every toolset and
    exit it once to disconnect them all. Construction fails fast: if an enabled
    toolset cannot be entered, the manager raises :class:`MCPConnectionError`
    naming the failing server and closes any toolsets it already opened.
    """

    def __init__(self, toolsets: Sequence[tuple[str, MCPToolset[Any]]] = ()) -> None:
        self._toolsets: list[tuple[str, MCPToolset[Any]]] = list(toolsets)
        self._stack: AsyncExitStack | None = None

    @property
    def toolsets(self) -> list[MCPToolset[Any]]:
        """The toolsets to mount on the agent (empty when none are enabled)."""
        return [toolset for _, toolset in self._toolsets]

    async def __aenter__(self) -> MCPToolsetManager:
        stack = AsyncExitStack()
        await stack.__aenter__()
        for name, toolset in self._toolsets:
            try:
                await stack.enter_async_context(toolset)
            except Exception as exc:
                await stack.aclose()
                raise MCPConnectionError(name, exc) from exc
            logger.info("MCP server %r connected", name)
        self._stack = stack
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        stack = self._stack
        self._stack = None
        if stack is None:
            return None
        return await stack.__aexit__(exc_type, exc, tb)
