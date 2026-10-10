"""Tests for MCP toolset construction and lifecycle management.

Everything here uses stubbed toolsets; no real MCP server is ever spawned or
contacted.
"""

import asyncio

import pytest

from docoracle.backends.mcp import (
    MCPConnectionError,
    MCPToolsetManager,
    build_mcp_toolsets,
)
from docoracle.core.config import (
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


class StubToolset:
    """Minimal async-context-manager stand-in for an ``MCPToolset``."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "StubToolset":
        if self.fail:
            raise RuntimeError("connection refused")
        self.entered += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited += 1


def test_build_skips_disabled_servers() -> None:
    servers = [
        StdioMCPServerConfig(name="on", transport="stdio", command="uvx"),
        StreamableHTTPMCPServerConfig(
            name="off", transport="streamable-http", url="https://x/mcp", enabled=False
        ),
    ]
    pairs = build_mcp_toolsets(servers)
    assert [name for name, _ in pairs] == ["on"]


def test_build_returns_empty_for_no_servers() -> None:
    assert build_mcp_toolsets([]) == []


def test_lifecycle_connects_on_entry_and_disconnects_on_exit() -> None:
    first, second = StubToolset(), StubToolset()
    manager = MCPToolsetManager([("a", first), ("b", second)])  # type: ignore[list-item]

    async def run() -> None:
        async with manager:
            assert first.entered == 1
            assert second.entered == 1
        assert first.exited == 1
        assert second.exited == 1

    asyncio.run(run())


def test_fail_fast_names_failing_server_and_closes_opened() -> None:
    opened = StubToolset()
    failing = StubToolset(fail=True)
    manager = MCPToolsetManager([("opened", opened), ("broken", failing)])  # type: ignore[list-item]

    async def run() -> None:
        with pytest.raises(MCPConnectionError) as exc:
            async with manager:
                pass
        assert exc.value.server_name == "broken"
        assert "connection refused" in str(exc.value)
        # The already-opened toolset is closed during cleanup.
        assert opened.exited == 1

    asyncio.run(run())


def test_toolsets_property_exposes_configured_toolsets() -> None:
    first = StubToolset()
    manager = MCPToolsetManager([("a", first)])  # type: ignore[list-item]
    assert manager.toolsets == [first]


def test_no_toolsets_is_a_noop_lifecycle() -> None:
    manager = MCPToolsetManager([])

    async def run() -> None:
        async with manager:
            assert manager.toolsets == []

    asyncio.run(run())
