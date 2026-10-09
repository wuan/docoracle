## Why

The `agent` backend can already retrieve documentation through a built-in tool, but
it cannot reach any other capability. A pydantic-ai `Agent` is a natural host for
external tools, and the Model Context Protocol (MCP) is the emerging standard for
exposing them. Users who already run MCP servers — for example a home-automation
bridge — want the agent to discover and call those tools alongside retrieval,
without DocOracle reimplementing each integration.

This change adds MCP server integration to the agent backend: a config-driven list
of MCP servers whose tools become available to the agent at construction, while the
retrieval tool stays a permanently-on internal built-in.

## What Changes

- **Add an `agent.mcp_servers` configuration list** under `docoracle.agent`. Each
  entry names one MCP server, selects a `transport` (`stdio` or `streamable-http`),
  and carries only that transport's fields (`command`/`env` for stdio,
  `url`/`headers` for streamable-http), plus `enabled` (default `true`). The list
  is modelled as a pydantic discriminated union on `transport`, so an unknown
  transport, a `stdio` entry missing `command`, a `streamable-http` entry missing
  `url`, or a duplicate `name` all fail at config load.
- **Add `src/docoracle/backends/mcp.py`**: a connection-management module that turns
  the enabled config entries into pydantic-ai `MCPToolset`s — built from FastMCP
  `StdioTransport` / `StreamableHttpTransport` — and owns their long-lived
  lifecycle.
- **Wire the toolsets into `QAAgent`**: the retrieval tool remains an always-on
  internal built-in and is never part of the toggleable list; enabled MCP toolsets
  are added alongside it. If any enabled MCP server fails to connect during agent
  construction, construction raises immediately with a clear error.
- **Own the MCP lifecycle at the entry points**: the FastAPI server keeps MCP
  connections open across requests via its async lifespan; the CLI keeps them open
  for the invocation. Connections are not opened or closed per request.
- **Runtime failures are lenient**: an MCP server that dies mid-tool-call surfaces
  its error to the model through pydantic-ai's normal tool-error handling instead
  of crashing the request.
- **No change to the `engine` backend or to retrieval behavior.** MCP tools only
  exist under the `agent` backend.

## Capabilities

### New Capabilities

- `agent-mcp-tools`: MCP server configuration, connection management, always-on
  retrieval, and the startup/runtime failure contract for the agent backend.

### Modified Capabilities

<!-- None: `qa-agent` and `answer-backends` behavior is unchanged; the agent gains
     tools without altering its existing answer, result-shape, or retrieval
     contracts. -->

## Impact

- **New**: `src/docoracle/backends/mcp.py` (config-to-toolset construction and
  lifecycle management).
- `src/docoracle/core/config.py`: add the `agent.mcp_servers` discriminated-union
  models and validation.
- `src/docoracle/backends/agent.py`: accept and mount MCP toolsets; keep the
  retrieval tool always on; fail fast when an enabled server cannot connect.
- `src/docoracle/backends/factory.py`: pass MCP configuration through when building
  the agent backend.
- `src/docoracle/server/main.py`: own the long-lived MCP connections via the async
  lifespan (connection survives across requests).
- `src/docoracle/cli.py`: keep MCP connections open for the invocation.
- `config.yaml` / `README.md`: document `docoracle.agent.mcp_servers`, the two
  transports, and that stdio runtimes (e.g. `uvx`) must be provided by the host.
- Tests: config-validation tests and agent tests against **mocked** MCP servers
  (never real ones).
