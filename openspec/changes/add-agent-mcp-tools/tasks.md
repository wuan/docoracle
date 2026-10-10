## 1. Configuration model and validation

- [x] 1.1 Add `StdioMCPServerConfig` and `StreamableHTTPMCPServerConfig` models and an `MCPServerConfig` discriminated union on `transport` to `src/docoracle/core/config.py`
- [x] 1.2 Add an `AgentConfig` (`mcp_servers: list[MCPServerConfig] = []`) and wire it under `docoracle.agent`; verify the field defaults to an empty list so existing configs are unaffected (test)
- [x] 1.3 Verify each entry carries only its transport's fields (a `stdio` entry may not supply `url`; a `streamable-http` entry may not supply `command`) (test)
- [x] 1.4 Verify an unknown `transport` value is rejected at config load with an error naming `stdio` and `streamable-http` (test)
- [x] 1.5 Verify a `stdio` entry missing `command` is rejected at config load (test)
- [x] 1.6 Verify a `streamable-http` entry missing `url` is rejected at config load (test)
- [x] 1.7 Verify duplicate `name` values across the list are rejected at config load with an error naming the duplicate (test)
- [x] 1.8 Verify `enabled` defaults to `true` when omitted (test)
- [x] 1.9 Verify `${VAR}` in a stdio `env` value and a streamable-http `headers` value is resolved by the existing `Config.from_yaml` pass (test)

## 2. MCP connection management (`src/docoracle/backends/mcp.py`)

- [x] 2.1 Create `src/docoracle/backends/mcp.py` with a builder that maps each enabled config entry to a pydantic-ai `MCPServerStdio` or `MCPServerStreamableHTTP` toolset
- [x] 2.2 Have the builder skip entries with `enabled: false` and verify skipped servers are never contacted (test, mocked)
- [x] 2.3 Provide a long-lived async lifecycle manager (async context manager) that enters every enabled toolset on start and exits them on stop
- [x] 2.4 Verify the lifecycle actually connects enabled servers on entry and disconnects on exit (test, mocked)
- [x] 2.5 Verify construction raises immediately, naming the failing server and cause, when an enabled server cannot connect (test, mocked)

## 3. Agent wiring (always-on retrieval + MCP toolsets, fail-fast)

- [x] 3.1 Extend `QAAgent` to accept MCP toolsets and mount them alongside the retrieval and summary tools
- [x] 3.2 Verify the retrieval tool is always mounted under the agent backend and is never removed by MCP configuration (test)
- [x] 3.3 Verify all enabled MCP toolsets are registered with the underlying pydantic-ai agent (test, mocked)
- [x] 3.4 Verify agent construction fails fast with a clear error naming the server when an enabled server cannot connect, and that no request is served in that case (test, mocked)
- [x] 3.5 Verify an explicitly disabled server is not required to connect at construction (test, mocked)
- [x] 3.6 Verify a tool call to an MCP server that dies mid-call surfaces the error to the model via pydantic-ai tool-error handling instead of crashing the run (test, mocked)
- [x] 3.7 Pass MCP configuration through `create_answer_backend` when building the agent backend; verify the engine backend ignores it entirely (test)

## 4. Entry point lifecycle ownership

- [x] 4.1 Add an async lifespan to the FastAPI app in `src/docoracle/server/main.py` that opens the MCP connections on startup and closes them on shutdown
- [x] 4.2 Verify the server's MCP connections are established once at startup and reused across multiple `/ask` requests, not reopened per request (test, mocked)
- [x] 4.3 Verify server shutdown closes the MCP connections cleanly (test, mocked)
- [x] 4.4 Wrap the CLI `ask`/`search` invocation in the MCP lifecycle so connections persist for the invocation and close afterward (test, mocked)
- [x] 4.5 Verify the CLI with no MCP servers configured starts without opening any MCP connection (test)

## 5. Tests against mocked MCP servers

- [x] 5.1 Ensure all MCP tests use mocked/in-process MCP servers (or stubbed toolsets) and never spawn real ones or hit real endpoints
- [x] 5.2 Verify an agent followed by a mocked MCP tool returns a validated `AnswerResponse` and records the tool call (test)
- [x] 5.3 Verify existing agent tests still pass with an empty `mcp_servers` list (no behavior change when MCP is unconfigured)
- [x] 5.4 Verify the `engine` backend never constructs MCP toolsets regardless of `agent.mcp_servers` contents (test)

## 6. Verification and documentation

- [x] 6.1 Run the full suite (`pytest`), `ruff check`, `ruff format --check`, and `basedpyright src/` and verify all pass
- [x] 6.2 Update `README.md` with an "MCP tools" section: the `agent.mcp_servers` list, the two transports, always-on retrieval, and the fail-fast/lenient runtime contract
- [x] 6.3 Update `config.yaml` (example) with a commented `docoracle.agent.mcp_servers` block matching the design
- [x] 6.4 Document that stdio servers assume the runtime (e.g. `uvx`) is provided by the host; DocOracle does not install them
- [x] 6.5 Confirm no `src/` or `tests/` change alters the `engine` backend or the `hybrid-retrieval` contract
