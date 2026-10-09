## ADDED Requirements

<!-- Implementation note: the enabled entries are realised as pydantic-ai
     `MCPToolset`s built from FastMCP transports — `StdioTransport(command,
     args, env)` for `stdio` and `StreamableHttpTransport(url, headers)` for
     `streamable-http`. The requirements below describe behavior; class names
     are illustrative of the pinned pydantic-ai API, not a normative contract. -->

### Requirement: MCP server configuration

The system SHALL expose an `agent.mcp_servers` list in configuration, where each
entry names one MCP server and selects a transport of either `stdio` or
`streamable-http`. A `stdio` entry SHALL carry `command` (and optional `args` and
`env`); a `streamable-http` entry SHALL carry `url` (and optional `headers`). Each
entry SHALL default `enabled` to `true`. Each entry MUST carry only the fields of
its own transport, and server `name` values MUST be unique across the list.

#### Scenario: Stdio server is configured
- **WHEN** the configuration contains an entry with `transport: stdio` and a `command`
- **THEN** the entry is accepted and describes a stdio MCP server

#### Scenario: Streamable-http server is configured
- **WHEN** the configuration contains an entry with `transport: streamable-http` and a `url`
- **THEN** the entry is accepted and describes a streamable-http MCP server

#### Scenario: Enabled defaults to true
- **WHEN** an entry omits `enabled`
- **THEN** the entry is treated as enabled

#### Scenario: Empty list is valid
- **WHEN** the configuration omits `agent.mcp_servers`
- **THEN** the configuration is accepted with no MCP servers

### Requirement: MCP configuration validation

The system MUST reject invalid MCP configuration at load time with a clear error
rather than starting with a partial or malformed configuration. An unknown
transport, a `stdio` entry missing `command`, a `streamable-http` entry missing
`url`, and a duplicate server `name` MUST each fail configuration loading.

#### Scenario: Unknown transport is rejected
- **WHEN** an entry sets `transport` to an unsupported value
- **THEN** configuration loading fails with an error naming `stdio` and `streamable-http`

#### Scenario: Stdio entry missing command is rejected
- **WHEN** an entry sets `transport: stdio` and omits `command`
- **THEN** configuration loading fails with an error identifying the missing `command`

#### Scenario: Streamable-http entry missing url is rejected
- **WHEN** an entry sets `transport: streamable-http` and omits `url`
- **THEN** configuration loading fails with an error identifying the missing `url`

#### Scenario: Duplicate server name is rejected
- **WHEN** two entries share the same `name`
- **THEN** configuration loading fails with an error naming the duplicate

#### Scenario: Cross-transport fields are rejected
- **WHEN** a `stdio` entry supplies `url` or a `streamable-http` entry supplies `command`
- **THEN** configuration loading fails because each entry carries only its own transport's fields

### Requirement: Env-var resolution for MCP configuration

The system SHALL resolve `${VAR}` syntax in MCP `env` values and `headers` using
the same resolution applied to the rest of the configuration.

#### Scenario: Env value is resolved
- **WHEN** a stdio entry sets an `env` value of `${HA_URL}` and that variable is set
- **THEN** the resolved value is the environment variable's value

#### Scenario: Header value is resolved
- **WHEN** a streamable-http entry sets a `headers` value of `${TOKEN}` and that variable is set
- **THEN** the resolved value is the environment variable's value

### Requirement: Retrieval tool is always on

Under the agent backend the built-in retrieval tool MUST always be mounted and
available to the model. MCP configuration MUST NOT be able to remove, disable, or
replace the retrieval tool.

#### Scenario: Retrieval tool survives MCP configuration
- **WHEN** the agent is constructed with any set of MCP servers
- **THEN** the retrieval tool is mounted and callable

#### Scenario: No configuration disables retrieval
- **WHEN** a caller configures one or more MCP servers
- **THEN** there is no configuration path that removes the retrieval tool

### Requirement: MCP tools are available to the agent

The agent backend SHALL register the tools of every enabled MCP server as tools
available to the model, so the model can discover and call them alongside the
built-in retrieval tool. A server with `enabled: false` MUST NOT be connected and
its tools MUST NOT be available.

#### Scenario: Enabled servers provide tools
- **WHEN** the agent is constructed with an enabled MCP server exposing tools
- **THEN** those tools are registered with the agent and can be called

#### Scenario: Disabled servers are skipped
- **WHEN** an MCP server entry sets `enabled: false`
- **THEN** the server is not connected and none of its tools are available

### Requirement: Fail fast at startup

If any enabled MCP server fails to connect during agent construction, the system
MUST raise immediately with an error identifying the failing server and the
underlying cause, rather than constructing an agent with missing tools.

#### Scenario: Unreachable enabled server aborts construction
- **WHEN** an enabled MCP server cannot be connected while the agent is being built
- **THEN** construction raises an error naming the failing server and no agent is returned

#### Scenario: Disabled server does not abort construction
- **WHEN** a server is disabled and would otherwise be unreachable
- **THEN** construction succeeds without contacting it

### Requirement: Lenient at runtime

Once the agent is running, a tool call to an MCP server that fails MUST surface
the error to the model through pydantic-ai's normal tool-error handling and MUST
NOT crash the in-flight request.

#### Scenario: Server dies mid-tool-call
- **WHEN** an MCP tool call fails because its server has become unavailable
- **THEN** the error is returned to the model as a tool error and the request completes

### Requirement: Long-lived MCP lifecycle

MCP connections SHALL be long-lived and owned by the agent's entry point. For the
HTTP server the connections MUST persist across requests (opened at application
startup, closed at shutdown); for the CLI they MUST persist for the invocation.
Connections MUST NOT be opened or closed per request.

#### Scenario: Server connections persist across requests
- **WHEN** the HTTP server is running with enabled MCP servers and serves multiple requests
- **THEN** the MCP connections established at startup are reused rather than reopened per request

#### Scenario: Server connections close on shutdown
- **WHEN** the HTTP server shuts down
- **THEN** its MCP connections are closed

#### Scenario: CLI connections last for the invocation
- **WHEN** a CLI command runs with enabled MCP servers
- **THEN** the MCP connections are open for the command and closed when it finishes

### Requirement: Engine backend is unaffected

MCP integration SHALL apply only to the agent backend. The engine backend MUST NOT
construct MCP toolsets or depend on MCP configuration, and retrieval behavior MUST
be unchanged.

#### Scenario: Engine ignores MCP configuration
- **WHEN** the backend is `engine` and `agent.mcp_servers` is non-empty
- **THEN** no MCP server is contacted and the engine answers as before
