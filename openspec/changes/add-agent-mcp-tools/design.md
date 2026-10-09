## Context

The `agent` backend (`src/docoracle/backends/agent.py::QAAgent`) is a pydantic-ai
`Agent[RetrievalState, AnswerResponse]` built in the factory
(`src/docoracle/backends/factory.py::create_answer_backend`) when
`docoracle.backend == "agent"`. It mounts a retrieval tool and a summarization
tool, returns a validated `AnswerResponse`, and relies on the same
OpenAI-compatible chat model as the engine.

pydantic-ai 2.x ships native MCP toolsets: `MCPServerStdio` (spawns a subprocess
speaking MCP over stdio) and `MCPServerStreamableHTTP` (connects to a remote MCP
endpoint over streamable HTTP). Both are context-managed: a long-lived `async with`
enters the connection, and their tools are passed to `Agent(toolsets=[...])`.

Configuration today is a single `docoracle.backend` field
(`src/docoracle/core/config.py::DocOracleConfig`). `Config.from_yaml` already
resolves `${VAR}` syntax recursively before model validation, so any string field
(including MCP `env` values and streamable-http `headers`) inherits env-var
resolution for free.

The server (`src/docoracle/server/main.py`) currently constructs the backend lazily
on first request and has no async lifespan; the CLI (`src/docoracle/cli.py`)
constructs a backend per command invocation. Both need to own the MCP connections
for a longer lifetime than a single request.

## Goals / Non-Goals

**Goals:**

- Let the agent call tools from external MCP servers listed in configuration.
- Keep the retrieval tool always on and internal; it is never toggleable.
- Fail fast when an enabled MCP server cannot connect at agent construction.
- Be lenient at runtime: a dead server errors one tool call, not the request.
- Support exactly two transports: `stdio` and `streamable-http`.
- Reuse the existing `${VAR}` env-var resolution for server env/headers.

**Non-Goals:**

- Per-tool toggles or per-tool options inside a server (whole-server granularity only).
- Per-server custom setups, auth flows beyond static headers/env, or retry policy.
- MCP under the `engine` backend.
- Legacy SSE transport.
- Installing or bundling MCP server runtimes (e.g. `uvx`, the server package).

## Decisions

### The retrieval tool is always on and never toggleable

Retrieval is the reason the agent exists; an agent without it cannot answer
documentation questions. The retrieval tool (and the summarization tool) stay as
internal built-ins wired in `QAAgent.__init__`, and `agent.mcp_servers` only ever
adds MCP toolsets alongside them. There is no configuration path that removes or
disables the retrieval tool.

*Alternatives considered:*
- **Represent retrieval as just another toggleable tool entry** — rejected; a
  misconfiguration could produce an agent that hallucinates without ever
  retrieving, which contradicts the backend's purpose.
- **Auto-detect a retrieval-like MCP server** — rejected as magic and fragile.

### Fail fast at startup, lenient at runtime

During construction, every *enabled* MCP server is entered and its tools
discovered. If any enabled server fails to connect, `QAAgent` construction raises
immediately with an error naming the failing server (`name`) and the underlying
cause. An explicitly `enabled: false` entry is skipped and never contacted.
Once the agent is running, a server that dies mid-tool-call does not crash the
request: the tool call raises, and pydantic-ai's standard tool-error handling
surfaces the error back to the model, which can explain the failure or retry.

This split matches the user's intent: misconfiguration is caught at boot (where it
is cheap to fix), while transient runtime failure degrades gracefully instead of
taking down the server.

*Alternatives considered:*
- **Lazy connect on first tool call** — rejected; a broken server would only fail
  once a request happens to need it, which is hard to diagnose.
- **Retry/backoff at startup** — rejected as scope creep; a clear error is enough.
- **Strict runtime failure (crash the request)** — rejected; one flaky external
  server should not fail answers that don't depend on it.

### Config is a simple enable/disable list

`docoracle.agent.mcp_servers` is a list; each entry names one server and is
connected when `enabled` is true (the default). There are no per-tool toggles or
per-tool options inside a server — presence in the list is the only granularity.

*Alternatives considered:*
- **Per-tool enable lists inside each server** — deferred; adds surface without a
  concrete need. Revisit if a server exposes tools we must hide.
- **A map keyed by server name instead of a list** — rejected; the list form
  matches the existing YAML style and preserves declaration order.

### Discriminated union on `transport`

Model each entry as a pydantic discriminated union keyed on `transport`:

```python
class StdioMCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio"]
    command: list[str]
    args: list[str] = []
    env: dict[str, str] = {}
    enabled: bool = True

class StreamableHTTPMCPServerConfig(BaseModel):
    name: str
    transport: Literal["streamable-http"]
    url: str
    headers: dict[str, str] = {}
    enabled: bool = True

MCPServerConfig = Annotated[
    StdioMCPServerConfig | StreamableHTTPMCPServerConfig,
    Field(discriminator="transport"),
]
```

This makes unknown transports, a `stdio` entry missing `command`, and a
`streamable-http` entry missing `url` fail at config load (via pydantic), and it
guarantees each branch carries only its own fields (no stray `url` on a stdio
entry). Duplicate `name` values across the list are rejected by a model validator
on the `agent` config (names are used in logs and errors and must be unique).

*Alternatives considered:*
- **A single flat model with optional fields** — rejected; it permits invalid
  combinations and pushes transport validation into ad-hoc code.
- **A plain `dict[str, Any]`** — rejected; no validation, no clear errors.

Configuration shape:

```yaml
docoracle:
  backend: agent
  agent:
    mcp_servers:
      - name: home_assistant         # required, unique, used in logs/errors
        transport: stdio             # Literal["stdio", "streamable-http"]
        command: ["uvx", "mcp-server-home-assistant"]   # stdio only
        env:                          # stdio only, optional
          HA_URL: ${HA_URL}         # resolved by the existing ${VAR} pass
        enabled: true                 # default true
      - name: web_search
        transport: streamable-http
        url: https://example.com/mcp  # streamable-http only
        headers: {}                   # streamable-http only, optional
        enabled: false
```

### Env-var resolution is inherited, not reimplemented

`Config.from_yaml` already runs `_resolve_env_vars_recursive` over the whole YAML
tree before constructing the models, so `${VAR}` works in `env` values and
`headers` without any MCP-specific code. The MCP module consumes already-resolved
strings.

*Alternatives considered:*
- **Resolve `${VAR}` inside the MCP module** — rejected; duplicates the existing
  mechanism and would behave inconsistently with the rest of the config.

### Long-lived connections owned by the agent

MCP toolsets are context-managed, so the agent must hold their live connection for
its whole working life — across requests for the server, and for the invocation
for the CLI. `src/docoracle/backends/mcp.py` provides the toolsets plus a lifecycle
manager (an async context manager) that the entry points drive:

- `src/docoracle/backends/agent.py`: `QAAgent` accepts the built toolsets and
  mounts them; construction validates that every enabled server connects.
- `src/docoracle/server/main.py`: an async lifespan opens the MCP connections when
  the app starts and closes them on shutdown, so a request never pays connect cost
  and connections are not churned per request.
- `src/docoracle/cli.py`: wraps the command in the same lifecycle for the duration
  of the invocation.

*Alternatives considered:*
- **Open/close per request** — rejected; adds latency and connection churn and
  contradicts MCP's session model.
- **Module-level global connections** — rejected; makes shutdown and test isolation
  hard. Entry points own the lifespan explicitly.

### Skip legacy SSE

Only `stdio` and `streamable-http` are supported. SSE is legacy in MCP and is
excluded; we do not add an SSE branch unless pydantic-ai exposes one for free
(today it does not, so there is nothing to wire).

*Alternatives considered:*
- **Include SSE for completeness** — rejected; extra surface for a deprecated
  transport with no known users.

## Risks / Trade-offs

- **A misconfigured MCP server blocks boot under the agent backend** → intended:
  fail-fast makes misconfiguration obvious. Mitigation: set `enabled: false` (or
  remove the entry) to skip a server; tests use mocked servers.
- **Startup now depends on external processes/endpoints** → the FastAPI lifespan
  and CLI startup block until enabled servers connect. Mitigation: keep MCP opt-in
  and document that each enabled server must be reachable at boot.
- **Agent behavior becomes non-deterministic once external tools exist** → the
  model may call tools we did not anticipate. Mitigation: retrieval remains always
  on and unchanged; the engine backend is unaffected and remains the default.
- **stdio servers depend on a host-provided runtime** (e.g. `uvx`, the server
  package) → Mitigation: document explicitly that DocOracle does not install MCP
  server runtimes; a missing runtime fails fast at construction with the underlying
  error.
- **Long-lived connections complicate tests** → Mitigation: tests enter the same
  lifecycle against mocked MCP servers; never spawn real ones.
- **Duplicate or malformed entries slip through** → Mitigation: discriminated
  union plus a unique-name validator reject them at config load.

## Migration Plan

1. Add `agent.mcp_servers` (discriminated union + unique-name validation) to
   `src/docoracle/core/config.py`; the field defaults to an empty list, so existing
   configs are unaffected.
2. Add `src/docoracle/backends/mcp.py` to build toolsets and manage their lifecycle.
3. Mount the toolsets in `QAAgent` (retrieval tool stays always on) and raise on any
   enabled-server connection failure.
4. Give the FastAPI server and the CLI ownership of the long-lived lifecycle.
5. Opt in by listing servers under `docoracle.agent.mcp_servers`; roll back by
   clearing the list or setting `enabled: false`.

No stored data changes; rollback requires no migration and the `engine` backend is
untouched, so `docoracle.backend: engine` sidesteps MCP entirely.

## Open Questions

- Should a failed MCP tool call be retried automatically before being surfaced to
  the model, or is pydantic-ai's default single-attempt tool-error handling
  sufficient? Deferrable; default to no retry.
- Should the server expose MCP server health (name + connected/failed) on
  `/health` or `/info`? Deferrable; useful for operators but not required.
- When a server is `enabled: false`, should DocOracle log it at startup so the
  omission is visible? Leaning yes (one log line); deferrable.
