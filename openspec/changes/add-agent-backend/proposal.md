## Why

The repository has two answer paths that are tangled together and uneven in
quality. `QAEngine` (used by the CLI and the HTTP server) is a fixed
retrieve-then-answer pipeline whose "structured" output is really a JSON blob
requested in the prompt and recovered with brittle regex/heuristic parsing
(`parse_structured`). `QAAgent` is a pydantic-ai agent that validates its output
natively but is unreachable from any entry point and does not obey the engine's
retrieval contract. There is no abstraction that lets a caller choose between
them, and no clean boundary between the two.

This change gives each answer path its own home under a new `src/docoracle/backends/`
package, exposes a single configurable backend selection to all entry points,
and replaces the engine's string-parsing approach with provider-native
structured output.

## What Changes

- **Introduce `src/docoracle/backends/`** as the single home for answer generation:
  - a shared `AnswerBackend` protocol (`ask`, `ask_async`,
    `get_related_chunks`, `default_mode`),
  - `engine.py` hosting `QAEngine` (moved from `src/docoracle/core/qa_engine.py`),
  - `agent.py` hosting `QAAgent` and its tools (moved from `src/docoracle/agents/`),
  - `factory.py` returning the configured backend.
- **Select the backend from configuration**: add a `docoracle.backend` setting
  (`engine` | `agent`, default `engine`) with validation that names the valid
  values. The CLI and the HTTP server both resolve their backend through the
  shared factory.
- **Modernise the engine's structured output**: request and validate
  `AnswerResponse` through pydantic-ai's native structured output instead of a
  prompt-embedded JSON schema. Remove `parse_structured`,
  `_extract_json_object`, and `build_structured_prompt` along with their
  heuristic fallbacks.
- **Bring `QAAgent` to behavioral parity** with the engine so it is a true
  drop-in: config-driven default mode, `k` defaulting to `retrieval.top_k`,
  deterministic `module`/`component`/`version` filters, `rrf_k`/`prefetch_k`
  applied to the searcher, a `QAEngine`-shaped detailed result, and
  `get_related_chunks`.
- **BREAKING (internal import paths)**: `src.core.qa_engine` becomes
  `src.backends.engine`; `src.agents` is removed in favor of
  `src.backends.agent`. No public API response schema, on-disk format, or
  retrieval-mode contract changes.

## Capabilities

### New Capabilities

- `answer-backends`: backend selection via `docoracle.backend`, the shared
  `AnswerBackend` protocol/factory, and the separation guarantee that the engine
  and agent are independent implementations.
- `qa-engine` (directory `specs/qa-engine`): the deterministic retrieve-then-answer pipeline, including its
  provider-native structured output and detailed result contract.
- `qa-agent`: the tool-using pydantic-ai answer backend, including retrieval
  parity, detailed result shape, and search parity with the engine.

### Modified Capabilities

<!-- None: hybrid-retrieval behavior is unchanged; both backends conform to it. -->

## Impact

- **New**: `src/docoracle/backends/` (`__init__.py`, `protocol.py`, `engine.py`,
  `agent.py`, `factory.py`).
- **Removed**: `src/docoracle/core/qa_engine.py`, `src/docoracle/agents/` (moved into `backends`).
- `src/docoracle/core/llm_outputs.py`: remove the heuristic parser and prompt builder;
  keep the response models.
- `src/api/structured_llm_client.py`: keep embeddings/chat/streaming; route
  structured answers through the native pydantic-ai path used by both backends.
- `src/docoracle/core/config.py`: add the `docoracle.backend` setting.
- `src/docoracle/cli.py`, `src/docoracle/server/main.py`: depend only on the `AnswerBackend`
  protocol via the factory.
- Tests: relocate engine/agent tests, delete parser-heuristic tests, add
  backend-selection, native-output, and engine/agent parity tests.
- Documentation: `README.md` and `config.yaml` describe `docoracle.backend`.
