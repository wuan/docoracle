## Context

Two answer paths exist today and have already drifted once:

- `src/docoracle/core/qa_engine.py::QAEngine` owns retrieval defaults and hyperparameters
  (`retrieval.mode`, `top_k`, `rrf_k`, `prefetch_k`), applies filters inline,
  returns a detailed dict from `ask`/`ask_async`, and exposes
  `get_related_chunks` for search. Its "structured" answer goes through
  `StructuredLLMClient.ask`, which embeds the `AnswerResponse` JSON schema in the
  prompt and then runs `parse_structured` — regex extraction, JSON coercion, and
  best-effort fallbacks that can silently stuff the whole response into
  `answer`.
- `src/docoracle/agents/qa_agent.py::QAAgent` is a pydantic-ai `Agent` that validates
  `AnswerResponse` natively via `output_type`, collects retrieved chunks per run
  through `RetrievalState`, and has `ask_detailed`/`ask_detailed_async`, but no
  entry point constructs it and its retrieval inputs (`mode`, `k`, filters) are
  prompt hints the model may ignore.

The CLI (`src/docoracle/cli.py`) and server (`src/docoracle/server/main.py`) import `QAEngine`
directly and mutate `default_mode` to apply `--retrieval`. `pydantic-ai` is
already a hard dependency and both paths build their model through
`build_chat_model` (the OpenAI-compatible provider).

## Goals / Non-Goals

**Goals:**

- A single home for answer generation (`src/docoracle/backends/`) with a shared
  `AnswerBackend` protocol so CLI and server are backend-agnostic.
- A config-driven backend choice (`docoracle.backend`, default `engine`) with no
  behavior change when unset.
- Provider-native structured output for the engine, removing the heuristic
  parser entirely.
- Agent parity with the engine: same retrieval inputs, same detailed result
  shape, same search behavior.
- Keep `engine` the default and keep both implementations independently
  usable.

**Non-Goals:**

- Changing retrieval behavior or the `hybrid-retrieval` contract.
- Changing API response schemas or on-disk formats.
- Per-request backend selection (see Decisions).
- Removing `QAEngine`.

## Decisions

### A dedicated `src/docoracle/backends/` package

Move both answer implementations out of `core`/`agents` into
`src/docoracle/backends/{protocol,engine,agent,factory}.py`. `core` becomes purely
retrieval/data concerns (models, config, loader, chunker, indices, searcher);
`backends` owns generation.

- `protocol.py`: the `AnswerBackend` Protocol plus a shared
  `build_answer_result(...)` helper that produces the detailed dict
  (`question`, `answer`, `confidence`, `citations`, `reasoning`, `sources`,
  `chunk_details`, `retrieved_count`, `retrieval_mode`). Centralizing this removes
  the three near-identical copies that exist today and guarantees both backends
  emit the same shape.
- `engine.py`: `QAEngine` (retrieve-then-answer).
- `agent.py`: `QAAgent`, `RetrievalTool`, `SummaryTool`, `RetrievalState`.
- `factory.py`: `create_answer_backend(config)`.

*Alternatives considered:*
- **Interface-only separation (no file moves)** — leaves generation mixed into
  `core`/`agents` and keeps the import-path ambiguity; rejected because the user
  asked for clear separation.
- **Replace the engine with the agent** — rejected as destructive; removes the
  deterministic fallback while the agent is unproven.

### Native structured output for the engine

Build the engine's answer through a pydantic-ai `Agent` with
`output_type=AnswerResponse` (no tools), reusing the same model/instructions
plumbing as `QAAgent`. This validates the model's response against the schema at
the provider boundary instead of asking for JSON in prose and reparsing it.
`parse_structured`, `_extract_json_object`, and `build_structured_prompt` are
deleted. `StructuredLLMClient` keeps embeddings, chat, and streaming; its
`ask`/`ask_async` structured helpers are replaced by the native path.

Consequence: when a model returns nothing schema-conformant, pydantic-ai raises
instead of silently degrading. This is intended — the previous fallback masked
failures. The `engine` default and `docoracle.backend` switch keep rollback cheap.

*Alternatives considered:*
- **Keep the prompt-embedded JSON schema + parser** — rejected; the heuristics
  are exactly what we want to remove.
- **Use the OpenAI SDK's `response_format` directly** — bypasses pydantic-ai and
  duplicates provider handling already present in both paths.

### Config-driven backend resolved by a shared factory

Add `docoracle.backend` (`engine` | `agent`, default `engine`) to `Config` as a
`Literal` so an unknown value fails at load with an error naming the valid
options. `create_answer_backend(config)` constructs the selected backend and
returns it typed as `AnswerBackend`. Assigning `default_mode` on the protocol
preserves how CLI/server currently apply `--retrieval`.

*Alternatives considered:*
- **Per-request backend field** — leaks backend choice into the public API and
  doubles the surface; deferred until the agent is proven.
- **Duplicating construction per entry point** — the existing drift bug; rejected.

### Retrieval inputs travel through `RetrievalState`

Extend `RetrievalState` to carry resolved `mode`, `k`, and filters alongside
`results`. `RetrievalTool.search` reads them from `RunContext.deps`, so filters
are applied deterministically when the model calls the tool rather than being
prompt hints it can ignore.

### Shared result builder and dedup

`build_answer_result` owns chunk-detail shaping and a single `_unique_chunks`
dedup pass, so `sources`, `chunk_details`, and `retrieved_count` agree in both
backends.

## Risks / Trade-offs

- **Model may skip retrieval under the agent** → the detailed result reports zero
  chunks. Mitigation: parity guarantees behavior *when* the tool is called; the
  engine remains the default and the tool prompt/instructions drive when that
  happens.
- **Native structured output can raise on non-conforming models** → Mitigation:
  it fails loudly instead of fabricating an answer; `docoracle.backend` allows instant
  rollback and the engine/agent parity tests run against stubs.
- **Moving modules breaks imports/tests** → Mitigation: mechanical move plus a
  grep-verified import update in the task list; `basedpyright`/`ruff`/`pytest`
  gate completion.
- **Non-determinism/cost of the agent** → Mitigation: opt-in via config, no data
  migration.
- **Interface drift between engine and agent** → Mitigation: shared protocol,
  shared result builder, and shared parity tests over identical fixtures.

## Migration Plan

1. Create `src/docoracle/backends/` and move `QAEngine`/`QAAgent` there; update imports.
2. Add `docoracle.backend` (default `engine`); route CLI/server through the factory.
3. Switch the engine to native structured output and delete the parser.
4. Enable `agent` in config where desired; roll back by setting `engine`.

No stored data changes, so rollback requires no migration.

## Open Questions

- Should per-request backend selection be added after the agent is validated?
  Deferrable; does not affect the specs or task breakdown.
- Should the agent's `SummaryTool` stay enabled by default, or be opt-in given
  its extra LLM call? Deferrable; defaults are documented in `config.yaml`.
