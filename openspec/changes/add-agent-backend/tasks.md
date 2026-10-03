## 1. Backends package and shared abstraction

- [x] 1.1 Create `src/docoracle/backends/` with `__init__.py`, `protocol.py`, `engine.py`, `agent.py`, `factory.py`
- [x] 1.2 Define the `AnswerBackend` Protocol (`ask`, `ask_async`, `get_related_chunks`, `default_mode`) in `protocol.py` and verify both backends structurally satisfy it (test)
- [x] 1.3 Move `QAEngine` from `src/docoracle/core/qa_engine.py` to `src/docoracle/backends/engine.py` and update every import (`cli.py`, `server/main.py`, tests) so `rg "core.qa_engine"` returns nothing
- [x] 1.4 Move `QAAgent`, `RetrievalTool`, `SummaryTool`, `RetrievalState`, and `create_qa_agent` from `src/docoracle/agents/` to `src/docoracle/backends/agent.py`, remove `src/docoracle/agents/`, and update every import
- [x] 1.5 Extract a shared `build_answer_result(...)` helper (and single chunk-dedup pass) into `protocol.py`; have both backends use it and verify the emitted field set is identical (test)

## 2. Configuration and factory

- [x] 2.1 Add a `docoracle.backend` setting (`Literal["engine", "agent"]`, default `engine`) to `src/docoracle/core/config.py` and verify the default with a config-parsing test
- [x] 2.2 Verify an unsupported backend value is rejected at config load with an error naming `engine` and `agent` (test)
- [x] 2.3 Implement `create_answer_backend(config)` in `src/docoracle/backends/factory.py`, constructing the searcher, LLM client, and selected backend, and verify it returns the engine by default and the agent when configured (test)
- [x] 2.4 Verify the factory logs or reports which backend is active on construction

## 3. Engine native structured output

- [x] 3.1 Route the engine's answer generation through a pydantic-ai `Agent` with `output_type=AnswerResponse` and `instructions=config.prompts.system`, removing the prompt-embedded JSON schema
- [x] 3.2 Remove `parse_structured`, `_extract_json_object`, and `build_structured_prompt` from `src/docoracle/core/llm_outputs.py`; keep the response models
- [x] 3.3 Update `StructuredLLMClient` so its structured answer path uses the native mechanism; keep embeddings, chat, and streaming unchanged
- [x] 3.4 Verify the engine surfaces `confidence`, `citations`, and `reasoning` from a validated response (test)
- [x] 3.5 Verify a non-conforming model response raises instead of degrading to raw text in `answer` (test)
- [x] 3.6 Delete the parser-heuristic tests in `tests/test_structured_llm_client.py` and `tests/test_response_schemas.py` that assert regex/JSON fallback behavior, and replace the prompt-provenance assertions with native-output equivalents
- [x] 3.7 Verify `config.prompts.system` reaches the engine's generation request (test)

## 4. Agent retrieval parity

- [x] 4.1 Extend `RetrievalState` to carry resolved `mode`, `k`, and filters alongside `results`, and verify the fields are populated per run (test)
- [x] 4.2 Apply `mode`/`k`/filters from `RetrievalState` in `RetrievalTool.search` instead of prompt hints, and verify a tool call with filters returns only matching chunks (test)
- [x] 4.3 Default the agent's retrieval mode to `retrieval.mode` from config and `k` to `retrieval.top_k`, and verify configured defaults are used when a request omits them (test)
- [x] 4.4 Apply `rrf_k` and `prefetch_k` to the agent's searcher at construction, matching the engine, and verify the searcher receives the configured values (test)
- [x] 4.5 Reject invalid retrieval modes in the agent with an error naming `hybrid`, `semantic`, `bm25`, and verify the error is raised even when the model skips retrieval (test)

## 5. Agent result and search parity

- [x] 5.1 Ensure `ask_detailed`/`ask_detailed_async` populate every field of the engine-shaped result via the shared builder, and verify against a fixture (test)
- [x] 5.2 Verify duplicates across multiple tool calls are collapsed so `sources`, `chunk_details`, and `retrieved_count` agree (test)
- [x] 5.3 Implement `get_related_chunks` on the agent using its default mode and lazy embedding, and verify it returns scored chunks with provenance (test)

## 6. Entry point integration

- [x] 6.1 Replace direct backend construction in `src/docoracle/server/main.py` with the factory, keeping the `AskResponse`/`SearchResult` output shape unchanged, and verify `/ask` and `/search` tests pass for both backends
- [x] 6.2 Use the async detailed path for the server's `/ask` under the agent backend, and verify no "event loop is already running" error occurs (async test)
- [x] 6.3 Replace direct backend construction in `src/docoracle/cli.py` `ask`/`search` with the factory, preserving the `--retrieval` override behavior via `default_mode`, and verify CLI tests pass for both backends

## 7. Legacy cleanup

- [x] 7.1 Remove the empty/dissolved `src/docoracle/agents/` package and any now-unused imports or symbols introduced by the move (`ruff check` clean)
- [x] 7.2 Remove the dead `Chunker(...)` instantiation in `src/docoracle/cli.py` and the no-op expression in `convert_to_ssh`
- [x] 7.3 Verify `rg "src.agents|core.qa_engine|parse_structured|build_structured_prompt"` returns no source references

## 8. Verification and documentation

- [x] 8.1 Add shared parity tests that run the same question/filter/mode fixtures against engine and agent and assert equivalent result shapes, and verify they pass
- [x] 8.2 Verify the default (`engine`) backend produces the same CLI/API output shape before and after the change
- [x] 8.3 Run the full suite (`pytest`), `ruff check`, `ruff format --check`, and `basedpyright src/` and verify all pass
- [x] 8.4 Update `README.md` and `config.yaml` to document `docoracle.backend`, the `src/docoracle/backends/` layout, and native structured output
- [x] 8.5 Update `docs/concepts.md` to describe the two backends as first-class and reachable
