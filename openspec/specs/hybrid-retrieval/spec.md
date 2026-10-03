## ADDED Requirements

### Requirement: Hybrid retrieval is the default mode

The system MUST perform hybrid retrieval (semantic + BM25 fused via RRF) as the default behavior when no explicit retrieval mode is requested. The default mode MUST be configurable via `retrieval.mode` in `config.yaml`.

#### Scenario: Default mode is hybrid when no override is given
- **WHEN** a query is processed without an explicit `--retrieval` CLI flag and without a `retrieval` field in the API request
- **THEN** both semantic and BM25 indices are searched and their results are fused via Reciprocal Rank Fusion

#### Scenario: Configured default is respected
- **WHEN** `retrieval.mode` is set to `semantic` in `config.yaml`
- **THEN** the system uses semantic-only retrieval for queries that do not specify an explicit mode

### Requirement: SearchIndex Protocol

The system MUST define a `SearchIndex` Protocol in `src/docoracle/core/search_index.py` with `add`, `search`, `save`, and `load` methods. Both `SemanticIndex` and `BM25Index` MUST satisfy this Protocol.

#### Scenario: Both indices satisfy the Protocol
- **WHEN** type checking is run on `SemanticIndex` and `BM25Index`
- **THEN** both are recognized as satisfying the `SearchIndex` Protocol

#### Scenario: Adding chunks updates the index
- **WHEN** `index.add(chunks)` is called with a non-empty list of `Chunk` objects
- **THEN** the chunks become searchable via subsequent `index.search(...)` calls

#### Scenario: Search returns ranked results
- **WHEN** `index.search(query, k=5, filters={"module": "api"})` is called
- **THEN** up to 5 `ScoredChunk` results are returned, ranked by the index's native scoring, with each chunk matching the `module == "api"` filter

#### Scenario: Save and load round-trips state
- **WHEN** `index.save()` is called, the process restarts, and `index.load()` is called
- **THEN** the index returns the same results for the same query as before save

### Requirement: Reciprocal Rank Fusion

The `HybridSearcher` MUST fuse semantic and BM25 rankings using RRF: `rrf_score(d) = Σ_i 1/(rrf_k + rank_i(d))`, where `rrf_k` defaults to 60 and is configurable via `retrieval.rrf_k`. A chunk absent from a retriever's results contributes 0 to the sum for that retriever.

#### Scenario: Chunk found by both retrievers sums both contributions
- **WHEN** a chunk is ranked 1 by semantic and 3 by BM25 with `rrf_k=60`
- **THEN** its RRF score is `1/(60+1) + 1/(60+3)`

#### Scenario: Chunk found by only one retriever contributes one term
- **WHEN** a chunk is ranked 5 by BM25 but absent from semantic results with `rrf_k=60`
- **THEN** its RRF score is `1/(60+5) + 0`

#### Scenario: rrf_k is configurable
- **WHEN** `retrieval.rrf_k` is set to `30` in `config.yaml`
- **THEN** RRF computes scores using `30` as the damping constant

### Requirement: BM25 tokenization with German stemming

The `BM25Index` MUST tokenize text by: (1) lowercasing, (2) splitting on word boundaries via `\w+`, (3) removing tokens shorter than `min_token_length` (default 2) and tokens present in a German stopword list, (4) applying Snowball German stemming via `PyStemmer`. The tokenizer MUST be applied identically at ingest time and at query time.

#### Scenario: German inflected forms within the same paradigm collapse to the same stem
- **WHEN** the tokenizer processes "Beispiel", "Beispiele", and "Beispielen" (or equivalently "Konfiguration" / "Konfigurationen")
- **THEN** all three produce the same stem token, allowing BM25 to match across singular and plural forms

#### Scenario: Stopwords are removed
- **WHEN** the tokenizer processes "Das ist ein Beispiel"
- **THEN** the token list is `["beispiel"]` after stopword removal and stemming

#### Scenario: Minimum token length is enforced
- **WHEN** `min_token_length` is `3` and the tokenizer processes "ab cdef ghi"
- **THEN** the token list is `["cdef", "ghi"]` after the length filter (and any stemming)

#### Scenario: Tokenization is consistent across ingest and query
- **WHEN** a chunk is ingested containing "konfigurierten" and a query is run for "konfigurieren"
- **THEN** both produce the same stem token, allowing BM25 to match them

### Requirement: Metadata filtering parity

The `HybridSearcher` MUST apply metadata filters (module, component, version, and any other filterable chunk attributes) consistently across both retrieval paths. A chunk excluded by the filter MUST NOT appear in either the semantic results or the BM25 results.

#### Scenario: Filter excludes chunks from both paths
- **WHEN** a query is run with `filters={"module": "module-api"}` and a chunk belongs to module `module-core`
- **THEN** that chunk does not appear in either semantic or BM25 results

#### Scenario: Filter accepts chunks eligible for both paths
- **WHEN** a query is run with `filters={"component": "core"}` and a chunk belongs to component `core`
- **THEN** that chunk is eligible to appear in results from either index

#### Scenario: Multiple filters combine with AND semantics
- **WHEN** a query is run with `filters={"component": "core", "version": "1.0"}`
- **THEN** only chunks matching both `component == "core"` AND `version == "1.0"` are returned

### Requirement: Retrieval mode selection

The system MUST allow callers to select between `hybrid`, `semantic`, and `bm25` retrieval modes via a `--retrieval` CLI flag and a `retrieval` field on the `AskRequest` and `SearchRequest` API models. Invalid values MUST be rejected with a clear error.

#### Scenario: CLI flag selects explicit mode
- **WHEN** `docoracle ask "question" --retrieval bm25` is invoked
- **THEN** only the BM25 path is queried and results are not fused

#### Scenario: API field selects explicit mode
- **WHEN** `POST /ask` is called with body `{"question": "...", "retrieval": "semantic"}`
- **THEN** only the semantic path is queried and results are not fused

#### Scenario: Default mode is applied when neither CLI nor API specifies one
- **WHEN** a request omits the `--retrieval` flag and the `retrieval` field
- **THEN** the system uses the value of `retrieval.mode` from `config.yaml` (defaulting to `hybrid`)

#### Scenario: Invalid mode is rejected
- **WHEN** `--retrieval invalid` is invoked or `retrieval: "invalid"` is sent in an API request
- **THEN** the system raises an error naming the valid options (`hybrid`, `semantic`, `bm25`)

### Requirement: Retrieval provenance in results

Each chunk in the final result list MUST carry provenance indicating which retriever(s) found it and its rank from each retriever. The provenance MUST be exposed in CLI `--show-context` output and in the API `chunk_details` field, but MUST NOT appear in the prompt sent to the LLM.

#### Scenario: Chunk found by both retrievers exposes both ranks
- **WHEN** a chunk is ranked 1 by semantic and 4 by BM25
- **THEN** its `ScoredChunk` carries `sources: {"semantic": 1, "bm25": 4}`

#### Scenario: Chunk found by only one retriever exposes single source
- **WHEN** a chunk is only in BM25 results at rank 3
- **THEN** its `ScoredChunk` carries `sources: {"bm25": 3}`

#### Scenario: Provenance does not appear in LLM context
- **WHEN** chunks are formatted into the user prompt sent to the LLM
- **THEN** the prompt contains only the chunk text and a `[Source N]` marker; the `sources` map and `score` field are excluded from the prompt

### Requirement: Config-driven retrieval hyperparameters

The `retrieval` block of `config.yaml` MUST expose `mode`, `rrf_k`, `prefetch_k`, and BM25 settings (`k1`, `b`, `stemming`, `stopwords`, `min_token_length`). The system MUST read these at startup and apply them to all subsequent retrievals. When the block is absent, defaults MUST be: `mode=hybrid`, `rrf_k=60`, `prefetch_k=top_k*4`, `k1=1.5`, `b=0.75`, `stemming=german`, `stopwords=true`, `min_token_length=2`.

#### Scenario: Defaults are applied when config block is absent
- **WHEN** `config.yaml` contains no `retrieval` block
- **THEN** the system uses the default values listed above for all retrievals

#### Scenario: Custom config overrides defaults
- **WHEN** `retrieval.rrf_k: 30` is set in `config.yaml`
- **THEN** RRF uses 30 as the damping constant instead of 60

#### Scenario: BM25 k1 and b are tunable
- **WHEN** `retrieval.bm25.k1: 2.0` and `retrieval.bm25.b: 0.5` are set
- **THEN** BM25 scoring uses k1=2.0 and b=0.5 instead of the defaults

#### Scenario: Stemming can be disabled
- **WHEN** `retrieval.bm25.stemming: none` is set
- **THEN** the BM25 tokenizer skips the stemming step and operates on the raw lowercased word tokens

### Requirement: On-disk persistence

The system MUST persist three files in `./data/vectorstore/`: `chunks.pkl` (canonical chunks, owned by `HybridSearcher`), `index.faiss` (semantic vectors, owned by `SemanticIndex`), `bm25.pkl` (tokenized corpus and BM25 parameters, owned by `BM25Index`). Each file MUST be owned by exactly one class.

#### Scenario: Save writes all three files
- **WHEN** `HybridSearcher.save()` is called after ingestion
- **THEN** all three files exist and contain data sufficient to restore the system state

#### Scenario: Load reads all three files
- **WHEN** the system starts and the store directory contains all three files
- **THEN** the chunks, semantic index, and BM25 index are restored and queryable

#### Scenario: Missing file produces a clear error
- **WHEN** the store directory contains only some of the three files (e.g., `index.faiss` and `chunks.pkl` but no `bm25.pkl`)
- **THEN** the system raises an error identifying which expected file is missing

### Requirement: HybridSearcher owns canonical chunks

The `HybridSearcher` MUST own the `List[Chunk]` as the canonical source of truth. Indices MUST receive read-only access to chunks at construction and MUST NOT mutate the canonical list. When an index returns a chunk reference from search, the `HybridSearcher` MUST resolve it against its canonical store before returning the result.

#### Scenario: Adding chunks updates the canonical store exactly once
- **WHEN** `hybrid_searcher.add_chunks(chunks)` is called
- **THEN** the chunks appear in `hybrid_searcher.chunks` exactly once, regardless of how many internal indices were updated

#### Scenario: Indices do not own the canonical chunk list
- **WHEN** a `SemanticIndex` instance is inspected after `hybrid_searcher.add_chunks(...)` has been called
- **THEN** the semantic index does not hold its own mutable copy of the chunks list

### Requirement: Backward-incompatible on-disk format

The system MUST NOT attempt to load data from the legacy `metadata.pkl` file. On first run after upgrade, the user MUST re-ingest their documentation to populate the new on-disk format.

#### Scenario: Legacy metadata.pkl is not loaded
- **WHEN** the system starts and `./data/vectorstore/metadata.pkl` exists but the three new files do not
- **THEN** the system treats the store as empty rather than attempting to migrate the legacy file
