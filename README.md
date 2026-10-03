# DocOracle

Answer questions about your Antora/AsciiDoc documentation using LLM APIs.

## Requirements

- **Python 3.12 or higher** (Pydantic v2 and pydantic-ai require Python 3.10+)
- New dependencies: `pydantic>=2.5.0`, `pydantic-ai>=2.0.0`

## Features

- **Antora-aware**: Leverages Antora's directory structure and metadata
- **Configurable chunking**: Split documents into embeddable chunks with configurable size
- **Local hybrid store**: FAISS vectors + BM25 index stored locally
- **Hybrid retrieval**: Semantic similarity fused with BM25 lexical search via Reciprocal Rank Fusion (RRF) by default; semantic-only and BM25-only modes are also available
- **German-aware BM25**: Snowball German stemmer and stopword filtering tuned for the German documentation corpus
- **Retrieval provenance**: Every retrieved chunk records which retriever(s) found it
- **Source linking**: Answers include links to the original documentation (module:pages:page#section)
- **Filtering**: Search within specific modules, components, or versions
- **Structured outputs**: Type-safe LLM responses with Pydantic validation

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure

Create a `config.yaml` file (or use the provided one):

```yaml
embedding:
  model: mistral-embed
  api_key: ${LLM_API_KEY}
  chunk_size: 512
  chunk_overlap: 50

vector_store:
  type: faiss
  path: ./data/vectorstore

sources:
  antora_root: ./antora-docs  # Path to your Antora project
```

Set your LLM API key. You can either:

**Option 1: Environment variable**
```bash
export LLM_API_KEY="your-api-key-here"
```

**Option 2: .env file**
```bash
# Create .env file
cp .env.example .env
# Edit .env and add your API key
LLM_API_KEY=your_api_key_here
```

The application automatically loads environment variables from a `.env` file if present.

### 3. Prepare your Antora project

Ensure your Antora project has this structure:

```
antora-docs/
├── antora.yml          # Component configuration
└── modules/
    └── module-name/
        ├── pages/
        │   └── page.adoc
        └── nav.adoc     # Navigation (optional)
```

### 4. Ingest documentation

```bash
# Using CLI
python -m src.cli ingest
docoracle ingest

# With options
docoracle ingest --antora-root ./docs --chunk-size 256 --overlap 25
```

This will:
1. Load all AsciiDoc files from your Antora project
2. Parse metadata (module, component, version, page role)
3. Split into chunks (512 tokens by default)
4. Generate embeddings using LLM
5. Store embeddings in a local FAISS index and chunks in a BM25 index

> **Note**: Re-ingest is required after upgrading from a pre-`0.2.0` install — the on-disk layout changed from `metadata.pkl` to `chunks.pkl` + `bm25.pkl` + `index.faiss`. See [On-disk layout](#on-disk-layout) below.

### 5. Start the web server

```bash
# Using CLI
docoracle serve
docoracle serve --host 0.0.0.0 --port 8080

# Or directly
python -m src.cli serve
```

### 6. Ask questions via CLI

```bash
# Simple question
docoracle ask "How do I configure the database?"

# Filter by module
docoracle ask "What are the API endpoints?" --module module-api

# Show sources
docoracle ask "How to install?" --show-sources

# Show full context
docoracle ask "What's new in v2?" --show-context
```

## Using the Web Server

### Start the server

```bash
# Using the installed script
docoracle-server

# Or directly with uvicorn
python -m src.server

# Or with custom host/port
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` or `/ui` | Web UI interface |
| GET | `/health` | Health check |
| GET | `/info` | Vector store statistics |
| POST | `/ingest` | Ingest documentation |
| POST | `/ask` | Ask a question |
| POST | `/search` | Search similar chunks |

### Web UI

Open `http://localhost:8000/ui` in your browser to access the web interface.

Features:
- Ask questions in natural language
- Filter by module
- View retrieved sources with links
- Question history with local storage
- Responsive design

#### UI Screenshot Description

The web UI provides:
- A text area for entering questions
- Module filter dropdown (populated from your documentation)
- Number of results selector
- Ask and Clear buttons
- Loading spinner during processing
- Answer display with formatted text
- Source references with module name and link to original documentation
- Recent question history

## Commands

### `ingest`

```bash
docoracle ingest [OPTIONS]

Options:
  --antora-root TEXT    Path to Antora project root
  --chunk-size INTEGER  Chunk size in tokens (default: 512)
  --overlap INTEGER     Overlap between chunks (default: 50)
  --force               Force re-ingest even if index exists
```

### `ask`

```bash
docoracle ask QUESTION [OPTIONS]

Arguments:
  QUESTION  The question to answer

Options:
  --module TEXT              Filter by module name
  --component TEXT           Filter by component name
  --version TEXT             Filter by version
  --k INTEGER                Number of chunks to retrieve
  --retrieval [hybrid|semantic|bm25]   Retrieval mode (default from config: hybrid)
  --show-sources             Show source links
  --show-context             Show retrieved context chunks
```

### `search`

```bash
docoracle search TEXT [OPTIONS]

Arguments:
  TEXT  Text to search for similar chunks

Options:
  --k INTEGER                       Number of results
  --module TEXT                     Filter by module
  --retrieval [hybrid|semantic|bm25]  Retrieval mode (default from config: hybrid)
```

### `info`

```bash
docoracle info
```

Shows information about the vector store (total chunks, count by module).

### `components`

```bash
docoracle components [OPTIONS]

Options:
  --antora-root TEXT      Path to Antora root (default: ./antora-docs)
  --list-sources          List content sources from playbook
  --clone                 Clone missing component repositories
  --ssh / --https         Use SSH URLs (default) or HTTPS
  --target-dir TEXT       Target directory for cloning
  --dry-run              Show what would be cloned without cloning
```

**Examples:**

```bash
# List all components from playbook
docoracle components --antora-root antora-docs --list-sources

# Clone all missing repos using SSH (default)
docoracle components --antora-root antora-docs --clone

# Clone using HTTPS URLs
docoracle components --antora-root antora-docs --clone --https

# Dry run to see what would be cloned
docoracle components --antora-root antora-docs --clone --dry-run

# Clone to a specific directory
docoracle components --antora-root antora-docs --clone --target-dir ./all-docs
```

**Note:** HTTPS URLs are automatically converted to SSH format (e.g.,
`https://git.tryb.de/doc/site` → `git@git.tryb.de:doc/site.git`).
Use `--https` to keep the original HTTPS URLs.

## Project Structure

```
docoracle/
├── src/
│   └── docoracle/
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py           # Data models (Chunk, Document) - Pydantic
│       │   ├── config.py          # Configuration management - Pydantic
│       │   ├── llm_outputs.py     # Structured LLM output models
│       │   ├── loader.py          # Antora-aware AsciiDoc loader
│       │   ├── chunker.py         # Text chunker
│       │   ├── converter.py       # AsciiDoc -> Markdown conversion
│       │   ├── search_index.py    # SearchIndex Protocol, ScoredChunk, matches_filters
│       │   ├── semantic_index.py  # FAISS-backed semantic index
│       │   ├── bm25_index.py      # BM25 lexical index (German tokenizer)
│       │   └── hybrid_searcher.py # Composes semantic + BM25 via RRF
│       ├── backends/              # Interchangeable answer backends
│       │   ├── protocol.py        # AnswerBackend Protocol + shared result builder
│       │   ├── engine.py          # QAEngine: retrieve-then-answer
│       │   ├── agent.py           # QAAgent: tool-using pydantic-ai agent
│       │   └── factory.py         # create_answer_backend (config-driven)
│       ├── api/
│       │   └── structured_llm_client.py  # Embeddings, chat, native structured output
│       ├── server/                # FastAPI app + static UI
│       └── cli.py                 # CLI interface
├── tests/
├── data/
│   ├── raw/                   # Original files (optional)
│   ├── processed/             # Processed files (optional)
│   └── vectorstore/           # chunks.pkl + index.faiss + bm25.pkl
├── config.yaml                # Configuration
└── pyproject.toml
```

## Configuration

### Embedding Settings

```yaml
embedding:
  model: mistral-embed      # Embedding model
  api_key: ${LLM_API_KEY}  # API key (from env var)
  chunk_size: 512           # Tokens per chunk
  chunk_overlap: 50         # Token overlap between chunks
  rate_limit_delay: 0.1     # Delay between batch requests (seconds)
```

### Vector Store Settings

```yaml
vector_store:
  type: faiss              # Currently only FAISS supported
  path: ./data/vectorstore  # Storage directory
```

### Retrieval & Generation

```yaml
retrieval:
  top_k: 5                 # Number of chunks to retrieve
  mode: hybrid             # hybrid | semantic | bm25 (default: hybrid)
  rrf_k: 60                # RRF damping constant
  prefetch_k: 20           # Per-retriever prefetch before fusion
  bm25:
    k1: 1.5
    b: 0.75
    stemming: german       # none | german | english | ...
    stopwords: true
    min_token_length: 2

generation:
  model: mistral-small-latest  # Chat model for answers
  temperature: 0.3        # Sampling temperature
  max_context_length: 4000 # Max context tokens

llm:
  api_url: https://api.mistral.ai/v1  # Base URL for LLM API
  timeout: 120           # API timeout in seconds
```

Any vendor with an OpenAI-compatible endpoint works here — the client
routes requests through pydantic-ai's bundled OpenAI provider against
``llm.api_url``, so no vendor-specific optional package is required
(e.g. no need for ``pydantic-ai-slim[mistral]``).

### Custom Prompts

Customize the system and user prompts:

```yaml
prompts:
  system: |
    You are a helpful assistant...
  user: |
    Question: {question}
    Context: {context}
    Answer:
```

## Answer Backends

Two interchangeable answer backends live under `src/docoracle/backends/`. Select one with
`docoracle.backend` in `config.yaml`; both honor the same retrieval inputs and return
the same detailed result shape, so the CLI and HTTP API are backend-agnostic.

```yaml
docoracle:
  backend: engine   # engine (default) | agent
```

- **`engine`** — deterministic retrieve-then-answer: one retrieval pass, one
  generation. This is the default and the fallback.
- **`agent`** — a pydantic-ai agent with a retrieval tool (and a summarization
  tool) that decides for itself whether to retrieve. Useful for multi-hop or
  adaptive retrieval, at the cost of determinism and extra LLM calls.

## Structured Outputs

Answers are requested and validated through pydantic-ai's **native structured
output** (`output_type=AnswerResponse`): the provider is told the response
schema, and the result is a validated Pydantic object — no prompt-embedded JSON
schema and no regex/heuristic parsing. If a model returns something that does
not conform to the schema, generation fails loudly instead of fabricating an
answer.

The default `AnswerResponse` model includes:
- `answer`: The generated answer text
- `confidence`: Optional confidence score (0.0-1.0)
- `citations`: List of source citations
- `reasoning`: Optional chain-of-thought

## How It Works

For a fuller mental model of the system — including the two phases, the core
concepts, and the [browser session workflow](docs/concepts.md) — see
[`docs/concepts.md`](docs/concepts.md).

1. **Ingestion**: Documents are loaded, parsed for Antora metadata, split into chunks, and embedded
2. **Storage**: Three files under `./data/vectorstore/`:
   - `chunks.pkl` — canonical chunk list (source of truth)
   - `index.faiss` — FAISS embedding vectors for semantic search
   - `bm25.pkl` — tokenized corpus for lexical search (German stemming applied)
3. **Query**: The question runs through one or both indices:
   - **Semantic**: embedding is compared via L2 nearest neighbor
   - **BM25**: the question is tokenized with the same German pipeline and scored
   - **Hybrid** (default): both paths run, fused via Reciprocal Rank Fusion
4. **Provenance**: Each retrieved chunk records which retriever(s) found it (e.g. `{semantic: 1, bm25: 4}`) — surfaced in CLI `--show-context` output and the API `chunk_details` field
5. **Answer**: Top-k chunks (after fusion) form the LLM context; provenance is excluded from the prompt
6. **Citation**: Answers include links to the source documentation (module:pages:page#section)

### On-disk layout

```
./data/vectorstore/
  ├── chunks.pkl     # canonical chunk list
  ├── index.faiss    # FAISS vectors (semantic index)
  └── bm25.pkl       # tokenized corpus + BM25 params
```

> **Breaking change**: prior versions stored only `metadata.pkl` + `index.faiss`. After upgrading, run `docoracle ingest --force` to populate the new layout.

## Document Linking

Each chunk stores metadata that allows linking back to the original documentation:

- `module`: Antora module name
- `component`: Component name from antora.yml
- `version`: Component version
- `page_id`: Antora page ID (e.g., `module:pages:installation`)
- `section_id`: AsciiDoc section ID

Links are formatted as `module:pages:page#section` which can be used to navigate to the source in your Antora site.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install the git hooks (runs ruff, basedpyright and pytest on commit)
pre-commit install

# Run all hooks on demand
pre-commit run --all-files

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type-check
basedpyright src/
```

## License

MIT License
