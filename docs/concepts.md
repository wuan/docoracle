# Concepts

A mental model of the DocOracle system: what it stores, how a question becomes
an answer, and how a browser session puts those pieces together.

## Overview

DocOracle answers natural-language questions about an Antora/AsciiDoc
documentation set. It does this in two phases:

1. **Ingest** (offline): read the documentation, split it into chunks, embed the
   chunks, and persist both a semantic index and a lexical (BM25) index.
2. **Query** (online): retrieve the chunks most relevant to a question and ask a
   language model to answer using only those chunks.

Nothing is generated from the model's own knowledge: answers are grounded in
retrieved documentation, and every answer can point back to its sources.

## Core concepts

- **Document** — one loaded Antora page, with its component/module metadata.
- **Chunk** — a passage of a document, sized in tokens, that is the unit of both
  storage and retrieval. A chunk carries its Antora metadata (component, module,
  version, page id, section id/title) so it can be linked back to the source.
- **Index** — a searchable store over chunks. There are two:
  - **Semantic index** (FAISS): compares the meaning of the question and the
    chunk via embedding vectors.
  - **BM25 index**: compares the words of the question and the chunk, with
    German stemming and stopword removal tuned for this corpus.
- **Scored chunk** — a retrieved chunk plus a score and **provenance**: which
  retriever(s) found it and at what rank (for example
  `{"semantic": 1, "bm25": 4}`). Provenance is shown to users but never sent to
  the model.
- **Retrieval mode** — how the two indices are used:
  - `hybrid` (default): query both, then fuse their rankings with Reciprocal
    Rank Fusion (RRF). Robust when either wording or meaning matches.
  - `semantic`: embeddings only. Good for paraphrases.
  - `bm25`: lexical only. Good for exact terms, identifiers, and rare tokens.

### Why fusion (RRF)

Semantic and BM25 scores are not comparable, so hybrid mode does not add them.
Instead each retriever produces a ranked list, and Reciprocal Rank Fusion scores
each chunk by `1 / (rrf_k + rank)` summed across retrievers. This rewards chunks
that both retrievers agree on without needing score normalization.

### On-disk state

Three files under `./data/vectorstore/` are the source of truth:

```
chunks.pkl    # canonical chunk list
index.faiss   # semantic vectors
bm25.pkl      # tokenized corpus + BM25 parameters
```

Re-ingesting after a format change is required; there is no automatic migration
from the legacy `metadata.pkl` layout.

## Answer generation

Retrieval is the same regardless of how the answer is produced. Two interchangeable
**answer backends** consume the retrieved chunks:

- **Engine** (`QAEngine`): a fixed retrieve-then-answer pipeline. Deterministic,
  the default.
- **Agent** (`QAAgent`): a pydantic-ai agent that decides for itself whether to
  call a retrieval tool before producing a structured answer. Its retrieved
  chunks are collected per run and reported with the same result shape.

Both backends return the same detailed result: the answer, model confidence and
citations, the reasoning (if any), the sources, per-chunk details with
provenance, the retrieved count, and the retrieval mode used. Which one runs is a
configuration choice (`docoracle.backend`, default `engine`); callers (CLI and web) do
not change. Both generate their answer through pydantic-ai's native structured
output, so the response is validated against the answer schema at the provider
boundary.

## The browser session workflow

The web UI (`src/docoracle/server/static/ui.html`, served at `/ui`) is a single page. A
typical session proceeds like this:

1. **Load.** The page opens and calls `GET /info`. The response reports the
   indexed chunk count and the modules grouped by component, which populate the
   module dropdown and the status line. Previously asked questions are restored
   from the browser's `localStorage` (`qaHistory`), so history is per-browser and
   survives reloads.

2. **Compose a question.** The user types a question, optionally picks a module
   filter and the number of chunks to retrieve (`k`), then submits. Enter submits;
   Shift+Enter inserts a newline.

3. **Ask.** The page issues `POST /ask` with `{question, module, k,
   show_sources: true}`. The server resolves the configured answer backend and
   runs the retrieval-plus-generation pipeline. While waiting, a loading state is
   shown.

4. **Answer and confidence.** The response's `answer` is rendered as sanitized
   markdown. If the model returned a confidence value, it is shown as a coloured
   badge; any model citations appear as chips.

5. **Reasoning.** If the response includes `reasoning`, it is shown in a
   collapsible block; otherwise the block stays hidden.

6. **Sources.** The response's `source_details` are listed with their Antora
   component/module/section, a text preview, and provenance. If the site URL is
   configured, each source links to the published page; otherwise clicking copies
   the page identifier to the clipboard.

7. **History.** The question is prepended to the local history (capped at the 10
   most recent). Clicking a history entry re-runs it.

Errors (for example an unreachable API or an invalid retrieval mode) surface in
an inline error area; the page returns to an idle state when the request
finishes.

### Session at a glance

```
browser                         server
  |  GET /info            ----->  read indices, group modules by component
  |  <---- total_chunks, modules_by_component, retrieval_mode
  |
  |  POST /ask {question,module,k} ----> select backend, retrieve, generate
  |  <---- answer, confidence, citations, reasoning,
  |        sources, source_details, retrieved_count, retrieval_mode
  |
  |  render + save question to localStorage
```

## CLI and HTTP are the same pipeline

The CLI (`docoracle ask`, `docoracle search`) and the HTTP API (`/ask`, `/search`) both go
through the same retrieval and answer backends. The CLI is a thin wrapper that
prints the same result fields; `--show-sources` and `--show-context` expose the
sources and per-chunk details that the browser UI renders. Behavior differences
between the interfaces are presentational, not architectural.
