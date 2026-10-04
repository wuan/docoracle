"""FastAPI backend server for DocOracle."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docoracle.api.structured_llm_client import StructuredLLMClient
from docoracle.backends.factory import create_answer_backend
from docoracle.backends.protocol import AnswerBackend
from docoracle.core.config import Config, get_config
from docoracle.core.hybrid_searcher import VALID_MODES, HybridSearcher
from docoracle.core.loader import AntoraLoader
from docoracle.core.models import Chunk

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("server.main")

# Initialize components (lazy loading)
_searcher: HybridSearcher | None = None
_backend: AnswerBackend | None = None
_llm_client: StructuredLLMClient | None = None
_config: Config | None = None


def _get_config(config_path: str = "config.yaml") -> Config:
    """Load and cache the validated configuration."""
    global _config
    if _config is None:
        _config = get_config(config_path)
    return _config


def get_searcher(config_path: str = "config.yaml") -> HybridSearcher:
    """Get or initialize the hybrid searcher and its answer backend."""
    global _searcher, _backend, _llm_client

    if _searcher is None:
        config = _get_config(config_path)
        _searcher = HybridSearcher()
        _searcher.load(config.vector_store.path)

        if _llm_client is None:
            _llm_client = StructuredLLMClient(config_path)

        if _backend is None:
            _backend = create_answer_backend(_searcher, _llm_client, config)

    return _searcher


def get_backend(config_path: str = "config.yaml") -> AnswerBackend:
    """Get or initialize the configured answer backend."""
    get_searcher(config_path)
    return _backend  # type: ignore[return-value]


# Models
class IngestRequest(BaseModel):
    antora_root: str = "./antora-docs"
    chunk_size: int | None = None
    overlap: int | None = None
    force: bool = False


class AskRequest(BaseModel):
    question: str
    module: str | None = None
    component: str | None = None
    version: str | None = None
    k: int | None = None
    retrieval: Literal["hybrid", "semantic", "bm25"] | None = Field(
        default=None, description="hybrid|semantic|bm25; default from config"
    )
    show_sources: bool = True
    show_context: bool = False


class SearchRequest(BaseModel):
    text: str
    k: int = 5
    module: str | None = None
    retrieval: Literal["hybrid", "semantic", "bm25"] | None = Field(
        default=None, description="hybrid|semantic|bm25; default from config"
    )


# Response models
class SourceLink(BaseModel):
    link: str
    url: str | None = None
    module: str
    component: str
    breadcrumb: str | None = None
    section_title: str | None = None
    text_preview: str
    sources: dict[str, int] = Field(
        default_factory=dict, description="retriever name -> 1-based rank"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence (0.0-1.0)",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Source references cited by the model",
    )
    reasoning: str | None = Field(
        None,
        description="Optional model reasoning / chain-of-thought",
    )
    sources: list[str]
    source_details: list[SourceLink]
    retrieved_count: int
    retrieval_mode: str


class SearchResult(BaseModel):
    text: str
    module: str
    page_id: str
    section_id: str | None = None
    section_title: str | None = None
    link: str
    score: float
    sources: dict[str, int] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    total_chunks: int
    message: str | None = None


# Create FastAPI app
app = FastAPI(
    title="DocOracle API",
    description="Answer questions about documentation",
    version="0.2.0",
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Log full traceback for any unhandled exception and return 500.

    Ensures we always see *why* a request 500'd instead of just the bare
    "Internal Server Error" line in the access log.
    """
    logger.exception("Unhandled exception serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_retrieval(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid retrieval mode {value!r}. Must be one of {VALID_MODES}.",
        )
    return value


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def get_info() -> dict[str, Any]:
    """Get store information."""
    config = _get_config()
    searcher = get_searcher()

    modules: dict[str, int] = {}
    components: dict[str, int] = {}
    modules_by_component: dict[str, dict[str, int]] = {}
    for chunk in searcher.chunks:
        modules[chunk.module] = modules.get(chunk.module, 0) + 1
        components[chunk.component] = components.get(chunk.component, 0) + 1
        if chunk.component not in modules_by_component:
            modules_by_component[chunk.component] = {}
        modules_by_component[chunk.component][chunk.module] = (
            modules_by_component[chunk.component].get(chunk.module, 0) + 1
        )

    return {
        "total_chunks": len(searcher),
        "semantic_chunks": len(searcher.semantic),
        "bm25_chunks": len(searcher.bm25),
        "retrieval_mode": config.retrieval.mode,
        "store_path": str(Path(config.vector_store.path)),
        "modules": modules,
        "components": components,
        "modules_by_component": modules_by_component,
    }


@app.post("/ingest")
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest AsciiDoc modules into the hybrid store."""
    config = _get_config()

    chunk_size: int = request.chunk_size or config.embedding.chunk_size
    overlap: int = request.overlap or config.embedding.chunk_overlap
    store_path = Path(config.vector_store.path)

    searcher = get_searcher()

    if not request.force and len(searcher) > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Store already has {len(searcher)} chunks. Use force=True to re-ingest.",
        )

    loader = AntoraLoader(request.antora_root)
    documents = loader.load_component()

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found at {request.antora_root}",
        )

    all_chunks: list[Chunk] = []
    all_texts: list[str] = []
    for doc in documents:
        chunks = doc.to_chunks(chunk_size=chunk_size, overlap=overlap)  # type: ignore[arg-type]
        all_chunks.extend(chunks)
        all_texts.extend([c.text for c in chunks])

    llm_client = StructuredLLMClient()
    embeddings = llm_client.embed_batch(all_texts, batch_size=32)

    searcher.add_chunks(all_chunks, embeddings)
    searcher.save(str(store_path))

    return IngestResponse(
        status="success",
        chunks_added=len(all_chunks),
        total_chunks=len(searcher),
        message=f"Ingested {len(documents)} documents, {len(all_chunks)} chunks",
    )


@app.post("/ask")
async def ask(request: AskRequest):
    """Answer a question about the documentation."""
    retrieval = _validate_retrieval(request.retrieval)
    backend = get_backend()

    # Must use the async path: this handler runs on FastAPI's event loop, and
    # the sync backend path uses pydantic-ai's sync wrappers, which raise
    # ``RuntimeError: This event loop is already running.`` when invoked from
    # async code.
    result = await backend.ask_detailed_async(
        question=request.question,
        module=request.module,
        component=request.component,
        version=request.version,
        k=request.k,
        retrieval=retrieval,
    )

    source_details: list[SourceLink] = []
    for chunk in result.get("chunk_details", []):
        source_details.append(
            SourceLink(
                link=chunk["link"],
                url=chunk.get("url"),
                module=chunk["module"],
                component=chunk.get("component", "unknown"),
                breadcrumb=chunk.get("breadcrumb"),
                section_title=chunk.get("section_title"),
                text_preview=chunk["text"][:200] + "..."
                if len(chunk["text"]) > 200
                else chunk["text"],
                sources=chunk.get("sources", {}),
            )
        )

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        confidence=result.get("confidence"),
        citations=result.get("citations", []),
        reasoning=result.get("reasoning"),
        sources=result.get("sources", []),
        source_details=source_details,
        retrieved_count=result.get("retrieved_count", 0),
        retrieval_mode=result.get("retrieval_mode", "hybrid"),
    )


@app.post("/search")
async def search(request: SearchRequest):
    """Search for similar documentation chunks."""
    retrieval = _validate_retrieval(request.retrieval)
    backend = get_backend()

    if retrieval:
        backend.default_mode = retrieval

    filters = {}
    if request.module:
        filters["module"] = request.module

    results = backend.get_related_chunks(
        request.text,
        k=request.k,
        **filters,
    )

    return [
        SearchResult(
            text=scored.chunk.text[:500] + "..."
            if len(scored.chunk.text) > 500
            else scored.chunk.text,
            module=scored.chunk.module,
            page_id=scored.chunk.page_id,
            section_id=scored.chunk.section_id,
            section_title=scored.chunk.section_title,
            link=scored.chunk.get_link(),
            score=scored.score,
            sources=scored.sources,
        )
        for scored in results
    ]


# Mount static files for frontend

# Get the directory of this file
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui.html", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend UI."""
    ui_path = static_dir / "ui.html"
    headers = {"Cache-Control": "no-store, max-age=0"}
    if ui_path.exists():
        # Read in a worker thread so the sync file I/O does not block the event loop.
        html_content = await asyncio.to_thread(ui_path.read_text, encoding="utf-8")
        return HTMLResponse(
            content=html_content,
            status_code=200,
            media_type="text/html",
            headers=headers,
        )
    return HTMLResponse(
        content="<h1>DocOracle API</h1><p>UI not found.</p>",
        status_code=404,
        headers=headers,
    )
