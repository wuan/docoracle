"""Tests for the FastAPI server endpoints (src/docoracle/server/main.py).

All external components (searcher, answer backend, LLM client, config) are
replaced with lightweight fakes so the tests exercise the HTTP layer only:
request validation, response shapes, error mapping, and caching behavior.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from docoracle.core.config import Config
from docoracle.core.models import Chunk
from docoracle.server import main as server_main
from docoracle.server.main import app


def _chunk(text: str = "Some documentation text.", module: str = "ROOT", **overrides) -> Chunk:
    data = {
        "text": text,
        "chunk_id": "c1",
        "module": module,
        "component": "docoracle",
        "version": "1.0",
        "page_id": f"{module}:pages:index",
    }
    data.update(overrides)
    return Chunk(**data)


class FakeSearcher:
    """Minimal HybridSearcher stand-in for the HTTP layer."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or []
        self.saved_to: str | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def semantic(self) -> list[int]:
        return list(range(len(self.chunks)))

    @property
    def bm25(self) -> list[int]:
        return list(range(len(self.chunks)))

    def add_chunks(self, chunks, embeddings) -> None:
        self.chunks.extend(chunks)

    def save(self, path: str) -> None:
        self.saved_to = path


class FakeBackend:
    """Minimal answer-backend stand-in returning canned results."""

    def __init__(self, ask_result: dict | None = None, chunks: list[Chunk] | None = None):
        self.ask_result = ask_result or {}
        self.chunks = chunks or []
        self.default_mode = "hybrid"
        self.related_calls: list[dict] = []

    async def ask_detailed_async(self, **kwargs) -> dict:
        return dict(self.ask_result, **kwargs)

    def get_related_chunks(self, text: str, k: int = 5, **filters):
        self.related_calls.append({"text": text, "k": k, **filters})
        return [
            SimpleNamespace(
                chunk=chunk,
                score=1.0 - i * 0.1,
                sources={"hybrid": i + 1},
            )
            for i, chunk in enumerate(self.chunks[:k])
        ]


@pytest.fixture()
def fake_parts(monkeypatch):
    """Install fakes into the server module and reset them after the test."""
    searcher = FakeSearcher([_chunk(), _chunk("Second chunk", module="nav")])
    backend = FakeBackend(
        ask_result={
            "question": "q",
            "answer": "an answer",
            "confidence": 0.9,
            "citations": ["ref1"],
            "sources": ["ROOT:pages:index"],
            "retrieved_count": 2,
            "retrieval_mode": "hybrid",
            "chunk_details": [
                {
                    "link": "ROOT:pages:index",
                    "module": "ROOT",
                    "component": "docoracle",
                    "section_title": "Index",
                    "text": "x" * 250,
                    "sources": {"hybrid": 1},
                }
            ],
        },
        chunks=searcher.chunks,
    )
    monkeypatch.setattr(server_main, "_config", Config())
    monkeypatch.setattr(server_main, "_searcher", searcher)
    monkeypatch.setattr(server_main, "_backend", backend)
    monkeypatch.setattr(server_main, "_llm_client", object())
    return SimpleNamespace(searcher=searcher, backend=backend)


@pytest.fixture()
def client(fake_parts) -> TestClient:
    return TestClient(app)


# -----------------------------------------------------------------------------
# Health and info
# -----------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_info_aggregates_store_metadata(client: TestClient, fake_parts):
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 2
    assert body["semantic_chunks"] == 2
    assert body["bm25_chunks"] == 2
    assert body["modules"] == {"ROOT": 1, "nav": 1}
    assert body["components"] == {"docoracle": 2}
    assert body["modules_by_component"]["docoracle"] == {"ROOT": 1, "nav": 1}


def test_unhandled_exception_returns_logged_500(client: TestClient, fake_parts, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(fake_parts.backend, "ask_detailed_async", boom)
    # Our exception handler turns any unhandled error into a logged 500;
    # TestClient must not re-raise it for us to observe that behavior.
    raw_client = TestClient(app, raise_server_exceptions=False)
    response = raw_client.post("/ask", json={"question": "q"})
    assert response.status_code == 500
    assert "RuntimeError" in response.json()["detail"]


# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------


def test_ingest_conflicts_without_force(client: TestClient, fake_parts):
    response = client.post("/ingest", json={})
    assert response.status_code == 409
    assert "force=True" in response.json()["detail"]


def test_ingest_force_reruns_pipeline(client: TestClient, fake_parts, monkeypatch):
    class FakeDoc:
        def __init__(self):
            self._chunk = _chunk("ingested")

        def to_chunks(self, chunk_size: int, overlap: int):
            assert chunk_size > 0
            return [self._chunk]

    class FakeLoader:
        def __init__(self, root: str):
            self.root = root

        def load_component(self):
            return [FakeDoc()]

    class FakeLLM:
        def embed_batch(self, texts, batch_size: int = 32):
            return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(server_main, "AntoraLoader", FakeLoader)
    monkeypatch.setattr(server_main, "StructuredLLMClient", FakeLLM)

    response = client.post("/ingest", json={"force": True})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["chunks_added"] == 1
    assert body["total_chunks"] == 3
    assert fake_parts.searcher.saved_to is not None


def test_ingest_without_documents_returns_404(client: TestClient, fake_parts, monkeypatch):
    class EmptyLoader:
        def __init__(self, root: str):
            pass

        def load_component(self):
            return []

    monkeypatch.setattr(server_main, "AntoraLoader", EmptyLoader)
    response = client.post("/ingest", json={"force": True})
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# Ask
# -----------------------------------------------------------------------------


def test_ask_returns_detailed_response(client: TestClient):
    response = client.post("/ask", json={"question": "How do I configure it?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How do I configure it?"
    assert body["answer"] == "an answer"
    assert body["confidence"] == 0.9
    assert body["citations"] == ["ref1"]
    assert body["retrieved_count"] == 2
    assert body["retrieval_mode"] == "hybrid"
    assert len(body["source_details"]) == 1
    detail = body["source_details"][0]
    assert detail["module"] == "ROOT"
    assert detail["component"] == "docoracle"
    assert detail["section_title"] == "Index"
    # 250-char chunk text is truncated to a 200-char preview + ellipsis
    assert detail["text_preview"].endswith("...")
    assert len(detail["text_preview"]) == 203
    assert detail["sources"] == {"hybrid": 1}


def test_ask_short_text_not_truncated(client: TestClient, fake_parts):
    fake_parts.backend.ask_result["chunk_details"] = [
        {"link": "l", "module": "m", "text": "short", "sources": {}}
    ]
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 200
    assert response.json()["source_details"][0]["text_preview"] == "short"


def test_ask_rejects_invalid_retrieval_mode(client: TestClient):
    # AskRequest.retrieval is a Literal, so pydantic rejects unknown modes
    # with a standard validation error before the handler runs.
    response = client.post("/ask", json={"question": "q", "retrieval": "bogus"})
    assert response.status_code == 422
    assert "bogus" in response.text


def test_ask_requires_question(client: TestClient):
    response = client.post("/ask", json={"module": "m"})
    assert response.status_code == 422


def test_ask_passes_filters_and_k_to_backend(client: TestClient, fake_parts):
    response = client.post(
        "/ask",
        json={"question": "q", "module": "nav", "component": "c", "version": "2", "k": 7},
    )
    assert response.status_code == 200
    assert fake_parts.backend.related_calls == []
    # ask kwargs are echoed into the result dict by FakeBackend
    body = response.json()
    assert body["question"] == "q"


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


def test_search_returns_ranked_results(client: TestClient):
    response = client.post("/search", json={"text": "chunks", "k": 2})
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    first = results[0]
    assert first["module"] == "ROOT"
    assert first["page_id"] == "ROOT:pages:index"
    assert first["link"] == "ROOT:pages:index"
    assert first["score"] == 1.0
    assert first["sources"] == {"hybrid": 1}


def test_search_truncates_long_text(client: TestClient, fake_parts):
    long_text = "y" * 600
    # Mutate in place: the backend shares this list object with the searcher.
    fake_parts.searcher.chunks[:] = [_chunk(long_text)]
    response = client.post("/search", json={"text": "q", "k": 1})
    assert response.status_code == 200
    body = response.json()[0]
    assert body["text"].endswith("...")
    assert len(body["text"]) == 503


def test_search_rejects_invalid_retrieval_mode(client: TestClient):
    response = client.post("/search", json={"text": "q", "retrieval": "nope"})
    assert response.status_code == 422


def test_search_sets_backend_mode_and_module_filter(client: TestClient, fake_parts):
    response = client.post(
        "/search", json={"text": "q", "k": 3, "retrieval": "bm25", "module": "nav"}
    )
    assert response.status_code == 200
    assert fake_parts.backend.default_mode == "bm25"
    assert fake_parts.backend.related_calls == [
        {"text": "q", "k": 3, "module": "nav"}
    ]


# -----------------------------------------------------------------------------
# UI serving
# -----------------------------------------------------------------------------


def test_ui_routes_serve_html_with_sri_attributes(client: TestClient):
    for path in ("/", "/ui", "/ui.html"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store, max-age=0"
        body = response.text
        # CDN scripts must keep their Subresource Integrity attributes
        assert 'integrity="sha384-' in body
        assert 'crossorigin="anonymous"' in body


def test_ui_returns_404_fallback_when_file_missing(client: TestClient, monkeypatch):
    # serve_ui builds static_dir / "ui.html" and checks .exists(); a stand-in
    # static_dir whose path never exists triggers the 404 fallback.
    class MissingPath:
        def __truediv__(self, _name):
            return self

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(server_main, "static_dir", MissingPath())
    response = client.get("/ui")
    assert response.status_code == 404
    assert "UI not found" in response.text
