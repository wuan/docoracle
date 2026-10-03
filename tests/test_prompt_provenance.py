"""Test that the LLM prompt excludes retrieval provenance."""

import uuid

from docoracle.backends.engine import QAEngine
from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.models import Chunk


class _FakeLLMClient:
    """Captures the structured prompt passed to .ask() without contacting any API."""

    def __init__(self) -> None:
        self.last_question = None
        self.last_context = None

    def embed(self, texts):
        # Return zero vectors regardless of input.
        return [[0.0] * 4 for _ in texts]

    def ask(
        self,
        question,
        context,
        output_model=None,
        model=None,
        temperature=None,
        max_tokens=None,
        system=None,
    ):
        from docoracle.core.llm_outputs import AnswerResponse

        self.last_question = question
        self.last_context = context
        model_cls = output_model or AnswerResponse
        return model_cls(answer="fake answer")


def test_prompt_does_not_include_provenance_or_score():
    chunks = [
        Chunk(
            text="alpha bravo charlie",
            chunk_id=str(uuid.uuid4()),
            module="m",
            component="c",
            version="v",
            page_id="p",
        ),
        Chunk(
            text="delta echo foxtrot",
            chunk_id=str(uuid.uuid4()),
            module="m",
            component="c",
            version="v",
            page_id="p",
        ),
    ]
    from docoracle.core.bm25_index import BM25Index
    from docoracle.core.semantic_index import SemanticIndex

    hs = HybridSearcher(semantic=SemanticIndex(), bm25=BM25Index())
    # BM25 only is enough to exercise the prompt-building path without
    # calling embed().
    hs.add_chunks(chunks)
    fake = _FakeLLMClient()
    engine = QAEngine(hs, fake, config_path="/nonexistent.yaml")

    engine.ask("anything", k=2, retrieval="bm25")
    # The captured context must not leak score/sources/provenance
    # to the LLM.
    ctx = fake.last_context
    assert ctx is not None
    for forbidden in ("sources", "score", "rrf", "provenance", "semantic=", "bm25="):
        assert forbidden not in ctx.lower(), (
            f"LLM prompt leaked provenance term: {forbidden!r} in {ctx!r}"
        )
    # The Source markers must be present.
    assert "[Source 1" in ctx
    assert "[Source 2" in ctx
    # The actual chunk text must be present.
    assert "alpha bravo charlie" in ctx
    assert "delta echo foxtrot" in ctx


def test_provenance_present_in_chunk_details():
    chunks = [
        Chunk(
            text="alpha bravo",
            chunk_id=str(uuid.uuid4()),
            module="m",
            component="c",
            version="v",
            page_id="p",
        ),
        Chunk(
            text="charlie delta",
            chunk_id=str(uuid.uuid4()),
            module="m",
            component="c",
            version="v",
            page_id="p",
        ),
    ]
    from docoracle.core.bm25_index import BM25Index
    from docoracle.core.semantic_index import SemanticIndex

    hs = HybridSearcher(semantic=SemanticIndex(), bm25=BM25Index())
    hs.add_chunks(chunks)
    fake = _FakeLLMClient()
    engine = QAEngine(hs, fake, config_path="/nonexistent.yaml")

    result = engine.ask("alpha", k=1, retrieval="bm25")
    assert result["chunk_details"][0]["sources"] == {"bm25": 1}
    assert result["retrieval_mode"] == "bm25"
