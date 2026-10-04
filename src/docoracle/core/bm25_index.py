"""BM25 lexical index.

Tokenizes text with a German-tuned pipeline (lowercase → word-split →
stopword filter → minimum-length filter → Snowball stem) and ranks
chunks with ``rank_bm25.BM25Okapi``.

Like ``SemanticIndex``, this class does not own the canonical chunks
list. It holds only ``chunk_id → tokenized_text`` so the
``HybridSearcher`` can resolve results back to chunks it owns.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-untyped]
import Stemmer  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from .models import Chunk
from .search_index import ScoredChunk, matches_filters

# Common German (and a few English) stopwords. Kept inline so the package
# has no extra data file. Covers determiners, common prepositions,
# auxiliary verbs, and a small set of English function words that
# frequently appear in technical documentation.
GERMAN_STOPWORDS: set[str] = {
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "and",
    "ander",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "anderm",
    "andern",
    "anderr",
    "anders",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "das",
    "dasselbe",
    "dazu",
    "dass",
    "dein",
    "deine",
    "deinem",
    "deinen",
    "deiner",
    "deines",
    "dem",
    "demselben",
    "den",
    "denn",
    "denselben",
    "der",
    "derer",
    "desselben",
    "dessen",
    "dich",
    "die",
    "dies",
    "diese",
    "dieselbe",
    "dieselben",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "dir",
    "doch",
    "dort",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "einig",
    "einige",
    "einigem",
    "einigen",
    "einiger",
    "einiges",
    "einmal",
    "er",
    "es",
    "etwas",
    "euch",
    "euer",
    "eure",
    "eurem",
    "euren",
    "eurer",
    "eures",
    "for",
    "from",
    "für",
    "gegen",
    "gewesen",
    "hab",
    "habe",
    "haben",
    "hat",
    "hatte",
    "hatten",
    "hier",
    "hin",
    "hinter",
    "ich",
    "ihm",
    "ihn",
    "ihnen",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "ihres",
    "im",
    "in",
    "indem",
    "ins",
    "ist",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jene",
    "jenem",
    "jenen",
    "jener",
    "jenes",
    "jetzt",
    "kann",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "keines",
    "können",
    "könnte",
    "machen",
    "man",
    "manche",
    "manchem",
    "manchen",
    "mancher",
    "manches",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "meiner",
    "meines",
    "mich",
    "mir",
    "mit",
    "muss",
    "musste",
    "nach",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "ohne",
    "sehr",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "seines",
    "selbst",
    "sich",
    "sie",
    "sind",
    "so",
    "solche",
    "solchem",
    "solchen",
    "solcher",
    "solches",
    "soll",
    "sollte",
    "sondern",
    "the",
    "über",
    "und",
    "uns",
    "unse",
    "unsem",
    "unsen",
    "unser",
    "unses",
    "unter",
    "von",
    "vor",
    "war",
    "waren",
    "warst",
    "wann",
    "warum",
    "was",
    "weg",
    "weil",
    "weiter",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "welches",
    "wenn",
    "wer",
    "werde",
    "werden",
    "weshalb",
    "wieso",
    "wie",
    "wieder",
    "will",
    "wir",
    "wird",
    "wirst",
    "wo",
    "wollen",
    "wollte",
    "würde",
    "würden",
    "zu",
    "zum",
    "zur",
    "zwar",
    "zwischen",
}


_TOKEN_RE = re.compile(r"\w+")


class BM25Index:
    """BM25-backed lexical index.

    Tokenization is German-aware: lowercase → word-split → stopword
    removal → length filter → Snowball stemming (via ``PyStemmer``).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stemming: str = "german",
        stopwords: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.stemming = stemming
        self.stopwords = stopwords
        self.min_token_length = min_token_length

        self._bm25: Any = None
        self._chunk_ids: list[str] = []
        # We retain the tokenized corpus so ``add()`` can rebuild the BM25
        # object across multiple ingest passes (``rank_bm25`` is not
        # natively appendable).
        self._tokenized_corpus: list[list[str]] = []

        self._stemmer: Any = None
        if stemming and stemming.lower() != "none":
            self._stemmer = Stemmer.Stemmer(stemming.lower())  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Apply the full German-aware tokenization pipeline to ``text``."""
        if not text:
            return []
        text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        if self.stopwords:
            tokens = [t for t in tokens if t not in GERMAN_STOPWORDS]
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self._stemmer is not None:
            tokens = [self._stemmer.stemWord(t) for t in tokens]  # type: ignore[attr-defined]
        return tokens

    # ------------------------------------------------------------------
    # SearchIndex interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        """Tokenize and index ``chunks``. May be called repeatedly to append."""
        if not chunks:
            return
        tokenized = [self.tokenize(c.text) for c in chunks]
        self._tokenized_corpus.extend(tokenized)
        self._chunk_ids.extend(c.chunk_id for c in chunks)
        # Rebuild BM25 over the full accumulated corpus. Cheap at our scale.
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)  # type: ignore[arg-type]

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        chunk_resolver: Any | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ranked by BM25 score for ``query``."""
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)  # type: ignore[attr-defined]

        # Sort all scores descending and walk through them, applying
        # filters and stopping once we have ``k`` results. The pre-filter
        # sweep is broader than ``k`` because filters can cull candidates.
        order = np.argsort(-scores)  # type: ignore[arg-type]

        results: list[ScoredChunk] = []
        for idx in order:  # type: ignore[assignment]
            if idx >= len(self._chunk_ids):
                continue
            chunk_id = self._chunk_ids[idx]  # type: ignore[index]
            chunk: Chunk | None = None
            if chunk_resolver is not None:
                chunk = chunk_resolver(chunk_id)
            if chunk is None:
                continue
            if filters and not matches_filters(chunk, filters):
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(scores[idx]),  # type: ignore[index]
                    sources={"bm25": len(results) + 1},
                )
            )
            if len(results) >= k:
                break

        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._bm25 is None:
            raise ValueError("No BM25 index to save. Call add() first.")
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "chunk_ids": self._chunk_ids,
                    "tokenized_corpus": self._tokenized_corpus,
                    "params": {
                        "k1": self.k1,
                        "b": self.b,
                        "stemming": self.stemming,
                        "stopwords": self.stopwords,
                        "min_token_length": self.min_token_length,
                    },
                },
                f,
            )

    def load(self, path: Path) -> bool:
        path = Path(path)
        bm25_file = path / "bm25.pkl"
        if not bm25_file.exists():
            return False
        with open(bm25_file, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data.get("tokenized_corpus", [])
        params = data.get("params", {})
        self.k1 = params.get("k1", self.k1)
        self.b = params.get("b", self.b)
        self.stemming = params.get("stemming", self.stemming)
        self.stopwords = params.get("stopwords", self.stopwords)
        self.min_token_length = params.get("min_token_length", self.min_token_length)
        # Recreate the stemmer with whatever was persisted.
        self._stemmer = None
        if self.stemming and self.stemming.lower() != "none":
            self._stemmer = Stemmer.Stemmer(self.stemming.lower())  # type: ignore[attr-defined]
        return True

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunk_ids)
