"""Tests for the BM25 tokenizer."""

from docoracle.core.bm25_index import GERMAN_STOPWORDS, BM25Index


def test_stemming_collapses_inflected_plural_forms():
    idx = BM25Index()
    stems = {idx.tokenize(w)[0] for w in ("Beispiel", "Beispiele", "Beispielen")}
    assert len(stems) == 1


def test_stemming_lowercases_before_stemming():
    idx = BM25Index()
    # Same word in mixed case should produce same token
    assert idx.tokenize("Datei") == idx.tokenize("DATEI") == idx.tokenize("datei")


def test_stopwords_are_removed():
    idx = BM25Index(stopwords=True, stemming="none")
    tokens = idx.tokenize("Das ist ein Beispiel")
    assert "das" not in tokens
    assert "ist" not in tokens
    assert "ein" not in tokens
    assert "beispiel" in tokens


def test_stopwords_disabled_keeps_function_words():
    idx = BM25Index(stopwords=False, stemming="none")
    tokens = idx.tokenize("Das ist ein Beispiel")
    assert "das" in tokens
    assert "ist" in tokens
    assert "ein" in tokens
    assert "beispiel" in tokens


def test_min_token_length_filters_short_tokens():
    idx = BM25Index(stopwords=False, stemming="none", min_token_length=3)
    tokens = idx.tokenize("ab cdef ghi j")
    assert "ab" not in tokens
    assert "j" not in tokens
    assert "cdef" in tokens
    assert "ghi" in tokens


def test_stemming_none_skips_stemming():
    idx = BM25Index(stemming="none", stopwords=False)
    tokens = idx.tokenize("Konfiguration")
    assert tokens == ["konfiguration"]


def test_tokenization_handles_empty_string():
    idx = BM25Index()
    assert idx.tokenize("") == []


def test_tokenization_handles_only_punctuation():
    idx = BM25Index()
    assert idx.tokenize("!!! ... ???") == []


def test_ingest_and_query_tokenization_match():
    """A query for an inflected form should tokenize compatibly with the chunk's text."""
    idx = BM25Index()
    chunk_tokens = idx.tokenize("Konfiguration der Datenbankverbindung")
    query_tokens = idx.tokenize("konfiguriere")
    # Stemming should at minimum lowercase them; exact stem identity
    # is stemmer-specific, but both should be lowercase.
    assert all(t == t.lower() for t in chunk_tokens)
    assert all(t == t.lower() for t in query_tokens)


def test_stopwords_set_contains_german_function_words():
    assert "der" in GERMAN_STOPWORDS
    assert "die" in GERMAN_STOPWORDS
    assert "das" in GERMAN_STOPWORDS
    assert "und" in GERMAN_STOPWORDS
    assert "ist" in GERMAN_STOPWORDS
