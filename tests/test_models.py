"""Tests for data models."""

from docoracle.core.models import Chunk, Document


def test_chunk_creation():
    """Test Chunk creation and serialization."""
    chunk = Chunk(
        text="Sample text",
        chunk_id="test-123",
        module="test-module",
        component="test-component",
        version="1.0.0",
        page_id="test:pages:test",
        section_id="section-1",
        section_title="Test Section",
        page_role="concept",
        source_file="modules/test/pages/test.adoc",
        hierarchy=["Getting Started", "Installation"],
        metadata={"custom": "value"},
    )

    assert chunk.text == "Sample text"
    assert chunk.chunk_id == "test-123"
    assert chunk.module == "test-module"
    assert chunk.get_link() == "test:pages:test#section-1"


def test_chunk_without_section():
    """Test Chunk without section ID."""
    chunk = Chunk(
        text="Sample text",
        chunk_id="test-123",
        module="test-module",
        component="test-component",
        version="1.0.0",
        page_id="test:pages:test",
        section_id=None,
        section_title=None,
    )

    assert chunk.get_link() == "test:pages:test"


def test_chunk_serialization():
    """Test Chunk to_dict and from_dict."""
    original = Chunk(
        text="Sample text",
        chunk_id="test-123",
        module="test-module",
        component="test-component",
        version="1.0.0",
        page_id="test:pages:test",
        section_id="section-1",
        section_title="Test Section",
        page_role="concept",
        source_file="modules/test/pages/test.adoc",
        hierarchy=["Getting Started", "Installation"],
        metadata={"custom": "value"},
    )

    data = original.to_dict()
    restored = Chunk.from_dict(data)

    assert restored.text == original.text
    assert restored.chunk_id == original.chunk_id
    assert restored.module == original.module
    assert restored.component == original.component
    assert restored.version == original.version
    assert restored.page_id == original.page_id
    assert restored.section_id == original.section_id
    assert restored.section_title == original.section_title
    assert restored.page_role == original.page_role
    assert restored.source_file == original.source_file
    assert restored.hierarchy == original.hierarchy
    assert restored.metadata == original.metadata


def test_document_creation():
    """Test Document creation."""
    doc = Document(
        content="Sample content",
        file_path="modules/test/pages/test.adoc",
        module="test-module",
        component="test-component",
        version="1.0.0",
        page_id="test:pages:test",
        title="Test Document",
        page_role="concept",
        sections=[
            {"title": "Section 1", "id": "section-1", "level": 2},
        ],
        metadata={"custom": "value"},
    )

    assert doc.content == "Sample content"
    assert doc.file_path == "modules/test/pages/test.adoc"
    assert doc.module == "test-module"
    assert doc.component == "test-component"
    assert doc.version == "1.0.0"
    assert doc.page_id == "test:pages:test"
    assert doc.title == "Test Document"
    assert doc.page_role == "concept"
    assert len(doc.sections) == 1
    assert doc.metadata["custom"] == "value"


def _chunk(**overrides) -> Chunk:
    defaults = {
        "text": "t",
        "chunk_id": "c",
        "module": "ROOT",
        "component": "finanzen",
        "version": "latest",
        "page_id": "ROOT:pages:Geldanlage",
        "section_id": None,
        "section_title": None,
        "source_file": "modules/ROOT/pages/Geldanlage.adoc",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def test_site_url_default_strategy():
    """Version is always included; ROOT module is elided."""
    url = _chunk().site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/finanzen/latest/Geldanlage.html"


def test_site_url_includes_non_root_module():
    # Mirrors the real-site URL the user supplied:
    # https://doc.tryb.de/allgemein/latest/Garten/Pflanzgeschichte.html
    url = _chunk(
        component="allgemein",
        module="Garten",
        source_file="modules/Garten/pages/Pflanzgeschichte.adoc",
    ).site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/allgemein/latest/Garten/Pflanzgeschichte.html"


def test_site_url_includes_non_latest_version():
    url = _chunk(version="2.1").site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/finanzen/2.1/Geldanlage.html"


def test_site_url_encodes_spaces_in_page_name():
    url = _chunk(
        module="Steuer",
        source_file="modules/Steuer/pages/dauerhafte Spenden.adoc",
    ).site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/finanzen/latest/Steuer/dauerhafte%20Spenden.html"


def test_site_url_preserves_subdirectories():
    url = _chunk(
        module="Steuer",
        source_file="modules/Steuer/pages/Jahre/Steuer 2024.adoc",
    ).site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/finanzen/latest/Steuer/Jahre/Steuer%202024.html"


def test_site_url_appends_section_anchor():
    url = _chunk(section_id="konten").site_url("https://doc.tryb.de")
    assert url == "https://doc.tryb.de/finanzen/latest/Geldanlage.html#konten"


def test_site_url_returns_none_without_base():
    assert _chunk().site_url(None) is None
    assert _chunk().site_url("") is None


def test_site_url_returns_none_without_source_file():
    assert _chunk(source_file=None).site_url("https://doc.tryb.de") is None


def test_site_url_returns_none_when_path_lacks_pages_segment():
    # Some loaders set source_file to just a filename; nothing safe to derive.
    assert _chunk(source_file="Geldanlage.adoc").site_url("https://doc.tryb.de") is None


def test_site_url_strips_trailing_slash_from_base():
    url = _chunk().site_url("https://doc.tryb.de/")
    assert url == "https://doc.tryb.de/finanzen/latest/Geldanlage.html"
