"""Tests for the text chunker."""

import pytest

from docoracle.core.chunker import Chunker
from docoracle.core.converter import convert_asciidoc_to_markdown
from docoracle.core.models import Chunk, Document


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        # The loader converts AsciiDoc to Markdown before chunking.
        content=convert_asciidoc_to_markdown("""= Sample Document

This is the introduction paragraph.

== First Section

This is the content of the first section. It has multiple sentences.

== Second Section

This is the second section with more content.
"""),
        file_path="modules/sample/pages/sample.adoc",
        module="sample",
        component="test-component",
        version="1.0.0",
        page_id="sample:pages:sample",
        title="Sample Document",
        page_role="concept",
        sections=[
            {"title": "Sample Document", "id": "sample-document", "level": 1},
            {"title": "First Section", "id": "first-section", "level": 2},
            {"title": "Second Section", "id": "second-section", "level": 2},
        ],
        hierarchy=["Getting Started"],
    )


@pytest.fixture
def sample_markdown_document():
    """Create a sample Markdown document for testing."""
    return Document(
        content="""# Sample Document

This is the introduction paragraph.

## First Section

This is the content of the first section. It has multiple sentences.

## Second Section

This is the second section with more content.
""",
        file_path="modules/sample/pages/sample.md",
        module="sample",
        component="test-component",
        version="1.0.0",
        page_id="sample:pages:sample",
        title="Sample Document",
        page_role="concept",
        sections=[
            {"title": "Sample Document", "id": "sample-document", "level": 1},
            {"title": "First Section", "id": "first-section", "level": 2},
            {"title": "Second Section", "id": "second-section", "level": 2},
        ],
        hierarchy=["Getting Started"],
    )


def test_chunker_init():
    """Test chunker initialization."""
    chunker = Chunker(chunk_size=512, overlap=50)
    assert chunker.chunk_size == 512
    assert chunker.overlap == 50

    chunker = Chunker()
    assert chunker.chunk_size == 512  # default
    assert chunker.overlap == 50  # default


def test_chunk_small_document(sample_document):
    """Test chunking a small document (single chunk)."""
    chunker = Chunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.module == "sample"
        assert chunk.component == "test-component"
        assert chunk.version == "1.0.0"
        assert chunk.page_id == "sample:pages:sample"
        assert chunk.text  # Should have text


def test_chunk_small_markdown_document(sample_markdown_document):
    """Test chunking a small Markdown document."""
    chunker = Chunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk_document(sample_markdown_document)

    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.module == "sample"
        assert chunk.component == "test-component"
        assert chunk.version == "1.0.0"
        assert chunk.page_id == "sample:pages:sample"
        assert chunk.text  # Should have text


def test_chunk_large_text():
    """Test chunking plain text."""
    chunker = Chunker(chunk_size=20, overlap=5)

    long_text = "word " * 50  # 100 words

    chunks = chunker.chunk_text(
        text=long_text,
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
    )

    # With 20 word chunk size and 5 word overlap
    # Expected: ceil((100 - 5) / (20 - 5)) + 1 = ceil(95/15) + 1 = 7 + 1 = 8 chunks
    assert len(chunks) > 1

    # Verify each chunk has text
    for chunk in chunks:
        assert chunk.text
        assert "word" in chunk.text


def test_chunk_metadata_preservation(sample_document):
    """Test that metadata is preserved in chunks."""
    chunker = Chunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk_document(sample_document)

    for chunk in chunks:
        assert chunk.module == sample_document.module
        assert chunk.component == sample_document.component
        assert chunk.version == sample_document.version
        assert chunk.page_id == sample_document.page_id
        assert chunk.source_file == sample_document.file_path
        assert chunk.page_role == sample_document.page_role


def test_clean_markdown():
    """Test Markdown cleanup for chunking."""
    chunker = Chunker()

    content = """# Title

This is **bold** text with *italic* words.

`code example`

```python
print("hello")
```

## Section

More content.
"""

    clean = chunker._clean_markdown(content)

    # Should remove markup
    assert "**" not in clean
    assert "*" not in clean or clean.count("*") == 0  # All asterisks removed
    assert "`" not in clean
    assert "[source" not in clean

    # Should preserve text content INCLUDING code block bodies and inline code
    assert "Title" in clean
    assert "bold" in clean
    assert "italic" in clean
    assert "code example" in clean  # inline code text preserved
    assert 'print("hello")' in clean  # code block body preserved
    assert "Section" in clean
    assert "More content" in clean


def test_chunk_document_with_sections():
    """Test that section information is preserved in chunks."""
    chunker = Chunker(chunk_size=100, overlap=10)

    document = Document(
        content="""# Test Document

## Section 1

Content of section 1.

## Section 2

Content of section 2.""",
        file_path="test.adoc",
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
        title="Test Document",
        sections=[
            {"title": "Test Document", "id": "test-doc", "level": 1},
            {"title": "Section 1", "id": "section-1", "level": 2},
            {"title": "Section 2", "id": "section-2", "level": 2},
        ],
    )

    chunks = chunker.chunk_document(document)

    # Each section should produce at least one chunk
    assert len(chunks) >= 2

    # Check that section titles appear in chunks
    section_titles = [chunk.section_title for chunk in chunks]
    assert "Section 1" in section_titles
    assert "Section 2" in section_titles


def test_nested_sections_hierarchy():
    """Test that nested sections preserve hierarchy paths."""
    chunker = Chunker(chunk_size=100, overlap=10)

    document = Document(
        content="""# Parent Section

Parent content.

## Child Section

Child content.

### Grandchild Section

Grandchild content.""",
        file_path="test.adoc",
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
        title="Test Document",
        hierarchy=["Getting Started"],
        sections=[
            {
                "title": "Parent Section",
                "id": "parent",
                "level": 1,
                "children": [
                    {
                        "title": "Child Section",
                        "id": "child",
                        "level": 2,
                        "children": [
                            {"title": "Grandchild Section", "id": "grandchild", "level": 3},
                        ],
                    },
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)

    # Should have chunks for each section
    assert len(chunks) >= 3

    # Find grandchild chunk and verify hierarchy
    grandchild_chunks = [c for c in chunks if c.section_id == "grandchild"]
    assert len(grandchild_chunks) > 0

    grandchild = grandchild_chunks[0]
    assert grandchild.section_title == "Grandchild Section"
    # Hierarchy should include document hierarchy + section path
    assert "parent" in grandchild.hierarchy
    assert "child" in grandchild.hierarchy
    assert "grandchild" in grandchild.hierarchy
    # Check section_path_titles in metadata
    assert "Parent Section" in grandchild.metadata.get("section_path_titles", [])
    assert "Child Section" in grandchild.metadata.get("section_path_titles", [])


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    chunker = Chunker(chunk_size=10, overlap=3)

    # Create text that will be split into chunks with overlap
    long_text = "word " * 30  # 30 words

    chunks = chunker.chunk_text(
        text=long_text,
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
    )

    assert len(chunks) > 1

    # Check that consecutive chunks overlap
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]

        # Split into words for comparison
        words1 = chunk1.text.split()
        words2 = chunk2.text.split()

        # Last 'overlap' words of chunk1 should be first 'overlap' words of chunk2
        overlap_words = words1[-3:] if len(words1) >= 3 else words1
        if overlap_words and len(words2) >= 3:
            assert words2[: len(overlap_words)] == overlap_words


def test_empty_document():
    """Test chunking an empty document."""
    chunker = Chunker(chunk_size=100, overlap=10)

    document = Document(
        content="",
        file_path="test.adoc",
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
    )

    chunks = chunker.chunk_document(document)
    assert chunks == []


def test_no_sections_document():
    """Test chunking a document without parsed sections."""
    chunker = Chunker(chunk_size=100, overlap=10)

    document = Document(
        content="""# Simple Document

This is the entire content without any sub-sections.""",
        file_path="test.adoc",
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
        title="Simple Document",
        sections=[],  # No sections
    )

    chunks = chunker.chunk_document(document)
    assert len(chunks) >= 1

    # Should treat entire document as one section
    for chunk in chunks:
        assert chunk.text
        assert chunk.module == "test"


def test_chunk_size_boundary():
    """Test chunking at exact chunk_size boundaries."""
    chunker = Chunker(chunk_size=10, overlap=0)

    # Exactly 10 words - should produce 1 chunk
    exact_text = "word " * 10
    chunks = chunker.chunk_text(
        text=exact_text.strip(),
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
    )
    assert len(chunks) == 1

    # 11 words - should produce 2 chunks
    overflow_text = "word " * 11
    chunks = chunker.chunk_text(
        text=overflow_text.strip(),
        module="test",
        component="test",
        version="1.0",
        page_id="test:pages:test",
    )
    assert len(chunks) == 2


def test_chunk_text_with_metadata():
    """Test chunk_text with all optional parameters."""
    chunker = Chunker(chunk_size=20, overlap=5)

    long_text = "word " * 40
    chunks = chunker.chunk_text(
        text=long_text,
        module="test",
        component="test-comp",
        version="2.0",
        page_id="test:page:id",
        section_id="section-123",
        section_title="Test Section",
        page_role="concept",
        source_file="test/file.adoc",
        hierarchy=["Parent", "Child"],
    )

    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        assert chunk.module == "test"
        assert chunk.component == "test-comp"
        assert chunk.version == "2.0"
        assert chunk.page_id == "test:page:id"
        assert chunk.section_id == "section-123"
        assert chunk.section_title == "Test Section"
        assert chunk.page_role == "concept"
        assert chunk.source_file == "test/file.adoc"
        assert chunk.hierarchy == ["Parent", "Child"]

        # First chunk should have section title prepended
        if i == 0:
            assert "Test Section" in chunk.text

        # Check metadata
        assert chunk.metadata.get("chunk_index") == i
        assert chunk.metadata.get("total_chunks") == len(chunks)


def test_clean_markdown_comprehensive():
    """Test comprehensive Markdown cleanup."""
    chunker = Chunker()

    content = """# Heading 1

This has **bold** and *italic* and `code`.

> A blockquote

- List item 1
- List item 2

[Link text](https://example.com)

<!---->

| Column 1 | Column 2 |
|---------|---------|
| Cell 1  | Cell 2  |

<div>HTML div</div>

![Image alt](image.png)

***bold italic***
"""

    clean = chunker._clean_markdown(content)

    # Should remove all formatting markers
    assert "**" not in clean
    assert "*" not in clean
    assert "`" not in clean
    assert "<" not in clean
    # Pipe characters are not stripped — markdown tables (and our
    # flattened-table separator) survive into the chunk text.

    # Should preserve text content (inline code text now preserved, tables kept)
    assert "Heading 1" in clean
    assert "bold" in clean
    assert "italic" in clean
    assert "code" in clean  # inline code text preserved
    assert "blockquote" in clean.lower() or "Blockquote" in clean
    assert "List item 1" in clean
    assert "List item 2" in clean
    assert "Link text" in clean
    assert "Image alt" in clean or "[image]" in clean
    assert "HTML div" in clean or "div" in clean.lower()


def test_breadcrumb_prepended_to_chunks():
    """Every chunk should start with a 'component > module > ... > section' breadcrumb."""
    chunker = Chunker(chunk_size=500, overlap=10)

    document = Document(
        content=convert_asciidoc_to_markdown("""= Wohnungskredite

== Kredit Oetztaler Str. 16

Höhe 200.000,00 EUR, Zinssatz 3,380 Prozent."""),
        file_path="finanzen/modules/ROOT/pages/Wohnungskredite.adoc",
        module="ROOT",
        component="finanzen",
        version="1.0",
        page_id="ROOT:pages:Wohnungskredite",
        title="Wohnungskredite",
        sections=[
            {
                "title": "Wohnungskredite",
                "id": "wohnungskredite",
                "level": 1,
                "children": [
                    {
                        "title": "Kredit Oetztaler Str. 16",
                        "id": "kredit-oetztaler-str-16",
                        "level": 2,
                    },
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)
    assert chunks, "should produce at least one chunk"

    kredit_chunks = [c for c in chunks if c.section_id == "kredit-oetztaler-str-16"]
    assert kredit_chunks, "child section should produce a chunk"

    breadcrumb = "finanzen > ROOT > Wohnungskredite > Kredit Oetztaler Str. 16"
    assert kredit_chunks[0].text.startswith(breadcrumb), (
        f"chunk should start with breadcrumb, got: {kredit_chunks[0].text[:120]}"
    )
    # Section's own content also present
    assert "Zinssatz" in kredit_chunks[0].text


def test_breadcrumb_uses_component_title_when_set():
    """When ``component_title`` is set, the breadcrumb shows the human label."""
    chunker = Chunker(chunk_size=500, overlap=10)

    document = Document(
        content=convert_asciidoc_to_markdown("""= Wohnungskredite

== Kredit

Some text."""),
        file_path="finanzen/pages/Wohnungskredite.adoc",
        module="ROOT",
        component="finanzen",
        component_title="Finanzen",
        version="1.0",
        page_id="ROOT:pages:Wohnungskredite",
        title="Wohnungskredite",
        sections=[
            {
                "title": "Wohnungskredite",
                "id": "wk",
                "level": 1,
                "children": [
                    {"title": "Kredit", "id": "kredit", "level": 2},
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)
    kredit = [c for c in chunks if c.section_id == "kredit"][0]

    # Breadcrumb opens with the human title, not the lowercase identifier.
    assert kredit.text.startswith("Finanzen >")
    assert "finanzen >" not in kredit.text
    # The stable identifier remains on the chunk for filtering.
    assert kredit.component == "finanzen"


def test_breadcrumb_collapses_duplicate_doc_title():
    """When the top section title equals the doc title, breadcrumb shouldn't repeat it."""
    chunker = Chunker(chunk_size=500, overlap=10)

    document = Document(
        content=convert_asciidoc_to_markdown("""= Geldanlage

Intro for Geldanlage.

== Konten

Konten content."""),
        file_path="finanzen/pages/Geldanlage.adoc",
        module="ROOT",
        component="finanzen",
        version="1.0",
        page_id="ROOT:pages:Geldanlage",
        title="Geldanlage",
        sections=[
            {
                "title": "Geldanlage",
                "id": "geldanlage",
                "level": 1,
                "children": [
                    {"title": "Konten", "id": "konten", "level": 2},
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)
    konten = [c for c in chunks if c.section_id == "konten"][0]

    # 'Geldanlage' should appear once in the breadcrumb prefix, not twice.
    breadcrumb_line = konten.text.split("\n", 1)[0]
    assert breadcrumb_line.count("Geldanlage") == 1


def test_subsection_content_not_included_in_parent():
    """Parent section should own only its own paragraphs, not subsection text."""
    chunker = Chunker(chunk_size=500, overlap=10)

    document = Document(
        content=convert_asciidoc_to_markdown("""= Doc

== Parent

Parent's own paragraph here.

=== Child

CHILD-MARKER-TEXT belongs only to the child.

== Sibling

Sibling content."""),
        file_path="test.adoc",
        module="m",
        component="c",
        version="1.0",
        page_id="m:pages:test",
        title="Doc",
        sections=[
            {
                "title": "Doc",
                "id": "doc",
                "level": 1,
                "children": [
                    {
                        "title": "Parent",
                        "id": "parent",
                        "level": 2,
                        "children": [
                            {"title": "Child", "id": "child", "level": 3},
                        ],
                    },
                    {"title": "Sibling", "id": "sibling", "level": 2},
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)

    parent_chunks = [c for c in chunks if c.section_id == "parent"]
    child_chunks = [c for c in chunks if c.section_id == "child"]

    assert parent_chunks, "parent should have a chunk"
    assert child_chunks, "child should have a chunk"

    # The child's text must appear only in the child chunk, not also in parent.
    parent_text = " ".join(c.text for c in parent_chunks)
    child_text = " ".join(c.text for c in child_chunks)

    assert "CHILD-MARKER-TEXT" in child_text
    assert "CHILD-MARKER-TEXT" not in parent_text
    assert "Parent's own paragraph" in parent_text


def test_asciidoc_table_converted_to_flat_text():
    """AsciiDoc tables should arrive at the chunk text as readable 'Header: Value' lines."""
    chunker = Chunker(chunk_size=500, overlap=10)

    asciidoc_content = """= Konten

== SSKM

[cols=",,",]
|===
|Buchung |Betrag |Bemerkung
|von Ann-Christine |€ 500,00 |
|Mieteinnahmen |€ 1060,00 |
|Grundsteuer |€ –15,74 |€ 46,62 pro Quartal
|===
"""

    document = Document(
        # The loader converts AsciiDoc to Markdown before chunking.
        content=convert_asciidoc_to_markdown(asciidoc_content),
        file_path="finanzen/pages/Geldanlage.adoc",
        module="ROOT",
        component="finanzen",
        version="1.0",
        page_id="ROOT:pages:Geldanlage",
        title="Konten",
        sections=[
            {
                "title": "Konten",
                "id": "konten",
                "level": 1,
                "children": [
                    {"title": "SSKM", "id": "sskm", "level": 2},
                ],
            },
        ],
    )

    chunks = chunker.chunk_document(document)
    sskm_chunks = [c for c in chunks if c.section_id == "sskm"]
    assert sskm_chunks, "SSKM section should produce a chunk"

    text = sskm_chunks[0].text
    # Table cell data must reach the chunk.
    assert "Ann-Christine" in text
    assert "500,00" in text
    assert "Mieteinnahmen" in text
    assert "1060,00" in text
    assert "Grundsteuer" in text
    assert "46,62 pro Quartal" in text
    # Header pairing is visible in the flattened format.
    assert "Buchung: von Ann-Christine" in text or "Buchung:" in text


def test_asciidoc_table_preserves_word_endings():
    """The cell-modifier regex must not eat the trailing letter of a word."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    content = """[cols=",,",]
|===
|Name |Detail |Date
|Versicherung |Unfall |2025
|===
"""
    result = convert_asciidoc_to_markdown(content)
    # The 'l' at the end of 'Unfall' must survive even though 'l|' looks like
    # an AsciiDoc literal-cell modifier.
    assert "Unfall" in result


def test_asciidoc_table_with_line_continuation():
    """Cells joined via `+` continuation should be merged into one cell."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    content = """[cols=",,",]
|===
|Name |Value |Date
|die Bayerische +
0390418209 +
Andreas Würl |€ 41.507,30 |17.10.2025
|===
"""
    result = convert_asciidoc_to_markdown(content)
    assert "die Bayerische 0390418209 Andreas Würl" in result
    assert "41.507,30" in result
    assert "17.10.2025" in result


def test_chunks_respect_token_budget():
    """Chunks must stay within the configured token budget, breadcrumb included."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunker = Chunker(chunk_size=80, overlap=10)

    # A few paragraphs that together blow past 80 tokens.
    paragraphs = [
        "Paragraph " + str(i) + " has multiple sentences of moderate length. "
        "It exists to push the section past the chunk size so the splitter "
        "must actually do work."
        for i in range(8)
    ]
    body = "\n\n".join(paragraphs)

    document = Document(
        content=f"# Long\n\n{body}",
        file_path="test.md",
        module="m",
        component="c",
        version="1.0",
        page_id="m:pages:test",
        title="Long",
        sections=[{"title": "Long", "id": "long", "level": 1}],
    )

    chunks = chunker.chunk_document(document)
    assert len(chunks) > 1, "should produce multiple chunks"

    for c in chunks:
        assert len(enc.encode(c.text)) <= chunker.chunk_size, (
            f"chunk exceeds budget: {len(enc.encode(c.text))} > {chunker.chunk_size}"
        )


def test_chunks_split_on_paragraph_boundaries():
    """Paragraph-aware splitting should keep individual paragraphs intact."""
    chunker = Chunker(chunk_size=60, overlap=5)

    p1 = "First paragraph with several sentences. " * 3
    p2 = "Second paragraph with distinct content. " * 3
    p3 = "Third paragraph wraps things up nicely. " * 3
    body = "\n\n".join([p1.strip(), p2.strip(), p3.strip()])

    document = Document(
        content=f"# Doc\n\n{body}",
        file_path="test.md",
        module="m",
        component="c",
        version="1.0",
        page_id="m:pages:test",
        title="Doc",
        sections=[{"title": "Doc", "id": "doc", "level": 1}],
    )

    chunks = chunker.chunk_document(document)
    assert len(chunks) > 1

    # No chunk should split a paragraph in the middle — each paragraph's first
    # sentence should appear intact in exactly one chunk.
    full_text = "\n".join(c.text for c in chunks)
    for marker in [
        "First paragraph with several sentences",
        "Second paragraph with distinct content",
        "Third paragraph wraps things up nicely",
    ]:
        assert marker in full_text


def test_xref_converted_to_markdown_link():
    """xref: macros should turn into markdown links with the page label."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    cases = {
        "xref:Geldanlage.adoc[]": "[Geldanlage](Geldanlage)",
        "xref:ROOT:Vorsorge.adoc[Vorsorge planen]": "[Vorsorge planen](Vorsorge)",
        "xref:Jahre/Steuer 2024.adoc[]": "[Steuer 2024](Steuer 2024)",
    }
    for src, expected in cases.items():
        result = convert_asciidoc_to_markdown(src)
        assert expected in result, f"{src!r} -> {result!r} (expected {expected!r})"


def test_star_lists_convert_to_nested_markdown():
    """Star bullets convert to indented Markdown list items."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    content = """* Top item
** Nested item
*** Deeply nested item"""
    result = convert_asciidoc_to_markdown(content)
    assert "- Top item" in result
    assert "  - Nested item" in result
    assert "    - Deeply nested item" in result


def test_inline_role_syntax_stripped():
    """``[.role]#text#`` should reduce to just ``text``."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    content = "notarielle [.underline]#Beglaubigung# der Unterschrift"
    result = convert_asciidoc_to_markdown(content)
    assert "Beglaubigung" in result
    assert "[.underline]" not in result
    assert "#Beglaubigung#" not in result


def test_dot_lists_convert_to_numbered_markdown():
    """Dot-numbered lists convert to Markdown ordered lists, block titles untouched."""
    from docoracle.core.converter import convert_asciidoc_to_markdown

    content = """. First step
. Second step
.. Sub-step
.Block title not a list"""
    result = convert_asciidoc_to_markdown(content)
    assert "1. First step" in result
    assert "1. Second step" in result
    assert "  1. Sub-step" in result
    # Block titles (no space after the dot) must not be converted.
    assert ".Block title not a list" in result
