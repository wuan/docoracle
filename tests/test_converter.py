"""Tests for the AsciiDoc to Markdown converter."""

from docoracle.core.converter import convert_asciidoc_to_markdown


class TestAsciiDocToMarkdown:
    """Tests for AsciiDoc to Markdown conversion."""

    def test_convert_empty_content(self):
        """Test conversion of empty content."""
        result = convert_asciidoc_to_markdown("")
        assert result == ""

    def test_convert_heading_level_1(self):
        """Test conversion of level 1 heading."""
        content = "= Title"
        result = convert_asciidoc_to_markdown(content)
        assert "# Title" in result

    def test_convert_heading_level_2(self):
        """Test conversion of level 2 heading."""
        content = "== Section Title"
        result = convert_asciidoc_to_markdown(content)
        assert "## Section Title" in result

    def test_convert_heading_level_3(self):
        """Test conversion of level 3 heading."""
        content = "=== Subsection"
        result = convert_asciidoc_to_markdown(content)
        assert "### Subsection" in result

    def test_convert_multiple_headings(self):
        """Test conversion of multiple headings."""
        content = """= Title

== Section 1

Some content.

== Section 2"""
        result = convert_asciidoc_to_markdown(content)
        assert "# Title" in result
        assert "## Section 1" in result
        assert "## Section 2" in result
        assert "Some content" in result

    def test_convert_bold_text(self):
        """Test conversion of bold text."""
        content = "This is ***bold*** text"
        result = convert_asciidoc_to_markdown(content)
        assert "**bold**" in result

    def test_convert_code_block(self):
        """Test conversion of code block."""
        content = '[source,python]\n----\nprint("hello")\n----'
        result = convert_asciidoc_to_markdown(content)
        assert "```python" in result
        assert 'print("hello")' in result
        assert "```" in result

    def test_convert_link(self):
        """Test conversion of link."""
        content = "link:https://example.com[Example]"
        result = convert_asciidoc_to_markdown(content)
        assert "[Example](https://example.com)" in result

    def test_convert_inline_link(self):
        """Test conversion of inline link."""
        content = "https://example.com[Example]"
        result = convert_asciidoc_to_markdown(content)
        assert "[Example](https://example.com)" in result

    def test_convert_attribute(self):
        """Test that attributes are removed."""
        content = """:author: John Doe
:version: 1.0

= Title"""
        result = convert_asciidoc_to_markdown(content)
        assert "author" not in result
        assert "version" not in result
        assert "Title" in result

    def test_convert_comment(self):
        """Test that comments are removed."""
        content = """// This is a comment
= Title"""
        result = convert_asciidoc_to_markdown(content)
        assert "This is a comment" not in result
        assert "Title" in result

    def test_convert_list(self):
        """Test that lists are preserved (syntax is compatible)."""
        content = """* Item 1
* Item 2
* Item 3"""
        result = convert_asciidoc_to_markdown(content)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Item 3" in result

    def test_convert_ordered_list(self):
        """Test that ordered lists are preserved."""
        content = """1. First
2. Second
3. Third"""
        result = convert_asciidoc_to_markdown(content)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_convert_block_quote(self):
        """Test conversion of block quotes."""
        content = "____\nQuoted text\n____"
        result = convert_asciidoc_to_markdown(content)
        assert "> Quoted text" in result

    def test_convert_complex_document(self):
        """Test conversion of a complex document."""
        content = """= Document Title
:author: John Doe

This is an introduction.

== First Section

This is the first section content.

=== Subsection

Subsection content.

== Second Section

More content here.

[source,python]
----
def hello():
    print("world")
----

That was code."""
        result = convert_asciidoc_to_markdown(content)

        # Check structure
        assert "# Document Title" in result
        assert "## First Section" in result
        assert "### Subsection" in result
        assert "## Second Section" in result

        # Check content
        assert "This is an introduction" in result
        assert "This is the first section content" in result
        assert "Subsection content" in result
        assert "More content here" in result
        assert "That was code" in result

        # Check code block
        assert "```python" in result
        assert "def hello():" in result

    def test_heading_with_explicit_id(self):
        """Test conversion of heading with explicit ID."""
        content = "== Section Title [[section-id]]"
        result = convert_asciidoc_to_markdown(content)
        assert "## Section Title" in result
        assert "{#section-id}" in result
