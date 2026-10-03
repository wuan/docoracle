"""Tests for the Antora-aware AsciiDoc loader."""

import shutil
import tempfile
from pathlib import Path

import pytest

from docoracle.core.loader import AntoraLoader
from docoracle.core.models import Document


@pytest.fixture
def temp_antora_project():
    """Create a temporary Antora project for testing."""
    tmpdir = tempfile.mkdtemp()

    # Create antora.yml
    antora_yml = Path(tmpdir) / "antora.yml"
    antora_yml.write_text("""name: test-component
version: 1.0.0
title: Test Component
""")

    # Create module structure
    modules_dir = Path(tmpdir) / "modules"
    modules_dir.mkdir()

    module_a_dir = modules_dir / "module-a"
    module_a_dir.mkdir()
    pages_dir = module_a_dir / "pages"
    pages_dir.mkdir()

    # Create a simple AsciiDoc page
    page1 = pages_dir / "page1.adoc"
    page1.write_text("""= Page 1
:page-role: concept
:page-description: A test page

This is the content of page 1.

== Section 1

Content for section 1.

== Section 2

Content for section 2.
""")

    # Create nav.adoc
    nav_file = module_a_dir / "nav.adoc"
    nav_file.write_text("""* xref:page1.adoc[Page 1]
""")

    yield tmpdir

    # Cleanup
    shutil.rmtree(tmpdir)


def test_load_component(temp_antora_project):
    """Test loading an Antora component."""
    loader = AntoraLoader(temp_antora_project)
    documents = loader.load_component()

    assert len(documents) == 1

    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.component == "test-component"
    assert doc.version == "1.0.0"
    assert doc.module == "module-a"
    assert doc.page_id == "module-a:pages:page1"  # Uses original filename with hyphens
    assert doc.title == "Page 1"
    assert doc.page_role == "concept"
    assert "Page 1" in doc.content


def test_extract_sections(temp_antora_project):
    """Test section extraction from AsciiDoc."""
    loader = AntoraLoader(temp_antora_project)
    loader._load_antora_config()

    page1 = Path(temp_antora_project) / "modules" / "module-a" / "pages" / "page1.adoc"
    doc = loader._load_document(page1, "module-a")

    assert doc is not None
    # The document title is the first section, with children for == sections
    assert len(doc.sections) >= 1

    # Collect all section titles including children
    all_titles = []

    def collect_titles(sections):
        for s in sections:
            all_titles.append(s["title"])
            if "children" in s:
                collect_titles(s["children"])

    collect_titles(doc.sections)

    assert "Page 1" in all_titles
    assert "Section 1" in all_titles
    assert "Section 2" in all_titles


def test_extract_metadata(temp_antora_project):
    """Test metadata extraction from AsciiDoc attributes."""
    loader = AntoraLoader(temp_antora_project)

    content = """:page-role: tutorial
:page-description: A tutorial page
:custom-attr: custom-value

= Title

Content
"""

    metadata = loader._extract_metadata(content)

    assert metadata["page-role"] == "tutorial"
    assert metadata["page-description"] == "A tutorial page"
    assert metadata["custom-attr"] == "custom-value"


def test_extract_title():
    """Test title extraction from AsciiDoc."""
    loader = AntoraLoader(".")

    content = "= My Document Title\n\nContent here"
    title = loader._extract_title(content)
    assert title == "My Document Title"

    content = "== Section Title\n\nContent"
    title = loader._extract_title(content)
    assert title == "Section Title"

    content = "No title here"
    title = loader._extract_title(content)
    assert title is None
