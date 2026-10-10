"""Data models for the DocOracle system.

Uses Pydantic BaseModel for type safety, validation, and built-in serialization.
"""

import urllib.parse
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text from an AsciiDoc document.

    Provides built-in JSON serialization, field validation, and type safety.
    """

    text: str = Field(..., description="The text content of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    module: str = Field(..., description="Antora module name")
    component: str = Field(..., description="Component name from antora.yml")
    version: str = Field(..., description="Component version")
    page_id: str = Field(..., description="Antora page ID (e.g., module:pages:page)")

    section_id: str | None = Field(None, description="AsciiDoc section ID")
    section_title: str | None = Field(None, description="Human-readable section title")
    page_role: str | None = Field(None, description="Antora page role")
    source_file: str | None = Field(None, description="Path to source AsciiDoc file")
    hierarchy: list[str] = Field(default_factory=list, description="Document hierarchy path")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(**data)

    def get_link(self) -> str:
        """Generate link to source document section."""
        if self.section_id:
            return f"{self.page_id}#{self.section_id}"
        return self.page_id

    @property
    def breadcrumb(self) -> str | None:
        """Human-readable breadcrumb for UI display.

        Mirrors the breadcrumb the chunker prepends to ``text``: component
        display title (falling back to the identifier), module (omitted for
        ``ROOT``), document title, then section titles. Consecutive duplicates
        are collapsed.
        """
        parts: list[str] = []
        label = self.metadata.get("component_title") or self.component
        if label:
            parts.append(label)
        if self.module and self.module not in ("ROOT", label):
            parts.append(self.module)
        document_title = self.metadata.get("document_title")
        if document_title:
            parts.append(document_title)
        section_path_titles: list[str] = self.metadata.get("section_path_titles") or []
        for t in section_path_titles:
            if t:
                parts.append(t)
        if self.section_title:
            parts.append(self.section_title)

        deduped: list[str] = []
        for p in parts:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        return " > ".join(deduped) or None

    def site_url(self, base_url: str | None) -> str | None:
        """Build the Antora-style URL for this chunk's page on the published site.

        URL shape: ``{base}/{component}/{version}/[{module}/]{page}.html``. The
        version segment is always included (Antora keeps it even for
        ``latest``). The ``ROOT`` module is omitted per Antora's default
        URL strategy. Returns ``None`` if no base URL is configured or the
        page path can't be derived.
        """
        if not base_url:
            return None
        page_path = self._page_path_from_source_file()
        if not page_path:
            return None

        segments = [
            base_url.rstrip("/"),
            urllib.parse.quote(self.component, safe=""),
            urllib.parse.quote(self.version or "latest", safe=""),
        ]
        if self.module and self.module != "ROOT":
            segments.append(urllib.parse.quote(self.module, safe=""))

        # Preserve subdirectory slashes within the page path.
        encoded_page = "/".join(urllib.parse.quote(part, safe="") for part in page_path.split("/"))
        segments.append(encoded_page + ".html")

        url = "/".join(segments)
        if self.section_id:
            url += "#" + urllib.parse.quote(self.section_id, safe="")
        return url

    def _page_path_from_source_file(self) -> str | None:
        """Extract the page's relative path (without ``.adoc``) from ``source_file``."""
        if not self.source_file:
            return None
        parts = self.source_file.split("/pages/", 1)
        if len(parts) != 2:
            return None
        page = parts[1]
        if page.endswith(".adoc"):
            page = page[:-5]
        return page or None


class Document(BaseModel):
    """A document from Antora, normalized to Markdown."""

    content: str = Field(..., description="Markdown content (converted from AsciiDoc if needed)")
    file_path: str = Field(..., description="Path to the source file")
    module: str = Field(..., description="Antora module name")
    component: str = Field(..., description="Component name (from antora.yml name)")
    version: str = Field(..., description="Component version")
    page_id: str = Field(..., description="Antora page ID")

    title: str | None = Field(None, description="Document title")
    # Human display label for the component (from antora.yml ``title``). The
    # ``component`` field above is the stable identifier (from ``name``); this
    # is what we surface in breadcrumbs.
    component_title: str | None = Field(None, description="Component display title")
    page_role: str | None = Field(None, description="Antora page role")
    sections: list[dict[str, Any]] = Field(default_factory=list, description="Parsed sections")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    hierarchy: list[str] = Field(default_factory=list, description="Document hierarchy path")

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary for storage."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(**data)

    def to_chunks(self, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
        """Split document into chunks. Delegates to Chunker."""
        from .chunker import Chunker

        chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
        return chunker.chunk_document(self)
