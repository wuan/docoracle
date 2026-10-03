"""Data models for the DocOracle system.

Uses Pydantic BaseModel for type safety, validation, and built-in serialization.
Maintains backward compatibility with the previous dataclass implementation.
"""

import urllib.parse
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text from an AsciiDoc document.

    Replaces the previous dataclass implementation with Pydantic BaseModel.
    Provides built-in JSON serialization, field validation, and type safety.

    The to_dict() and from_dict() methods are preserved for backward compatibility.
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

    # -------------------------------------------------------------------------
    # Backward compatibility methods
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for storage.

        Preserved for backward compatibility with existing serialization code.
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary.

        Preserved for backward compatibility with existing deserialization code.
        """
        return cls(**data)

    # -------------------------------------------------------------------------
    # Link generation methods (unchanged from original)
    # -------------------------------------------------------------------------

    def get_link(self) -> str:
        """Generate link to source document section."""
        if self.section_id:
            return f"{self.page_id}#{self.section_id}"
        return self.page_id

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
    """A document from Antora, normalized to Markdown.

    Replaces the previous dataclass implementation with Pydantic BaseModel.
    """

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
        """Convert document to dictionary for storage.

        Preserved for backward compatibility.
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary.

        Preserved for backward compatibility.
        """
        return cls(**data)

    def to_chunks(self, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
        """Split document into chunks. Delegates to Chunker."""
        from .chunker import Chunker

        chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
        return chunker.chunk_document(self)
