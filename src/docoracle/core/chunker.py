"""Text chunker for splitting Markdown documents into embeddable chunks.

This chunker respects the document structure by:
- Extracting only this section's own content (excluding subsections)
- Splitting oversized sections on paragraph, then sentence, then token-window
  boundaries so chunks remain semantically coherent
- Sizing chunks by real BPE tokens (via tiktoken) so the embedding model's
  budget is what's actually budgeted
- Prepending a breadcrumb header to every chunk so embeddings carry
  hierarchical context

Documents are expected to be Markdown already; AsciiDoc sources are
converted to Markdown by the loader.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .models import Chunk, Document

# tiktoken encoders are not cheap to construct; share one per encoding name.
_ENCODER_CACHE: dict[str, Any] = {}


def _get_encoder(name: str) -> Any:
    enc = _ENCODER_CACHE.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)  # type: ignore[attr-defined]
        _ENCODER_CACHE[name] = enc
    return enc


@dataclass
class Chunker:
    """Splits documents into chunks for embedding."""

    chunk_size: int = 512
    overlap: int = 50
    encoding_name: str = "cl100k_base"
    _encoder: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._encoder = _get_encoder(self.encoding_name)

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))  # type: ignore[attr-defined]

    def _take_last_tokens(self, text: str, n: int) -> str:
        if n <= 0 or not text:
            return ""
        ids = self._encoder.encode(text)  # type: ignore[attr-defined]
        if len(ids) <= n:
            return text
        return self._encoder.decode(ids[-n:])  # type: ignore[attr-defined]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks respecting section structure.

        Args:
            document: Document to chunk (content is expected to be Markdown)

        Returns:
            List of Chunk objects with metadata
        """
        chunks: list[Chunk] = []

        if not document.content:
            return chunks

        # Get sections with their actual content extracted from the document
        sections: list[dict[str, Any]] = self._extract_sections_from_document(
            document, document.content
        )

        # Create chunks from sections
        for section in sections:
            section_chunks = self._chunk_section(section, document, self.chunk_size, self.overlap)
            chunks.extend(section_chunks)  # type: ignore[arg-type]

        return chunks

    def _clean_markdown(self, content: str) -> str:
        """Clean Markdown for chunking while preserving structure and text.

        Drops pure markup noise but keeps the underlying text — including code
        block bodies and inline code — so the embedding sees what the reader
        sees.
        """
        # Strip fenced code-block delimiters but keep the code text.
        content = re.sub(
            r"```[ \t]*\w*[ \t]*\n(.*?)```",
            r"\1",
            content,
            flags=re.DOTALL,
        )
        # Inline code: keep the text inside the backticks.
        content = re.sub(r"`([^`]+)`", r"\1", content)

        # Remove HTML tables wholesale (rare in this corpus and noisy when present).
        content = re.sub(
            r"<table>.*?</table>",
            " [table] ",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Strip blockquote markers but keep the quoted text.
        content = re.sub(r"^[ \t]*>[ \t]?", "", content, flags=re.MULTILINE)

        # Inline emphasis markers — drop the markers, keep the text.
        content = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", content)
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
        content = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", content)
        content = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", content)

        # Heading markers — strip the leading hashes; we surface structure via
        # the breadcrumb instead.
        content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)

        # List markers — strip the bullets/numbers, keep the item text.
        content = re.sub(r"^\s*[-*+•·]\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s+", "", content, flags=re.MULTILINE)

        # Link syntax — keep the link text.
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        # HTML tags — drop entirely.
        content = re.sub(r"<[^>]+>", " ", content)

        # Image syntax — replace with placeholder so we know something was there.
        content = re.sub(r"!\[[^\]]*\]\([^)]+\)", " [image] ", content)

        # Normalize whitespace.
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"\n\s+\n", "\n\n", content)

        return content.strip()

    def _extract_sections_from_document(
        self, document: Document, markdown_content: str
    ) -> list[dict[str, Any]]:
        """Extract sections with each section's own content (no subsection text).

        Uses ``document.sections`` for hierarchy when available, otherwise falls
        back to treating the whole document as one section.
        """
        if not document.sections:
            cleaned = self._clean_markdown(markdown_content)
            return [
                {
                    "title": document.title or "Document",
                    "id": "document",
                    "level": 0,
                    "content": cleaned,
                    "section_path": [],
                    "section_path_titles": [],
                }
            ]

        sections_with_hierarchy: list[dict[str, Any]] = self._build_section_hierarchy(
            document.sections
        )

        sections_with_content: list[dict[str, Any]] = []
        for section_info in sections_with_hierarchy:
            content = self._extract_section_content(
                markdown_content,
                section_info["title"],
                section_info["level"],
            )
            cleaned = self._clean_markdown(content)

            sections_with_content.append(
                {
                    "title": section_info["title"],
                    "id": section_info["id"],
                    "level": section_info["level"],
                    "content": cleaned,
                    "section_path": section_info["path"],
                    "section_path_titles": section_info["path_titles"],
                }
            )

        return sections_with_content

    def _extract_section_content(
        self,
        markdown_content: str,
        section_title: str,
        section_level: int,
    ) -> str:
        """Extract content for a specific section, excluding its subsections.

        The end boundary is the next heading of *any* level — so each section
        owns only its own paragraphs, not its children's text.
        """
        heading_pattern = re.compile(
            r"^(#{1,6})\s*(.+?)\s*(?:\{.+?\})?\s*$",
            re.MULTILINE,
        )
        matches = list(heading_pattern.finditer(markdown_content))

        for i, match in enumerate(matches):
            match_level = len(match.group(1))
            match_title = match.group(2).strip()

            if match_title == section_title and match_level == section_level:
                # Start after this heading line so the heading text isn't
                # duplicated in the chunk (the breadcrumb already carries it).
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_content)
                return markdown_content[start_pos:end_pos]

        return ""

    def _build_section_hierarchy(
        self,
        sections: list[dict[str, Any]],
        parent_path: list[str] | None = None,
        parent_titles: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a flat list of sections with their full hierarchy paths."""
        if parent_path is None:
            parent_path = []
        if parent_titles is None:
            parent_titles = []

        result: list[dict[str, Any]] = []

        for section in sections:
            section_id = section.get("id", "")
            title = section.get("title", "")
            level = section.get("level", 0)
            children = section.get("children", [])

            result.append(
                {
                    "id": section_id,
                    "title": title,
                    "level": level,
                    "path": parent_path + [section_id],
                    "path_titles": parent_titles + [title],
                }
            )

            if children:
                result.extend(
                    self._build_section_hierarchy(
                        children,
                        parent_path + [section_id],
                        parent_titles + [title],
                    )
                )

        return result

    def _build_breadcrumb(
        self,
        document: Document,
        section_path_titles: list[str],
        section_title: str | None,
    ) -> str:
        """Build a 'Component > Module > Doc > ... > Section' breadcrumb.

        Consecutive duplicates (common when a doc's level-1 heading repeats as
        the document title) are collapsed.
        """
        parts: list[str] = []
        # Prefer the human-readable component title for breadcrumbs; fall back
        # to the identifier when no title is available.
        component_label = document.component_title or document.component
        if component_label:
            parts.append(component_label)
        if document.module and document.module != component_label:
            parts.append(document.module)
        if document.title:
            parts.append(document.title)
        for t in section_path_titles or []:
            if t:
                parts.append(t)
        if section_title:
            parts.append(section_title)

        deduped: list[str] = []
        for p in parts:
            if not deduped or deduped[-1] != p:
                deduped.append(p)

        return " > ".join(deduped)

    def _chunk_section(
        self,
        section: dict[str, Any],
        document: Document,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        """Split a section into chunks, each prefixed with a breadcrumb.

        Sizing is in BPE tokens. Sections that exceed ``chunk_size`` are split
        on paragraph boundaries first, then on sentence boundaries within an
        oversized paragraph, then as a last resort on a token window.
        """
        chunks: list[Chunk] = []
        content = section.get("content", "")
        if not content:
            return chunks

        breadcrumb = self._build_breadcrumb(
            document,
            section.get("section_path_titles", []),
            section.get("title"),
        )
        # Reserve headroom for the breadcrumb so the total chunk doesn't exceed
        # ``chunk_size`` tokens once the prefix is added.
        breadcrumb_tokens = self._count_tokens(breadcrumb + "\n\n") if breadcrumb else 0
        body_budget = max(1, chunk_size - breadcrumb_tokens)

        bodies = self._split_into_bodies(content, body_budget, overlap)
        total = len(bodies)

        for i, body in enumerate(bodies):
            text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
            chunks.append(
                self._create_chunk(
                    text,
                    document,
                    section.get("id"),
                    section.get("title"),
                    chunk_index=i,
                    total_chunks=total,
                    section_path=section.get("section_path", []),
                    section_path_titles=section.get("section_path_titles", []),
                )
            )

        return chunks

    def _split_into_bodies(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split ``text`` into pieces of at most ``max_tokens`` BPE tokens.

        The strategy is recursive — paragraphs first, then sentences, then a
        token-window fallback for runs that are still too large. Adjacent
        units are packed together greedily so chunks are filled efficiently.
        Overlap is implemented by carrying the tail of the previous chunk into
        the start of the next.
        """
        if self._count_tokens(text) <= max_tokens:
            return [text]

        # 1. Split into paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # 2. Recursively break paragraphs that are still oversized.
        units: list[str] = []
        for p in paragraphs:
            if self._count_tokens(p) <= max_tokens:
                units.append(p)
            else:
                units.extend(self._split_paragraph(p, max_tokens))

        # 3. Pack units into chunks with overlap.
        return self._pack_units(units, max_tokens, overlap_tokens)

    def _split_paragraph(self, paragraph: str, max_tokens: int) -> list[str]:
        """Break an oversized paragraph into sentence-sized (or smaller) units."""
        # Sentence split — keeps the terminator with the sentence.
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        sentences = [s for s in sentences if s]

        units: list[str] = []
        for s in sentences:
            if self._count_tokens(s) <= max_tokens:
                units.append(s)
            else:
                units.extend(self._token_window(s, max_tokens))
        return units

    def _token_window(self, text: str, max_tokens: int) -> list[str]:
        """Hard fallback — slice on the BPE token stream itself."""
        ids = self._encoder.encode(text)
        pieces: list[str] = []
        for start in range(0, len(ids), max_tokens):
            piece = self._encoder.decode(ids[start : start + max_tokens])
            if piece.strip():
                pieces.append(piece)
        return pieces

    def _pack_units(
        self,
        units: list[str],
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Greedily pack units into chunks of <= ``max_tokens`` with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        # ``\n\n`` between joined units adds a token or two; count it.
        join_cost = self._count_tokens("\n\n")

        def flush():
            nonlocal current, current_tokens
            if not current:
                return
            chunks.append("\n\n".join(current))
            if overlap_tokens > 0:
                tail = self._take_last_tokens(chunks[-1], overlap_tokens)
                current = [tail] if tail else []
                current_tokens = self._count_tokens(tail) if tail else 0
            else:
                current = []
                current_tokens = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)
            added = unit_tokens + (join_cost if current else 0)
            if current and current_tokens + added > max_tokens:
                flush()
                # If the unit alone would still overflow (shouldn't, since we
                # pre-split, but be defensive) skip the overlap prefix and put
                # the unit in by itself.
                added = unit_tokens + (join_cost if current else 0)
                if current_tokens + added > max_tokens:
                    current = []
                    current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens + (join_cost if len(current) > 1 else 0)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _create_chunk(
        self,
        text: str,
        document: Document,
        section_id: str | None,
        section_title: str | None,
        chunk_index: int = 0,
        total_chunks: int = 1,
        section_path: list[str] | None = None,
        section_path_titles: list[str] | None = None,
    ) -> Chunk:
        """Create a Chunk object with proper metadata including hierarchy."""
        chunk_id = str(uuid.uuid4())

        full_hierarchy = (document.hierarchy or []) + (section_path or [])

        return Chunk(
            text=text,
            chunk_id=chunk_id,
            module=document.module,
            component=document.component,
            version=document.version,
            page_id=document.page_id,
            section_id=section_id,
            section_title=section_title,
            page_role=document.page_role,
            source_file=document.file_path,
            hierarchy=full_hierarchy,
            metadata={
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "document_title": document.title,
                "section_path": section_path or [],
                "section_path_titles": section_path_titles or [],
            },
        )

    def chunk_text(
        self,
        text: str,
        module: str,
        component: str,
        version: str,
        page_id: str,
        section_id: str | None = None,
        section_title: str | None = None,
        page_role: str | None = None,
        source_file: str | None = None,
        hierarchy: list[str] | None = None,
    ) -> list[Chunk]:
        """
        Chunk plain text with minimal metadata.

        Useful for chunking without full Document structure.
        Text can be either Markdown or plain text.
        """
        bodies = self._split_into_bodies(text, self.chunk_size, self.overlap)
        chunks: list[Chunk] = []
        total = len(bodies)

        for i, body in enumerate(bodies):
            chunk_text = (
                f"{section_title}: {body}" if i == 0 and section_title and total > 1 else body
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=str(uuid.uuid4()),
                    module=module,
                    component=component,
                    version=version,
                    page_id=page_id,
                    section_id=section_id,
                    section_title=section_title,
                    page_role=page_role,
                    source_file=source_file,
                    hierarchy=hierarchy or [],
                    metadata=({"chunk_index": i, "total_chunks": total} if total > 1 else {}),
                )
            )

        return chunks
