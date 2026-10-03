"""Antora-aware AsciiDoc loader.

Supports both single-component Antora sites and multi-component
sites defined by an Antora playbook (antora-playbook.yml).
"""

import re
from pathlib import Path
from typing import Any

import yaml

from .models import Document


class AntoraLoader:
    """Loads AsciiDoc documents from Antora project structure.

    Supports:
    - Single component: antora.yml with modules/
    - Multi-component: antora-playbook.yml with content.sources
    - Mixed: playbook with some local, some remote sources
    """

    def __init__(self, antora_root: str = "./antora-docs"):
        """
        Initialize loader.

        Args:
            antora_root: Path to Antora project root or playbook directory
        """
        self.antora_root = Path(antora_root)
        self.component_sources: list[dict[str, Any]] = []
        self.components_loaded: list[str] = []

    def load_component(self) -> list[Document]:
        """
        Load all documents from Antora components.

        Detects if antora_root contains a playbook and loads all
        components, or loads a single component if no playbook found.

        Returns:
            List of Document objects
        """
        # Try to detect and parse playbook
        playbook_path = self._find_playbook()

        if playbook_path:
            # Multi-component site via playbook
            self._parse_playbook(playbook_path)
            return self._load_all_components()
        else:
            # Single component site
            return self._load_single_component()

    def _find_playbook(self) -> Path | None:
        """Find Antora playbook in the root directory."""
        # Check for common playbook filenames
        playbook_files = [
            "antora-playbook.yml",
            "antora-playbook.yaml",
            "playbook.yml",
            "playbook.yaml",
        ]

        for filename in playbook_files:
            path = self.antora_root / filename
            if path.exists():
                return path

        return None

    def _parse_playbook(self, playbook_path: Path):
        """Parse Antora playbook to extract content sources."""
        with open(playbook_path) as f:
            playbook: dict[str, Any] = yaml.safe_load(f) or {}

        # Extract content sources
        content: dict[str, Any] = playbook.get("content", {})
        sources: list[dict[str, Any]] = content.get("sources", [])

        if not sources:
            # Maybe it's a v1 playbook format
            sources = playbook.get("sources", [])

        for source in sources:
            source_dict: dict[str, Any] = {
                "url": source.get("url", ""),
                "branches": source.get("branches", ["HEAD"]),
                "start_path": source.get("start_path", "."),
                "tags": source.get("tags", []),
            }

            # Extract component name from URL
            # URL formats: https://git.xxx/owner/repo or git@git.xxx:owner/repo
            component_name = self._extract_component_name(source_dict["url"])
            source_dict["name"] = component_name

            self.component_sources.append(source_dict)

    def _extract_component_name(self, url: str) -> str:
        """Extract component name from repository URL.

        Handles formats:
        - https://git.example.com/owner/repo
        - https://git.example.com/owner/repo.git
        - https://git.example.com/owner/repo/
        - git@git.example.com:owner/repo.git
        """
        url = url.strip()

        # Remove trailing slashes
        while url.endswith("/"):
            url = url[:-1]

        # Remove .git suffix
        if url.endswith(".git"):
            url = url[:-4]

        # Handle HTTPS URLs
        if "https://" in url or "http://" in url:
            # Remove protocol
            clean_url = url.replace("https://", "").replace("http://", "")
            # Get path parts
            parts = clean_url.split("/")
            # Filter empty parts from trailing slashes
            parts = [p for p in parts if p]
            return parts[-1] if parts else "unknown"

        # Handle SSH URLs (git@host:owner/repo)
        if "@" in url:
            # Split host from path
            if ":" in url:
                _host_part, path_part = url.split(":", 1)
                # Get repo name from path
                parts = path_part.split("/")
                parts = [p for p in parts if p]
                return parts[-1] if parts else "unknown"
            return url.split("@")[-1]

        # Fallback for any other format
        parts = url.replace(".git", "").split("/")
        parts = [p for p in parts if p]
        return parts[-1] if parts else "unknown"

    def _load_all_components(self) -> list[Document]:
        """Load documents from all components in the playbook."""
        all_documents: list[Document] = []

        for source in self.component_sources:
            component_name = source["name"]

            # Try to find the component directory
            # Look in: antora_root/{name}, antora_root/components/{name}, antora_root/{name}-component
            possible_paths = [
                self.antora_root / source["name"],
                self.antora_root / "components" / source["name"],
                self.antora_root.parent / source["name"],
                self.antora_root.parent / "components" / source["name"],
                self.antora_root.parent.parent / source["name"],
                self.antora_root.parent.parent / "components" / source["name"],
            ]

            component_dir = None
            for path in possible_paths:
                if path.exists() and (path / "modules").exists():
                    component_dir = path
                    break

            # Also check if antora_root itself is this component
            # (e.g., the playbook references a component whose files are in the playbook dir)
            if (
                not component_dir
                and self.antora_root.exists()
                and (self.antora_root / "modules").exists()
            ):
                # Check if antora_root has an antora.yml that matches this component
                antora_yml = self.antora_root / "antora.yml"
                if antora_yml.exists():
                    with open(antora_yml) as f:
                        config: dict[str, Any] = yaml.safe_load(f) or {}
                    comp_name: str | None = config.get("name") or config.get("title")  # type: ignore[union-attr]
                    if comp_name and comp_name == component_name:
                        component_dir = self.antora_root

            if component_dir:
                # Load documents from this component
                docs = self._load_component_from_dir(component_dir)
                all_documents.extend(docs)
                self.components_loaded.append(component_name)
            else:
                print(
                    f"Warning: Component '{component_name}' not found locally. "
                    f"Expected at one of: {[str(p) for p in possible_paths]}"
                )

        return all_documents

    def _load_single_component(self) -> list[Document]:
        """Load documents from a single Antora component."""
        # Check if this is a component directory (has antora.yml)
        component_yml = self.antora_root / "antora.yml"
        if component_yml.exists():
            return self._load_component_from_dir(self.antora_root)

        # Maybe it's a directory with modules/
        modules_dir = self.antora_root / "modules"
        if modules_dir.exists():
            # Create a temporary component context
            return self._load_component_from_dir(self.antora_root)

        raise FileNotFoundError(
            f"No Antora component or playbook found at {self.antora_root}. "
            f"Expected antora.yml or modules/ directory."
        )

    def _load_component_from_dir(self, component_dir: Path) -> list[Document]:
        """Load documents from a single component directory."""
        # Save original root
        original_root = self.antora_root

        try:
            # Temporarily set root to component directory
            self.antora_root = component_dir

            # Load component config
            self._load_antora_config()

            documents: list[Document] = []
            modules_dir = component_dir / "modules"

            if not modules_dir.exists():
                print(f"Warning: No modules directory in {component_dir}")
                return documents

            for module_dir in sorted(modules_dir.iterdir()):
                if not module_dir.is_dir():
                    continue

                module_name = module_dir.name
                pages_dir = module_dir / "pages"

                if not pages_dir.exists():
                    continue

                for adoc_file in sorted(pages_dir.rglob("*.adoc")):
                    document = self._load_document(adoc_file, module_name)
                    if document:
                        documents.append(document)

            return documents

        finally:
            # Restore original root
            self.antora_root = original_root

    def _load_antora_config(self):
        """Load antora.yml for component metadata."""
        antora_yml = self.antora_root / "antora.yml"
        self.component_name = None
        self.component_title = None
        self.component_version = None

        if antora_yml.exists():
            with open(antora_yml) as f:
                config = yaml.safe_load(f)

            # Antora convention: ``name`` is the component identifier (URL
            # slug), ``title`` is the human display label. Prefer ``name`` so
            # filters and page IDs use a stable identifier; expose ``title``
            # separately for breadcrumbs / UI.
            self.component_name = config.get("name") or config.get("title")
            self.component_title = config.get("title")
            self.component_version = config.get("version")

            if not self.component_name:
                name_config = config.get("asciidoc", {}).get("config", {})
                self.component_name = name_config.get("name")

            if not self.component_version:
                self.component_version = "latest"

        # Fallback
        if not self.component_name:
            self.component_name = self.antora_root.name
        if not self.component_version:
            self.component_version = "1.0.0"

    def _load_document(self, file_path: Path, module: str) -> Document | None:
        """Load a single AsciiDoc file."""
        try:
            with open(file_path) as f:
                raw_content = f.read()

            # Extract metadata from AsciiDoc attributes
            metadata = self._extract_metadata(raw_content)

            # Extract title
            title = metadata.get("page-title") or self._extract_title(raw_content)

            # Extract page role
            page_role = metadata.get("page-role")

            # Generate page ID (Antora format: module:pages:filename-without-ext)
            stem = file_path.stem.replace("-", "_")
            page_id = f"{module}:pages:{stem}"

            # Extract sections with IDs
            sections = self._extract_sections(raw_content)

            return Document(
                content=raw_content,
                file_path=str(file_path.relative_to(self.antora_root)),
                module=module,
                component=self.component_name or "unknown",
                component_title=getattr(self, "component_title", None),
                version=self.component_version or "1.0.0",
                page_id=page_id,
                title=title,
                page_role=page_role,
                sections=sections,
                metadata=metadata,
                hierarchy=[],
            )

        except Exception as e:
            print(f"Warning: Could not load document {file_path}: {e}")
            return None

    def _extract_metadata(self, content: str) -> dict[str, str]:
        """Extract AsciiDoc attributes from content."""
        metadata: dict[str, str] = {}

        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(":") and ":" in line[1:]:
                parts = line[1:].split(":", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    value = parts[1].strip()
                    metadata[name] = value

        return metadata

    def _extract_title(self, content: str) -> str | None:
        """Extract document title from AsciiDoc."""
        title_match = re.search(r"^=+\s*(.+?)\s*$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        return None

    def _extract_sections(self, content: str) -> list[dict[str, Any]]:
        """Extract sections with their IDs and titles."""
        sections: list[dict[str, Any]] = []

        # Match an AsciiDoc heading and optional inline anchor on the SAME line.
        # Whitespace classes are restricted to [ \t] so the optional anchor
        # group can't reach across newlines and grab an unrelated [cols=...]
        # config line that follows the heading.
        section_pattern = re.compile(
            r"^(={1,6})[ \t]+(.+?)[ \t]*(?:\[([^\]\n]+)\])?[ \t]*$", re.MULTILINE
        )

        stack: list[dict[str, Any]] = []

        for match in section_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            explicit_id = match.group(3)

            # Pop stack to current level
            while stack and stack[-1]["level"] >= level:  # type: ignore[operator]
                stack.pop()

            # Generate ID if not explicit
            if explicit_id:
                section_id = explicit_id
            else:
                section_id = re.sub(r"[^\w\-]", "-", title.lower())
                section_id = re.sub(r"-+", "-", section_id).strip("-")

            section: dict[str, Any] = {
                "title": title,
                "id": section_id,
                "level": level,
                "parent_id": stack[-1]["id"] if stack else None,
            }

            if stack:
                parent = stack[-1]
                if "children" not in parent:
                    parent["children"] = []
                parent["children"].append(section)  # type: ignore[arg-type]
            else:
                sections.append(section)

            stack.append(section)

        return sections

    def get_loaded_components(self) -> list[str]:
        """Get list of loaded component names."""
        return self.components_loaded

    def get_content_sources(self) -> list[dict[str, Any]]:
        """Get list of content sources from playbook."""
        return self.component_sources
