"""CLI interface for DocOracle."""

from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from docoracle.api.structured_llm_client import StructuredLLMClient
from docoracle.backends.factory import create_answer_backend
from docoracle.backends.mcp import MCPToolsetManager, build_mcp_toolsets
from docoracle.backends.protocol import AnswerBackend
from docoracle.core.bm25_index import BM25Index
from docoracle.core.config import Config, get_config
from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.loader import AntoraLoader
from docoracle.core.models import Chunk
from docoracle.core.search_index import ScoredChunk

# Load environment variables from .env file
load_dotenv()


async def _run_with_mcp_lifecycle(
    config: Config,
    coro_factory: Any,
) -> Any:
    """Run ``coro_factory(toolsets)`` with the MCP connections owned for the call.

    Enabled MCP servers are connected for the duration of the invocation and
    closed afterward. With no servers configured this is a no-op.
    """
    manager = MCPToolsetManager(build_mcp_toolsets(config.docoracle.agent.mcp_servers))
    async with manager:
        return await coro_factory(manager.toolsets)


def _get_pydantic_config(config_path: str) -> Config:
    """Load and return the Pydantic config.

    Surfaces a single-line, actionable error if ``config_path`` exists but
    is invalid (wrong type, out-of-range value, etc.) so the user can fix
    their config without reading a Pydantic traceback.

    A missing file is treated as "use defaults" — that's the expected
    behavior for first-run / CI / minimal setups (handled inside
    ``Config.from_yaml``).
    """
    import sys

    from pydantic import ValidationError

    try:
        return get_config(config_path)
    except ValidationError as e:
        # Render just the first error in a CLI-friendly form; the full
        # pydantic error is available on stderr via traceback if -v is set.
        err = e.errors()[0]
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        click.echo(
            f"Error: invalid value in {config_path}: {loc}: {msg}",
            err=True,
        )
        sys.exit(2)


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str, debug: bool) -> None:
    """DocOracle CLI tool."""
    ctx.ensure_object(dict)  # type: ignore[arg-type]
    ctx.obj["config"] = config  # type: ignore[arg-type]
    ctx.obj["debug"] = debug  # type: ignore[arg-type]

    if debug:
        import http.client
        import logging

        # Configure root logger to DEBUG level
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Enable HTTP connection debugging
        http.client.HTTPConnection.debuglevel = 1

        # Enable detailed logging for common libraries
        logging.getLogger("requests").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("uvicorn").setLevel(logging.DEBUG)
        logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
        logging.getLogger("fastapi").setLevel(logging.DEBUG)

        click.echo("Debug logging enabled", err=True)


@cli.command()
@click.option("--antora-root", default=None, help="Path to Antora project root")
@click.option("--chunk-size", default=None, type=int, help="Chunk size in tokens")
@click.option("--overlap", default=None, type=int, help="Overlap between chunks")
@click.option("--force", is_flag=True, help="Force re-ingest even if index exists")
@click.option(
    "--test-mode",
    is_flag=True,
    help="Test mode: show chunks without creating embeddings or updating store",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    antora_root: str | None,
    chunk_size: int | None,
    overlap: int | None,
    force: bool,
    test_mode: bool,
) -> None:
    """Ingest AsciiDoc modules into the hybrid store."""
    config_path = ctx.obj["config"]  # type: ignore[arg-type]
    pydantic_config = _get_pydantic_config(config_path)

    antora_root = antora_root or pydantic_config.sources.antora_root
    store_path = pydantic_config.vector_store.path

    chunk_size = chunk_size or pydantic_config.embedding.chunk_size
    overlap = overlap or pydantic_config.embedding.chunk_overlap

    loader = AntoraLoader(antora_root)
    bm25 = BM25Index(
        k1=pydantic_config.retrieval.bm25.k1,
        b=pydantic_config.retrieval.bm25.b,
        stemming=pydantic_config.retrieval.bm25.stemming,
        stopwords=pydantic_config.retrieval.bm25.stopwords,
        min_token_length=pydantic_config.retrieval.bm25.min_token_length,
    )
    searcher = HybridSearcher(bm25=bm25)
    llm_client = StructuredLLMClient(config_path)

    # Check if a store already exists (skip in test mode).
    if not test_mode and not force and searcher.load(store_path):
        click.echo(f"Store already exists at {store_path}")
        click.echo(f"Loaded {len(searcher)} chunks")
        click.confirm("Continue and add to existing index?", abort=True)

    click.echo(f"Loading documents from {antora_root}...")

    documents = loader.load_component()
    click.echo(f"Loaded {len(documents)} documents")

    all_chunks: list[Chunk] = []
    all_texts: list[str] = []

    for doc in documents:
        chunks = doc.to_chunks(chunk_size=chunk_size, overlap=overlap)
        page_path = doc.file_path
        module_prefix = f"modules/{doc.module}/pages/"
        if page_path.startswith(module_prefix):
            page_path = page_path[len(module_prefix) :]
        collection = doc.component_title or doc.component
        module_label = doc.module if doc.module != "ROOT" else ""
        parts = [collection, module_label, page_path]
        click.echo(f"  Processing: {' > '.join(p for p in parts if p)} - {len(chunks)} chunks")
        all_chunks.extend(chunks)
        all_texts.extend([c.text for c in chunks])

    click.echo(f"Created {len(all_chunks)} chunks")

    # Test mode: display all chunks and exit without creating embeddings or updating the store.
    if test_mode:
        click.echo("\n" + "=" * 70)
        click.echo("TEST MODE: Showing all chunks (no embeddings created, no store update)")
        click.echo("=" * 70)

        for idx, chunk in enumerate(all_chunks, 1):
            click.echo(f"\nChunk {idx}/{len(all_chunks)}")
            click.echo(f"  ID: {chunk.chunk_id}")
            click.echo(f"  Component: {chunk.component}")
            click.echo(f"  Module: {chunk.module}")
            click.echo(f"  Version: {chunk.version}")
            click.echo(f"  Page ID: {chunk.page_id}")
            if chunk.section_id:
                click.echo(f"  Section ID: {chunk.section_id}")
            if chunk.section_title:
                click.echo(f"  Section Title: {chunk.section_title}")
            if chunk.source_file:
                click.echo(f"  Source File: {chunk.source_file}")
            if chunk.hierarchy:
                click.echo(f"  Hierarchy: {' > '.join(chunk.hierarchy)}")
            if chunk.page_role:
                click.echo(f"  Page Role: {chunk.page_role}")
            click.echo("  Text:")
            for line in chunk.text.split("\n"):
                click.echo(f"    {line}")
            click.echo("-" * 70)

        click.echo(
            f"\nTest mode complete. Displayed {len(all_chunks)} chunks from {len(documents)} documents."
        )
        return

    click.echo("Generating embeddings...")
    embeddings = llm_client.embed_batch(all_texts, batch_size=32)
    click.echo(f"Generated {len(embeddings)} embeddings")

    click.echo("Adding to hybrid store...")
    searcher.add_chunks(all_chunks, embeddings)

    click.echo("Saving index...")
    searcher.save(store_path)

    click.echo("\nIngestion complete!")
    click.echo(f"  Total chunks: {len(all_chunks)}")
    click.echo(f"  Index saved to: {store_path}")

    components_loaded = loader.get_loaded_components()
    if components_loaded:
        click.echo(f"  Components loaded: {', '.join(components_loaded)}")

    sources = loader.get_content_sources()
    if sources:
        missing = [s["name"] for s in sources if s["name"] not in components_loaded]
        if missing:
            click.echo(f"  Warning: Missing components: {', '.join(missing)}")
            click.echo(f"  To index them, clone to: {loader.antora_root}")


@cli.command()
@click.argument("question")
@click.option("--module", default=None, help="Filter by module name")
@click.option("--component", default=None, help="Filter by component name")
@click.option("--version", default=None, help="Filter by version")
@click.option("--k", default=None, type=int, help="Number of chunks to retrieve")
@click.option(
    "--retrieval",
    default=None,
    type=click.Choice(["hybrid", "semantic", "bm25"], case_sensitive=False),
    help="Retrieval mode (default from config: hybrid)",
)
@click.option("--show-sources", is_flag=True, help="Show source links")
@click.option("--show-context", is_flag=True, help="Show retrieved context")
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    module: str | None,
    component: str | None,
    version: str | None,
    k: int | None,
    retrieval: str | None,
    show_sources: bool,
    show_context: bool,
) -> None:
    """Ask a question about the documentation."""
    config_path = ctx.obj["config"]  # type: ignore[arg-type]
    pydantic_config = _get_pydantic_config(config_path)

    store_path = pydantic_config.vector_store.path

    searcher = HybridSearcher()
    if not searcher.load(store_path):
        click.echo("Error: No store found. Run 'ingest' first.")
        return

    click.echo(f"Loaded {len(searcher)} chunks from store")

    import asyncio

    llm_client = StructuredLLMClient(config_path)

    click.echo(f"\nQuestion: {question}")
    click.echo("-" * 60)

    async def _answer(toolsets: list[Any]) -> dict[str, Any]:
        backend: AnswerBackend = create_answer_backend(
            searcher, llm_client, pydantic_config, toolsets=toolsets
        )
        return await backend.ask_detailed_async(
            question=question,
            module=module,
            component=component,
            version=version,
            k=k,
            retrieval=retrieval,
        )

    result: dict[str, Any] = asyncio.run(
        _run_with_mcp_lifecycle(pydantic_config, _answer)
    )

    click.echo(f"\nAnswer:\n{result['answer']}")

    if show_sources and result.get("sources"):
        click.echo("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            click.echo(f"  {i}. {source}")

    if show_context and result.get("chunk_details"):
        click.echo("\nContext chunks:")
        chunk_details: list[dict[str, Any]] = result["chunk_details"]
        for i, chunk in enumerate(chunk_details, 1):
            click.echo(f"  {i}. [{chunk['module']}] {chunk['section_title'] or 'N/A'}")
            click.echo(f"     Link: {chunk['link']}")
            srcs: dict[str, int] = chunk.get("sources") or {}
            if srcs:
                src_str = ", ".join(f"{key}={val}" for key, val in sorted(srcs.items()))
                click.echo(f"     Retrieved by: {src_str}")
            click.echo(f"     Text: {chunk['text'][:100]}...")
            click.echo()


@cli.command()
@click.argument("text")
@click.option("--k", default=5, type=int, help="Number of results")
@click.option("--module", default=None, help="Filter by module")
@click.option(
    "--retrieval",
    default=None,
    type=click.Choice(["hybrid", "semantic", "bm25"], case_sensitive=False),
    help="Retrieval mode (default from config: hybrid)",
)
@click.pass_context
def search(
    ctx: click.Context,
    text: str,
    k: int,
    module: str | None,
    retrieval: str | None,
) -> None:
    """Search for similar documentation chunks."""
    config_path = ctx.obj["config"]  # type: ignore[arg-type]
    pydantic_config = _get_pydantic_config(config_path)

    store_path = pydantic_config.vector_store.path

    searcher = HybridSearcher()
    if not searcher.load(store_path):
        click.echo("Error: No store found. Run 'ingest' first.")
        return

    import asyncio

    llm_client = StructuredLLMClient(config_path)

    click.echo(f"Searching for: '{text}'")
    click.echo("-" * 60)

    filters: dict[str, Any] = {}
    if module:
        filters["module"] = module

    async def _search(toolsets: list[Any]) -> list[ScoredChunk]:
        backend: AnswerBackend = create_answer_backend(
            searcher, llm_client, pydantic_config, toolsets=toolsets
        )
        # Override the backend's default mode if the user passed --retrieval.
        if retrieval:
            backend.default_mode = retrieval
        return await asyncio.to_thread(backend.get_related_chunks, text, k=k, **filters)

    results: list[ScoredChunk] = asyncio.run(
        _run_with_mcp_lifecycle(pydantic_config, _search)
    )

    for i, scored in enumerate(results, 1):
        chunk = scored.chunk
        click.echo(f"\nResult {i} (score: {scored.score:.4f}):")
        click.echo(f"  Module: {chunk.module}")
        click.echo(f"  Page: {chunk.page_id}")
        if chunk.section_title:
            click.echo(f"  Section: {chunk.section_title}")
        if scored.sources:
            src_str = ", ".join(f"{key}={val}" for key, val in sorted(scored.sources.items()))
            click.echo(f"  Retrieved by: {src_str}")
        click.echo(f"  Link: {chunk.get_link()}")
        click.echo(f"  Text: {chunk.text[:200]}...")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show information about the store."""
    config_path = ctx.obj["config"]  # type: ignore[arg-type]
    pydantic_config = _get_pydantic_config(config_path)

    store_path = pydantic_config.vector_store.path

    searcher = HybridSearcher()
    if searcher.load(store_path):
        click.echo(f"Store: {store_path}")
        click.echo(f"Total chunks: {len(searcher)}")
        click.echo(f"  Semantic index: {len(searcher.semantic)}")
        click.echo(f"  BM25 index: {len(searcher.bm25)}")
        mode = pydantic_config.retrieval.mode
        click.echo(f"Default retrieval mode: {mode}")

        modules: dict[str, int] = {}
        components: dict[str, int] = {}
        modules_by_component: dict[str, dict[str, int]] = {}
        for chunk in searcher.chunks:
            modules[chunk.module] = modules.get(chunk.module, 0) + 1
            components[chunk.component] = components.get(chunk.component, 0) + 1
            if chunk.component not in modules_by_component:
                modules_by_component[chunk.component] = {}
            modules_by_component[chunk.component][chunk.module] = (
                modules_by_component[chunk.component].get(chunk.module, 0) + 1
            )

        click.echo("\nChunks by component:")
        for component, count in sorted(components.items()):
            click.echo(f"  {component}: {count}")

        click.echo("\nChunks by module (grouped by component):")
        for component in sorted(modules_by_component.keys()):
            click.echo(f"  {component}:")
            for module, count in sorted(modules_by_component[component].items()):
                click.echo(f"    {module}: {count}")
    else:
        click.echo("No store found. Run 'ingest' first.")


@cli.command()
@click.option("--antora-root", default=None, help="Path to Antora root")
@click.option("--list-sources", is_flag=True, help="List content sources from playbook")
@click.option("--clone", is_flag=True, help="Clone missing component repositories")
@click.option("--ssh/--https", default=True, help="Use SSH URLs (default) or HTTPS")
@click.option(
    "--target-dir", default=None, help="Target directory for cloning (default: antora-root)"
)
@click.option("--dry-run", is_flag=True, help="Show what would be cloned without actually cloning")
@click.pass_context
def components(
    ctx: click.Context,
    antora_root: str | None,
    list_sources: bool,
    clone: bool,
    ssh: bool,
    target_dir: str | None,
    dry_run: bool,
) -> None:
    """List or clone Antora component repositories from playbook."""
    config_path = ctx.obj["config"]  # type: ignore[arg-type]
    pydantic_config = _get_pydantic_config(config_path)

    antora_root_str: str = antora_root or pydantic_config.sources.antora_root
    antora_root_path: Path = Path(antora_root_str)

    loader = AntoraLoader(str(antora_root_path))

    playbook = loader._find_playbook()  # type: ignore[attr-defined]
    if not playbook:
        click.echo("No playbook found. Single component mode.")
        if list_sources:
            click.echo("Use --antora-root to specify a directory with a playbook.")
        return

    loader._parse_playbook(playbook)  # type: ignore[attr-defined]
    sources = loader.get_content_sources()

    if list_sources or not (clone or ssh or target_dir or dry_run):
        # Default: list sources
        click.echo("Content sources from playbook:")
        for s in sources:
            comp_path = antora_root_path / s["name"]
            exists = comp_path.exists()
            status = "✓" if exists else "✗"
            click.echo(f"  {status} {s['name']:15} {s['url']}")

        click.echo(f"\nTo index all components, clone them to: {antora_root_path}")

        if not clone:
            return

    if clone:
        target = Path(target_dir) if target_dir else antora_root_path
        target.mkdir(parents=True, exist_ok=True)

        click.echo(f"Cloning component repositories to: {target}")
        click.echo(f"Using {'SSH' if ssh else 'HTTPS'} URLs\n")

        if dry_run:
            click.echo("Dry run - no repositories will be cloned\n")

        import subprocess

        cloned: list[str] = []
        updated: list[str] = []
        failed: list[str] = []

        for source in sources:
            comp_name: str = source["name"]
            comp_dir: Path = target / comp_name

            # Get the Git URL
            url: str = source["url"]

            # Convert to SSH if requested
            ssh_url = convert_to_ssh(url) if ssh else url

            if dry_run:
                if comp_dir.exists():
                    click.echo(f"  [DRY] git -C {comp_dir} pull")
                else:
                    click.echo(f"  [DRY] git clone {ssh_url} {comp_name}")
            else:
                try:
                    if comp_dir.exists():
                        click.echo(f"  Updating {comp_name}...")
                        result = subprocess.run(
                            ["git", "-C", str(comp_dir), "pull"],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            click.echo(f"  ✓ {comp_name:15} Updated successfully")
                            updated.append(comp_name)
                        else:
                            click.echo(f"  ✗ {comp_name:15} Failed: {result.stderr}")
                            failed.append(comp_name)
                    else:
                        click.echo(f"  Cloning {comp_name}...")
                        result = subprocess.run(
                            ["git", "clone", ssh_url, str(comp_dir)],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            click.echo(f"  ✓ {comp_name:15} Cloned successfully")
                            cloned.append(comp_name)
                        else:
                            click.echo(f"  ✗ {comp_name:15} Failed: {result.stderr}")
                            failed.append(comp_name)
                except Exception as e:
                    click.echo(f"  ✗ {comp_name:15} Error: {e}")
                    failed.append(comp_name)

        if not dry_run:
            click.echo("\nSummary:")
            click.echo(f"  Cloned: {len(cloned)} ({', '.join(cloned) if cloned else 'none'})")
            click.echo(f"  Updated: {len(updated)} ({', '.join(updated) if updated else 'none'})")
            if failed:
                click.echo(f"  Failed: {len(failed)} ({', '.join(failed)})")
            click.echo(f"\nAll repositories at: {target}")

            # Now show what's loaded
            docs = loader.load_component()
            components_loaded = loader.get_loaded_components()
            click.echo(f"Loaded {len(docs)} documents from {len(components_loaded)} components")


def convert_to_ssh(git_url: str) -> str:
    """Convert HTTPS Git URL to SSH format.

    Examples:
        https://git.example.com/doc/site -> git@git.example.com:doc/site.git
        https://git.example.com/doc/site.git -> git@git.example.com:doc/site.git
        https://git.example.com/doc/site/ -> git@git.example.com:doc/site.git
        git@git.example.com:doc/site.git -> git@git.example.com:doc/site.git
    """
    git_url = git_url.strip()

    # Already SSH format
    if git_url.startswith("git@"):
        # Ensure it ends with .git
        if not git_url.endswith(".git"):
            return git_url + ".git"
        return git_url

    # Remove .git suffix if present
    has_git_suffix = git_url.endswith(".git")
    clean_url = git_url[:-4] if has_git_suffix else git_url

    # Remove trailing slashes
    clean_url = clean_url.rstrip("/")

    # HTTPS to SSH conversion
    if clean_url.startswith("https://"):
        clean_url = clean_url[8:]
    elif clean_url.startswith("http://"):
        clean_url = clean_url[7:]

    # Split host from path
    parts = clean_url.split("/", 1)
    if len(parts) == 2:
        host = parts[0]
        path = parts[1]
        if path.endswith(".git"):
            path = path[:-4]
        return f"git@{host}:{path}.git"

    # Fallback: return original
    return git_url + ".git" if not git_url.endswith(".git") else git_url


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, default=True, help="Enable auto-reload")
@click.option("--debug", is_flag=True, help="Enable debug logging (also enabled by global --debug)")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    reload: bool,
    debug: bool,
) -> None:
    """Start the web server."""
    import uvicorn

    ctx.obj["config"]  # type: ignore[arg-type]
    # Local --debug on `serve` takes precedence over the group-level --debug,
    # so `docoracle serve --debug` works the same as `docoracle --debug serve`.
    debug_mode = debug or ctx.obj.get("debug", False)  # type: ignore[arg-type]

    click.echo("Starting DocOracle server...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Open: http://{host}:{port}/ui")
    click.echo("  Press Ctrl+C to stop")

    # Get the server module path
    from pathlib import Path

    server_dir = Path(__file__).parent / "server"

    # Use debug log level if debug flag is set
    log_level = "debug" if debug_mode else "info"

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        app_dir=str(server_dir.parent),
    )


if __name__ == "__main__":
    cli(obj={})
