"""Configuration management for DocOracle.

Uses Pydantic for type-safe configuration loading and validation.
Supports loading from YAML files, environment variables, and provides defaults.
"""

import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field("mistral-embed", description="Embedding model name")
    api_key: str | None = Field(
        None, description="API key for embedding service (resolved from env if ${VAR} syntax)"
    )
    chunk_size: int = Field(
        512, ge=1, le=4096, description="Tokens per chunk (BPE tokens, cl100k_base)"
    )
    chunk_overlap: int = Field(50, ge=0, le=1024, description="Token overlap between chunks")
    rate_limit_delay: float = Field(
        0.1,
        ge=0.0,
        le=10.0,
        description="Delay in seconds between batch embedding requests (for rate limiting)",
    )


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    type: str = Field("faiss", description="Vector store type")
    path: str = Field("./data/vectorstore", description="Storage directory for vector store files")


class SourcesConfig(BaseModel):
    """Documentation sources configuration."""

    antora_root: str = Field("./antora-docs", description="Path to Antora project root")


class AntoraSiteConfig(BaseModel):
    """Published Antora site configuration."""

    url: str | None = Field(
        None, description="Base URL of published Antora site for link generation"
    )


class LLMConfig(BaseModel):
    """LLM API configuration."""

    api_url: str = Field("https://api.mistral.ai/v1", description="Base URL for LLM API")
    api_key: str | None = Field(
        None, description="API key for LLM service (resolved from env if ${VAR} syntax)"
    )
    timeout: int = Field(120, ge=1, le=600, description="API timeout in seconds")


class BM25Config(BaseModel):
    """BM25 retrieval configuration."""

    k1: float = Field(1.5, ge=0.0, description="BM25 k1 parameter")
    b: float = Field(0.75, ge=0.0, le=1.0, description="BM25 b parameter")
    stemming: str = Field("german", description="Snowball stemmer language (or 'none')")
    stopwords: bool = Field(True, description="Enable stopword removal")
    min_token_length: int = Field(2, ge=1, le=10, description="Minimum token length")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = Field(5, ge=1, le=100, description="Number of chunks to retrieve")
    mode: Literal["hybrid", "semantic", "bm25"] = Field(
        "hybrid", description="Default retrieval mode"
    )
    rrf_k: int = Field(60, ge=1, description="RRF damping constant")
    prefetch_k: int | None = Field(
        None, ge=1, description="Per-retriever prefetch count (defaults to 4 * top_k)"
    )
    bm25_min_score_ratio: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "During hybrid fusion, BM25 results scoring below this ratio of the "
            "best BM25 score are dropped (and the whole BM25 list when its best "
            "score is 0), so lexical noise cannot outvote strong semantic hits. "
            "Set to 0 to disable."
        ),
    )
    bm25: BM25Config = Field(
        default_factory=lambda: BM25Config(),  # type: ignore[call-arg]
        description="BM25-specific configuration",
    )


class GenerationConfig(BaseModel):
    """LLM generation configuration."""

    model: str = Field(
        "mistral-small-latest",
        description=(
            "Chat model name. Sent to the configured ``llm.api_url`` via "
            "pydantic-ai's OpenAI-compatible provider, so any vendor with "
            "an OpenAI-compatible endpoint (e.g. Mistral's ``/v1`` API) "
            "works without an extra provider package."
        ),
    )
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_context_length: int = Field(
        4000, ge=1, le=32768, description="Maximum context tokens for LLM"
    )


class PromptsConfig(BaseModel):
    """Prompt configuration."""

    system: str = Field(
        "You are an expert assistant answering questions about technical documentation.",
        description="System prompt",
    )
    user: str = Field(
        "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nAnswer the question based only on the context above.",
        description="User prompt template",
    )


class DocOracleConfig(BaseModel):
    """Answer-backend selection."""

    backend: Literal["engine", "agent"] = Field(
        "engine",
        description="Answer backend: 'engine' (deterministic) or 'agent' (tool-using)",
    )


# =============================================================================
# MAIN CONFIG WITH ENV VAR RESOLUTION
# =============================================================================


def _resolve_env_var(value: Any) -> Any:
    """Resolve ``${ENV_VAR}`` syntax in a single value.

    Returns the original string unchanged if the env var is not set; the
    caller (``Config.from_yaml``) will then accept it as a literal value
    and any downstream consumer (``resolve_api_key``) will report the
    missing key with a clear error.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, value)
    return value


def resolve_api_key(config: "Config") -> str:
    """Resolve the LLM API key from config, with environment fallback.

    Lookup order:
      1. ``config.embedding.api_key`` if set (literal or ``${VAR}`` syntax).
      2. ``config.llm.api_key`` if set (literal or ``${VAR}`` syntax).
      3. ``LLM_API_KEY`` environment variable.
      4. ``MISTRAL_API_KEY`` environment variable (backward compatibility).

    Raises:
        ValueError: if no key is found anywhere.
    """
    # Try embedding API key first
    api_key = config.embedding.api_key
    if api_key and api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var)

    if api_key:
        return api_key

    # Try LLM API key
    api_key = config.llm.api_key
    if api_key and api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var)

    if api_key:
        return api_key

    # Try environment variables
    if env_key := os.environ.get("LLM_API_KEY") or os.environ.get("MISTRAL_API_KEY"):
        return env_key

    raise ValueError(
        "API key not found. Set LLM_API_KEY environment variable "
        "(or MISTRAL_API_KEY for backward compatibility)."
    )


def _resolve_env_vars_recursive(data: Any) -> Any:
    """Recursively resolve ${ENV_VAR} syntax in config data."""
    if isinstance(data, dict):
        return {k: _resolve_env_vars_recursive(v) for k, v in data.items()}  # type: ignore[return-value]
    elif isinstance(data, list):
        return [_resolve_env_vars_recursive(item) for item in data]  # type: ignore[return-value]
    else:
        return _resolve_env_var(data)


class Config(BaseModel):
    """Main configuration for DocOracle.

    This is the root configuration model that contains all nested configurations.
    It can be loaded from a YAML file or created with defaults.

    Environment variables in ${VAR} syntax are resolved automatically.
    """

    embedding: EmbeddingConfig = Field(default_factory=lambda: EmbeddingConfig())  # type: ignore[arg-type]
    vector_store: VectorStoreConfig = Field(default_factory=lambda: VectorStoreConfig())  # type: ignore[arg-type]
    sources: SourcesConfig = Field(default_factory=lambda: SourcesConfig())  # type: ignore[arg-type]
    antora_site: AntoraSiteConfig = Field(default_factory=lambda: AntoraSiteConfig())  # type: ignore[arg-type]
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig())  # type: ignore[arg-type]
    retrieval: RetrievalConfig = Field(default_factory=lambda: RetrievalConfig())  # type: ignore[arg-type]
    generation: GenerationConfig = Field(default_factory=lambda: GenerationConfig())  # type: ignore[arg-type]
    prompts: PromptsConfig = Field(default_factory=lambda: PromptsConfig())  # type: ignore[arg-type]
    docoracle: DocOracleConfig = Field(default_factory=lambda: DocOracleConfig())  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated Config instance
        """
        yaml_config: dict[str, Any] = {}
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            yaml_config = {}

        # Resolve ${ENV_VAR} syntax recursively
        yaml_config = _resolve_env_vars_recursive(yaml_config)

        # Create config from dict. If the YAML has invalid values (wrong
        # type, out-of-range number, etc.), pydantic raises ValidationError.
        # We let it propagate so the caller sees a clear error rather than
        # silently running with defaults that ignore their config.
        return cls(**yaml_config)


# =============================================================================
# CONFIG LOADING
# =============================================================================


def get_config(config_path: str = "config.yaml") -> Config:
    """Get the configuration, loading from YAML if the file exists.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated Config instance
    """
    return Config.from_yaml(config_path)
