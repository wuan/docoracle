"""Factory selecting the configured answer backend.

Both entry points (CLI and HTTP server) resolve their answer backend here so
they never construct a concrete backend directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic_ai.toolsets import AbstractToolset

from ..api.structured_llm_client import StructuredLLMClient
from ..core.config import Config
from ..core.hybrid_searcher import HybridSearcher
from .protocol import AnswerBackend

logger = logging.getLogger("backends.factory")


def create_answer_backend(
    searcher: HybridSearcher,
    llm_client: StructuredLLMClient,
    config: Config,
    toolsets: Sequence[AbstractToolset] | None = None,
) -> AnswerBackend:
    """Return the answer backend selected by ``config.docoracle.backend``.

    ``toolsets`` are the live MCP toolsets to mount; they are only used by the
    agent backend and are ignored by the engine backend. The agent module is
    imported lazily so the engine backend does not depend on the agent (and
    vice versa).
    """
    if config.docoracle.backend == "agent":
        from .agent import QAAgent

        logger.info("Answer backend: agent")
        return QAAgent(searcher, llm_client=llm_client, config=config, toolsets=toolsets)

    from .engine import QAEngine

    logger.info("Answer backend: engine")
    return QAEngine(searcher, llm_client, config=config)
