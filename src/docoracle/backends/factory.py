"""Factory selecting the configured answer backend.

Both entry points (CLI and HTTP server) resolve their answer backend here so
they never construct a concrete backend directly.
"""

from __future__ import annotations

import logging

from ..api.structured_llm_client import StructuredLLMClient
from ..core.config import Config
from ..core.hybrid_searcher import HybridSearcher
from .protocol import AnswerBackend

logger = logging.getLogger("backends.factory")


def create_answer_backend(
    searcher: HybridSearcher,
    llm_client: StructuredLLMClient,
    config: Config,
) -> AnswerBackend:
    """Return the answer backend selected by ``config.docoracle.backend``.

    The agent module is imported lazily so the engine backend does not depend on
    the agent (and vice versa).
    """
    if config.docoracle.backend == "agent":
        from .agent import QAAgent

        logger.info("Answer backend: agent")
        return QAAgent(searcher, llm_client=llm_client, config=config)

    from .engine import QAEngine

    logger.info("Answer backend: engine")
    return QAEngine(searcher, llm_client, config=config)
