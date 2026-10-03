"""Answer backends for DocOracle.

Houses the two interchangeable answer paths behind the shared
:class:`~docoracle.backends.protocol.AnswerBackend` interface:

- :mod:`docoracle.backends.engine` — deterministic retrieve-then-answer pipeline.
- :mod:`docoracle.backends.agent` — pydantic-ai agent that decides when to retrieve.

Use :func:`~docoracle.backends.factory.create_answer_backend` to build the backend
selected by configuration.
"""
