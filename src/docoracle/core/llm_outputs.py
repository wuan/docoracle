"""Structured LLM outputs for DocOracle.

Provides Pydantic models for validating and parsing LLM responses,
enabling type-safe interactions with the LLM API.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Structured response from LLM for question answering.

    This is the default response shape for Q&A flows. Other flows may use
    their own Pydantic models (e.g. comparison, listing).
    """

    answer: str = Field(..., description="The generated answer text")
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence (0.0-1.0)",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of source citations (e.g. ['module:pages:install#setup', ...])",
    )
    reasoning: str | None = Field(
        None,
        description="Optional reasoning/chain-of-thought",
    )


class ComparisonResponse(BaseModel):
    """Structured response for comparison questions."""

    items: list[str] = Field(..., description="List of compared items")
    comparison: dict[str, dict[str, str]] = Field(
        ...,
        description="Feature-by-feature comparison (feature -> {item: value})",
    )
    summary: str = Field(..., description="Brief summary of differences")
    recommendation: str | None = Field(None, description="Recommended choice if applicable")


class ListResponse(BaseModel):
    """Structured response for listing questions."""

    items: list[dict[str, Any]] = Field(..., description="Listed items")
    total_count: int = Field(..., description="Total number of items")
    categories: list[str] | None = Field(None, description="Category breakdown")


class DecisionResponse(BaseModel):
    """Structured response for decision questions."""

    decision: Literal["yes", "no", "maybe"] = Field(..., description="Final decision")
    reasoning: str = Field(..., description="Detailed reasoning")
    pros: list[str] = Field(default_factory=list, description="Arguments in favor")
    cons: list[str] = Field(default_factory=list, description="Arguments against")
    confidence: float = Field(..., description="Confidence in decision (0.0-1.0)")
