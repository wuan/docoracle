"""Tests for extended response schemas.

Tests the Pydantic models for different types of structured responses.
"""

import pytest
from pydantic import ValidationError

# =============================================================================
# COMPARISON RESPONSE TESTS
# =============================================================================


class TestComparisonResponse:
    """Tests for ComparisonResponse model."""

    def test_valid_comparison_response(self):
        """Test that a valid ComparisonResponse can be created."""
        from docoracle.core.llm_outputs import ComparisonResponse

        response = ComparisonResponse(
            items=["Option A", "Option B"],
            comparison={
                "Price": {"Option A": "$10", "Option B": "$20"},
                "Speed": {"Option A": "Fast", "Option B": "Faster"},
            },
            summary="Option B is more expensive but faster.",
            recommendation="Option B",
        )

        assert response.items == ["Option A", "Option B"]
        assert response.comparison["Price"]["Option A"] == "$10"
        assert response.summary == "Option B is more expensive but faster."
        assert response.recommendation == "Option B"

    def test_comparison_response_without_recommendation(self):
        """Test ComparisonResponse without optional recommendation."""
        from docoracle.core.llm_outputs import ComparisonResponse

        response = ComparisonResponse(
            items=["A", "B"],
            comparison={"feature": {"A": "x", "B": "y"}},
            summary="Different",
        )

        assert response.recommendation is None

    def test_comparison_response_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        from docoracle.core.llm_outputs import ComparisonResponse

        with pytest.raises(ValidationError):
            ComparisonResponse()

    def test_comparison_response_missing_items(self):
        """Test that missing items field raises ValidationError."""
        from docoracle.core.llm_outputs import ComparisonResponse

        with pytest.raises(ValidationError) as exc_info:
            ComparisonResponse(
                comparison={"f": {"a": "b"}},
                summary="test",
            )

        assert "items" in str(exc_info.value)

    def test_comparison_response_missing_comparison(self):
        """Test that missing comparison field raises ValidationError."""
        from docoracle.core.llm_outputs import ComparisonResponse

        with pytest.raises(ValidationError) as exc_info:
            ComparisonResponse(
                items=["A", "B"],
                summary="test",
            )

        assert "comparison" in str(exc_info.value)

    def test_comparison_response_missing_summary(self):
        """Test that missing summary field raises ValidationError."""
        from docoracle.core.llm_outputs import ComparisonResponse

        with pytest.raises(ValidationError) as exc_info:
            ComparisonResponse(
                items=["A", "B"],
                comparison={"f": {"a": "b"}},
            )

        assert "summary" in str(exc_info.value)

    def test_comparison_response_to_dict(self):
        """Test that ComparisonResponse can be serialized to dict."""
        from docoracle.core.llm_outputs import ComparisonResponse

        response = ComparisonResponse(
            items=["A", "B"],
            comparison={"f": {"A": "x", "B": "y"}},
            summary="summary",
            recommendation="A",
        )

        data = response.model_dump()

        assert data["items"] == ["A", "B"]
        assert data["comparison"] == {"f": {"A": "x", "B": "y"}}
        assert data["summary"] == "summary"
        assert data["recommendation"] == "A"

    def test_comparison_response_from_dict(self):
        """Test that ComparisonResponse can be created from dict."""
        from docoracle.core.llm_outputs import ComparisonResponse

        data = {
            "items": ["A", "B"],
            "comparison": {"f": {"A": "x", "B": "y"}},
            "summary": "summary",
        }

        response = ComparisonResponse(**data)

        assert response.items == ["A", "B"]
        assert response.comparison["f"]["A"] == "x"


# =============================================================================
# LIST RESPONSE TESTS
# =============================================================================


class TestListResponse:
    """Tests for ListResponse model."""

    def test_valid_list_response(self):
        """Test that a valid ListResponse can be created."""
        from docoracle.core.llm_outputs import ListResponse

        response = ListResponse(
            items=[
                {"name": "Item 1", "value": 100},
                {"name": "Item 2", "value": 200},
            ],
            total_count=2,
            categories=["Category A", "Category B"],
        )

        assert len(response.items) == 2
        assert response.items[0]["name"] == "Item 1"
        assert response.total_count == 2
        assert response.categories == ["Category A", "Category B"]

    def test_list_response_without_categories(self):
        """Test ListResponse without optional categories."""
        from docoracle.core.llm_outputs import ListResponse

        response = ListResponse(
            items=[{"name": "Test"}],
            total_count=1,
        )

        assert response.categories is None

    def test_list_response_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        from docoracle.core.llm_outputs import ListResponse

        with pytest.raises(ValidationError):
            ListResponse()

    def test_list_response_missing_items(self):
        """Test that missing items field raises ValidationError."""
        from docoracle.core.llm_outputs import ListResponse

        with pytest.raises(ValidationError) as exc_info:
            ListResponse(total_count=0)

        assert "items" in str(exc_info.value)

    def test_list_response_missing_total_count(self):
        """Test that missing total_count field raises ValidationError."""
        from docoracle.core.llm_outputs import ListResponse

        with pytest.raises(ValidationError) as exc_info:
            ListResponse(items=[])

        assert "total_count" in str(exc_info.value)

    def test_list_response_empty_items(self):
        """Test ListResponse with empty items list."""
        from docoracle.core.llm_outputs import ListResponse

        response = ListResponse(items=[], total_count=0)

        assert response.items == []
        assert response.total_count == 0

    def test_list_response_to_dict(self):
        """Test that ListResponse can be serialized to dict."""
        from docoracle.core.llm_outputs import ListResponse

        response = ListResponse(
            items=[{"id": 1}, {"id": 2}],
            total_count=2,
            categories=["cat1"],
        )

        data = response.model_dump()

        assert len(data["items"]) == 2
        assert data["total_count"] == 2
        assert data["categories"] == ["cat1"]

    def test_list_response_from_dict(self):
        """Test that ListResponse can be created from dict."""
        from docoracle.core.llm_outputs import ListResponse

        data = {
            "items": [{"id": 1}, {"id": 2}],
            "total_count": 2,
        }

        response = ListResponse(**data)

        assert len(response.items) == 2
        assert response.total_count == 2


# =============================================================================
# DECISION RESPONSE TESTS
# =============================================================================


class TestDecisionResponse:
    """Tests for DecisionResponse model."""

    def test_valid_decision_response_yes(self):
        """Test DecisionResponse with 'yes' decision."""
        from docoracle.core.llm_outputs import DecisionResponse

        response = DecisionResponse(
            decision="yes",
            reasoning="It is the best option",
            pros=["Pro 1", "Pro 2"],
            cons=["Con 1"],
            confidence=0.95,
        )

        assert response.decision == "yes"
        assert response.reasoning == "It is the best option"
        assert response.pros == ["Pro 1", "Pro 2"]
        assert response.cons == ["Con 1"]
        assert response.confidence == 0.95

    def test_valid_decision_response_no(self):
        """Test DecisionResponse with 'no' decision."""
        from docoracle.core.llm_outputs import DecisionResponse

        response = DecisionResponse(
            decision="no",
            reasoning="It is not recommended",
            confidence=0.8,
        )

        assert response.decision == "no"
        assert response.pros == []
        assert response.cons == []

    def test_valid_decision_response_maybe(self):
        """Test DecisionResponse with 'maybe' decision."""
        from docoracle.core.llm_outputs import DecisionResponse

        response = DecisionResponse(
            decision="maybe",
            reasoning="Uncertain",
            confidence=0.5,
        )

        assert response.decision == "maybe"

    def test_decision_response_with_empty_pros_cons(self):
        """Test DecisionResponse with empty pros and cons lists."""
        from docoracle.core.llm_outputs import DecisionResponse

        response = DecisionResponse(
            decision="yes",
            reasoning="Test",
            confidence=0.9,
        )

        assert response.pros == []
        assert response.cons == []

    def test_decision_response_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        from docoracle.core.llm_outputs import DecisionResponse

        with pytest.raises(ValidationError):
            DecisionResponse()

    def test_decision_response_invalid_decision(self):
        """Test that invalid decision value raises ValidationError."""
        from docoracle.core.llm_outputs import DecisionResponse

        with pytest.raises(ValidationError) as exc_info:
            DecisionResponse(
                decision="invalid",
                reasoning="test",
                confidence=0.5,
            )

        assert "decision" in str(exc_info.value)

    def test_decision_response_missing_decision(self):
        """Test that missing decision field raises ValidationError."""
        from docoracle.core.llm_outputs import DecisionResponse

        with pytest.raises(ValidationError) as exc_info:
            DecisionResponse(
                reasoning="test",
                confidence=0.5,
            )

        assert "decision" in str(exc_info.value)

    def test_decision_response_missing_reasoning(self):
        """Test that missing reasoning field raises ValidationError."""
        from docoracle.core.llm_outputs import DecisionResponse

        with pytest.raises(ValidationError) as exc_info:
            DecisionResponse(
                decision="yes",
                confidence=0.5,
            )

        assert "reasoning" in str(exc_info.value)

    def test_decision_response_missing_confidence(self):
        """Test that missing confidence field raises ValidationError."""
        from docoracle.core.llm_outputs import DecisionResponse

        with pytest.raises(ValidationError) as exc_info:
            DecisionResponse(
                decision="yes",
                reasoning="test",
            )

        assert "confidence" in str(exc_info.value)

    def test_decision_response_confidence_range(self):
        """Test that confidence must be between 0.0 and 1.0."""
        from docoracle.core.llm_outputs import DecisionResponse

        # Test valid range
        response = DecisionResponse(
            decision="yes",
            reasoning="test",
            confidence=0.5,
        )
        assert response.confidence == 0.5

        # Test edge cases
        response_low = DecisionResponse(
            decision="yes",
            reasoning="test",
            confidence=0.0,
        )
        assert response_low.confidence == 0.0

        response_high = DecisionResponse(
            decision="yes",
            reasoning="test",
            confidence=1.0,
        )
        assert response_high.confidence == 1.0

    def test_decision_response_to_dict(self):
        """Test that DecisionResponse can be serialized to dict."""
        from docoracle.core.llm_outputs import DecisionResponse

        response = DecisionResponse(
            decision="yes",
            reasoning="Good choice",
            pros=["Pro"],
            cons=["Con"],
            confidence=0.9,
        )

        data = response.model_dump()

        assert data["decision"] == "yes"
        assert data["reasoning"] == "Good choice"
        assert data["pros"] == ["Pro"]
        assert data["cons"] == ["Con"]
        assert data["confidence"] == 0.9

    def test_decision_response_from_dict(self):
        """Test that DecisionResponse can be created from dict."""
        from docoracle.core.llm_outputs import DecisionResponse

        data = {
            "decision": "no",
            "reasoning": "Bad choice",
            "pros": [],
            "cons": ["Con 1", "Con 2"],
            "confidence": 0.2,
        }

        response = DecisionResponse(**data)

        assert response.decision == "no"
        assert response.reasoning == "Bad choice"
        assert response.cons == ["Con 1", "Con 2"]


# =============================================================================
# JSON SERIALIZATION TESTS
# ==============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization and deserialization of response schemas."""

    def test_comparison_response_json_roundtrip(self):
        """Test JSON roundtrip for ComparisonResponse."""
        from docoracle.core.llm_outputs import ComparisonResponse

        original = ComparisonResponse(
            items=["A", "B"],
            comparison={"f": {"A": "x", "B": "y"}},
            summary="summary",
        )

        json_str = original.model_dump_json()
        restored = ComparisonResponse.model_validate_json(json_str)

        assert restored == original

    def test_list_response_json_roundtrip(self):
        """Test JSON roundtrip for ListResponse."""
        from docoracle.core.llm_outputs import ListResponse

        original = ListResponse(
            items=[{"id": 1}, {"id": 2}],
            total_count=2,
            categories=["cat1"],
        )

        json_str = original.model_dump_json()
        restored = ListResponse.model_validate_json(json_str)

        assert restored == original

    def test_decision_response_json_roundtrip(self):
        """Test JSON roundtrip for DecisionResponse."""
        from docoracle.core.llm_outputs import DecisionResponse

        original = DecisionResponse(
            decision="yes",
            reasoning="Good",
            pros=["Pro"],
            cons=["Con"],
            confidence=0.9,
        )

        json_str = original.model_dump_json()
        restored = DecisionResponse.model_validate_json(json_str)

        assert restored == original


# =============================================================================
# SCHEMA VALIDATION TESTS
# ==============================================================================


class TestSchemaValidation:
    """Tests for schema validation and field constraints."""

    def test_comparison_response_field_descriptions(self):
        """Test that ComparisonResponse has correct field descriptions."""
        from docoracle.core.llm_outputs import ComparisonResponse

        schema = ComparisonResponse.model_json_schema()

        assert "items" in schema["properties"]
        assert "comparison" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "recommendation" in schema["properties"]

        assert schema["properties"]["items"]["description"] == "List of compared items"
        assert (
            schema["properties"]["recommendation"]["description"]
            == "Recommended choice if applicable"
        )

    def test_list_response_field_descriptions(self):
        """Test that ListResponse has correct field descriptions."""
        from docoracle.core.llm_outputs import ListResponse

        schema = ListResponse.model_json_schema()

        assert "items" in schema["properties"]
        assert "total_count" in schema["properties"]
        assert "categories" in schema["properties"]

        assert schema["properties"]["items"]["description"] == "Listed items"
        assert schema["properties"]["total_count"]["description"] == "Total number of items"

    def test_decision_response_field_descriptions(self):
        """Test that DecisionResponse has correct field descriptions."""
        from docoracle.core.llm_outputs import DecisionResponse

        schema = DecisionResponse.model_json_schema()

        assert "decision" in schema["properties"]
        assert "reasoning" in schema["properties"]
        assert "pros" in schema["properties"]
        assert "cons" in schema["properties"]
        assert "confidence" in schema["properties"]

        assert schema["properties"]["decision"]["description"] == "Final decision"
        assert (
            schema["properties"]["confidence"]["description"] == "Confidence in decision (0.0-1.0)"
        )
