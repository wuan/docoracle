"""Tests for the metadata filter primitive."""

from docoracle.core.models import Chunk
from docoracle.core.search_index import matches_filters


def _chunk(module: str = "m", component: str = "c", version: str = "1.0") -> Chunk:
    return Chunk(
        text="x",
        chunk_id=f"id-{module}-{component}-{version}",
        module=module,
        component=component,
        version=version,
        page_id="p",
    )


def test_no_filter_matches_everything():
    chunk = _chunk()
    assert matches_filters(chunk, None) is True
    assert matches_filters(chunk, {}) is True


def test_single_scalar_filter_matches():
    chunk = _chunk(module="api")
    assert matches_filters(chunk, {"module": "api"}) is True
    assert matches_filters(chunk, {"module": "core"}) is False


def test_list_filter_matches_membership():
    chunk = _chunk(module="api")
    assert matches_filters(chunk, {"module": ["api", "core"]}) is True
    assert matches_filters(chunk, {"module": ["core", "db"]}) is False


def test_multiple_filters_combine_with_and():
    chunk = _chunk(module="api", component="core", version="1.0")
    assert matches_filters(chunk, {"module": "api", "component": "core"}) is True
    assert matches_filters(chunk, {"module": "api", "component": "db"}) is False
    assert matches_filters(chunk, {"module": "api", "version": "2.0"}) is False


def test_metadata_fallback_when_attribute_missing():
    chunk = _chunk()
    chunk.metadata["custom_key"] = "value"
    assert matches_filters(chunk, {"custom_key": "value"}) is True
    assert matches_filters(chunk, {"custom_key": "other"}) is False
