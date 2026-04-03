"""Tests for the cross-reference resolver."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from src.models import DocumentMap, FileCoordinates, PageClassification
from src.resolution.crossref_resolver import (
    ResolvedReference,
    _get_agent,
    resolve_crossrefs,
)


@pytest.fixture
def file_coords() -> FileCoordinates:
    """Minimal FileCoordinates for testing."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "sample_file_coordinates.json"
    with open(fixture_path) as f:
        return FileCoordinates(**json.load(f))


@pytest.fixture
def doc_map() -> DocumentMap:
    return DocumentMap(
        classifications=[
            PageClassification(
                page_index=0,
                page_type="product_table",
                has_vision_table=True,
                vision_confidence=0.95,
                relevant=True,
            ),
            PageClassification(
                page_index=1,
                page_type="terms_and_conditions",
                has_vision_table=False,
                vision_confidence=0.0,
                relevant=False,
            ),
        ],
        total_pages=2,
        relevant_pages=[0],
    )


@pytest.fixture(autouse=True)
def _inject_test_agent():
    """Replace the lazy crossref_agent with a TestModel-backed agent for all tests."""
    import src.resolution.crossref_resolver as cr_mod

    test_agent: Agent[None, list[ResolvedReference]] = Agent(
        TestModel(),
        output_type=list[ResolvedReference],
    )
    original = cr_mod.crossref_agent
    cr_mod.crossref_agent = test_agent
    yield test_agent
    cr_mod.crossref_agent = original
    _get_agent.cache_clear()


@pytest.mark.asyncio
async def test_resolve_crossrefs_with_mock(doc_map, file_coords, _inject_test_agent):
    """Mock Gemini and verify cross-ref resolution returns target page indices."""
    mock_result = [
        ResolvedReference(
            original_text="see Section 3.2",
            target_page_indices=[1],
            resolved_content="Terms and conditions section",
        )
    ]

    custom_args = [r.model_dump() for r in mock_result]
    with _inject_test_agent.override(model=TestModel(custom_output_args=custom_args)):
        results = await resolve_crossrefs(
            crossrefs=["see Section 3.2"],
            doc_map=doc_map,
            file_coords=file_coords,
        )

    assert len(results) == 1
    assert results[0].original_text == "see Section 3.2"
    assert results[0].target_page_indices == [1]
    assert results[0].resolved_content == "Terms and conditions section"


@pytest.mark.asyncio
async def test_resolve_empty_crossrefs(doc_map, file_coords):
    """No cross-references should return empty list without calling the LLM."""
    results = await resolve_crossrefs(
        crossrefs=[],
        doc_map=doc_map,
        file_coords=file_coords,
    )
    assert results == []
