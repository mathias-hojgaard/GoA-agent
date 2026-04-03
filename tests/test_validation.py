"""Tests for the AdvancedValidation agent."""

from __future__ import annotations

import uuid

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from src.grounding.validation import ValidationResult, validate_extractions
from src.models.ga_models import Extraction, Rectangle

_ZERO_UUID = uuid.UUID(int=0)


def _make_extraction(field_name: str, value: str) -> Extraction:
    return Extraction(
        source_of_extraction="pdf",
        filename="test.pdf",
        extraction_certainty=0.9,
        similarity_to_confirmed_extractions=0.0,
        genai_score=0.9,
        coordinate_id=_ZERO_UUID,
        field_name=field_name,
        field_name_raw="",
        field_name_coordinates_id=_ZERO_UUID,
        raw_saga_extraction="",
        raw_extracted_value=value,
        extracted_value=value,
        relations=[],
        coordinates_rectangle=[
            Rectangle(top=0.3, left=0.1, width=0.2, height=0.01, page=0)
        ],
        message="",
        advanced_validation=[],
    )


def _make_test_agent(result: ValidationResult) -> Agent[None, ValidationResult]:
    return Agent(
        TestModel(custom_output_args=result.model_dump()),
        output_type=ValidationResult,
    )


@pytest.mark.asyncio
async def test_validation_flags_non_product_line():
    """Mock Gemini: feed extractions with an obvious non-product line → flagged as delete."""
    extractions = [
        _make_extraction("product_number", "WG-4420-BLK"),
        _make_extraction("product_number", "SUBTOTAL"),
    ]

    mock_result = ValidationResult(
        issues=[
            {
                "extraction_index": 1,
                "status": "delete",
                "message": "SUBTOTAL is not a product line",
                "details": "auto_apply",
            }
        ]
    )

    agent = _make_test_agent(mock_result)
    validations = await validate_extractions(extractions, agent=agent)

    assert len(validations) == 1
    assert validations[0].status == "delete"
    assert validations[0].details == "auto_apply"
    # Check it was mapped back to the extraction
    assert len(extractions[1].advanced_validation) == 1
    assert extractions[0].advanced_validation == []


@pytest.mark.asyncio
async def test_validation_empty_extractions():
    result = await validate_extractions([])
    assert result == []


@pytest.mark.asyncio
async def test_validation_no_issues():
    """When agent finds no issues, returns empty list."""
    extractions = [_make_extraction("product_number", "WG-4420-BLK")]
    mock_result = ValidationResult(issues=[])

    agent = _make_test_agent(mock_result)
    validations = await validate_extractions(extractions, agent=agent)

    assert validations == []
    assert extractions[0].advanced_validation == []
