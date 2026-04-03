"""Tests for Tier 3 Gemini LLM extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.extraction.tier3_gemini import build_tier3_prompt, extract_with_gemini
from src.models import (
    PageCoordinates,
    PageExtractionResult,
    RawExtraction,
    Rectangle,
    WordBox,
)


@pytest.fixture
def narrative_page() -> PageCoordinates:
    return PageCoordinates(
        image_url="https://example.com/img.png",
        image_url_v2="https://example.com/v2/img",
        page_height=842.0,
        page_width=595.0,
        page_name="",
        extraction_word_boxes=[],
        word_boxes=[
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.10, left=0.05, width=0.08, height=0.012, page=0)
                ],
                coordinate_id="01914a3b-aa01-7f00-1111-aaaaaaaaaaaa",
                value="Product",
            ),
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.10, left=0.15, width=0.10, height=0.012, page=0)
                ],
                coordinate_id="01914a3b-aa02-7f00-1111-bbbbbbbbbbbb",
                value="WG-4420-BLK",
            ),
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.13, left=0.05, width=0.06, height=0.012, page=0)
                ],
                coordinate_id="01914a3b-aa03-7f00-1111-cccccccccccc",
                value="Quantity:",
            ),
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.13, left=0.15, width=0.03, height=0.012, page=0)
                ],
                coordinate_id="01914a3b-aa04-7f00-1111-dddddddddddd",
                value="500",
            ),
        ],
        tables=[],
        layout_objects=[],
    )


@pytest.fixture
def empty_page() -> PageCoordinates:
    return PageCoordinates(
        image_url="",
        image_url_v2="",
        page_height=842.0,
        page_width=595.0,
        page_name="",
        extraction_word_boxes=[],
        word_boxes=[],
        tables=[],
        layout_objects=[],
    )


@pytest.fixture
def mock_gemini_result() -> PageExtractionResult:
    return PageExtractionResult(
        extractions=[
            RawExtraction(
                field_name="product_number",
                value="WG-4420-BLK",
                field_label_text="Product",
                confidence=0.92,
            ),
            RawExtraction(
                field_name="quantity",
                value="500",
                field_label_text="Quantity:",
                confidence=0.88,
            ),
        ],
        is_product_table=False,
        cross_references=["see Section 3.2"],
    )


class TestBuildPrompt:
    def test_contains_word_box_text(self, narrative_page: PageCoordinates):
        prompt = build_tier3_prompt(narrative_page)
        assert "Product" in prompt
        assert "WG-4420-BLK" in prompt
        assert "500" in prompt

    def test_contains_image_url(self, narrative_page: PageCoordinates):
        prompt = build_tier3_prompt(narrative_page)
        assert "https://example.com/v2/img" in prompt

    def test_contains_kg_terms(self, narrative_page: PageCoordinates):
        kg = {"WG-4420": "Widget Model 4420"}
        prompt = build_tier3_prompt(narrative_page, kg_terms=kg)
        assert "WG-4420" in prompt
        assert "Widget Model 4420" in prompt

    def test_empty_page(self, empty_page: PageCoordinates):
        prompt = build_tier3_prompt(empty_page)
        assert "Page text content:" in prompt


class TestExtractWithGemini:
    @pytest.mark.asyncio
    async def test_parses_mock_response(
        self, narrative_page: PageCoordinates, mock_gemini_result: PageExtractionResult
    ):
        mock_run_result = AsyncMock()
        mock_run_result.data = mock_gemini_result

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_run_result

        with patch(
            "src.extraction.tier3_gemini._get_agent",
            return_value=mock_agent,
        ):
            result = await extract_with_gemini(narrative_page)

        mock_agent.run.assert_called_once()
        prompt_arg = mock_agent.run.call_args[0][0]
        assert "Product" in prompt_arg
        assert "WG-4420-BLK" in prompt_arg

        assert len(result.extractions) == 2
        assert result.extractions[0].field_name == "product_number"
        assert result.extractions[0].value == "WG-4420-BLK"
        assert result.extractions[1].field_name == "quantity"
        assert result.cross_references == ["see Section 3.2"]

    @pytest.mark.asyncio
    async def test_empty_page_returns_empty(
        self, empty_page: PageCoordinates
    ):
        empty_result = PageExtractionResult(
            extractions=[], is_product_table=False, cross_references=[]
        )
        mock_run_result = AsyncMock()
        mock_run_result.data = empty_result

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_run_result

        with patch(
            "src.extraction.tier3_gemini._get_agent",
            return_value=mock_agent,
        ):
            result = await extract_with_gemini(empty_page)

        assert len(result.extractions) == 0
        assert result.cross_references == []
