"""Tests for the extraction tier router."""

from __future__ import annotations

import pytest

from src.extraction.router import CONFIDENCE_THRESHOLD, route_page
from src.models import PageClassification, PageCoordinates, Rectangle, WordBox


@pytest.fixture
def dummy_page() -> PageCoordinates:
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


def _classification(
    has_table: bool, confidence: float, page_type: str = "product_table"
) -> PageClassification:
    return PageClassification(
        page_index=0,
        page_type=page_type,
        has_vision_table=has_table,
        vision_confidence=confidence,
        relevant=True,
    )


class TestRouterDecisions:
    def test_high_confidence_table_routes_tier1(self, dummy_page: PageCoordinates):
        cls = _classification(has_table=True, confidence=0.95)
        assert route_page(dummy_page, cls) == "tier1"

    def test_exactly_at_threshold_routes_tier2(self, dummy_page: PageCoordinates):
        """Confidence == 0.85 does NOT exceed threshold (strict >)."""
        cls = _classification(has_table=True, confidence=CONFIDENCE_THRESHOLD)
        assert route_page(dummy_page, cls) == "tier2"

    def test_just_above_threshold_routes_tier1(self, dummy_page: PageCoordinates):
        cls = _classification(has_table=True, confidence=0.8500001)
        assert route_page(dummy_page, cls) == "tier1"

    def test_low_confidence_table_routes_tier2(self, dummy_page: PageCoordinates):
        cls = _classification(has_table=True, confidence=0.60)
        assert route_page(dummy_page, cls) == "tier2"

    def test_no_table_but_product_table_page_type_routes_tier2(
        self, dummy_page: PageCoordinates
    ):
        """has_vision_table=False but page_type contains 'table' → tier2."""
        cls = _classification(
            has_table=False, confidence=0.0, page_type="product_table"
        )
        assert route_page(dummy_page, cls) == "tier2"

    def test_narrative_page_routes_tier3(self, dummy_page: PageCoordinates):
        cls = _classification(has_table=False, confidence=0.0, page_type="narrative")
        assert route_page(dummy_page, cls) == "tier3"

    def test_cover_page_routes_tier3(self, dummy_page: PageCoordinates):
        cls = _classification(has_table=False, confidence=0.0, page_type="cover_page")
        assert route_page(dummy_page, cls) == "tier3"

    def test_boundary_084_routes_tier2(self, dummy_page: PageCoordinates):
        """0.84 with has_vision_table → tier2 (below threshold)."""
        cls = _classification(has_table=True, confidence=0.84)
        assert route_page(dummy_page, cls) == "tier2"

    def test_confidence_threshold_value(self):
        assert CONFIDENCE_THRESHOLD == 0.85
