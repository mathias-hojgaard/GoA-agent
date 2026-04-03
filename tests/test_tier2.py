"""Tests for Tier 2 spatial clustering extraction."""

from __future__ import annotations

import uuid

import pytest

from src.extraction.tier2_spatial import (
    X_GAP_THRESHOLD,
    Y_TOLERANCE,
    extract_from_word_boxes,
    match_field_label,
    merge_bounding_boxes,
)
from src.models import Rectangle, WordBox


def _wb(top: float, left: float, width: float, value: str, page: int = 0) -> WordBox:
    """Helper to create a WordBox with minimal boilerplate."""
    return WordBox(
        coordinates_rectangle=[
            Rectangle(top=top, left=left, width=width, height=0.012, page=page)
        ],
        coordinate_id=uuid.uuid4(),
        value=value,
    )


@pytest.fixture
def table_word_boxes() -> list[WordBox]:
    """3 rows x 4 columns arranged with clear gaps.

    Row 0 (header): Pos | Article | Qty | Price
    Row 1 (data):   10  | WG-4420 | 500 | 12.50
    Row 2 (data):   20  | BL-1100 | 200 | 8.75
    """
    boxes = [
        # Header row (top=0.10)
        _wb(0.100, 0.05, 0.03, "Pos"),
        _wb(0.100, 0.15, 0.06, "Article"),
        _wb(0.100, 0.35, 0.03, "Qty"),
        _wb(0.100, 0.55, 0.04, "Price"),
        # Data row 1 (top=0.13)
        _wb(0.130, 0.05, 0.02, "10"),
        _wb(0.130, 0.15, 0.06, "WG-4420"),
        _wb(0.130, 0.35, 0.03, "500"),
        _wb(0.130, 0.55, 0.04, "12.50"),
        # Data row 2 (top=0.16)
        _wb(0.160, 0.05, 0.02, "20"),
        _wb(0.160, 0.15, 0.06, "BL-1100"),
        _wb(0.160, 0.35, 0.03, "200"),
        _wb(0.160, 0.55, 0.04, "8.75"),
    ]
    return boxes


class TestRowClustering:
    def test_correct_number_of_rows(self, table_word_boxes: list[WordBox]):
        extractions = extract_from_word_boxes(
            table_word_boxes, 842.0, 595.0, "test.pdf"
        )
        # 2 data rows x 4 columns = 8 extractions
        assert len(extractions) == 8

    def test_header_identified(self, table_word_boxes: list[WordBox]):
        extractions = extract_from_word_boxes(
            table_word_boxes, 842.0, 595.0, "test.pdf"
        )
        # All extractions should have field names derived from header
        field_names = {ext.field_name for ext in extractions}
        assert "position_number" in field_names
        assert "quantity" in field_names


class TestColumnBoundaries:
    def test_four_columns_detected(self, table_word_boxes: list[WordBox]):
        extractions = extract_from_word_boxes(
            table_word_boxes, 842.0, 595.0, "test.pdf"
        )
        # Should have 4 distinct field names
        field_names = {ext.field_name for ext in extractions}
        assert len(field_names) == 4

    def test_values_correct(self, table_word_boxes: list[WordBox]):
        extractions = extract_from_word_boxes(
            table_word_boxes, 842.0, 595.0, "test.pdf"
        )
        values = {ext.extracted_value for ext in extractions}
        assert "10" in values
        assert "WG-4420" in values
        assert "500" in values
        assert "12.50" in values


class TestMergeBoundingBoxes:
    def test_single_box(self):
        wb = _wb(0.1, 0.2, 0.05, "hello")
        rect = merge_bounding_boxes([wb])
        assert rect.top == pytest.approx(0.1)
        assert rect.left == pytest.approx(0.2)
        assert rect.width == pytest.approx(0.05)
        assert rect.height == pytest.approx(0.012)

    def test_multiple_boxes(self):
        boxes = [
            _wb(0.10, 0.20, 0.05, "hello"),
            _wb(0.10, 0.30, 0.08, "world"),
        ]
        rect = merge_bounding_boxes(boxes)
        assert rect.top == pytest.approx(0.10)
        assert rect.left == pytest.approx(0.20)
        assert rect.width == pytest.approx(0.18)  # 0.38 - 0.20
        assert rect.height == pytest.approx(0.012)

    def test_different_rows(self):
        boxes = [
            _wb(0.10, 0.20, 0.05, "a"),
            _wb(0.15, 0.25, 0.05, "b"),
        ]
        rect = merge_bounding_boxes(boxes)
        assert rect.top == pytest.approx(0.10)
        assert rect.left == pytest.approx(0.20)
        # height spans from 0.10 to 0.15+0.012 = 0.162
        assert rect.height == pytest.approx(0.062)


class TestMatchFieldLabel:
    def test_exact_match(self):
        assert match_field_label("Pos") == "position_number"
        assert match_field_label("qty") == "quantity"
        assert match_field_label("Price") == "unit_price"

    def test_fuzzy_match(self):
        assert match_field_label("Positon") == "position_number"  # typo
        assert match_field_label("Quantty") == "quantity"  # typo

    def test_no_match(self):
        assert match_field_label("foobar") is None
        assert match_field_label("xyz") is None


class TestEdgeCases:
    def test_empty_word_boxes(self):
        result = extract_from_word_boxes([], 842.0, 595.0, "test.pdf")
        assert result == []

    def test_single_column_no_gaps(self):
        """Single column page — no X-gaps to split on."""
        boxes = [
            _wb(0.10, 0.05, 0.03, "Position"),
            _wb(0.13, 0.05, 0.02, "10"),
            _wb(0.16, 0.05, 0.02, "20"),
        ]
        # Only 1 header cell matches known labels → not enough (need ≥2)
        result = extract_from_word_boxes(boxes, 842.0, 595.0, "test.pdf")
        assert result == []

    def test_certainty_is_tier2_value(self, table_word_boxes: list[WordBox]):
        extractions = extract_from_word_boxes(
            table_word_boxes, 842.0, 595.0, "test.pdf"
        )
        for ext in extractions:
            assert ext.extraction_certainty == 0.7
