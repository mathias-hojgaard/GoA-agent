"""Tests for coordinate grounding, write-back, and similarity scoring."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.grounding.coordinate_matcher import (
    ground_extraction,
    ground_field_label,
    ground_tier3_results,
    merge_rectangles,
)
from src.grounding.similarity import FewShotStore, compute_similarity_scores
from src.grounding.writeback import write_extraction_word_boxes
from src.models.ga_models import (
    Extraction,
    FileCoordinates,
    PageCoordinates,
    Rectangle,
    WordBox,
)
from src.models.internal_models import RawExtraction

FIXTURES = Path(__file__).parent / "fixtures"
_ZERO_UUID = uuid.UUID(int=0)


@pytest.fixture
def file_coords() -> FileCoordinates:
    data = json.loads((FIXTURES / "sample_file_coordinates.json").read_text())
    return FileCoordinates(**data)


@pytest.fixture
def page(file_coords: FileCoordinates) -> PageCoordinates:
    return file_coords.pages_coordinates[0]


@pytest.fixture
def word_boxes(page: PageCoordinates) -> list[WordBox]:
    return page.word_boxes


# ── ground_extraction tests ──────────────────────────────────────


class TestGroundExtraction:
    def test_exact_match(self, word_boxes: list[WordBox]):
        """WG-4420-BLK is in word_boxes and should match exactly."""
        coord_id, rects = ground_extraction("WG-4420-BLK", word_boxes)
        assert coord_id is not None
        assert coord_id == uuid.UUID("01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1")
        assert len(rects) == 1

    def test_fuzzy_match_spaces(self, word_boxes: list[WordBox]):
        """WG 4420 BLK (spaces) should fuzzy-match WG-4420-BLK."""
        coord_id, rects = ground_extraction("WG 4420 BLK", word_boxes, threshold=0.7)
        assert coord_id is not None

    def test_multi_word(self, word_boxes: list[WordBox]):
        """'Purchase Order' spans two word_boxes → merged rectangle."""
        coord_id, rects = ground_extraction("Purchase Order", word_boxes)
        assert coord_id is not None
        assert len(rects) == 2
        merged = merge_rectangles(rects)
        # Should span from "Purchase" to "Order"
        assert merged.width > rects[0].width

    def test_no_match(self, word_boxes: list[WordBox]):
        """Nonexistent value returns (None, [])."""
        coord_id, rects = ground_extraction("NONEXISTENT", word_boxes)
        assert coord_id is None
        assert rects == []

    def test_number_format(self, word_boxes: list[WordBox]):
        """'12.50' (dot) should match '12,50' (comma) in word_boxes."""
        coord_id, rects = ground_extraction("12.50", word_boxes)
        assert coord_id is not None
        assert coord_id == uuid.UUID("01914a3b-5e6f-7f00-b8c9-d0e1f2a3b4c5")

    def test_duplicate_values_hint(self, word_boxes: list[WordBox]):
        """Two '500' on page — hint_rectangle selects correct one."""
        # There's only one "500" in our fixture, so let's add a duplicate
        # at a different Y-coordinate
        extra_500 = WordBox(
            coordinates_rectangle=[
                Rectangle(top=0.800, left=0.620, width=0.045, height=0.012, page=0)
            ],
            coordinate_id=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            value="500",
        )
        boxes = list(word_boxes) + [extra_500]

        # Hint near Y=0.345 should prefer the original
        hint = Rectangle(top=0.345, left=0.600, width=0.05, height=0.01, page=0)
        coord_id, rects = ground_extraction("500", boxes, hint_rectangle=hint)
        assert coord_id == uuid.UUID("01914a3b-3c4d-7f00-f6a7-b8c9d0e1f2a3")

    def test_empty_inputs(self):
        assert ground_extraction("", []) == (None, [])
        assert ground_extraction("test", []) == (None, [])
        assert ground_extraction("", [WordBox(
            coordinates_rectangle=[Rectangle(top=0, left=0, width=0.1, height=0.01, page=0)],
            coordinate_id=uuid.uuid4(),
            value="test",
        )]) == (None, [])


# ── merge_rectangles tests ───────────────────────────────────────


class TestMergeRectangles:
    def test_single(self):
        r = Rectangle(top=0.1, left=0.2, width=0.3, height=0.04, page=0)
        merged = merge_rectangles([r])
        assert merged.top == r.top
        assert merged.left == r.left
        assert merged.width == r.width
        assert merged.height == pytest.approx(r.height)

    def test_multiple(self):
        r1 = Rectangle(top=0.1, left=0.2, width=0.1, height=0.02, page=0)
        r2 = Rectangle(top=0.1, left=0.35, width=0.1, height=0.02, page=0)
        merged = merge_rectangles([r1, r2])
        assert merged.left == 0.2
        assert abs(merged.width - 0.25) < 1e-9

    def test_empty(self):
        merged = merge_rectangles([])
        assert merged.width == 0.0


# ── ground_field_label tests ─────────────────────────────────────


class TestGroundFieldLabel:
    def test_label_above_value(self, word_boxes: list[WordBox]):
        """'Article No.' should be found above the WG-4420-BLK value rect."""
        value_rect = Rectangle(
            top=0.345, left=0.112, width=0.098, height=0.012, page=0
        )
        label_id = ground_field_label("Article No.", word_boxes, value_rect)
        assert label_id is not None
        assert label_id == uuid.UUID("01914a3b-2b3c-7f00-e5f6-a7b8c9d0e1f2")

    def test_no_label(self, word_boxes: list[WordBox]):
        """No matching label returns None."""
        value_rect = Rectangle(top=0.5, left=0.5, width=0.1, height=0.01, page=0)
        label_id = ground_field_label("NonexistentLabel", word_boxes, value_rect)
        assert label_id is None

    def test_multi_word_label_single_box(self, word_boxes: list[WordBox]):
        """'Unit Price' is a single word_box in the fixture — should match."""
        value_rect = Rectangle(
            top=0.345, left=0.720, width=0.055, height=0.012, page=0
        )
        label_id = ground_field_label("Unit Price", word_boxes, value_rect)
        assert label_id is not None
        assert label_id == uuid.UUID("01914a3b-6f70-7f00-c9d0-e1f2a3b4c5d6")

    def test_multi_word_label_split_boxes(self):
        """Label spans two separate word_boxes — sliding window should match."""
        wb_unit = WordBox(
            coordinates_rectangle=[
                Rectangle(top=0.10, left=0.50, width=0.04, height=0.01, page=0)
            ],
            coordinate_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            value="Unit",
        )
        wb_price = WordBox(
            coordinates_rectangle=[
                Rectangle(top=0.10, left=0.55, width=0.05, height=0.01, page=0)
            ],
            coordinate_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            value="Price",
        )
        value_rect = Rectangle(
            top=0.20, left=0.50, width=0.06, height=0.01, page=0
        )
        label_id = ground_field_label(
            "Unit Price", [wb_unit, wb_price], value_rect
        )
        assert label_id == uuid.UUID("11111111-1111-1111-1111-111111111111")

    def test_empty_label(self, word_boxes: list[WordBox]):
        value_rect = Rectangle(top=0.5, left=0.5, width=0.1, height=0.01, page=0)
        assert ground_field_label("", word_boxes, value_rect) is None


# ── ground_tier3_results tests ───────────────────────────────────


class TestGroundTier3:
    def test_grounded_extraction(self, page: PageCoordinates):
        raw = [
            RawExtraction(
                field_name="product_number",
                value="WG-4420-BLK",
                field_label_text="Article No.",
                confidence=0.95,
            )
        ]
        results = ground_tier3_results(raw, page, "test.pdf")
        assert len(results) == 1
        ext = results[0]
        assert ext.coordinate_id != _ZERO_UUID
        assert ext.field_name_coordinates_id != _ZERO_UUID
        assert ext.extraction_certainty == pytest.approx(0.95 * 0.9)
        assert ext.coordinates_rectangle

    def test_ungrounded_extraction(self, page: PageCoordinates):
        raw = [
            RawExtraction(
                field_name="mystery",
                value="NONEXISTENT_VALUE",
                confidence=0.8,
            )
        ]
        results = ground_tier3_results(raw, page, "test.pdf")
        assert len(results) == 1
        ext = results[0]
        assert ext.coordinate_id == _ZERO_UUID
        assert ext.extraction_certainty == pytest.approx(0.8 * 0.5)


# ── write_extraction_word_boxes tests ────────────────────────────


class TestWriteback:
    def test_writeback_adds_entries(self, page: PageCoordinates):
        initial_count = len(page.extraction_word_boxes)
        coord_id = uuid.UUID("01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1")
        ext = Extraction(
            source_of_extraction="pdf",
            filename="test.pdf",
            extraction_certainty=0.95,
            similarity_to_confirmed_extractions=0.0,
            genai_score=0.9,
            coordinate_id=coord_id,
            field_name="product_number",
            field_name_raw="Article No.",
            field_name_coordinates_id=_ZERO_UUID,
            raw_saga_extraction="",
            raw_extracted_value="WG-4420-BLK",
            extracted_value="WG-4420-BLK",
            relations=[],
            coordinates_rectangle=[
                Rectangle(top=0.345, left=0.112, width=0.098, height=0.012, page=0)
            ],
            message="",
            advanced_validation=[],
        )
        updated = write_extraction_word_boxes(page, [ext])
        assert len(updated.extraction_word_boxes) == initial_count + 1
        new_wb = updated.extraction_word_boxes[-1]
        assert new_wb.coordinate_id == coord_id
        assert new_wb.value == "WG-4420-BLK"

    def test_writeback_skips_zero_uuid(self, page: PageCoordinates):
        initial_count = len(page.extraction_word_boxes)
        ext = Extraction(
            source_of_extraction="pdf",
            filename="test.pdf",
            extraction_certainty=0.5,
            similarity_to_confirmed_extractions=0.0,
            genai_score=0.5,
            coordinate_id=_ZERO_UUID,
            field_name="mystery",
            field_name_raw="",
            field_name_coordinates_id=_ZERO_UUID,
            raw_saga_extraction="",
            raw_extracted_value="unknown",
            extracted_value="unknown",
            relations=[],
            coordinates_rectangle=[],
            message="",
            advanced_validation=[],
        )
        updated = write_extraction_word_boxes(page, [ext])
        assert len(updated.extraction_word_boxes) == initial_count

    def test_writeback_skips_duplicates(self, page: PageCoordinates):
        """Don't add if coordinate_id already exists in extraction_word_boxes."""
        # The fixture already has one entry in extraction_word_boxes
        existing_id = page.extraction_word_boxes[0].coordinate_id
        ext = Extraction(
            source_of_extraction="pdf",
            filename="test.pdf",
            extraction_certainty=0.95,
            similarity_to_confirmed_extractions=0.0,
            genai_score=0.9,
            coordinate_id=existing_id,
            field_name="test",
            field_name_raw="",
            field_name_coordinates_id=_ZERO_UUID,
            raw_saga_extraction="",
            raw_extracted_value="duplicate",
            extracted_value="duplicate",
            relations=[],
            coordinates_rectangle=[
                Rectangle(top=0.05, left=0.06, width=0.1, height=0.01, page=0)
            ],
            message="",
            advanced_validation=[],
        )
        initial_count = len(page.extraction_word_boxes)
        write_extraction_word_boxes(page, [ext])
        assert len(page.extraction_word_boxes) == initial_count


# ── Similarity scoring tests ─────────────────────────────────────


class TestSimilarity:
    def test_empty_store(self):
        store = FewShotStore()
        assert store.find_similar("WG-4420-BLK", "product_number") == 0.0

    def test_matching_entries(self):
        store = FewShotStore(
            entries=[
                {"field_name": "product_number", "value": "WG-4420-BLK"},
                {"field_name": "product_number", "value": "WG-4420-WHT"},
            ]
        )
        score = store.find_similar("WG-4420-BLK", "product_number")
        assert score > 0.9

    def test_different_field(self):
        store = FewShotStore(
            entries=[{"field_name": "quantity", "value": "500"}]
        )
        score = store.find_similar("WG-4420-BLK", "product_number")
        assert score == 0.0

    def test_compute_similarity_scores(self):
        store = FewShotStore(
            entries=[{"field_name": "product_number", "value": "WG-4420-BLK"}]
        )
        ext = Extraction(
            source_of_extraction="pdf",
            filename="test.pdf",
            extraction_certainty=0.95,
            similarity_to_confirmed_extractions=0.0,
            genai_score=0.9,
            coordinate_id=_ZERO_UUID,
            field_name="product_number",
            field_name_raw="",
            field_name_coordinates_id=_ZERO_UUID,
            raw_saga_extraction="",
            raw_extracted_value="WG-4420-BLK",
            extracted_value="WG-4420-BLK",
            relations=[],
            coordinates_rectangle=[],
            message="",
            advanced_validation=[],
        )
        updated = compute_similarity_scores([ext], store)
        assert updated[0].similarity_to_confirmed_extractions > 0.9
