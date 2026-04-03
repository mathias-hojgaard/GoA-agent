"""Tests for the page classifier."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.ingestion.page_classifier import classify_page, classify_pages, classify_table_type
from src.models import (
    DocumentMap,
    FileCoordinates,
    PageClassification,
    PageCoordinates,
    TaskBrief,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def page_with_table() -> PageCoordinates:
    data = json.loads((FIXTURES / "sample_page_with_table.json").read_text())
    return PageCoordinates.model_validate(data)


@pytest.fixture
def page_narrative() -> PageCoordinates:
    data = json.loads((FIXTURES / "sample_page_narrative.json").read_text())
    return PageCoordinates.model_validate(data)


# ── classify_table_type tests ──────────────────────────────────────


class TestClassifyTableType:
    def test_product_number_column(self, page_with_table: PageCoordinates):
        """Table with a 'product_number' column → 'product_table'."""
        table = page_with_table.tables[0]
        assert classify_table_type(table) == "product_table"

    def test_crossref_cells(self, page_with_table: PageCoordinates):
        """Table with cross-ref text in cells → 'cross_reference'."""
        table = page_with_table.tables[0]
        # Mutate: remove product_number label, add cross-ref content
        for col in table.columns:
            col.classification.label = "generic"
        table.cells_flatten[0].content = "see Annex B for details"
        assert classify_table_type(table) == "cross_reference"

    def test_generic_table(self, page_with_table: PageCoordinates):
        """Table with generic columns and no cross-refs → 'other'."""
        table = page_with_table.tables[0]
        for col in table.columns:
            col.classification.label = "generic"
        for cell in table.cells_flatten:
            cell.content = "plain value"
        assert classify_table_type(table) == "other"


# ── classify_page tests ───────────────────────────────────────────


class TestClassifyPage:
    def test_page_with_high_confidence_product_table(
        self, page_with_table: PageCoordinates
    ):
        """Page with high-confidence table containing product_number → product_table."""
        result = classify_page(page_with_table, page_index=0)
        assert isinstance(result, PageClassification)
        assert result.page_type == "product_table"
        assert result.has_vision_table is True
        assert result.vision_confidence > 0.85
        assert result.relevant is True

    def test_page_with_low_confidence_table(
        self, page_with_table: PageCoordinates
    ):
        """Page with table below certainty threshold → 'other'."""
        # Lower the detection certainty below threshold
        page_with_table.tables[0].detection_certainty = 0.5
        result = classify_page(page_with_table, page_index=0)
        assert result.page_type == "other"
        assert result.has_vision_table is True
        assert result.vision_confidence == 0.5

    def test_narrative_page_few_words_no_toc(self, page_narrative: PageCoordinates):
        """Page with no tables, few word_boxes, no TOC patterns → 'other'."""
        result = classify_page(page_narrative, page_index=4)
        assert result.page_type == "other"
        assert result.has_vision_table is False
        assert result.vision_confidence == 0.0

    def test_narrative_page_many_words(self, page_narrative: PageCoordinates):
        """Page with no tables and >50 word_boxes → narrative."""
        # Add enough word boxes to exceed threshold
        from src.models.ga_models import Rectangle, WordBox

        for i in range(60):
            page_narrative.word_boxes.append(
                WordBox(
                    coordinates_rectangle=[
                        Rectangle(
                            top=0.1 + i * 0.01,
                            left=0.1,
                            width=0.05,
                            height=0.01,
                        )
                    ],
                    coordinate_id=uuid.uuid4(),
                    value=f"word{i}",
                )
            )
        result = classify_page(page_narrative, page_index=4)
        assert result.page_type == "narrative"

    def test_toc_page_detected_as_cover(self, page_narrative: PageCoordinates):
        """Page with TOC-like patterns and few word_boxes → 'cover_page'."""
        from src.models.ga_models import Rectangle, WordBox

        # Replace word boxes with TOC-like content
        page_narrative.word_boxes = [
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.05, left=0.05, width=0.3, height=0.01)
                ],
                coordinate_id=uuid.uuid4(),
                value="Table",
            ),
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.05, left=0.36, width=0.1, height=0.01)
                ],
                coordinate_id=uuid.uuid4(),
                value="of",
            ),
            WordBox(
                coordinates_rectangle=[
                    Rectangle(top=0.05, left=0.47, width=0.2, height=0.01)
                ],
                coordinate_id=uuid.uuid4(),
                value="Contents",
            ),
        ]
        result = classify_page(page_narrative, page_index=0)
        assert result.page_type == "cover_page"

    def test_page_index_preserved(self, page_with_table: PageCoordinates):
        """Page index is correctly stored in classification."""
        result = classify_page(page_with_table, page_index=7)
        assert result.page_index == 7


# ── classify_pages tests ──────────────────────────────────────────


def _make_file_coords(pages: list[PageCoordinates]) -> FileCoordinates:
    """Helper to build a FileCoordinates with the given pages."""
    return FileCoordinates(
        filename="test.pdf",
        content_id=uuid.uuid4(),
        content_type="application/pdf",
        doc_class="purchase_order",
        size=1024,
        date_archetype="EU",
        date_format="DD/MM/YYYY",
        decimal_separator=",",
        pages_coordinates=pages,
    )


class TestClassifyPages:
    def test_all_pages_relevant_when_no_filters(
        self,
        page_with_table: PageCoordinates,
        page_narrative: PageCoordinates,
    ):
        """Without page_filters, all pages are marked relevant."""
        fc = _make_file_coords([page_with_table, page_narrative])
        task = TaskBrief()

        doc_map = classify_pages(fc, task)
        assert isinstance(doc_map, DocumentMap)
        assert doc_map.total_pages == 2
        assert len(doc_map.classifications) == 2
        assert doc_map.relevant_pages == [0, 1]
        assert all(c.relevant for c in doc_map.classifications)

    def test_page_filters_1indexed_to_0indexed(
        self,
        page_with_table: PageCoordinates,
        page_narrative: PageCoordinates,
    ):
        """page_filters=[2] (1-indexed from email) → only page index 1 is relevant."""
        fc = _make_file_coords([page_with_table, page_narrative])
        task = TaskBrief(page_filters=[2])

        doc_map = classify_pages(fc, task)
        assert doc_map.relevant_pages == [1]
        assert doc_map.classifications[0].relevant is False
        assert doc_map.classifications[1].relevant is True

    def test_multiple_page_filters(
        self,
        page_with_table: PageCoordinates,
        page_narrative: PageCoordinates,
    ):
        """page_filters=[1, 2] → both pages relevant."""
        fc = _make_file_coords([page_with_table, page_narrative])
        task = TaskBrief(page_filters=[1, 2])

        doc_map = classify_pages(fc, task)
        assert doc_map.relevant_pages == [0, 1]

    def test_page_filter_out_of_range(
        self, page_with_table: PageCoordinates
    ):
        """page_filters=[99] with only 1 page → no pages relevant."""
        fc = _make_file_coords([page_with_table])
        task = TaskBrief(page_filters=[99])

        doc_map = classify_pages(fc, task)
        assert doc_map.relevant_pages == []
        assert doc_map.classifications[0].relevant is False

    def test_page_filter_zero_ignored(
        self, page_with_table: PageCoordinates
    ):
        """page_filters=[0] (invalid 1-indexed) is silently skipped, not converted to -1."""
        fc = _make_file_coords([page_with_table])
        task = TaskBrief(page_filters=[0])

        doc_map = classify_pages(fc, task)
        # 0 is invalid in 1-indexed space → no pages match → none relevant
        assert doc_map.relevant_pages == []
        assert doc_map.classifications[0].relevant is False

    def test_page_filter_zero_with_valid(
        self, page_with_table: PageCoordinates
    ):
        """page_filters=[0, 1] — the 0 is skipped, page 1 (index 0) is kept."""
        fc = _make_file_coords([page_with_table])
        task = TaskBrief(page_filters=[0, 1])

        doc_map = classify_pages(fc, task)
        assert doc_map.relevant_pages == [0]
        assert doc_map.classifications[0].relevant is True

    def test_empty_file_coordinates(self):
        """FileCoordinates with 0 pages → empty DocumentMap."""
        fc = _make_file_coords([])
        task = TaskBrief()

        doc_map = classify_pages(fc, task)
        assert doc_map.total_pages == 0
        assert doc_map.classifications == []
        assert doc_map.relevant_pages == []

    def test_classification_types_preserved(
        self,
        page_with_table: PageCoordinates,
        page_narrative: PageCoordinates,
    ):
        """Verify each page gets the correct classification type."""
        fc = _make_file_coords([page_with_table, page_narrative])
        task = TaskBrief()

        doc_map = classify_pages(fc, task)
        assert doc_map.classifications[0].page_type == "product_table"
        # Narrative fixture has <50 word_boxes and no TOC patterns → other
        assert doc_map.classifications[1].page_type == "other"
