"""Tests for Tier 1 vision table extraction."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.extraction.tier1_vision import extract_from_table
from src.models import (
    Cell,
    CellTypes,
    Classification,
    Column,
    PageCoordinates,
    Prediction,
    Rectangle,
    Row,
    Table,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def page_with_table() -> PageCoordinates:
    data = json.loads((FIXTURES / "sample_page_with_table.json").read_text())
    return PageCoordinates.model_validate(data)


class TestExtractFromTable:
    def test_correct_number_of_extractions(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, source_id = extract_from_table(
            table, page_with_table, "test.pdf"
        )
        # 1 data row x 2 columns = 2 extractions
        assert len(extractions) == 2

    def test_source_id_is_table_coordinate_id(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        _, source_id = extract_from_table(table, page_with_table, "test.pdf")
        assert source_id == table.coordinate_id

    def test_coordinate_id_matches_cell(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")

        data_cell_ids = {
            cell.coordinate_id
            for row_cells in table.cells[1:]  # skip header
            for cell in row_cells
        }
        for ext in extractions:
            assert ext.coordinate_id in data_cell_ids

    def test_field_name_matches_column_label(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")

        column_labels = {col.index: col.classification.label for col in table.columns}
        for ext in extractions:
            # Find the cell to get column_index
            cell = next(
                c
                for row in table.cells
                for c in row
                if c.coordinate_id == ext.coordinate_id
            )
            assert ext.field_name == column_labels[cell.column_index]

    def test_field_name_coordinates_id_matches_header(
        self, page_with_table: PageCoordinates
    ):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")

        header_ids = {cell.coordinate_id for cell in table.cells[0]}
        for ext in extractions:
            assert ext.field_name_coordinates_id in header_ids

    def test_extracted_value_matches_cell_content(
        self, page_with_table: PageCoordinates
    ):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")

        cell_contents = {
            cell.coordinate_id: cell.content
            for row in table.cells
            for cell in row
        }
        for ext in extractions:
            assert ext.extracted_value == cell_contents[ext.coordinate_id]

    def test_extraction_certainty_from_table(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")
        for ext in extractions:
            assert ext.extraction_certainty == table.detection_certainty

    def test_genai_score_is_zero(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")
        for ext in extractions:
            assert ext.genai_score == 0.0

    def test_source_of_extraction_is_pdf(self, page_with_table: PageCoordinates):
        table = page_with_table.tables[0]
        extractions, _ = extract_from_table(table, page_with_table, "test.pdf")
        for ext in extractions:
            assert ext.source_of_extraction == "pdf"

    def test_spanning_cell_skipped(self):
        """Spanning cells should be excluded from extractions."""
        table = _build_table_with_spanning_cell()
        page = _empty_page(table)
        extractions, _ = extract_from_table(table, page, "test.pdf")
        # Only non-spanning cells in data rows produce extractions
        assert all(
            ext.coordinate_id != uuid.UUID("00000000-0000-0000-0000-000000000099")
            for ext in extractions
        )


    def test_no_header_classification_falls_back_to_row_zero(self):
        """When no row is classified as 'header', row 0 is used as header."""
        table = _build_table_no_header_classification()
        page = _empty_page(table)
        extractions, _ = extract_from_table(table, page, "test.pdf")
        # 1 data row x 2 columns = 2 extractions
        assert len(extractions) == 2
        # field_name_raw should come from row 0 cells
        raw_names = {ext.field_name_raw for ext in extractions}
        assert raw_names == {"Col A", "Col B"}

    def test_header_only_table_returns_empty(self):
        """A table with only header row(s) produces no extractions."""
        table = _build_header_only_table()
        page = _empty_page(table)
        extractions, source_id = extract_from_table(table, page, "test.pdf")
        assert extractions == []
        assert source_id == table.coordinate_id


def _empty_page(table: Table) -> PageCoordinates:
    return PageCoordinates(
        image_url="",
        image_url_v2="",
        page_height=842.0,
        page_width=595.0,
        page_name="",
        extraction_word_boxes=[],
        word_boxes=[],
        tables=[table],
        layout_objects=[],
    )


def _build_header_only_table() -> Table:
    """Table with a single header row and no data rows."""
    rect = Rectangle(top=0.1, left=0.1, width=0.1, height=0.02, page=0)
    row_id = uuid.uuid4()
    col_id = uuid.uuid4()
    pred = [Prediction(label="header", certainty=0.99)]

    cell = Cell(
        type=CellTypes.NON_SPANNING_CELL,
        coordinates_rectangle=rect,
        coordinate_id=uuid.uuid4(),
        content="Header",
        column_index=0,
        row_index=0,
        column_id=col_id,
        row_id=row_id,
    )
    return Table(
        classification=Classification(label="table", certainty=0.95, predictions=pred),
        coordinates_rectangle=rect,
        coordinate_id=uuid.uuid4(),
        rows=[
            Row(
                classification=Classification(
                    label="header", certainty=0.99, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=row_id,
                index=0,
                detection_certainty=0.99,
                relations=[],
            ),
        ],
        columns=[
            Column(
                classification=Classification(
                    label="col_a", certainty=0.95, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=col_id,
                index=0,
                detection_certainty=0.95,
                relations=[],
            ),
        ],
        cells=[[cell]],
        cells_flatten=[cell],
        relations=[],
        detection_certainty=0.95,
    )


def _build_table_no_header_classification() -> Table:
    """2x2 table where NO row is classified as 'header' — fallback to row 0."""
    rect = Rectangle(top=0.1, left=0.1, width=0.1, height=0.02, page=0)
    row0_id = uuid.uuid4()
    row1_id = uuid.uuid4()
    col0_id = uuid.uuid4()
    col1_id = uuid.uuid4()
    pred = [Prediction(label="table_line", certainty=0.99)]

    row0_cells = [
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="Col A",
            column_index=0,
            row_index=0,
            column_id=col0_id,
            row_id=row0_id,
        ),
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="Col B",
            column_index=1,
            row_index=0,
            column_id=col1_id,
            row_id=row0_id,
        ),
    ]
    row1_cells = [
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="val1",
            column_index=0,
            row_index=1,
            column_id=col0_id,
            row_id=row1_id,
        ),
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="val2",
            column_index=1,
            row_index=1,
            column_id=col1_id,
            row_id=row1_id,
        ),
    ]

    return Table(
        classification=Classification(label="table", certainty=0.95, predictions=pred),
        coordinates_rectangle=rect,
        coordinate_id=uuid.uuid4(),
        rows=[
            Row(
                classification=Classification(
                    label="table_line", certainty=0.99, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=row0_id,
                index=0,
                detection_certainty=0.99,
                relations=[],
            ),
            Row(
                classification=Classification(
                    label="table_line", certainty=0.99, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=row1_id,
                index=1,
                detection_certainty=0.99,
                relations=[],
            ),
        ],
        columns=[
            Column(
                classification=Classification(
                    label="col_a", certainty=0.95, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=col0_id,
                index=0,
                detection_certainty=0.95,
                relations=[],
            ),
            Column(
                classification=Classification(
                    label="col_b", certainty=0.95, predictions=pred
                ),
                coordinates_rectangle=rect,
                coordinate_id=col1_id,
                index=1,
                detection_certainty=0.95,
                relations=[],
            ),
        ],
        cells=[row0_cells, row1_cells],
        cells_flatten=row0_cells + row1_cells,
        relations=[],
        detection_certainty=0.95,
    )


def _build_table_with_spanning_cell() -> Table:
    """Helper: 2x2 table where data row has one spanning cell."""
    rect = Rectangle(top=0.1, left=0.1, width=0.1, height=0.02, page=0)
    header_row_id = uuid.uuid4()
    data_row_id = uuid.uuid4()
    col0_id = uuid.uuid4()
    col1_id = uuid.uuid4()

    header_cells = [
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="Col A",
            column_index=0,
            row_index=0,
            column_id=col0_id,
            row_id=header_row_id,
        ),
        Cell(
            type=CellTypes.NON_SPANNING_CELL,
            coordinates_rectangle=rect,
            coordinate_id=uuid.uuid4(),
            content="Col B",
            column_index=1,
            row_index=0,
            column_id=col1_id,
            row_id=header_row_id,
        ),
    ]

    spanning_cell = Cell(
        type=CellTypes.SPANNING_CELL,
        coordinates_rectangle=rect,
        coordinate_id=uuid.UUID("00000000-0000-0000-0000-000000000099"),
        content="merged",
        column_index=0,
        row_index=1,
        column_id=col0_id,
        row_id=data_row_id,
    )
    normal_cell = Cell(
        type=CellTypes.NON_SPANNING_CELL,
        coordinates_rectangle=rect,
        coordinate_id=uuid.uuid4(),
        content="value",
        column_index=1,
        row_index=1,
        column_id=col1_id,
        row_id=data_row_id,
    )

    pred_h = [Prediction(label="header", certainty=0.99)]
    pred_d = [Prediction(label="table_line", certainty=0.99)]

    return Table(
        classification=Classification(
            label="table", certainty=0.95, predictions=pred_h
        ),
        coordinates_rectangle=rect,
        coordinate_id=uuid.uuid4(),
        rows=[
            Row(
                classification=Classification(
                    label="header", certainty=0.99, predictions=pred_h
                ),
                coordinates_rectangle=rect,
                coordinate_id=header_row_id,
                index=0,
                detection_certainty=0.99,
                relations=[],
            ),
            Row(
                classification=Classification(
                    label="table_line", certainty=0.99, predictions=pred_d
                ),
                coordinates_rectangle=rect,
                coordinate_id=data_row_id,
                index=1,
                detection_certainty=0.99,
                relations=[],
            ),
        ],
        columns=[
            Column(
                classification=Classification(
                    label="col_a", certainty=0.95, predictions=pred_h
                ),
                coordinates_rectangle=rect,
                coordinate_id=col0_id,
                index=0,
                detection_certainty=0.95,
                relations=[],
            ),
            Column(
                classification=Classification(
                    label="col_b", certainty=0.95, predictions=pred_h
                ),
                coordinates_rectangle=rect,
                coordinate_id=col1_id,
                index=1,
                detection_certainty=0.95,
                relations=[],
            ),
        ],
        cells=[header_cells, [spanning_cell, normal_cell]],
        cells_flatten=header_cells + [spanning_cell, normal_cell],
        relations=[],
        detection_certainty=0.95,
    )
