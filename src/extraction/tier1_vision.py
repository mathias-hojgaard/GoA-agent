"""Tier 1: Deterministic extraction from vision-detected tables."""

from __future__ import annotations

import uuid

from src.models import (
    Cell,
    CellTypes,
    Extraction,
    ExtractionSource,
    PageCoordinates,
    Rectangle,
    Relation,
    Table,
)


def extract_from_table(
    table: Table, page: PageCoordinates, filename: str
) -> tuple[list[Extraction], uuid.UUID]:
    """Extract structured data from a vision-detected table.

    Returns a list of Extraction objects (one per data cell) and the
    table's coordinate_id for use as OrderLine.source_id.
    """
    header_row_indices = _find_header_rows(table)
    if not header_row_indices:
        header_row_indices = {0}
    else:
        header_row_indices = set(header_row_indices)

    primary_header_idx = min(header_row_indices)
    columns_by_index = {col.index: col for col in table.columns}
    rows_by_index = {row.index: row for row in table.rows}
    extractions: list[Extraction] = []

    for row_idx, row_cells in enumerate(table.cells):
        row = rows_by_index.get(row_idx)
        if row is None or row.index in header_row_indices:
            continue

        for cell in row_cells:
            if cell.type == CellTypes.SPANNING_CELL:
                continue

            header_cell = _get_header_cell(table, primary_header_idx, cell.column_index)
            if header_cell is None:
                continue

            column = columns_by_index.get(cell.column_index)
            if column is None:
                continue

            row_relations = [
                Relation(
                    related_id=other_cell.coordinate_id,
                    relation_type="relation",
                    head_id=cell.coordinate_id,
                    tail_id=other_cell.coordinate_id,
                    relation_certainty=row.detection_certainty,
                )
                for other_cell in row_cells
                if other_cell.coordinate_id != cell.coordinate_id
                and other_cell.type != CellTypes.SPANNING_CELL
            ]

            extraction = Extraction(
                coordinate_id=cell.coordinate_id,
                field_name=column.classification.label,
                field_name_raw=header_cell.content,
                field_name_coordinates_id=header_cell.coordinate_id,
                raw_extracted_value=cell.content,
                extracted_value=cell.content,
                raw_saga_extraction="",
                coordinates_rectangle=[cell.coordinates_rectangle],
                source_of_extraction=ExtractionSource.PDF,
                filename=filename,
                extraction_certainty=table.detection_certainty,
                genai_score=0.0,
                similarity_to_confirmed_extractions=0.0,
                relations=row_relations,
                message="",
                advanced_validation=[],
            )
            extractions.append(extraction)

    return extractions, table.coordinate_id


def _find_header_rows(table: Table) -> list[int]:
    """Return indices of rows classified as headers."""
    return [
        row.index
        for row in table.rows
        if row.classification.label == "header"
    ]


def _get_header_cell(
    table: Table, header_row_idx: int, column_index: int
) -> Cell | None:
    """Get the header cell for a given column index."""
    if header_row_idx >= len(table.cells):
        return None
    for cell in table.cells[header_row_idx]:
        if cell.column_index == column_index:
            return cell
    return None
