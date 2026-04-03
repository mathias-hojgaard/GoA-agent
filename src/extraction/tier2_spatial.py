"""Tier 2: Deterministic extraction via spatial clustering of word boxes."""

from __future__ import annotations

from rapidfuzz import fuzz

from src.config import X_GAP_THRESHOLD as _X_GAP, Y_CLUSTERING_TOLERANCE as _Y_TOL
from src.models import (
    Extraction,
    ExtractionSource,
    Rectangle,
    WordBox,
)

Y_TOLERANCE = _Y_TOL
X_GAP_THRESHOLD = _X_GAP
TIER2_CERTAINTY = 0.7

KNOWN_LABELS: set[str] = {
    "pos",
    "position",
    "article",
    "product",
    "description",
    "qty",
    "quantity",
    "price",
    "unit",
    "amount",
    "total",
}

_LABEL_MAP: dict[str, str] = {
    "pos": "position_number",
    "position": "position_number",
    "article": "product_number",
    "product": "product_number",
    "description": "description",
    "qty": "quantity",
    "quantity": "quantity",
    "price": "unit_price",
    "unit": "unit_price",
    "amount": "total_amount",
    "total": "total_amount",
}


def extract_from_word_boxes(
    word_boxes: list[WordBox],
    page_height: float,  # reserved — coordinates are already normalized 0.0–1.0
    page_width: float,  # reserved — coordinates are already normalized 0.0–1.0
    filename: str,
) -> list[Extraction]:
    """Extract structured data by spatially clustering word boxes into a table."""
    if not word_boxes:
        return []

    rows = _cluster_into_rows(word_boxes)
    rows = [_sort_row_by_left(row) for row in rows]
    grid = [_split_row_into_cells(row) for row in rows]

    if len(grid) < 2:
        return []

    header_row_idx = _identify_header_row(grid)
    if header_row_idx is None:
        return []

    header_cells = grid[header_row_idx]
    extractions: list[Extraction] = []

    for row_idx, row_cells in enumerate(grid):
        if row_idx == header_row_idx:
            continue
        for col_idx, cell_boxes in enumerate(row_cells):
            if col_idx >= len(header_cells) or not cell_boxes:
                continue

            header_text = _cell_text(header_cells[col_idx])
            field_name = match_field_label(header_text) or header_text.lower().strip()
            cell_text = _cell_text(cell_boxes)
            bbox = merge_bounding_boxes(cell_boxes)
            header_bbox_first = header_cells[col_idx][0]

            extraction = Extraction(
                coordinate_id=cell_boxes[0].coordinate_id,
                field_name=field_name,
                field_name_raw=header_text,
                field_name_coordinates_id=header_bbox_first.coordinate_id,
                raw_extracted_value=cell_text,
                extracted_value=cell_text,
                raw_saga_extraction="",
                coordinates_rectangle=[bbox],
                source_of_extraction=ExtractionSource.PDF,
                filename=filename,
                extraction_certainty=TIER2_CERTAINTY,
                genai_score=0.0,
                similarity_to_confirmed_extractions=0.0,
                relations=[],
                message="",
                advanced_validation=[],
            )
            extractions.append(extraction)

    return extractions


def merge_bounding_boxes(word_boxes: list[WordBox]) -> Rectangle:
    """Compute the merged bounding box for a list of WordBoxes."""
    tops = [wb.coordinates_rectangle[0].top for wb in word_boxes]
    lefts = [wb.coordinates_rectangle[0].left for wb in word_boxes]
    rights = [
        wb.coordinates_rectangle[0].left + wb.coordinates_rectangle[0].width
        for wb in word_boxes
    ]
    bottoms = [
        wb.coordinates_rectangle[0].top + wb.coordinates_rectangle[0].height
        for wb in word_boxes
    ]

    min_top = min(tops)
    min_left = min(lefts)
    max_right = max(rights)
    max_bottom = max(bottoms)

    return Rectangle(
        top=min_top,
        left=min_left,
        width=max_right - min_left,
        height=max_bottom - min_top,
        page=word_boxes[0].coordinates_rectangle[0].page,
    )


def match_field_label(text: str) -> str | None:
    """Fuzzy-match text against known field labels. Returns canonical name or None."""
    text_lower = text.lower().strip()

    # Exact match first
    if text_lower in KNOWN_LABELS:
        return _LABEL_MAP[text_lower]

    # Fuzzy match
    best_score = 0.0
    best_label = None
    for label in KNOWN_LABELS:
        score = fuzz.ratio(text_lower, label)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= 75 and best_label is not None:
        return _LABEL_MAP[best_label]

    return None


# ── Internal helpers ──────────────────────────────────────────────


def _cluster_into_rows(word_boxes: list[WordBox]) -> list[list[WordBox]]:
    """Group word boxes into rows by vertical proximity.

    Uses running mean of the cluster's top values so that gradual drift
    within a row doesn't cause a spurious split.
    """
    sorted_boxes = sorted(word_boxes, key=lambda wb: wb.coordinates_rectangle[0].top)

    rows: list[list[WordBox]] = []
    current_row: list[WordBox] = [sorted_boxes[0]]
    current_top_sum = sorted_boxes[0].coordinates_rectangle[0].top

    for wb in sorted_boxes[1:]:
        wb_top = wb.coordinates_rectangle[0].top
        mean_top = current_top_sum / len(current_row)
        if abs(wb_top - mean_top) < Y_TOLERANCE:
            current_row.append(wb)
            current_top_sum += wb_top
        else:
            rows.append(current_row)
            current_row = [wb]
            current_top_sum = wb_top

    rows.append(current_row)
    return rows


def _sort_row_by_left(row: list[WordBox]) -> list[WordBox]:
    return sorted(row, key=lambda wb: wb.coordinates_rectangle[0].left)


def _split_row_into_cells(row: list[WordBox]) -> list[list[WordBox]]:
    """Split a sorted row into cells based on X-gaps."""
    if not row:
        return []

    cells: list[list[WordBox]] = [[row[0]]]

    for wb in row[1:]:
        prev = cells[-1][-1]
        prev_right = (
            prev.coordinates_rectangle[0].left + prev.coordinates_rectangle[0].width
        )
        current_left = wb.coordinates_rectangle[0].left
        gap = current_left - prev_right

        if gap > X_GAP_THRESHOLD:
            cells.append([wb])
        else:
            cells[-1].append(wb)

    return cells


def _cell_text(boxes: list[WordBox]) -> str:
    """Concatenate word box values with spaces."""
    return " ".join(wb.value for wb in boxes)


def _identify_header_row(grid: list[list[list[WordBox]]]) -> int | None:
    """Check if the first row contains known field labels."""
    first_row = grid[0]
    matches = sum(
        1
        for cell_boxes in first_row
        if cell_boxes and match_field_label(_cell_text(cell_boxes)) is not None
    )
    if matches >= 2:
        return 0
    return None
