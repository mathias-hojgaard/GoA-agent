"""Stage 2: Page classifier.

Deterministic classification of pages using Vision Model output (tables,
word_boxes) plus heuristics. No LLM calls — this is pure Python logic.
"""

from __future__ import annotations

import re

from src.config import VISION_CONFIDENCE_THRESHOLD
from src.models import (
    DocumentMap,
    FileCoordinates,
    PageClassification,
    PageCoordinates,
    Table,
    TaskBrief,
)

_CROSSREF_PATTERN = re.compile(
    r"\b(see|ref|annex|section|appendix|page)\b", re.IGNORECASE
)

_TOC_PATTERN = re.compile(
    r"(table\s+of\s+contents|contents|index|\.{3,}|\d+\s*\.{2,}\s*\d+)",
    re.IGNORECASE,
)


def classify_table_type(table: Table) -> str:
    """Classify a table's type by examining its column labels.

    Known GA column labels: "position", "product_number", "quantity",
    "unit_price", "total_price".

    Returns:
        "product_table" if a "product_number" column exists,
        "cross_reference" if cells contain cross-ref patterns,
        "other" for generic/description tables.
    """
    for col in table.columns:
        if col.classification.label == "product_number":
            return "product_table"

    # Check cells for cross-reference patterns
    for cell in table.cells_flatten:
        if cell.content and _CROSSREF_PATTERN.search(cell.content):
            return "cross_reference"

    return "other"


def classify_page(
    page: PageCoordinates, page_index: int
) -> PageClassification:
    """Classify a single page based on its Vision Model output.

    Args:
        page: The PageCoordinates from FileCoordinates.
        page_index: 0-indexed page position.

    Returns:
        A PageClassification with the detected page type.
    """
    has_tables = len(page.tables) > 0
    max_certainty = 0.0
    page_type = "other"

    if has_tables:
        max_certainty = max(t.detection_certainty for t in page.tables)

        if max_certainty > VISION_CONFIDENCE_THRESHOLD:
            # High-confidence table — classify by column labels
            table_types = [classify_table_type(t) for t in page.tables]
            if "product_table" in table_types:
                page_type = "product_table"
            elif "cross_reference" in table_types:
                page_type = "cross_reference"
            else:
                # Generic table with no product_number column → "other"
                page_type = "other"
        else:
            # Low-confidence table → needs Tier 2 extraction
            page_type = "other"
    else:
        # No tables — classify by word_box count and content
        word_count = len(page.word_boxes)

        if word_count > 50:
            page_type = "narrative"
        else:
            # Few word boxes — check for TOC-like patterns
            text = " ".join(wb.value for wb in page.word_boxes)
            if _TOC_PATTERN.search(text):
                page_type = "cover_page"
            else:
                page_type = "other"

    return PageClassification(
        page_index=page_index,
        page_type=page_type,
        has_vision_table=has_tables,
        vision_confidence=max_certainty,
        relevant=True,  # Default; classify_pages applies filters
    )


def classify_pages(
    file_coords: FileCoordinates, task_brief: TaskBrief
) -> DocumentMap:
    """Classify all pages and apply page filters from the TaskBrief.

    Page filter conversion: TaskBrief.page_filters uses 1-indexed page
    numbers (as written in the email by the sender). FileCoordinates uses
    0-indexed. This function converts 1-indexed → 0-indexed internally.

    Args:
        file_coords: The FileCoordinates with all pages.
        task_brief: The TaskBrief with optional page_filters.

    Returns:
        A DocumentMap with classifications for every page and a list of
        relevant (filtered) page indices.
    """
    total_pages = len(file_coords.pages_coordinates)

    # Convert 1-indexed email page numbers to 0-indexed (skip invalid ≤0)
    allowed_indices: set[int] | None = None
    if task_brief.page_filters:
        allowed_indices = {p - 1 for p in task_brief.page_filters if p >= 1}

    classifications: list[PageClassification] = []

    for idx, page in enumerate(file_coords.pages_coordinates):
        classification = classify_page(page, idx)

        # Apply page filters: mark non-matching pages as irrelevant
        if allowed_indices is not None and idx not in allowed_indices:
            classification.relevant = False

        classifications.append(classification)

    relevant_pages = [c.page_index for c in classifications if c.relevant]

    return DocumentMap(
        classifications=classifications,
        total_pages=total_pages,
        relevant_pages=relevant_pages,
    )
