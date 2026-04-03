"""Write-back: update PageCoordinates.extraction_word_boxes with grounded extractions."""

from __future__ import annotations

import uuid

from src.models.ga_models import Extraction, PageCoordinates, WordBox

_ZERO_UUID = uuid.UUID(int=0)


def write_extraction_word_boxes(
    page: PageCoordinates,
    extractions: list[Extraction],
) -> PageCoordinates:
    """Append grounded extractions to page.extraction_word_boxes.

    Skips ungrounded extractions (coordinate_id == Zero UUID) and duplicates
    (coordinate_id already present in extraction_word_boxes).

    Mutates page in place and returns it.
    """
    existing_ids = {wb.coordinate_id for wb in page.extraction_word_boxes}

    for ext in extractions:
        if ext.coordinate_id == _ZERO_UUID:
            continue
        if ext.coordinate_id in existing_ids:
            continue

        rect = ext.coordinates_rectangle[:1] if ext.coordinates_rectangle else []
        wb = WordBox(
            value=ext.extracted_value,
            coordinate_id=ext.coordinate_id,
            coordinates_rectangle=rect,
        )
        page.extraction_word_boxes.append(wb)
        existing_ids.add(ext.coordinate_id)

    return page
