"""Coordinate grounding: map extracted values back to word_boxes in FileCoordinates."""

from __future__ import annotations

import re
import uuid

from rapidfuzz import fuzz

from src.config import GROUNDING_THRESHOLD, LABEL_GROUNDING_THRESHOLD
from src.models.enums import ExtractionSource
from src.models.ga_models import (
    Extraction,
    PageCoordinates,
    Rectangle,
    WordBox,
)
from src.models.internal_models import RawExtraction


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, normalize number separators."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_number(text: str) -> str:
    """Normalize decimal separators: '12.50' → '12,50' for EU-format matching.

    Only replaces a dot that looks like a decimal separator (dot followed by
    1-2 digits at end of string), not thousand-separator dots or IP addresses.
    """
    normalized = _normalize(text)
    normalized = re.sub(r"\.(\d{1,2})$", r",\1", normalized)
    return normalized


def merge_rectangles(rects: list[Rectangle]) -> Rectangle:
    """Merge multiple bounding boxes into one encompassing rectangle."""
    if not rects:
        return Rectangle(top=0.0, left=0.0, width=0.0, height=0.0, page=0)

    min_top = min(r.top for r in rects)
    min_left = min(r.left for r in rects)
    max_right = max(r.left + r.width for r in rects)
    max_bottom = max(r.top + r.height for r in rects)

    return Rectangle(
        top=min_top,
        left=min_left,
        width=max_right - min_left,
        height=max_bottom - min_top,
        page=rects[0].page,
    )


def ground_extraction(
    raw_value: str,
    word_boxes: list[WordBox],
    threshold: float = GROUNDING_THRESHOLD,
    hint_rectangle: Rectangle | None = None,
) -> tuple[uuid.UUID | None, list[Rectangle]]:
    """Ground an extracted value against word_boxes using fuzzy matching.

    Returns (coordinate_id, rectangles) of the best match, or (None, []) if
    no match exceeds the threshold.
    """
    if not word_boxes or not raw_value.strip():
        return (None, [])

    normalized_value = _normalize(raw_value)
    normalized_value_num = _normalize_number(raw_value)

    best_score = 0.0
    best_match: tuple[uuid.UUID | None, list[Rectangle]] = (None, [])

    max_window = min(len(word_boxes), 10)

    for window_size in range(1, max_window + 1):
        for start in range(len(word_boxes) - window_size + 1):
            window = word_boxes[start : start + window_size]
            concatenated = " ".join(wb.value for wb in window)
            norm_concat = _normalize(concatenated)
            norm_concat_num = _normalize_number(concatenated)

            # Score using both normal and number-normalized forms
            score_normal = fuzz.ratio(normalized_value, norm_concat) / 100.0
            score_number = fuzz.ratio(normalized_value_num, norm_concat_num) / 100.0
            score = max(score_normal, score_number)

            if score < threshold:
                continue

            # Apply hint_rectangle proximity penalty to distant matches
            effective_score = score
            if hint_rectangle is not None:
                rects = [wb.coordinates_rectangle[0] for wb in window]
                merged = merge_rectangles(rects)
                y_distance = abs(merged.top - hint_rectangle.top)
                if y_distance > 0.1:
                    effective_score -= 0.1  # penalize distant matches

            if effective_score > best_score:
                best_score = effective_score
                coord_id = window[0].coordinate_id
                rects = [wb.coordinates_rectangle[0] for wb in window]
                best_match = (coord_id, rects)

                if best_score >= 0.98:
                    return best_match  # perfect enough, stop searching

    return best_match


def ground_field_label(
    label_text: str,
    word_boxes: list[WordBox],
    value_rect: Rectangle,
    threshold: float = LABEL_GROUNDING_THRESHOLD,
) -> uuid.UUID | None:
    """Ground a field label (e.g. column header) against word_boxes.

    Supports multi-word labels like "Unit Price" by using a sliding window
    over spatially-filtered candidates.
    Prefers candidates above the value (column header) over those to the left
    (key-value pattern).
    """
    if not label_text.strip() or not word_boxes:
        return None

    normalized_label = _normalize(label_text)
    best_score = 0.0
    best_id: uuid.UUID | None = None

    # Partition candidates by spatial relationship to value
    above: list[WordBox] = []
    left: list[WordBox] = []

    for wb in word_boxes:
        wb_rect = wb.coordinates_rectangle[0]
        is_above = wb_rect.top < value_rect.top
        is_left = (
            wb_rect.left < value_rect.left
            and abs(wb_rect.top - value_rect.top) < 0.05
        )
        if is_above:
            above.append(wb)
        elif is_left:
            left.append(wb)

    # Check each group with sliding windows; above gets position bonus
    for candidates, position_bonus in [(above, 0.1), (left, 0.0)]:
        max_window = min(len(candidates), 4)
        for window_size in range(1, max_window + 1):
            for start in range(len(candidates) - window_size + 1):
                window = candidates[start : start + window_size]
                concatenated = " ".join(wb.value for wb in window)
                norm_concat = _normalize(concatenated)
                score = fuzz.ratio(normalized_label, norm_concat) / 100.0

                if score < threshold:
                    continue

                effective_score = score + position_bonus
                if effective_score > best_score:
                    best_score = effective_score
                    best_id = window[0].coordinate_id

    return best_id


_ZERO_UUID = uuid.UUID(int=0)


def ground_tier3_results(
    raw_extractions: list[RawExtraction],
    page: PageCoordinates,
    filename: str,
) -> list[Extraction]:
    """Ground Tier 3 (LLM) raw extractions against page word_boxes.

    Builds full Extraction objects with coordinate_id linkage.
    Ungrounded extractions get Zero UUID.
    """
    results: list[Extraction] = []

    for raw in raw_extractions:
        coord_id, rects = ground_extraction(raw.value, page.word_boxes)
        grounded = coord_id is not None

        # Ground the field label if available
        label_coord_id = _ZERO_UUID
        if raw.field_label_text and rects:
            merged = merge_rectangles(rects)
            found_label = ground_field_label(
                raw.field_label_text, page.word_boxes, merged
            )
            if found_label is not None:
                label_coord_id = found_label

        # Penalize ungrounded extractions
        certainty = raw.confidence * (0.9 if grounded else 0.5)

        extraction = Extraction(
            source_of_extraction=ExtractionSource.PDF,
            filename=filename,
            extraction_certainty=certainty,
            similarity_to_confirmed_extractions=0.0,
            genai_score=raw.confidence,
            coordinate_id=coord_id or _ZERO_UUID,
            field_name=raw.field_name,
            field_name_raw=raw.field_label_text or "",
            field_name_coordinates_id=label_coord_id,
            raw_saga_extraction="",
            raw_extracted_value=raw.value,
            extracted_value=raw.value,
            relations=[],
            coordinates_rectangle=rects,
            message="",
            advanced_validation=[],
        )
        results.append(extraction)

    return results
