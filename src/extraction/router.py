"""Confidence-based router that selects the extraction tier for each page."""

from __future__ import annotations

from typing import Literal

from src.config import VISION_CONFIDENCE_THRESHOLD
from src.models import PageClassification, PageCoordinates

CONFIDENCE_THRESHOLD = VISION_CONFIDENCE_THRESHOLD


def route_page(
    page: PageCoordinates,  # reserved for future heuristics (e.g. word_box count)
    classification: PageClassification,
) -> Literal["tier1", "tier2", "tier3"]:
    """Decide which extraction tier to use for a page.

    - tier1: High-confidence vision table detected
    - tier2: Table detected but below confidence threshold
    - tier3: No table structure — use LLM extraction
    """
    if (
        classification.has_vision_table
        and classification.vision_confidence > CONFIDENCE_THRESHOLD
    ):
        return "tier1"

    if classification.has_vision_table or "table" in classification.page_type:
        return "tier2"

    return "tier3"
