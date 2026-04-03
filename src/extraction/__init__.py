"""Extraction tiers, router, and parallel runner."""

from .parallel import extract_pages_parallel
from .router import CONFIDENCE_THRESHOLD, route_page
from .tier1_vision import extract_from_table
from .tier2_spatial import (
    extract_from_word_boxes,
    match_field_label,
    merge_bounding_boxes,
)
from .tier3_gemini import extract_with_gemini

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "extract_from_table",
    "extract_from_word_boxes",
    "extract_pages_parallel",
    "extract_with_gemini",
    "match_field_label",
    "merge_bounding_boxes",
    "route_page",
]
