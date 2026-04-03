"""Coordinate grounding, write-back, similarity scoring, and validation."""

from .coordinate_matcher import (
    ground_extraction,
    ground_field_label,
    ground_tier3_results,
    merge_rectangles,
)
from .similarity import FewShotStore, compute_similarity_scores
from .validation import validate_extractions
from .writeback import write_extraction_word_boxes

__all__ = [
    "FewShotStore",
    "compute_similarity_scores",
    "ground_extraction",
    "ground_field_label",
    "ground_tier3_results",
    "merge_rectangles",
    "validate_extractions",
    "write_extraction_word_boxes",
]
