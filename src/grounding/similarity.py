"""Similarity scoring against confirmed extractions (few-shot store)."""

from __future__ import annotations

from rapidfuzz import fuzz

from src.models.ga_models import Extraction


class FewShotStore:
    """Simple in-memory store of confirmed extraction values for similarity scoring."""

    def __init__(self, entries: list[dict] | None = None):
        self.entries = entries or []

    def find_similar(self, value: str, field_name: str) -> float:
        """Return max similarity score (0.0-1.0) against confirmed extractions."""
        if not self.entries:
            return 0.0

        matching = [e for e in self.entries if e.get("field_name") == field_name]
        if not matching:
            return 0.0

        best = max(
            fuzz.ratio(value.lower(), e.get("value", "").lower()) for e in matching
        )
        return best / 100.0


def compute_similarity_scores(
    extractions: list[Extraction],
    store: FewShotStore,
) -> list[Extraction]:
    """Update similarity_to_confirmed_extractions for each extraction."""
    for ext in extractions:
        score = store.find_similar(ext.extracted_value, ext.field_name)
        ext.similarity_to_confirmed_extractions = score
    return extractions
