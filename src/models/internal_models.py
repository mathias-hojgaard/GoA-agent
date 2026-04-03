"""Internal pipeline models not part of the GA schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .ga_models import Extraction


class TaskBrief(BaseModel):
    page_filters: list[int] | None = None
    deadline: datetime | None = None
    special_instructions: list[str] = []
    sender_id: str | None = None
    attachment_filenames: list[str] = []


class PageClassification(BaseModel):
    page_index: int
    page_type: Literal[
        "product_table",
        "narrative",
        "cross_reference",
        "cover_page",
        "terms_and_conditions",
        "other",
    ]
    has_vision_table: bool
    vision_confidence: float
    relevant: bool


class DocumentMap(BaseModel):
    classifications: list[PageClassification] = []
    total_pages: int = 0
    relevant_pages: list[int] = []


class RawExtraction(BaseModel):
    """Used by Tier 3 before grounding."""

    field_name: str
    value: str
    field_label_text: str | None = None
    confidence: float = 0.0


class PageExtractionResult(BaseModel):
    extractions: list[RawExtraction] = []
    is_product_table: bool = False
    cross_references: list[str] = []


class TierResult(BaseModel):
    tier: Literal["tier1", "tier2", "tier3"]
    extractions: list[Extraction] = []
    page_index: int = 0
    grounded: bool = False
