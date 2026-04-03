"""Parallel page extraction runner with semaphore-limited concurrency."""

from __future__ import annotations

import asyncio
import logging

from src.models import (
    Extraction,
    PageClassification,
    PageCoordinates,
    TierResult,
)

from .router import route_page
from .tier1_vision import extract_from_table
from .tier2_spatial import extract_from_word_boxes
from .tier3_gemini import extract_with_gemini

logger = logging.getLogger(__name__)


async def extract_pages_parallel(
    pages: list[tuple[int, PageCoordinates, PageClassification]],
    filename: str,
    semaphore_limit: int = 15,
) -> list[TierResult]:
    """Extract data from multiple pages concurrently.

    Each page is routed to the appropriate tier, extracted, and wrapped
    in a TierResult. Individual page failures are logged and skipped.
    """
    semaphore = asyncio.Semaphore(semaphore_limit)

    async def _process_page(
        page_index: int,
        page: PageCoordinates,
        classification: PageClassification,
    ) -> TierResult | None:
        try:
            return await _extract_single_page(
                page_index, page, classification, filename, semaphore
            )
        except Exception:
            logger.exception("Failed to extract page %d", page_index)
            return None

    tasks = [
        _process_page(page_index, page, classification)
        for page_index, page, classification in pages
    ]

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def _extract_single_page(
    page_index: int,
    page: PageCoordinates,
    classification: PageClassification,
    filename: str,
    semaphore: asyncio.Semaphore,
) -> TierResult:
    """Route and extract a single page."""
    tier = route_page(page, classification)

    if tier == "tier1":
        all_extractions: list[Extraction] = []
        for table in page.tables:
            extractions, _ = extract_from_table(table, page, filename)
            all_extractions.extend(extractions)
        return TierResult(
            tier="tier1",
            extractions=all_extractions,
            page_index=page_index,
            grounded=True,
        )

    if tier == "tier2":
        extractions = extract_from_word_boxes(
            page.word_boxes, page.page_height, page.page_width, filename
        )
        return TierResult(
            tier="tier2",
            extractions=extractions,
            page_index=page_index,
            grounded=True,
        )

    # tier3 — semaphore limits concurrent Gemini API calls
    async with semaphore:
        page_result = await extract_with_gemini(page)
    return TierResult(
        tier="tier3",
        raw_extractions=page_result.extractions,
        cross_references=page_result.cross_references,
        page_index=page_index,
        grounded=False,
    )
