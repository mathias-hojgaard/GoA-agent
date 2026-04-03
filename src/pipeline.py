"""Main pipeline: wire all modules into process_tender()."""

from __future__ import annotations

import logging
import time
import uuid

from src.config import GEMINI_SEMAPHORE
from src.extraction import extract_pages_parallel
from src.grounding import (
    compute_similarity_scores,
    ground_tier3_results,
    write_extraction_word_boxes,
)
from src.grounding.similarity import FewShotStore as GroundingFewShotStore
from src.ingestion import classify_pages, parse_email
from src.knowledge import NullKGClient, TenantKGClient, get_sender_patterns
from src.knowledge import FewShotStore as KnowledgeFewShotStore
from src.models import (
    DocumentMap,
    ExtractionOutput,
    FileCoordinates,
    TierResult,
)
from src.resolution import assemble_output

logger = logging.getLogger(__name__)


async def process_tender(
    email_body: str,
    file_coordinates: FileCoordinates,
    tenant_id: str = "default",
    kg_client: TenantKGClient | NullKGClient | None = None,
    fewshot_store: KnowledgeFewShotStore | None = None,
    gemini_semaphore: int = GEMINI_SEMAPHORE,
) -> tuple[ExtractionOutput, FileCoordinates]:
    """Run the full tender extraction pipeline.

    Stages:
        1. Email parsing → TaskBrief
        2. Page classification → DocumentMap
        3. Parallel extraction (3 tiers) → TierResults
        4. Coordinate grounding → updated PageCoordinates
        5. Validation + scoring
        6. Cross-reference resolution (MVP: log only)
        7. Product resolution (MVP: skip)
        8. Assembly → ExtractionOutput + FileCoordinates
    """
    pipeline_start = time.monotonic()

    # Deep-copy so callers can safely reuse the original FileCoordinates.
    file_coordinates = file_coordinates.model_copy(deep=True)

    if kg_client is None:
        kg_client = NullKGClient(tenant_id=tenant_id)

    # ── Stage 1: Email parsing ──────────────────────────────────────────
    stage_start = time.monotonic()
    try:
        sender_patterns = await get_sender_patterns(kg_client, tenant_id)
        task_brief = await parse_email(email_body, kg_context=sender_patterns)
        logger.info(
            "Stage 1 (email parsing) complete in %.2fs — "
            "page_filters=%s, deadline=%s, instructions=%d",
            time.monotonic() - stage_start,
            task_brief.page_filters,
            task_brief.deadline,
            len(task_brief.special_instructions),
        )
    except Exception:
        logger.exception("Stage 1 (email parsing) failed — using empty TaskBrief")
        from src.models import TaskBrief

        task_brief = TaskBrief()

    # ── Stage 2: Page classification ────────────────────────────────────
    stage_start = time.monotonic()
    try:
        doc_map = classify_pages(file_coordinates, task_brief)
        _log_page_distribution(doc_map, stage_start)
    except Exception:
        logger.exception("Stage 2 (page classification) failed — treating all pages as relevant")
        doc_map = _fallback_doc_map(file_coordinates)

    # ── Stage 3: Parallel extraction ────────────────────────────────────
    stage_start = time.monotonic()
    tier_results: list[TierResult] = []
    table_source_ids: dict[int, uuid.UUID] = {}
    try:
        pages = [
            (idx, file_coordinates.pages_coordinates[idx], doc_map.classifications[idx])
            for idx in doc_map.relevant_pages
            if idx < len(file_coordinates.pages_coordinates)
        ]
        tier_results = await extract_pages_parallel(
            pages, file_coordinates.filename, gemini_semaphore
        )
        # Track table_source_ids for Tier 1 pages.
        # Limitation: only captures the first table per page. Multi-table pages
        # flatten all extractions into one TierResult (see parallel.py).
        for tr in tier_results:
            if tr.tier == "tier1":
                page = file_coordinates.pages_coordinates[tr.page_index]
                if page.tables:
                    table_source_ids[tr.page_index] = page.tables[0].coordinate_id
        logger.info(
            "Stage 3 (extraction) complete in %.2fs — %d tier results: %s",
            time.monotonic() - stage_start,
            len(tier_results),
            _tier_summary(tier_results),
        )
    except Exception:
        logger.exception("Stage 3 (extraction) failed")

    # ── Stage 4: Coordinate grounding ───────────────────────────────────
    stage_start = time.monotonic()
    grounded_count = 0
    ungrounded_count = 0
    try:
        for tr in tier_results:
            page = file_coordinates.pages_coordinates[tr.page_index]

            # Ground Tier 3 raw extractions
            if not tr.grounded and tr.raw_extractions:
                grounded_extractions = ground_tier3_results(
                    tr.raw_extractions, page, file_coordinates.filename
                )
                tr.extractions = grounded_extractions
                tr.grounded = True

            # Write extraction word_boxes for ALL tiers
            if tr.extractions:
                updated_page = write_extraction_word_boxes(page, tr.extractions)
                file_coordinates.pages_coordinates[tr.page_index] = updated_page

            for ext in tr.extractions:
                if ext.coordinate_id != uuid.UUID(int=0):
                    grounded_count += 1
                else:
                    ungrounded_count += 1

        total = grounded_count + ungrounded_count
        rate = (grounded_count / total * 100) if total else 0
        logger.info(
            "Stage 4 (grounding) complete in %.2fs — "
            "%d grounded, %d ungrounded (%.1f%% rate)",
            time.monotonic() - stage_start,
            grounded_count,
            ungrounded_count,
            rate,
        )
    except Exception:
        logger.exception("Stage 4 (grounding) failed")

    # ── Stage 5: Validation + scoring ───────────────────────────────────
    stage_start = time.monotonic()
    try:
        all_extractions = [ext for tr in tier_results for ext in tr.extractions]

        if fewshot_store:
            # Build a GroundingFewShotStore from the knowledge store's entries
            grounding_store = GroundingFewShotStore(entries=fewshot_store.entries)
            compute_similarity_scores(all_extractions, grounding_store)

        # MVP: skip LLM validation to avoid API calls.
        # Enable with: from src.grounding import validate_extractions
        logger.info(
            "Stage 5 (validation) complete in %.2fs — %d extractions scored",
            time.monotonic() - stage_start,
            len(all_extractions),
        )
    except Exception:
        logger.exception("Stage 5 (validation/scoring) failed")

    # ── Stage 6: Cross-reference resolution ─────────────────────────────
    stage_start = time.monotonic()
    try:
        all_crossrefs = [
            ref for tr in tier_results for ref in tr.cross_references
        ]
        if all_crossrefs:
            # Enable with: from src.resolution import resolve_crossrefs
            logger.info(
                "Stage 6 (cross-refs) — %d references found: %s (MVP: skipping re-extraction)",
                len(all_crossrefs),
                all_crossrefs,
            )
        else:
            logger.info("Stage 6 (cross-refs) — no cross-references found")
    except Exception:
        logger.exception("Stage 6 (cross-reference resolution) failed")

    # ── Stage 7: Product resolution ─────────────────────────────────────
    # MVP: skip — product_number comes directly from extraction
    logger.info("Stage 7 (product resolution) — MVP skip")

    # ── Stage 8: Assembly ───────────────────────────────────────────────
    stage_start = time.monotonic()
    try:
        output, updated_fc = assemble_output(
            file_coordinates, tier_results, table_source_ids
        )
        logger.info(
            "Stage 8 (assembly) complete in %.2fs — "
            "%d products, %d attribute fields, %d addresses",
            time.monotonic() - stage_start,
            len(output.products),
            len(output.attributes.extractions),
            len(output.address),
        )
    except Exception:
        logger.exception("Stage 8 (assembly) failed — returning empty output")
        output, updated_fc = _empty_output(file_coordinates)

    total_extractions = sum(len(tr.extractions) for tr in tier_results)
    logger.info(
        "Pipeline complete in %.2fs — %d products, %d total extractions, "
        "grounding rate %.1f%%",
        time.monotonic() - pipeline_start,
        len(output.products),
        total_extractions,
        (grounded_count / (grounded_count + ungrounded_count) * 100)
        if (grounded_count + ungrounded_count)
        else 0,
    )

    return output, updated_fc


# ── Helpers ─────────────────────────────────────────────────────────────


def _log_page_distribution(doc_map: DocumentMap, stage_start: float) -> None:
    """Log stage 2 results."""
    type_counts: dict[str, int] = {}
    for cls in doc_map.classifications:
        type_counts[cls.page_type] = type_counts.get(cls.page_type, 0) + 1
    logger.info(
        "Stage 2 (classification) complete in %.2fs — "
        "%d total pages, %d relevant, types: %s",
        time.monotonic() - stage_start,
        doc_map.total_pages,
        len(doc_map.relevant_pages),
        dict(type_counts),
    )


def _tier_summary(tier_results: list[TierResult]) -> str:
    """Summarize tier distribution."""
    counts: dict[str, int] = {}
    for tr in tier_results:
        counts[tr.tier] = counts.get(tr.tier, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _fallback_doc_map(file_coordinates: FileCoordinates) -> DocumentMap:
    """Create a DocumentMap that treats all pages as relevant product_table pages."""
    from src.models import PageClassification

    classifications = []
    for i, page in enumerate(file_coordinates.pages_coordinates):
        has_table = bool(page.tables)
        classifications.append(
            PageClassification(
                page_index=i,
                page_type="product_table" if has_table else "narrative",
                has_vision_table=has_table,
                vision_confidence=0.9 if has_table else 0.0,
                relevant=True,
            )
        )
    return DocumentMap(
        classifications=classifications,
        total_pages=len(file_coordinates.pages_coordinates),
        relevant_pages=list(range(len(file_coordinates.pages_coordinates))),
    )


def _empty_output(
    file_coordinates: FileCoordinates,
) -> tuple[ExtractionOutput, FileCoordinates]:
    """Build an empty ExtractionOutput for error fallback."""
    from src.models import (
        Attributes,
        ExtractionOutput,
        ExtractionSource,
        ImageResolution,
        Meta,
    )

    ct = file_coordinates.content_type.lower()
    if "pdf" in ct:
        file_type = "PDF"
        source = ExtractionSource.PDF
    elif "image" in ct:
        file_type = "IMAGE"
        source = ExtractionSource.IMAGE
    elif "excel" in ct or "spreadsheet" in ct:
        file_type = "EXCEL"
        source = ExtractionSource.EXCEL
    else:
        file_type = ct.split("/")[-1].upper()
        source = ExtractionSource.PDF

    meta = Meta(
        filename=file_coordinates.filename,
        file_type=file_type,
        content_id=file_coordinates.content_id,
        doc_class=file_coordinates.doc_class,
        accuracy="",
        doc_class_certainty=0.0,
        image_resolution=ImageResolution(x_pixel=2480, y_pixel=3508),
        document_warnings=[],
    )
    output = ExtractionOutput(
        products=[],
        attributes=Attributes(extractions={}),
        address=[],
        meta=meta,
        keyword_extractions={},
        source_of_extraction=source,
    )
    return output, file_coordinates
