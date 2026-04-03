"""Final ExtractionOutput assembler.

Groups tier results into OrderLines, Attributes, and Addresses,
applies value normalization, builds Meta, and produces a valid
ExtractionOutput with correct FileCoordinates linkage.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from src.models import (
    Address,
    Attributes,
    Extraction,
    ExtractionOutput,
    ExtractionSource,
    FileCoordinates,
    ImageResolution,
    Meta,
    OrderLine,
    TierResult,
)

from .normalizer import normalize_value

# Document-level attribute fields.
_ATTRIBUTE_FIELDS = {"order_number", "order_date", "currency", "delivery_terms"}

# Address-related fields.
_ADDRESS_FIELDS = {"company_name", "street", "city", "postal_code", "country"}

# Tolerance for grouping extractions by vertical position (fraction of page).
_ROW_Y_TOLERANCE = 0.005


def assemble_output(
    file_coords: FileCoordinates,
    tier_results: list[TierResult],
    table_source_ids: dict[int, uuid.UUID],
) -> tuple[ExtractionOutput, FileCoordinates]:
    """Assemble a valid ExtractionOutput from tier results.

    Args:
        file_coords: The document's FileCoordinates.
        tier_results: List of TierResult from all extraction tiers.
        table_source_ids: Mapping of page_index -> Table.coordinate_id for Tier 1.

    Returns:
        Tuple of (ExtractionOutput, FileCoordinates) with validated linkage.
    """
    # Step 1 — Group extractions into OrderLines.
    order_lines = _build_order_lines(tier_results, table_source_ids)

    # Step 2 — Build Attributes from document-level fields.
    attributes = _build_attributes(tier_results)

    # Step 3 — Build Address list.
    addresses = _build_addresses(tier_results)

    # Step 4 — Apply normalization to all extractions.
    date_format = file_coords.date_format
    decimal_sep = file_coords.decimal_separator

    for ol in order_lines:
        _normalize_extraction_dict(ol.extractions, date_format, decimal_sep)
    _normalize_extraction_dict(attributes.extractions, date_format, decimal_sep)
    for addr in addresses:
        _normalize_extraction_dict(addr.extractions, date_format, decimal_sep)

    # Step 5 — Build Meta.
    meta = _build_meta(file_coords)

    # Step 6 — Build ExtractionOutput.
    output = ExtractionOutput(
        products=order_lines,
        attributes=attributes,
        address=addresses,
        meta=meta,
        keyword_extractions={},
        source_of_extraction=_derive_extraction_source(file_coords.content_type),
    )

    # Step 7 — Validate round-trip and content_id linkage.
    ExtractionOutput(**output.model_dump())
    if meta.content_id != file_coords.content_id:
        raise ValueError(
            f"content_id mismatch: meta={meta.content_id}, "
            f"file_coords={file_coords.content_id}"
        )

    return output, file_coords


def _row_key(ext: Extraction) -> float:
    """Extract the vertical position (top) used to group extractions by row.

    Returns the top coordinate of the first rectangle, or -1.0 if none.
    """
    if ext.coordinates_rectangle:
        return ext.coordinates_rectangle[0].top
    return -1.0


def _group_by_row(extractions: list[Extraction]) -> list[list[Extraction]]:
    """Group extractions into rows by vertical position proximity.

    Extractions whose top coordinates are within _ROW_Y_TOLERANCE of each
    other are considered to be on the same row.
    """
    if not extractions:
        return []

    sorted_exts = sorted(extractions, key=_row_key)
    rows: list[list[Extraction]] = [[sorted_exts[0]]]

    for ext in sorted_exts[1:]:
        current_top = _row_key(ext)
        group_top = _row_key(rows[-1][0])

        if abs(current_top - group_top) <= _ROW_Y_TOLERANCE:
            rows[-1].append(ext)
        else:
            rows.append([ext])

    return rows


def _build_order_lines(
    tier_results: list[TierResult],
    table_source_ids: dict[int, uuid.UUID],
) -> list[OrderLine]:
    """Group extractions into OrderLines — one per product row."""
    zero_uuid = uuid.UUID(int=0)

    # Collect product extractions per (tier, page_index).
    tier_page_groups: dict[tuple[str, int], list[Extraction]] = defaultdict(list)

    for tr in tier_results:
        for ext in tr.extractions:
            field = ext.field_name.lower()
            if field in _ATTRIBUTE_FIELDS or field in _ADDRESS_FIELDS:
                continue
            tier_page_groups[(tr.tier, tr.page_index)].append(ext)

    if not tier_page_groups:
        return []

    order_lines: list[OrderLine] = []

    # Process each (tier, page) group.  Sort keys for determinism.
    for (tier, page_index) in sorted(tier_page_groups.keys()):
        exts = tier_page_groups[(tier, page_index)]

        # Determine source_id for this tier/page.
        if tier == "tier1":
            source_id = table_source_ids.get(page_index, zero_uuid)
        else:
            source_id = zero_uuid

        # Group extractions by row (vertical position).
        rows = _group_by_row(exts)

        for row_exts in rows:
            row_dict: dict[str, list[Extraction]] = {}
            for ext in row_exts:
                row_dict.setdefault(ext.field_name, []).append(ext)

            order_lines.append(
                OrderLine(
                    extractions=row_dict,
                    source_id=source_id,
                    advanced_validation=[],
                )
            )

    return order_lines


def _build_attributes(tier_results: list[TierResult]) -> Attributes:
    """Extract document-level attribute fields."""
    attr_extractions: dict[str, list[Extraction]] = {}

    for tr in tier_results:
        for ext in tr.extractions:
            if ext.field_name.lower() in _ATTRIBUTE_FIELDS:
                attr_extractions.setdefault(ext.field_name, []).append(ext)

    return Attributes(extractions=attr_extractions)


def _build_addresses(tier_results: list[TierResult]) -> list[Address]:
    """Extract address-related fields into Address objects."""
    addr_extractions: dict[str, list[Extraction]] = {}

    for tr in tier_results:
        for ext in tr.extractions:
            if ext.field_name.lower() in _ADDRESS_FIELDS:
                addr_extractions.setdefault(ext.field_name, []).append(ext)

    if not addr_extractions:
        return []

    return [Address(extractions=addr_extractions)]


def _normalize_extraction_dict(
    extractions: dict[str, list[Extraction]],
    date_format: str,
    decimal_separator: str,
) -> None:
    """Apply normalization in-place to all extractions in a dict."""
    for field_name, ext_list in extractions.items():
        for ext in ext_list:
            ext.extracted_value = normalize_value(
                ext.raw_extracted_value,
                field_name,
                date_format,
                decimal_separator,
            )


def _derive_extraction_source(content_type: str) -> ExtractionSource:
    """Derive the ExtractionSource enum from the document's MIME content_type."""
    ct = content_type.lower()
    if "pdf" in ct:
        return ExtractionSource.PDF
    if "image" in ct:
        return ExtractionSource.IMAGE
    if "excel" in ct or "spreadsheet" in ct:
        return ExtractionSource.EXCEL
    return ExtractionSource.PDF


def _build_meta(file_coords: FileCoordinates) -> Meta:
    """Build Meta from FileCoordinates."""
    # Derive file_type from content_type.
    content_type = file_coords.content_type.lower()
    if "pdf" in content_type:
        file_type = "PDF"
    elif "image" in content_type:
        file_type = "IMAGE"
    elif "excel" in content_type or "spreadsheet" in content_type:
        file_type = "EXCEL"
    else:
        file_type = content_type.split("/")[-1].upper()

    # Default image resolution if no pages available.
    image_resolution = ImageResolution(x_pixel=2480, y_pixel=3508)

    return Meta(
        filename=file_coords.filename,
        file_type=file_type,
        content_id=file_coords.content_id,
        doc_class=file_coords.doc_class,
        accuracy="",
        doc_class_certainty=0.0,
        image_resolution=image_resolution,
        document_warnings=[],
    )
