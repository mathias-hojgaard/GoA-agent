"""Product Resolver integration.

Builds context from extractions and calls the GA Product Resolver API.
MVP returns a mock response; production would HTTP POST to the real API.
"""

from __future__ import annotations

from src.models import Extraction


def _first_value(extractions: dict[str, list[Extraction]], field: str) -> str | None:
    """Get the extracted_value of the first Extraction for a field, or None."""
    entries = extractions.get(field, [])
    return entries[0].extracted_value if entries else None


def build_resolver_context(
    order_line_extractions: dict[str, list[Extraction]],
    attributes: dict[str, list[Extraction]],
    kg_past_resolutions: list[dict] | None = None,
) -> dict:
    """Assemble context dict for the Product Resolver API.

    Args:
        order_line_extractions: Extractions from a single OrderLine.
        attributes: Document-level attribute extractions.
        kg_past_resolutions: Optional past resolutions from the knowledge graph.

    Returns:
        Context dict ready for the Product Resolver API.

    Expected request format for production API:
        POST /api/v1/resolve-product
        {
            "product_number": "WG-4420-BLK",
            "description": "...",
            "quantity": "500",
            "unit_price": "12.50",
            "currency": "EUR",
            "delivery_terms": "...",
            "field_name_raw": {"product_number": "Article No.", ...},
            "past_resolutions": [...]
        }
    """
    # Build field_name_raw mapping for all order line fields.
    field_name_raw: dict[str, str] = {}
    for field, exts in order_line_extractions.items():
        if exts:
            field_name_raw[field] = exts[0].field_name_raw

    context: dict = {
        "product_number": _first_value(order_line_extractions, "product_number"),
        "description": _first_value(order_line_extractions, "description"),
        "quantity": _first_value(order_line_extractions, "quantity"),
        "unit_price": _first_value(order_line_extractions, "unit_price"),
        "currency": _first_value(attributes, "currency"),
        "delivery_terms": _first_value(attributes, "delivery_terms"),
        "field_name_raw": field_name_raw,
    }

    if kg_past_resolutions:
        context["past_resolutions"] = kg_past_resolutions

    return context


async def call_product_resolver(context: dict) -> dict:
    """Call the Product Resolver API.

    MVP: Returns a mock response.
    Production: HTTP POST to GA's Product Resolver API.

    Args:
        context: Context dict from build_resolver_context().

    Returns:
        Dict with resolved_product and confidence.

    Expected response format:
        {
            "resolved_product": "WG-4420-BLK",
            "confidence": 0.9,
            "matched_catalog_entry": {...}  # optional
        }
    """
    # MVP mock response
    return {
        "resolved_product": context.get("product_number") or "UNKNOWN",
        "confidence": 0.9,
    }
