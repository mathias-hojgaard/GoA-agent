"""Tier 3: LLM extraction using PydanticAI with Gemini."""

from __future__ import annotations

from pydantic_ai import Agent

from src.models import PageCoordinates, PageExtractionResult, WordBox

SYSTEM_PROMPT = """\
You are extracting product and tender data from a document page.
The page may contain product descriptions, quantities, prices, and specifications
in paragraph/narrative form rather than a structured table.

Extract all product-related information you can find:
- product_number: vendor-specific part numbers
- description: product descriptions
- quantity: amounts requested
- unit_price: price per unit
- position_number: line item numbers

Also identify any cross-references to other sections (e.g., "see Section 3.2",
"as per Annex B", "refer to page 45").

Return the raw text values exactly as they appear in the document — do NOT
normalize dates, numbers, or formats."""

_agent: Agent[None, PageExtractionResult] | None = None


def _get_agent() -> Agent[None, PageExtractionResult]:
    global _agent
    if _agent is None:
        _agent = Agent(
            "google-gla:gemini-2.0-flash",
            system_prompt=SYSTEM_PROMPT,
            result_type=PageExtractionResult,
        )
    return _agent


def build_tier3_prompt(
    page: PageCoordinates, kg_terms: dict[str, str] | None = None
) -> str:
    """Build the user prompt from page word boxes in reading order."""
    sorted_boxes = sorted(
        page.word_boxes,
        key=lambda wb: (wb.coordinates_rectangle[0].top, wb.coordinates_rectangle[0].left),
    )

    lines: list[str] = ["Page text content:"]
    lines.append(_boxes_to_text(sorted_boxes))

    if page.image_url_v2:
        lines.append(f"\nPage image: {page.image_url_v2}")

    if kg_terms:
        lines.append("\nKnown terminology mappings:")
        for term, canonical in kg_terms.items():
            lines.append(f"  {term} → {canonical}")

    return "\n".join(lines)


async def extract_with_gemini(
    page: PageCoordinates, kg_terms: dict[str, str] | None = None
) -> PageExtractionResult:
    """Run Tier 3 LLM extraction on a single page."""
    prompt = build_tier3_prompt(page, kg_terms)
    result = await _get_agent().run(prompt)
    return result.data


def _boxes_to_text(boxes: list[WordBox]) -> str:
    """Concatenate word boxes into readable text, grouping by approximate rows."""
    if not boxes:
        return ""

    rows: list[list[str]] = []
    current_row: list[str] = [boxes[0].value]
    current_top = boxes[0].coordinates_rectangle[0].top

    for wb in boxes[1:]:
        wb_top = wb.coordinates_rectangle[0].top
        if abs(wb_top - current_top) > 0.005:
            rows.append(current_row)
            current_row = [wb.value]
            current_top = wb_top
        else:
            current_row.append(wb.value)

    rows.append(current_row)
    return "\n".join(" ".join(row) for row in rows)
