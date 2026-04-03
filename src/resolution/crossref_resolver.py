"""Cross-reference resolver using PydanticAI + Gemini.

Resolves textual cross-references like "see Section 3.2" to target page indices
so the pipeline can re-enter extraction on those pages.
"""

from __future__ import annotations

import functools

from pydantic import BaseModel
from pydantic_ai import Agent

from src.models import DocumentMap, FileCoordinates

_SYSTEM_PROMPT = (
    "You are a document cross-reference resolver. Given a list of cross-reference "
    "texts from a tender document and a table of contents with page info, determine "
    "which page indices each reference points to. Return a list of resolved references "
    "with the original text, target page indices, and optionally the resolved content "
    "summary. If a reference cannot be resolved, return an empty target_page_indices list."
)

# Fraction of page height: word_boxes above this line are treated as potential
# section headers when building the table of contents for the LLM.
_HEADER_TOP_THRESHOLD = 0.15


class ResolvedReference(BaseModel):
    """A resolved cross-reference pointing to specific pages."""

    original_text: str
    target_page_indices: list[int]
    resolved_content: str | None = None


@functools.lru_cache(maxsize=1)
def _get_agent() -> Agent[None, list[ResolvedReference]]:
    """Lazy, thread-safe singleton for the crossref agent."""
    return Agent(
        "google-gla:gemini-2.0-flash",
        output_type=list[ResolvedReference],
        system_prompt=_SYSTEM_PROMPT,
    )


def get_crossref_agent() -> Agent[None, list[ResolvedReference]]:
    """Return the crossref agent (creates on first call).

    The module-level ``crossref_agent`` variable, if set to a non-None Agent,
    takes precedence.  This lets tests inject a TestModel-backed agent.
    """
    if crossref_agent is not None:
        return crossref_agent
    return _get_agent()


# Settable override — tests inject a TestModel-backed agent here.
crossref_agent: Agent[None, list[ResolvedReference]] | None = None


def _build_toc(doc_map: DocumentMap, file_coords: FileCoordinates) -> str:
    """Build a table of contents string from document structure.

    Extracts section headers from word_boxes and combines with page
    classification info to give the LLM context for resolving references.
    """
    lines: list[str] = []

    for cls in doc_map.classifications:
        page_label = f"Page {cls.page_index} ({cls.page_type})"

        # Extract potential section headers from word_boxes on this page.
        if cls.page_index < len(file_coords.pages_coordinates):
            page = file_coords.pages_coordinates[cls.page_index]
            headers = [
                wb.value
                for wb in page.word_boxes
                if wb.coordinates_rectangle
                and wb.coordinates_rectangle[0].top < _HEADER_TOP_THRESHOLD
            ]
            if headers:
                page_label += f": {' '.join(headers)}"

        lines.append(page_label)

    return "\n".join(lines) if lines else "No table of contents available."


async def resolve_crossrefs(
    crossrefs: list[str],
    doc_map: DocumentMap,
    file_coords: FileCoordinates,
    kg_patterns: dict | None = None,
) -> list[ResolvedReference]:
    """Resolve cross-reference texts to target page indices.

    Args:
        crossrefs: List of cross-reference strings (e.g. ["see Section 3.2"]).
        doc_map: The document map with page classifications.
        file_coords: The file coordinates for word_box context.
        kg_patterns: Optional knowledge graph patterns for resolution hints.

    Returns:
        List of ResolvedReference with target page indices.
    """
    if not crossrefs:
        return []

    toc = _build_toc(doc_map, file_coords)
    total_pages = doc_map.total_pages or len(file_coords.pages_coordinates)

    kg_hint = ""
    if kg_patterns:
        kg_hint = f"\n\nKnowledge graph patterns: {kg_patterns}"

    prompt = (
        f"Document has {total_pages} pages.\n\n"
        f"Table of contents:\n{toc}\n\n"
        f"Cross-references to resolve:\n"
        + "\n".join(f"- {ref}" for ref in crossrefs)
        + kg_hint
    )

    agent = get_crossref_agent()
    result = await agent.run(prompt)
    return result.output
