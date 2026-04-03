"""KG reader — query functions for pipeline enrichment."""

from __future__ import annotations

import asyncio

from .kg_client import NullKGClient, TenantKGClient


def _extract_fact(result: object) -> str:
    return result.fact if hasattr(result, "fact") else str(result)


async def get_sender_patterns(
    client: TenantKGClient | NullKGClient, sender_id: str
) -> dict:
    """Query sender document structure patterns.

    Used by stage 1 (email parser) to prime the TaskBrief.
    """
    if isinstance(client, NullKGClient):
        return {"patterns": [], "known_aliases": [], "section_mappings": []}

    results = await client.graphiti.search(
        f"sender {sender_id} document structure patterns"
    )
    # TODO: replace keyword heuristic with structured episode metadata
    # once Graphiti episodes carry typed labels
    patterns = []
    known_aliases = []
    section_mappings = []
    for result in results:
        fact = _extract_fact(result)
        if "alias" in fact.lower():
            known_aliases.append(fact)
        elif "section" in fact.lower() or "mapping" in fact.lower():
            section_mappings.append(fact)
        else:
            patterns.append(fact)

    return {
        "patterns": patterns,
        "known_aliases": known_aliases,
        "section_mappings": section_mappings,
    }


async def get_term_mappings(
    client: TenantKGClient | NullKGClient, terms: list[str]
) -> dict[str, str]:
    """Map tenant-specific terminology to canonical field names.

    Used by stage 3, Tier 3 (Gemini extraction) to translate
    tenant-specific column headers.
    """
    if isinstance(client, NullKGClient):
        return {}

    async def _lookup(term: str) -> tuple[str, str | None]:
        results = await client.graphiti.search(f"terminology mapping {term}")
        if results:
            return term, _extract_fact(results[0])
        return term, None

    pairs = await asyncio.gather(*[_lookup(t) for t in terms])
    return {term: fact for term, fact in pairs if fact is not None}


async def get_past_resolutions(
    client: TenantKGClient | NullKGClient, description: str
) -> list[dict]:
    """Retrieve past product resolutions for a description.

    Used by stage 7 (Product Resolver) to provide prior resolution context.
    """
    if isinstance(client, NullKGClient):
        return []

    results = await client.graphiti.search(f"product resolution {description}")
    resolutions = []
    for result in results:
        fact = _extract_fact(result)
        score = result.score if hasattr(result, "score") else 0.5
        resolutions.append(
            {
                "description": description,
                "resolved_product": fact,
                "confidence": score,
            }
        )
    return resolutions


async def get_crossref_patterns(
    client: TenantKGClient | NullKGClient, sender_id: str
) -> list[dict]:
    """Retrieve cross-reference patterns for a sender.

    Used by stage 6 (cross-ref resolver).
    """
    if isinstance(client, NullKGClient):
        return []

    results = await client.graphiti.search(
        f"cross reference patterns {sender_id}"
    )
    patterns = []
    for result in results:
        fact = _extract_fact(result)
        patterns.append(
            {
                "pattern": fact,
                "typical_content": "",
                "typical_pages": [],
            }
        )
    return patterns
