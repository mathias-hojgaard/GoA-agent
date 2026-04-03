"""KG writer — correction ingestion after HITL review."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypedDict
from uuid import uuid4

if TYPE_CHECKING:
    from .kg_client import TenantKGClient


class CorrectionDict(TypedDict):
    field: str
    wrong_value: str
    correct_value: str
    context: str


async def ingest_extraction_correction(
    client: TenantKGClient, correction: CorrectionDict
) -> None:
    """Ingest a human-corrected extraction into the knowledge graph."""
    from graphiti_core import EpisodeType

    await client.graphiti.add_episode(
        name=f"correction_{correction['field']}_{uuid4()}",
        episode_body=(
            f"Human corrected {correction['field']}: "
            f"'{correction['wrong_value']}' was changed to "
            f"'{correction['correct_value']}'. "
            f"Context: {correction['context']}"
        ),
        source=EpisodeType.text,
        source_description="HITL_correction",
        reference_time=datetime.now(timezone.utc),
    )


async def ingest_product_resolution(
    client: TenantKGClient,
    description: str,
    resolved_product: str,
    sender_id: str,
) -> None:
    """Record a product resolution for future lookup."""
    from graphiti_core import EpisodeType

    await client.graphiti.add_episode(
        name=f"resolution_{sender_id}_{uuid4()}",
        episode_body=(
            f"For {sender_id}, the description '{description}' "
            f"resolves to product '{resolved_product}'"
        ),
        source=EpisodeType.text,
        source_description="product_resolution",
        reference_time=datetime.now(timezone.utc),
    )


async def ingest_sender_pattern(
    client: TenantKGClient, sender_id: str, pattern: str
) -> None:
    """Record a sender-specific document pattern."""
    from graphiti_core import EpisodeType

    await client.graphiti.add_episode(
        name=f"sender_pattern_{sender_id}_{uuid4()}",
        episode_body=f"Sender {sender_id} uses pattern: {pattern}",
        source=EpisodeType.text,
        source_description="sender_pattern",
        reference_time=datetime.now(timezone.utc),
    )
