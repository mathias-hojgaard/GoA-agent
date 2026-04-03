"""Tenant-scoped knowledge graph and few-shot store."""

from .fewshot_store import FewShotStore
from .kg_client import NullKGClient, TenantKGClient
from .kg_reader import (
    get_crossref_patterns,
    get_past_resolutions,
    get_sender_patterns,
    get_term_mappings,
)
from .kg_writer import (
    CorrectionDict,
    ingest_extraction_correction,
    ingest_product_resolution,
    ingest_sender_pattern,
)

__all__ = [
    "CorrectionDict",
    "FewShotStore",
    "NullKGClient",
    "TenantKGClient",
    "get_crossref_patterns",
    "get_past_resolutions",
    "get_sender_patterns",
    "get_term_mappings",
    "ingest_extraction_correction",
    "ingest_product_resolution",
    "ingest_sender_pattern",
]
