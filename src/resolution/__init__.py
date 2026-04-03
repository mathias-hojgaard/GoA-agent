"""Resolution module: cross-ref resolver, product resolver, normalizer, assembler."""

from .assembler import assemble_output
from .crossref_resolver import ResolvedReference, get_crossref_agent, resolve_crossrefs
from .normalizer import normalize_value
from .product_resolver import build_resolver_context, call_product_resolver

__all__ = [
    "ResolvedReference",
    "assemble_output",
    "get_crossref_agent",
    "build_resolver_context",
    "call_product_resolver",
    "normalize_value",
    "resolve_crossrefs",
]
