"""Ingestion stage: email parsing and page classification."""

from .email_parser import parse_email
from .page_classifier import classify_page, classify_pages, classify_table_type

__all__ = [
    "classify_page",
    "classify_pages",
    "classify_table_type",
    "parse_email",
]
