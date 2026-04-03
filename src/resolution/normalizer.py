"""Value normalization for extracted fields.

Handles date formats (EU/US → ISO), decimal separators (EU → dot),
quantity unit stripping, and whitespace collapsing.
"""

from __future__ import annotations

import re
from datetime import datetime

# Fields that should have decimal normalization applied.
_NUMERIC_FIELDS = {"unit_price", "total_price", "quantity", "amount"}

# Fields that should have date normalization applied.
_DATE_FIELDS = {"order_date", "delivery_date", "invoice_date", "date", "ship_date"}


def normalize_value(
    raw_value: str,
    field_name: str,
    date_format: str,
    decimal_separator: str,
) -> str:
    """Normalize a raw extracted value based on field type and document format settings.

    Args:
        raw_value: The raw string extracted from the document.
        field_name: The canonical field name (e.g. "order_date", "unit_price").
        date_format: The document's date format (e.g. "DD.MM.YYYY", "MM/DD/YYYY").
        decimal_separator: The document's decimal separator ("," or ".").

    Returns:
        Normalized string value.
    """
    # Whitespace: strip and collapse
    value = " ".join(raw_value.split())

    if not value:
        return value

    # Date normalization — only for known date fields or fields ending with "_date"
    field_lower = field_name.lower()
    if field_lower in _DATE_FIELDS or field_lower.endswith("_date"):
        return _normalize_date(value, date_format)

    # Decimal / quantity normalization — only for known numeric fields
    if field_lower in _NUMERIC_FIELDS:
        value = _strip_units(value)
        return _normalize_decimal(value, decimal_separator)

    return value


def _normalize_date(value: str, date_format: str) -> str:
    """Convert a date string from the document's format to ISO 8601 (YYYY-MM-DD)."""
    fmt_upper = date_format.upper()

    # Build a strptime format string from the declared date_format.
    py_fmt = (
        fmt_upper
        .replace("YYYY", "%Y")
        .replace("YY", "%y")
        .replace("DD", "%d")
        .replace("MM", "%m")
    )

    try:
        dt = datetime.strptime(value, py_fmt)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # If parsing fails, return as-is rather than crashing.
        return value


def _normalize_decimal(value: str, decimal_separator: str) -> str:
    """Normalize a numeric string to use '.' as the decimal separator.

    Handles EU-style thousands separators (dots) with comma decimal,
    US-style commas-as-thousands with dot decimal, and space-separated
    thousands (common in some EU documents).
    """
    # Strip spaces first — some EU documents use space as thousands separator.
    value = value.replace(" ", "")

    if decimal_separator == ",":
        # EU format: "1.250,00" → "1250.00", "12,50" → "12.50"
        # Remove dots (thousands separators), then replace comma with dot.
        value = value.replace(".", "").replace(",", ".")
    elif decimal_separator == ".":
        # US format: "1,250.00" → "1250.00"
        # Remove commas (thousands separators).
        value = value.replace(",", "")

    return value


def _strip_units(value: str) -> str:
    """Strip trailing unit suffixes from quantity values.

    "500 pcs" → "500", "10 m" → "10", "1 250 pcs" → "1250",
    "12.50" → "12.50" (no unit).
    """
    match = re.match(r"^([\d.,\s]+)\s+[a-zA-Z]+\.?$", value)
    if match:
        # Collapse interior spaces (e.g. "1 250" → "1250") since some EU
        # documents use space as a thousands separator.
        return match.group(1).replace(" ", "")
    return value
