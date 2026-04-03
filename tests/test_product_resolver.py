"""Tests for the product resolver."""

import uuid

import pytest

from src.models import Extraction, ExtractionSource, Rectangle
from src.resolution.product_resolver import build_resolver_context, call_product_resolver


def _make_extraction(field_name: str, value: str, field_name_raw: str = "") -> Extraction:
    return Extraction(
        source_of_extraction=ExtractionSource.PDF,
        filename="test.pdf",
        extraction_certainty=0.95,
        similarity_to_confirmed_extractions=0.0,
        genai_score=0.9,
        coordinate_id=uuid.UUID(int=0),
        field_name=field_name,
        field_name_raw=field_name_raw or field_name,
        field_name_coordinates_id=uuid.UUID(int=0),
        raw_saga_extraction="",
        raw_extracted_value=value,
        extracted_value=value,
        relations=[],
        coordinates_rectangle=[Rectangle(top=0.3, left=0.1, width=0.1, height=0.01)],
        message="",
        advanced_validation=[],
    )


class TestBuildResolverContext:
    def test_basic_context(self):
        order_exts = {
            "product_number": [_make_extraction("product_number", "WG-4420-BLK", "Article No.")],
            "quantity": [_make_extraction("quantity", "500", "Qty")],
            "unit_price": [_make_extraction("unit_price", "12.50", "Unit Price")],
        }
        attr_exts = {
            "currency": [_make_extraction("currency", "EUR")],
        }
        ctx = build_resolver_context(order_exts, attr_exts)

        assert ctx["product_number"] == "WG-4420-BLK"
        assert ctx["quantity"] == "500"
        assert ctx["unit_price"] == "12.50"
        assert ctx["currency"] == "EUR"
        assert ctx["field_name_raw"]["product_number"] == "Article No."

    def test_missing_fields_are_none(self):
        ctx = build_resolver_context({}, {})
        assert ctx["product_number"] is None
        assert ctx["description"] is None
        assert ctx["currency"] is None

    def test_past_resolutions_included(self):
        past = [{"product": "WG-4420", "resolved_to": "WG-4420-BLK"}]
        ctx = build_resolver_context({}, {}, kg_past_resolutions=past)
        assert ctx["past_resolutions"] == past

    def test_past_resolutions_excluded_when_none(self):
        ctx = build_resolver_context({}, {})
        assert "past_resolutions" not in ctx

    def test_field_name_raw_mapping(self):
        order_exts = {
            "product_number": [_make_extraction("product_number", "X", "Artikelnr.")],
            "quantity": [_make_extraction("quantity", "1", "Menge")],
        }
        ctx = build_resolver_context(order_exts, {})
        assert ctx["field_name_raw"] == {
            "product_number": "Artikelnr.",
            "quantity": "Menge",
        }


class TestCallProductResolver:
    @pytest.mark.asyncio
    async def test_mock_response_with_product(self):
        result = await call_product_resolver({"product_number": "WG-4420-BLK"})
        assert result["resolved_product"] == "WG-4420-BLK"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_mock_response_without_product(self):
        result = await call_product_resolver({})
        assert result["resolved_product"] == "UNKNOWN"
