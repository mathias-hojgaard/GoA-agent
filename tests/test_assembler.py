"""Tests for the ExtractionOutput assembler."""

import json
import uuid
from pathlib import Path

import pytest

from src.models import (
    Extraction,
    ExtractionOutput,
    ExtractionSource,
    FileCoordinates,
    Rectangle,
    TierResult,
)
from src.resolution.assembler import assemble_output

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def file_coords() -> FileCoordinates:
    with open(FIXTURES / "sample_file_coordinates.json") as f:
        return FileCoordinates(**json.load(f))


def _make_extraction(
    field_name: str,
    value: str,
    coordinate_id: str,
    filename: str = "PO-2024-00891_Acme_Corp.pdf",
    field_name_raw: str = "",
) -> Extraction:
    """Helper to build a minimal Extraction."""
    return Extraction(
        source_of_extraction=ExtractionSource.PDF,
        filename=filename,
        extraction_certainty=0.95,
        similarity_to_confirmed_extractions=0.0,
        genai_score=0.9,
        coordinate_id=uuid.UUID(coordinate_id),
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


@pytest.fixture
def tier_results() -> list[TierResult]:
    """Build TierResults matching the fixture data."""
    product_extractions = [
        _make_extraction("product_number", "WG-4420-BLK", "01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1", field_name_raw="Article No."),
        _make_extraction("quantity", "500", "01914a3b-3c4d-7f00-f6a7-b8c9d0e1f2a3", field_name_raw="Qty"),
        _make_extraction("unit_price", "12,50", "01914a3b-5e6f-7f00-b8c9-d0e1f2a3b4c5", field_name_raw="Unit Price"),
    ]
    attribute_extractions = [
        _make_extraction("order_number", "PO-2024-00891", "01914a3b-9e3f-7f00-c2d3-e4f5a6b7c8d9"),
        _make_extraction("order_date", "15.03.2024", "01914a3b-f495-7f00-c8d9-e0f1a2b3c4d5"),
        _make_extraction("currency", "EUR", "01914a3b-7081-7f00-d0e1-f2a3b4c5d6e7"),
    ]
    address_extractions = [
        _make_extraction("company_name", "Acme Corporation GmbH", "01914a3b-92a3-7f00-f2a3-b4c5d6e7f8a9"),
    ]

    return [
        TierResult(
            tier="tier1",
            extractions=product_extractions + attribute_extractions + address_extractions,
            page_index=0,
            grounded=True,
        ),
    ]


@pytest.fixture
def table_source_ids() -> dict[int, uuid.UUID]:
    return {0: uuid.UUID("01914a3b-af40-7f00-d3e4-f5a6b7c8d9e0")}


class TestAssembler:
    def test_produces_valid_extraction_output(self, file_coords, tier_results, table_source_ids):
        output, fc = assemble_output(file_coords, tier_results, table_source_ids)
        assert isinstance(output, ExtractionOutput)

    def test_order_line_source_id_matches_table(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        assert len(output.products) == 1
        assert output.products[0].source_id == uuid.UUID("01914a3b-af40-7f00-d3e4-f5a6b7c8d9e0")

    def test_content_id_matches(self, file_coords, tier_results, table_source_ids):
        output, fc = assemble_output(file_coords, tier_results, table_source_ids)
        assert output.meta.content_id == fc.content_id

    def test_round_trip(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        dumped = output.model_dump()
        roundtripped = ExtractionOutput(**dumped)
        assert roundtripped.meta.content_id == output.meta.content_id
        assert len(roundtripped.products) == len(output.products)

    def test_attributes_extracted(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        attr_fields = set(output.attributes.extractions.keys())
        assert "order_number" in attr_fields
        assert "order_date" in attr_fields
        assert "currency" in attr_fields

    def test_address_extracted(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        assert len(output.address) == 1
        assert "company_name" in output.address[0].extractions

    def test_date_normalization_applied(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        order_date_ext = output.attributes.extractions["order_date"][0]
        assert order_date_ext.extracted_value == "2024-03-15"
        assert order_date_ext.raw_extracted_value == "15.03.2024"

    def test_decimal_normalization_applied(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        unit_price_ext = output.products[0].extractions["unit_price"][0]
        assert unit_price_ext.extracted_value == "12.50"
        assert unit_price_ext.raw_extracted_value == "12,50"

    def test_coordinate_ids_reference_valid_entries(self, file_coords, tier_results, table_source_ids):
        """All Extraction.coordinate_ids should exist in FileCoordinates."""
        output, fc = assemble_output(file_coords, tier_results, table_source_ids)

        # Collect all valid coordinate_ids from FileCoordinates.
        valid_ids: set[uuid.UUID] = set()
        zero_uuid = uuid.UUID(int=0)
        for page in fc.pages_coordinates:
            for wb in page.word_boxes:
                valid_ids.add(wb.coordinate_id)
            for wb in page.extraction_word_boxes:
                valid_ids.add(wb.coordinate_id)
            for table in page.tables:
                valid_ids.add(table.coordinate_id)
                for cell in table.cells_flatten:
                    valid_ids.add(cell.coordinate_id)
            for lo in page.layout_objects:
                valid_ids.add(lo.coordinate_id)

        # Check all extraction coordinate_ids.
        all_extractions: list[Extraction] = []
        for ol in output.products:
            for ext_list in ol.extractions.values():
                all_extractions.extend(ext_list)
        for ext_list in output.attributes.extractions.values():
            all_extractions.extend(ext_list)
        for addr in output.address:
            for ext_list in addr.extractions.values():
                all_extractions.extend(ext_list)

        for ext in all_extractions:
            assert ext.coordinate_id in valid_ids or ext.coordinate_id == zero_uuid, (
                f"coordinate_id {ext.coordinate_id} for field '{ext.field_name}' "
                f"not found in FileCoordinates"
            )

    def test_meta_fields(self, file_coords, tier_results, table_source_ids):
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        assert output.meta.filename == "PO-2024-00891_Acme_Corp.pdf"
        assert output.meta.file_type == "PDF"
        assert output.meta.doc_class == "PurchaseOrder"

    def test_no_tier_results_produces_empty_output(self, file_coords):
        output, _ = assemble_output(file_coords, [], {})
        assert output.products == []
        assert output.attributes.extractions == {}
        assert output.address == []
        assert output.meta.content_id == file_coords.content_id

    def test_multi_row_produces_multiple_order_lines(self, file_coords, table_source_ids):
        """Multiple product rows should produce multiple OrderLines."""
        row1 = [
            _make_extraction("product_number", "WG-4420-BLK", "01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1"),
            _make_extraction("quantity", "500", "01914a3b-3c4d-7f00-f6a7-b8c9d0e1f2a3"),
        ]
        # Place row 2 at a different vertical position.
        row2 = [
            _make_extraction("product_number", "WG-5530-WHT", "01914a3b-d273-7f00-a6b7-c8d9e0f1a2b3"),
            _make_extraction("quantity", "200", "01914a3b-d273-7f00-a6b7-c8d9e0f1a2b3"),
        ]
        for ext in row2:
            ext.coordinates_rectangle[0].top = 0.38  # different row

        tier_results = [
            TierResult(tier="tier1", extractions=row1 + row2, page_index=0, grounded=True),
        ]
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        assert len(output.products) == 2

    def test_tier3_gets_zero_source_id(self, file_coords):
        """Tier 2/3 extractions should get UUID(int=0) as source_id."""
        tier_results = [
            TierResult(
                tier="tier3",
                extractions=[
                    _make_extraction("product_number", "ABC-123", "01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1"),
                ],
                page_index=0,
                grounded=True,
            ),
        ]
        output, _ = assemble_output(file_coords, tier_results, {})
        assert len(output.products) == 1
        assert output.products[0].source_id == uuid.UUID(int=0)

    def test_row_grouping_at_tolerance_boundary(self, file_coords, table_source_ids):
        """Extractions exactly at the tolerance boundary (0.005) should be in the same row."""
        ext1 = _make_extraction("product_number", "A", "01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1")
        ext2 = _make_extraction("quantity", "100", "01914a3b-3c4d-7f00-f6a7-b8c9d0e1f2a3")
        ext1.coordinates_rectangle[0].top = 0.300
        ext2.coordinates_rectangle[0].top = 0.305  # exactly at tolerance

        ext3 = _make_extraction("product_number", "B", "01914a3b-d273-7f00-a6b7-c8d9e0f1a2b3")
        ext3.coordinates_rectangle[0].top = 0.306  # just beyond tolerance from ext1

        tier_results = [
            TierResult(tier="tier1", extractions=[ext1, ext2, ext3], page_index=0, grounded=True),
        ]
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        # ext1 (0.300) and ext2 (0.305) are within tolerance → same row
        # ext3 (0.306) is 0.006 from ext1 → new row
        assert len(output.products) == 2

    def test_mixed_tiers(self, file_coords, table_source_ids):
        """Mixed tier1 + tier3 results should produce OrderLines with correct source_ids."""
        tier1_ext = _make_extraction("product_number", "WG-4420-BLK", "01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1")
        tier3_ext = _make_extraction("product_number", "MANUAL-001", "01914a3b-d273-7f00-a6b7-c8d9e0f1a2b3")
        tier3_ext.coordinates_rectangle[0].top = 0.5

        tier_results = [
            TierResult(tier="tier1", extractions=[tier1_ext], page_index=0, grounded=True),
            TierResult(tier="tier3", extractions=[tier3_ext], page_index=1, grounded=True),
        ]
        output, _ = assemble_output(file_coords, tier_results, table_source_ids)
        assert len(output.products) == 2

        # First OrderLine (tier1, page 0) should have table source_id.
        assert output.products[0].source_id == uuid.UUID("01914a3b-af40-7f00-d3e4-f5a6b7c8d9e0")
        # Second OrderLine (tier3, page 1) should have zero UUID.
        assert output.products[1].source_id == uuid.UUID(int=0)
