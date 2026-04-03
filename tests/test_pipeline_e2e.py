"""End-to-end pipeline tests.

All LLM calls are mocked via monkeypatching — no real Gemini API calls.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models import (
    Extraction,
    ExtractionOutput,
    FileCoordinates,
    PageCoordinates,
    Rectangle,
    TaskBrief,
    WordBox,
)
from src.models.ga_models import (
    Cell,
    CellTypes,
    Classification,
    Column,
    Prediction,
    Row,
    Table,
)
from src.pipeline import process_tender

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_email() -> str:
    return (FIXTURES / "sample_email.txt").read_text()


def _mock_task_brief(**overrides) -> TaskBrief:
    """Build a TaskBrief with sensible defaults."""
    defaults = {
        "page_filters": [3, 5],
        "sender_id": "procurement@acme-corp.de",
        "attachment_filenames": ["PO-2024-00891_Acme_Corp.pdf"],
        "special_instructions": ["All prices must be in EUR"],
        "deadline": "2026-04-15T00:00:00",
    }
    defaults.update(overrides)
    return TaskBrief(**defaults)


def _patch_email_parser(task_brief: TaskBrief | None = None):
    """Return a patch that replaces parse_email with an async mock."""
    tb = task_brief or _mock_task_brief()

    async def _mock_parse(email_body, kg_context=None, *, model=None):
        return tb

    return patch("src.pipeline.parse_email", side_effect=_mock_parse)


# ── Test 1: Tier 1 table extraction ────────────────────────────────────


def _make_tier1_fc() -> FileCoordinates:
    """Build a FileCoordinates with a table containing header + data rows."""
    table_id = uuid.UUID("01914a3b-af40-7f00-d3e4-f5a6b7c8d9e0")
    header_row_id = uuid.UUID("01914a3b-b051-7f00-e4f5-a6b7c8d9e0f1")
    data_row_id = uuid.UUID("01914a3b-b052-7f00-e4f5-a6b7c8d9e0f2")
    col0_id = uuid.UUID("01914a3b-c162-7f00-f5a6-b7c8d9e0f1a2")
    col1_id = uuid.UUID("01914a3b-c163-7f00-f5a6-b7c8d9e0f1a3")
    col2_id = uuid.UUID("01914a3b-c164-7f00-f5a6-b7c8d9e0f1a4")
    hdr_cell0_id = uuid.UUID("01914a3b-d273-7f00-a6b7-c8d9e0f1a2b3")
    hdr_cell1_id = uuid.UUID("01914a3b-d274-7f00-a6b7-c8d9e0f1a2b4")
    hdr_cell2_id = uuid.UUID("01914a3b-d275-7f00-a6b7-c8d9e0f1a2b5")
    data_cell0_id = uuid.UUID("01914a3b-e001-7f00-a6b7-c8d9e0f1a2c0")
    data_cell1_id = uuid.UUID("01914a3b-e002-7f00-a6b7-c8d9e0f1a2c1")
    data_cell2_id = uuid.UUID("01914a3b-e003-7f00-a6b7-c8d9e0f1a2c2")

    pred = lambda label, cert: [Prediction(label=label, certainty=cert)]

    header_cells = [
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.31, left=0.05, width=0.15, height=0.025, page=0),
             coordinate_id=hdr_cell0_id, content="Article No.", column_index=0, row_index=0, column_id=col0_id, row_id=header_row_id),
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.31, left=0.30, width=0.10, height=0.025, page=0),
             coordinate_id=hdr_cell1_id, content="Qty", column_index=1, row_index=0, column_id=col1_id, row_id=header_row_id),
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.31, left=0.55, width=0.10, height=0.025, page=0),
             coordinate_id=hdr_cell2_id, content="Unit Price", column_index=2, row_index=0, column_id=col2_id, row_id=header_row_id),
    ]
    data_cells = [
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.345, left=0.05, width=0.15, height=0.025, page=0),
             coordinate_id=data_cell0_id, content="WG-4420-BLK", column_index=0, row_index=1, column_id=col0_id, row_id=data_row_id),
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.345, left=0.30, width=0.10, height=0.025, page=0),
             coordinate_id=data_cell1_id, content="500", column_index=1, row_index=1, column_id=col1_id, row_id=data_row_id),
        Cell(type=CellTypes.NON_SPANNING_CELL, coordinates_rectangle=Rectangle(top=0.345, left=0.55, width=0.10, height=0.025, page=0),
             coordinate_id=data_cell2_id, content="12,50", column_index=2, row_index=1, column_id=col2_id, row_id=data_row_id),
    ]

    table = Table(
        classification=Classification(label="po_table", certainty=0.95, predictions=pred("po_table", 0.95)),
        coordinates_rectangle=Rectangle(top=0.31, left=0.04, width=0.91, height=0.10, page=0),
        coordinate_id=table_id,
        rows=[
            Row(classification=Classification(label="header", certainty=0.99, predictions=pred("header", 0.99)),
                coordinates_rectangle=Rectangle(top=0.31, left=0.04, width=0.91, height=0.025, page=0),
                coordinate_id=header_row_id, index=0, detection_certainty=0.99, relations=[]),
            Row(classification=Classification(label="table_line", certainty=0.97, predictions=pred("table_line", 0.97)),
                coordinates_rectangle=Rectangle(top=0.345, left=0.04, width=0.91, height=0.025, page=0),
                coordinate_id=data_row_id, index=1, detection_certainty=0.97, relations=[]),
        ],
        columns=[
            Column(classification=Classification(label="product_number", certainty=0.98, predictions=pred("product_number", 0.98)),
                   coordinates_rectangle=Rectangle(top=0.31, left=0.05, width=0.15, height=0.10, page=0),
                   coordinate_id=col0_id, index=0, detection_certainty=0.98, relations=[]),
            Column(classification=Classification(label="quantity", certainty=0.96, predictions=pred("quantity", 0.96)),
                   coordinates_rectangle=Rectangle(top=0.31, left=0.30, width=0.10, height=0.10, page=0),
                   coordinate_id=col1_id, index=1, detection_certainty=0.96, relations=[]),
            Column(classification=Classification(label="unit_price", certainty=0.95, predictions=pred("unit_price", 0.95)),
                   coordinates_rectangle=Rectangle(top=0.31, left=0.55, width=0.10, height=0.10, page=0),
                   coordinate_id=col2_id, index=2, detection_certainty=0.95, relations=[]),
        ],
        cells=[header_cells, data_cells],
        cells_flatten=header_cells + data_cells,
        relations=[],
        detection_certainty=0.95,
    )

    page = PageCoordinates(
        image_url="https://storage.example.com/p0.png",
        image_url_v2="https://storage.example.com/v2/p0",
        page_height=842.0, page_width=595.0, page_name="",
        extraction_word_boxes=[],
        word_boxes=[
            WordBox(coordinates_rectangle=[Rectangle(top=0.345, left=0.05, width=0.15, height=0.012, page=0)],
                    coordinate_id=uuid.UUID("01914a3b-1a2b-7f00-d4e5-f6a7b8c9d0e1"), value="WG-4420-BLK"),
            WordBox(coordinates_rectangle=[Rectangle(top=0.345, left=0.30, width=0.045, height=0.012, page=0)],
                    coordinate_id=uuid.UUID("01914a3b-3c4d-7f00-f6a7-b8c9d0e1f2a3"), value="500"),
            WordBox(coordinates_rectangle=[Rectangle(top=0.345, left=0.55, width=0.055, height=0.012, page=0)],
                    coordinate_id=uuid.UUID("01914a3b-5e6f-7f00-b8c9-d0e1f2a3b4c5"), value="12,50"),
        ],
        tables=[table],
        layout_objects=[],
    )

    return FileCoordinates(
        filename="PO-2024-00891_Acme_Corp.pdf",
        content_id=uuid.UUID("01914a3b-7c5d-7f8e-9a2b-3d4e5f6a7b8c"),
        content_type="application/pdf",
        doc_class="PurchaseOrder",
        size=284731,
        date_archetype="EU",
        date_format="DD.MM.YYYY",
        decimal_separator=",",
        pages_coordinates=[page],
    )


@pytest.mark.asyncio
async def test_e2e_tier1_table(sample_email: str):
    """Full pipeline on fixture with a vision-detected table → Tier 1 extraction."""
    fc = _make_tier1_fc()
    tb = _mock_task_brief(page_filters=None)

    with _patch_email_parser(tb):
        output, updated_fc = await process_tender(
            sample_email, fc, gemini_semaphore=1
        )

    assert isinstance(output, ExtractionOutput)

    # Should have at least 1 product extracted from the table
    assert len(output.products) >= 1

    # Each OrderLine should have source_id matching the Table's coordinate_id
    table_coord_id = fc.pages_coordinates[0].tables[0].coordinate_id
    for ol in output.products:
        assert ol.source_id == table_coord_id

    # Each Extraction should have a non-zero coordinate_id
    zero_uuid = uuid.UUID(int=0)
    for ol in output.products:
        for field_exts in ol.extractions.values():
            for ext in field_exts:
                assert ext.coordinate_id != zero_uuid, (
                    f"Extraction {ext.field_name}={ext.extracted_value!r} is ungrounded"
                )

    # content_id linkage
    assert output.meta.content_id == fc.content_id

    # Updated FC should have extraction_word_boxes entries
    updated_page = updated_fc.pages_coordinates[0]
    assert len(updated_page.extraction_word_boxes) > 0

    # Round-trip validation
    ExtractionOutput(**output.model_dump())


# ── Test 2: Tier 2 spatial extraction ──────────────────────────────────


def _make_spatial_page() -> PageCoordinates:
    """Build a page with word_boxes arranged as a table but no Table objects."""
    # Header row at y=0.10
    headers = [
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.10, left=0.05, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0001-0000-0000-0000-000000000001"),
            value="Article No.",
        ),
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.10, left=0.30, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0001-0000-0000-0000-000000000002"),
            value="Quantity",
        ),
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.10, left=0.55, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0001-0000-0000-0000-000000000003"),
            value="Unit Price",
        ),
    ]
    # Data row at y=0.14
    data = [
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.14, left=0.05, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0002-0000-0000-0000-000000000001"),
            value="ABC-123",
        ),
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.14, left=0.30, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0002-0000-0000-0000-000000000002"),
            value="100",
        ),
        WordBox(
            coordinates_rectangle=[Rectangle(top=0.14, left=0.55, width=0.15, height=0.02, page=0)],
            coordinate_id=uuid.UUID("aaaa0002-0000-0000-0000-000000000003"),
            value="25.00",
        ),
    ]
    return PageCoordinates(
        image_url="https://example.com/p0.png",
        image_url_v2="https://example.com/v2/p0",
        page_height=842.0,
        page_width=595.0,
        page_name="",
        extraction_word_boxes=[],
        word_boxes=headers + data,
        tables=[],
        layout_objects=[],
    )


@pytest.mark.asyncio
async def test_e2e_tier2_spatial():
    """Page with word_boxes but no Table → Tier 2 spatial extraction."""
    page = _make_spatial_page()
    fc = FileCoordinates(
        filename="test_spatial.pdf",
        content_id=uuid.uuid4(),
        content_type="application/pdf",
        doc_class="PurchaseOrder",
        size=1000,
        date_archetype="US",
        date_format="MM/DD/YYYY",
        decimal_separator=".",
        pages_coordinates=[page],
    )

    tb = _mock_task_brief(page_filters=None)
    with _patch_email_parser(tb):
        output, updated_fc = await process_tender(
            "test email", fc, gemini_semaphore=1
        )

    assert isinstance(output, ExtractionOutput)
    # Tier 2 produces extractions grounded to word_boxes (not cells)
    zero_uuid = uuid.UUID(int=0)
    all_exts = [
        ext
        for ol in output.products
        for exts in ol.extractions.values()
        for ext in exts
    ]
    # If tier2 produced extractions, they should be grounded to word_box coordinate_ids
    wb_ids = {wb.coordinate_id for wb in page.word_boxes}
    for ext in all_exts:
        if ext.coordinate_id != zero_uuid:
            assert ext.coordinate_id in wb_ids


# ── Test 3: Normalization ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_normalization(sample_email: str):
    """EU date and decimal formats are normalized correctly."""
    fc = _make_tier1_fc()  # Has date_format="DD.MM.YYYY" and decimal_separator=","
    tb = _mock_task_brief(page_filters=None)

    with _patch_email_parser(tb):
        output, _ = await process_tender(sample_email, fc, gemini_semaphore=1)

    # Collect all extractions
    all_exts: list[Extraction] = []
    for ol in output.products:
        for exts in ol.extractions.values():
            all_exts.extend(exts)
    for exts in output.attributes.extractions.values():
        all_exts.extend(exts)

    # Check that EU decimals were normalized (raw "12,50" → "12.50")
    decimal_checked = False
    for ext in all_exts:
        if ext.raw_extracted_value == "12,50":
            assert ext.extracted_value == "12.50", (
                f"Decimal not normalized: raw={ext.raw_extracted_value!r} "
                f"extracted={ext.extracted_value!r}"
            )
            decimal_checked = True
    assert decimal_checked, "No extraction with raw_extracted_value '12,50' — test is vacuous"

    # Check that EU dates were normalized (raw "15.03.2024" → "2024-03-15")
    # Note: the tier1 fixture doesn't contain date values, so this is best-effort.
    for ext in all_exts:
        if ext.raw_extracted_value == "15.03.2024":
            assert ext.extracted_value == "2024-03-15", (
                f"Date not normalized: raw={ext.raw_extracted_value!r} "
                f"extracted={ext.extracted_value!r}"
            )


# ── Test 4: Page filters ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_page_filters():
    """Email says 'only quote page 1' → only page 0 (0-indexed) extracted; page 1 excluded."""
    # Build a 2-page FC. Page 0 has a table with data, page 1 has a table with data.
    fc = _make_tier1_fc()
    # Duplicate page 0 as page 1 (both have extractable data)
    page1 = fc.pages_coordinates[0].model_copy(deep=True)
    fc.pages_coordinates.append(page1)

    # page_filters=[1] means "only page 1" (1-indexed) → page 0 (0-indexed)
    tb = _mock_task_brief(page_filters=[1])

    with _patch_email_parser(tb):
        output_filtered, _ = await process_tender("test email", fc, gemini_semaphore=1)

    # Now run without filters for comparison
    tb_all = _mock_task_brief(page_filters=None)
    fc_all = _make_tier1_fc()
    fc_all.pages_coordinates.append(fc_all.pages_coordinates[0].model_copy(deep=True))

    with _patch_email_parser(tb_all):
        output_all, _ = await process_tender("test email", fc_all, gemini_semaphore=1)

    # Filtered should have fewer products than unfiltered (1 page vs 2 pages)
    assert len(output_all.products) > len(output_filtered.products), (
        f"Filtering didn't reduce products: all={len(output_all.products)}, "
        f"filtered={len(output_filtered.products)}"
    )
    assert len(output_filtered.products) >= 1, "Filtered page should still produce extractions"


# ── Test 5: Empty document ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_empty_document():
    """FileCoordinates with 0 pages → no crash, 0 products."""
    fc = FileCoordinates(
        filename="empty.pdf",
        content_id=uuid.uuid4(),
        content_type="application/pdf",
        doc_class="PurchaseOrder",
        size=0,
        date_archetype="US",
        date_format="MM/DD/YYYY",
        decimal_separator=".",
        pages_coordinates=[],
    )
    tb = _mock_task_brief(page_filters=None)

    with _patch_email_parser(tb):
        output, updated_fc = await process_tender("empty", fc, gemini_semaphore=1)

    assert isinstance(output, ExtractionOutput)
    assert len(output.products) == 0
    assert output.meta.content_id == fc.content_id


# ── Test 6: Output round-trip ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_output_roundtrip(sample_email: str):
    """ExtractionOutput survives JSON serialize → deserialize → Pydantic validate."""
    fc = _make_tier1_fc()
    tb = _mock_task_brief(page_filters=None)

    with _patch_email_parser(tb):
        output, _ = await process_tender(sample_email, fc, gemini_semaphore=1)

    # Round-trip through JSON
    json_str = json.dumps(output.model_dump(mode="json"))
    parsed = json.loads(json_str)
    reconstructed = ExtractionOutput(**parsed)

    # Compare via model_dump(mode="json") since Union[UUID, str] types
    # may differ after JSON round-trip
    assert str(reconstructed.meta.content_id) == str(output.meta.content_id)
    assert len(reconstructed.products) == len(output.products)
    assert reconstructed.model_dump(mode="json") == output.model_dump(mode="json")
