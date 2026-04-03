"""Tests for GA and internal data models."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from src.models import (
    DocumentMap,
    ExtractionOutput,
    FileCoordinates,
    PageClassification,
    PageCoordinates,
    PageExtractionResult,
    RawExtraction,
    TaskBrief,
    TierResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ── Round-trip tests ───────────────────────────────────────────────


class TestFileCoordinatesRoundtrip:
    def test_load_and_dump_matches(self):
        raw = _load("sample_file_coordinates.json")
        fc = FileCoordinates(**raw)
        dumped = json.loads(fc.model_dump_json())
        assert dumped == raw

    def test_page_with_table(self):
        raw = _load("sample_page_with_table.json")
        pc = PageCoordinates(**raw)
        assert len(pc.tables) == 1
        assert len(pc.tables[0].rows) == 2
        assert len(pc.tables[0].columns) == 2
        assert len(pc.tables[0].cells) == 2
        assert len(pc.tables[0].cells_flatten) == 4

    def test_page_narrative(self):
        raw = _load("sample_page_narrative.json")
        pc = PageCoordinates(**raw)
        assert len(pc.tables) == 0
        assert len(pc.word_boxes) == 10


class TestExtractionOutputRoundtrip:
    def test_load_and_dump_matches(self):
        raw = _load("sample_extraction_output.json")
        eo = ExtractionOutput(**raw)
        dumped = json.loads(eo.model_dump_json())
        assert dumped == raw


# ── Cross-reference linkage tests ──────────────────────────────────


class TestCoordinateLinkage:
    @pytest.fixture()
    def fc(self) -> FileCoordinates:
        return FileCoordinates(**_load("sample_file_coordinates.json"))

    @pytest.fixture()
    def eo(self) -> ExtractionOutput:
        return ExtractionOutput(**_load("sample_extraction_output.json"))

    def _all_coordinate_ids(self, fc: FileCoordinates) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        for page in fc.pages_coordinates:
            for wb in page.word_boxes:
                ids.add(wb.coordinate_id)
            for wb in page.extraction_word_boxes:
                ids.add(wb.coordinate_id)
            for table in page.tables:
                ids.add(table.coordinate_id)
                for cell in table.cells_flatten:
                    ids.add(cell.coordinate_id)
                for row in table.rows:
                    ids.add(row.coordinate_id)
                for col in table.columns:
                    ids.add(col.coordinate_id)
            for lo in page.layout_objects:
                ids.add(lo.coordinate_id)
        return ids

    def _all_extraction_coord_ids(self, eo: ExtractionOutput) -> set[uuid.UUID]:
        zero = uuid.UUID(int=0)
        ids: set[uuid.UUID] = set()
        containers = [eo.attributes] + eo.address
        for ol in eo.products:
            containers.append(ol)
        for container in containers:
            for field_extractions in container.extractions.values():
                for ext in field_extractions:
                    if ext.coordinate_id != zero:
                        ids.add(ext.coordinate_id)
        return ids

    def _all_field_name_coord_ids(self, eo: ExtractionOutput) -> set[uuid.UUID]:
        zero = uuid.UUID(int=0)
        ids: set[uuid.UUID] = set()
        containers = [eo.attributes] + eo.address
        for ol in eo.products:
            containers.append(ol)
        for container in containers:
            for field_extractions in container.extractions.values():
                for ext in field_extractions:
                    if ext.field_name_coordinates_id != zero:
                        ids.add(ext.field_name_coordinates_id)
        return ids

    def test_coordinate_id_linkage(self, fc, eo):
        """Every non-zero Extraction.coordinate_id must exist in FileCoordinates."""
        fc_ids = self._all_coordinate_ids(fc)
        ext_ids = self._all_extraction_coord_ids(eo)
        missing = ext_ids - fc_ids
        assert not missing, f"Extraction coordinate_ids not found in FileCoordinates: {missing}"

    def test_field_name_coordinates_id_linkage(self, fc, eo):
        """Every non-zero field_name_coordinates_id must exist in FileCoordinates."""
        fc_ids = self._all_coordinate_ids(fc)
        field_ids = self._all_field_name_coord_ids(eo)
        missing = field_ids - fc_ids
        assert not missing, f"field_name_coordinates_ids not found in FileCoordinates: {missing}"

    def test_source_id_linkage(self, fc, eo):
        """Every OrderLine.source_id must match a Table.coordinate_id."""
        table_ids = set()
        for page in fc.pages_coordinates:
            for table in page.tables:
                table_ids.add(table.coordinate_id)
        for ol in eo.products:
            assert ol.source_id in table_ids, (
                f"OrderLine.source_id {ol.source_id} not found in any Table"
            )

    def test_content_id_match(self, fc, eo):
        """ExtractionOutput.meta.content_id must match FileCoordinates.content_id."""
        assert str(eo.meta.content_id) == str(fc.content_id)


# ── Internal model tests ──────────────────────────────────────────


class TestInternalModels:
    def test_task_brief(self):
        tb = TaskBrief(
            page_filters=[3, 5],
            deadline=datetime(2026, 4, 15),
            special_instructions=["All prices in EUR"],
            sender_id="procurement@acme-corp.de",
            attachment_filenames=["PO-2024-00891_Acme_Corp.pdf"],
        )
        data = json.loads(tb.model_dump_json())
        tb2 = TaskBrief(**data)
        assert tb2.page_filters == [3, 5]
        assert tb2.sender_id == "procurement@acme-corp.de"

    def test_document_map(self):
        dm = DocumentMap(
            classifications=[
                PageClassification(
                    page_index=0,
                    page_type="cover_page",
                    has_vision_table=False,
                    vision_confidence=0.12,
                    relevant=False,
                ),
                PageClassification(
                    page_index=1,
                    page_type="product_table",
                    has_vision_table=True,
                    vision_confidence=0.95,
                    relevant=True,
                ),
            ],
            total_pages=2,
            relevant_pages=[1],
        )
        data = json.loads(dm.model_dump_json())
        dm2 = DocumentMap(**data)
        assert dm2.total_pages == 2
        assert dm2.relevant_pages == [1]
        assert dm2.classifications[1].page_type == "product_table"

    def test_raw_extraction(self):
        re = RawExtraction(
            field_name="product_number",
            value="WG-4420-BLK",
            field_label_text="Article No.",
            confidence=0.92,
        )
        data = json.loads(re.model_dump_json())
        re2 = RawExtraction(**data)
        assert re2.field_name == "product_number"
        assert re2.value == "WG-4420-BLK"
        assert re2.field_label_text == "Article No."
        assert re2.confidence == 0.92

    def test_raw_extraction_defaults(self):
        re = RawExtraction(field_name="qty", value="500")
        assert re.field_label_text is None
        assert re.confidence == 0.0

    def test_page_extraction_result(self):
        per = PageExtractionResult(
            extractions=[
                RawExtraction(field_name="product_number", value="WG-4420-BLK"),
                RawExtraction(field_name="quantity", value="500"),
            ],
            is_product_table=True,
            cross_references=["See Section 4.2"],
        )
        data = json.loads(per.model_dump_json())
        per2 = PageExtractionResult(**data)
        assert len(per2.extractions) == 2
        assert per2.is_product_table is True
        assert per2.cross_references == ["See Section 4.2"]

    def test_tier_result(self):
        tr = TierResult(tier="tier1", page_index=3, grounded=True)
        data = json.loads(tr.model_dump_json())
        tr2 = TierResult(**data)
        assert tr2.tier == "tier1"
        assert tr2.grounded is True
