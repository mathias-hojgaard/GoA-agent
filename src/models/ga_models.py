"""Go Autonomous data models transcribed from the case study appendix.

All models inherit from GABaseModel (alias for pydantic.BaseModel).
Coordinates are percentages (0.0-1.0) of page dimensions.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict

from .enums import CellTypes, ExtractionSource


class GABaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ── Shared primitives ──────────────────────────────────────────────


class Rectangle(GABaseModel):
    """Bounding box primitive. All values are percentages (0.0-1.0)."""

    top: float
    left: float
    width: float
    height: float
    page: int = 0


class Prediction(GABaseModel):
    label: str
    certainty: float


class Classification(GABaseModel):
    label: str
    certainty: float
    predictions: List[Prediction]


class Relation(GABaseModel):
    related_id: uuid.UUID
    relation_type: str
    head_id: uuid.UUID
    tail_id: uuid.UUID
    relation_certainty: float


# ── FileCoordinates hierarchy ──────────────────────────────────────


class WordBox(GABaseModel):
    coordinates_rectangle: List[Rectangle]  # Always single-element list
    coordinate_id: uuid.UUID
    value: str


class Row(GABaseModel):
    classification: Classification
    coordinates_rectangle: Rectangle
    coordinate_id: uuid.UUID
    index: int
    detection_certainty: float
    relations: List[Relation]


class Column(GABaseModel):
    classification: Classification
    coordinates_rectangle: Rectangle
    coordinate_id: uuid.UUID
    index: int
    detection_certainty: float
    relations: List[Relation]


class Cell(GABaseModel):
    type: CellTypes
    coordinates_rectangle: Rectangle
    coordinate_id: uuid.UUID
    content: str
    column_index: int
    row_index: int
    column_id: uuid.UUID
    row_id: uuid.UUID


class Table(GABaseModel):
    classification: Classification
    coordinates_rectangle: Rectangle
    coordinate_id: uuid.UUID
    rows: List[Row]
    columns: List[Column]
    cells: List[List[Cell]]
    cells_flatten: List[Cell]
    relations: List[Relation]
    detection_certainty: float


class LayoutObject(GABaseModel):
    classification: Classification
    coordinates_rectangle: Rectangle
    coordinate_id: uuid.UUID
    relations: List[Relation]
    content: str


class PageCoordinates(GABaseModel):
    image_url: str
    image_url_v2: str
    page_height: float
    page_width: float
    page_name: str
    extraction_word_boxes: List[WordBox]
    word_boxes: List[WordBox]
    tables: List[Table]
    layout_objects: List[LayoutObject]


class FileCoordinates(GABaseModel):
    filename: str
    content_id: uuid.UUID
    content_type: str
    doc_class: str
    size: int
    date_archetype: str
    date_format: str
    decimal_separator: str
    pages_coordinates: List[PageCoordinates]


# ── ExtractionOutput hierarchy ─────────────────────────────────────


class AdvancedValidation(GABaseModel):
    status: Literal["delete", "add", "modify"]
    message: str
    details: Literal["suggestion", "auto_apply"]
    url: str


class Extraction(GABaseModel):
    source_of_extraction: ExtractionSource
    filename: str
    extraction_certainty: float
    similarity_to_confirmed_extractions: float
    genai_score: float
    coordinate_id: uuid.UUID
    field_name: str
    field_name_raw: str
    field_name_coordinates_id: uuid.UUID
    raw_saga_extraction: str
    raw_extracted_value: str
    extracted_value: str
    relations: List[Relation]
    coordinates_rectangle: List[Rectangle]
    message: str
    advanced_validation: List[AdvancedValidation]


class OrderLine(GABaseModel):
    extractions: Dict[str, List[Extraction]]
    source_id: uuid.UUID
    advanced_validation: List[AdvancedValidation]


class Attributes(GABaseModel):
    extractions: Dict[str, List[Extraction]]


class Address(GABaseModel):
    extractions: Dict[str, List[Extraction]]


class ImageResolution(GABaseModel):
    x_pixel: int
    y_pixel: int


class DocumentWarning(GABaseModel):
    pass


class Meta(GABaseModel):
    filename: str
    file_type: str
    content_id: Union[uuid.UUID, str]
    doc_class: str
    accuracy: str
    doc_class_certainty: float
    image_resolution: ImageResolution
    document_warnings: List[DocumentWarning]


class ExtractionOutput(GABaseModel):
    products: List[OrderLine]
    attributes: Attributes
    address: List[Address]
    meta: Meta
    keyword_extractions: dict
    source_of_extraction: ExtractionSource
