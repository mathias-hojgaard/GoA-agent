"""Enums used across GA and internal models."""

from enum import Enum


class CellTypes(str, Enum):
    SPANNING_CELL = "spanning_cell"
    NON_SPANNING_CELL = "non_spanning_cell"


class ExtractionSource(str, Enum):
    PDF = "pdf"
    FREETEXT = "freetext"
    IMAGE = "image"
    EXCEL = "excel"
