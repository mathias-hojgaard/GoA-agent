"""AdvancedValidation agent: flag non-product lines and misaligned values."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from src.models.ga_models import AdvancedValidation, Extraction

VALIDATION_SYSTEM_PROMPT = """\
Review these product line extractions from a tender document.
Flag any lines that are NOT actual product lines (shipping fees, subtotals,
tax lines, section headers mistakenly extracted).
Flag values that appear misaligned (e.g., a product description in a price column).
For each flagged line, specify: status (delete/modify), reason, and whether
this should be a suggestion (human confirms) or auto_apply (safe to auto-fix).

Respond with a list of validation issues found. If no issues, return an empty list.
"""


class ValidationIssue(BaseModel):
    """A single validation issue found by the agent."""

    extraction_index: int
    status: Literal["delete", "add", "modify"]
    message: str
    details: Literal["suggestion", "auto_apply"]


class ValidationResult(BaseModel):
    """Result from the validation agent."""

    issues: list[ValidationIssue] = []


def _get_validation_agent() -> Agent[None, ValidationResult]:
    """Lazily create the validation agent to avoid requiring API key at import time."""
    return Agent(
        "google-gla:gemini-2.0-flash",
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        output_type=ValidationResult,
    )


async def validate_extractions(
    extractions: list[Extraction],
    agent: Agent[None, ValidationResult] | None = None,
) -> list[AdvancedValidation]:
    """Run the validation agent on extractions and return AdvancedValidation items.

    Each returned AdvancedValidation is also mapped back onto the corresponding
    Extraction's advanced_validation field.

    Pass an agent explicitly (e.g. with TestModel override) to avoid needing
    a real API key.
    """
    if not extractions:
        return []

    if agent is None:
        agent = _get_validation_agent()

    # Build a textual summary for the agent
    lines = []
    for i, ext in enumerate(extractions):
        lines.append(
            f"[{i}] field={ext.field_name} value={ext.extracted_value!r} "
            f"certainty={ext.extraction_certainty:.3f}"
        )
    prompt = "Extractions to validate:\n" + "\n".join(lines)

    result = await agent.run(prompt)
    validations: list[AdvancedValidation] = []

    for issue in result.output.issues:
        av = AdvancedValidation(
            status=issue.status,
            message=issue.message,
            details=issue.details,
            url="",
        )
        validations.append(av)

        # Map back to the extraction
        if 0 <= issue.extraction_index < len(extractions):
            extractions[issue.extraction_index].advanced_validation.append(av)

    return validations
