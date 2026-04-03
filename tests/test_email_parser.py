"""Tests for the email parser agent."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic_ai.models.test import TestModel

from src.ingestion.email_parser import parse_email
from src.models import TaskBrief


@pytest.fixture
def sample_email() -> str:
    return (
        "From: procurement@acme-corp.de\n"
        "To: tenders@goautonomous.io\n"
        "Subject: RFQ - Industrial Workwear Supply 2024/Q2\n"
        "Date: Mon, 11 Mar 2024 09:14:32 +0100\n\n"
        "Dear Go Autonomous Team,\n\n"
        "Please find attached our Purchase Order PO-2024-00891.\n\n"
        "Only quote products on page 3 and page 5.\n"
        "We need to have the bid no later than 15/04/2026.\n\n"
        "Special instructions:\n"
        "- All prices must be in EUR\n"
        "- Delivery to our Hamburg warehouse (address on page 1)\n"
        "- Lead time must not exceed 6 weeks\n\n"
        "Best regards,\n"
        "Klaus Weber\n"
        "Procurement Department\n"
        "Acme Corporation GmbH\n\n"
        "Attachment: PO-2024-00891_Acme_Corp.pdf (278 KB)\n"
    )


@pytest.mark.asyncio
async def test_parse_email_with_page_filters(sample_email: str):
    """Simulated extraction with page filters returns correct list."""
    model = TestModel(custom_output_args={
        "page_filters": [3, 5],
        "sender_id": "procurement@acme-corp.de",
        "attachment_filenames": ["PO-2024-00891_Acme_Corp.pdf"],
        "special_instructions": [
            "All prices must be in EUR",
            "Delivery to our Hamburg warehouse (address on page 1)",
            "Lead time must not exceed 6 weeks",
        ],
        "deadline": "2026-04-15T00:00:00",
    })
    result = await parse_email(sample_email, model=model)
    assert isinstance(result, TaskBrief)
    assert result.page_filters == [3, 5]
    assert result.sender_id == "procurement@acme-corp.de"
    assert result.deadline == datetime(2026, 4, 15)
    assert len(result.special_instructions) == 3
    assert "All prices must be in EUR" in result.special_instructions
    assert result.attachment_filenames == ["PO-2024-00891_Acme_Corp.pdf"]


@pytest.mark.asyncio
async def test_parse_email_with_eu_date_deadline():
    """EU date format (DD/MM/YYYY) is correctly parsed into datetime."""
    model = TestModel(custom_output_args={
        "deadline": "2026-04-15T00:00:00",
    })
    result = await parse_email("bid no later than 15/04/2026", model=model)
    assert result.deadline == datetime(2026, 4, 15, 0, 0)


@pytest.mark.asyncio
async def test_parse_email_no_constraints():
    """Email with no page filters, deadline, or instructions returns empty defaults."""
    model = TestModel(custom_output_args={})
    email = (
        "From: info@company.com\n\n"
        "Hello, please review the attached document.\n\n"
        "Regards,\nJohn\n"
    )
    result = await parse_email(email, model=model)
    assert isinstance(result, TaskBrief)
    assert result.page_filters is None
    assert result.deadline is None
    assert result.special_instructions == []
    assert result.sender_id is None
    assert result.attachment_filenames == []


@pytest.mark.asyncio
async def test_parse_email_special_instructions():
    """Rush order with special constraints are captured."""
    model = TestModel(custom_output_args={
        "special_instructions": [
            "Rush order",
            "Quote only stainless steel items",
            "All dimensions must be in metric",
        ],
        "sender_id": "buyer@steel-co.com",
    })
    result = await parse_email(
        "Rush order — quote only stainless steel items.\n"
        "All dimensions must be in metric.\n",
        model=model,
    )
    assert len(result.special_instructions) == 3
    assert "Rush order" in result.special_instructions
    assert result.sender_id == "buyer@steel-co.com"


@pytest.mark.asyncio
async def test_parse_email_with_kg_context():
    """kg_context is accepted and does not break the pipeline."""
    model = TestModel(custom_output_args={
        "page_filters": [1, 2],
        "sender_id": "procurement@acme-corp.de",
    })
    kg_context = {
        "sender": "acme-corp",
        "previous_orders": 42,
        "preferred_products": ["WG-4420"],
    }
    result = await parse_email(
        "Please quote the attached tender.",
        kg_context=kg_context,
        model=model,
    )
    assert isinstance(result, TaskBrief)
    assert result.page_filters == [1, 2]


@pytest.mark.asyncio
async def test_parse_email_without_kg_context():
    """kg_context=None works without error."""
    model = TestModel(custom_output_args={"sender_id": "test@example.com"})
    result = await parse_email(
        "Please quote the attached tender.",
        kg_context=None,
        model=model,
    )
    assert isinstance(result, TaskBrief)
    assert result.sender_id == "test@example.com"


@pytest.mark.asyncio
async def test_task_brief_schema_completeness():
    """All TaskBrief fields are present and populated from model output."""
    model = TestModel(custom_output_args={
        "page_filters": [12, 15],
        "deadline": "2026-06-01T00:00:00",
        "special_instructions": ["metric only"],
        "sender_id": "buyer@corp.de",
        "attachment_filenames": ["tender.pdf", "annex.pdf"],
    })
    result = await parse_email("test email", model=model)
    assert result.page_filters == [12, 15]
    assert result.deadline == datetime(2026, 6, 1)
    assert result.special_instructions == ["metric only"]
    assert result.sender_id == "buyer@corp.de"
    assert result.attachment_filenames == ["tender.pdf", "annex.pdf"]
