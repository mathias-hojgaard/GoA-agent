"""Stage 1: Email parser agent.

Uses PydanticAI with Gemini to extract a TaskBrief from the incoming
tender email. The agent identifies page filters, deadlines, special
instructions, and sender identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from src.models import TaskBrief

#: Default model used in production. Override in tests via the model parameter.
DEFAULT_MODEL = "google-gla:gemini-2.0-flash"


@dataclass
class EmailParserDeps:
    """Dependencies injected into the email parser agent."""

    kg_context: dict[str, Any] | None = None


SYSTEM_PROMPT = """\
You are an email parser for a tender intelligence pipeline. Your job is to
extract structured information from incoming procurement/tender emails.

Extract the following fields:

1. **page_filters** — If the sender specifies particular pages to quote or
   process (e.g. "only quote products on page 3 and page 5"), return them as
   a list of 1-indexed page numbers (integers). If no specific pages are
   mentioned, return null.

2. **deadline** — If a bid/quote deadline is mentioned, parse it as an ISO
   datetime. Handle European date formats (DD/MM/YYYY). If no deadline is
   mentioned, return null.

3. **special_instructions** — A list of any special requirements or
   instructions (e.g. "All prices must be in EUR", "Lead time must not
   exceed 6 weeks", "quote only stainless steel"). Return an empty list if
   none are found.

4. **sender_id** — Identify the sender. Use their email address if visible
   in a From: header, otherwise use their name and company. Return null if
   the sender cannot be identified.

5. **attachment_filenames** — List any attachment filenames mentioned in the
   email. Return an empty list if none are found.

Return a JSON object matching the TaskBrief schema exactly.
"""

email_parser_agent: Agent[EmailParserDeps, TaskBrief] = Agent(
    DEFAULT_MODEL,
    output_type=TaskBrief,
    system_prompt=SYSTEM_PROMPT,
    deps_type=EmailParserDeps,
    defer_model_check=True,
)


@email_parser_agent.system_prompt
async def add_kg_context(ctx) -> str:
    """Append knowledge-graph context about the sender when available."""
    deps: EmailParserDeps = ctx.deps
    if deps.kg_context:
        return (
            "\n\nAdditional context about the sender from the knowledge graph:\n"
            + json.dumps(deps.kg_context, default=str, ensure_ascii=False)
        )
    return ""


async def parse_email(
    email_body: str,
    kg_context: dict[str, Any] | None = None,
    *,
    model: Model | str | None = None,
) -> TaskBrief:
    """Parse an incoming email and return a TaskBrief.

    Args:
        email_body: The raw email text (headers + body).
        kg_context: Optional knowledge-graph context about the sender,
            e.g. previous ordering patterns or preferred products.
        model: LLM model to use. Defaults to DEFAULT_MODEL (Gemini 2.0 Flash).
            Pass a TestModel instance in tests.

    Returns:
        A TaskBrief with page filters, deadline, special instructions,
        sender identity, and attachment filenames extracted from the email.

    Note:
        page_filters are returned as 1-indexed page numbers (as written in
        the email). The page classifier is responsible for converting to
        0-indexed when matching against FileCoordinates.
    """
    deps = EmailParserDeps(kg_context=kg_context)
    kwargs: dict[str, Any] = {"deps": deps}
    if model is not None:
        kwargs["model"] = model
    result = await email_parser_agent.run(email_body, **kwargs)
    return result.output
