"""Tests for the knowledge graph client and reader/writer functions."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from src.knowledge.kg_client import NullKGClient
from src.knowledge.kg_reader import (
    get_crossref_patterns,
    get_past_resolutions,
    get_sender_patterns,
    get_term_mappings,
)
from src.knowledge.kg_writer import (
    ingest_extraction_correction,
    ingest_product_resolution,
    ingest_sender_pattern,
)


# ---------------------------------------------------------------------------
# NullKGClient — returns empty results for all queries
# ---------------------------------------------------------------------------


class TestNullKGClient:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self):
        client = NullKGClient()
        await client.initialize()
        await client.close()

    @pytest.mark.asyncio
    async def test_sender_patterns_empty(self):
        client = NullKGClient()
        result = await get_sender_patterns(client, "sender_123")
        assert result == {"patterns": [], "known_aliases": [], "section_mappings": []}

    @pytest.mark.asyncio
    async def test_term_mappings_empty(self):
        client = NullKGClient()
        result = await get_term_mappings(client, ["Bestell-Nr.", "Menge"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_past_resolutions_empty(self):
        client = NullKGClient()
        result = await get_past_resolutions(client, "Widget X")
        assert result == []

    @pytest.mark.asyncio
    async def test_crossref_patterns_empty(self):
        client = NullKGClient()
        result = await get_crossref_patterns(client, "sender_123")
        assert result == []


# ---------------------------------------------------------------------------
# Reader functions with mocked Graphiti
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    fact: str
    score: float = 0.9


class TestKGReaderMocked:
    def _make_client(self) -> AsyncMock:
        """Create a mock TenantKGClient with a mocked graphiti.search."""
        client = AsyncMock()
        client.tenant_id = "test_tenant"
        # Make isinstance check for NullKGClient fail
        client.__class__ = type("TenantKGClient", (), {})
        return client

    @pytest.mark.asyncio
    async def test_get_sender_patterns_returns_dict(self):
        client = self._make_client()
        client.graphiti.search = AsyncMock(
            return_value=[
                FakeSearchResult(fact="Orders always have cover page"),
                FakeSearchResult(fact="Known alias: ACME Inc"),
                FakeSearchResult(fact="Section 4 mapping to specs"),
            ]
        )
        result = await get_sender_patterns(client, "acme_corp")
        assert "patterns" in result
        assert "known_aliases" in result
        assert "section_mappings" in result
        assert len(result["known_aliases"]) == 1
        assert len(result["section_mappings"]) == 1
        assert len(result["patterns"]) == 1

    @pytest.mark.asyncio
    async def test_get_term_mappings_returns_mapping(self):
        client = self._make_client()
        client.graphiti.search = AsyncMock(
            return_value=[FakeSearchResult(fact="product_number")]
        )
        result = await get_term_mappings(client, ["Bestell-Nr."])
        assert result == {"Bestell-Nr.": "product_number"}

    @pytest.mark.asyncio
    async def test_get_term_mappings_empty_for_unknown(self):
        client = self._make_client()
        client.graphiti.search = AsyncMock(return_value=[])
        result = await get_term_mappings(client, ["unknown_term"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_past_resolutions_returns_list(self):
        client = self._make_client()
        client.graphiti.search = AsyncMock(
            return_value=[FakeSearchResult(fact="SKU-123", score=0.95)]
        )
        result = await get_past_resolutions(client, "Widget X")
        assert len(result) == 1
        assert result[0]["resolved_product"] == "SKU-123"
        assert result[0]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_get_crossref_patterns_returns_list(self):
        client = self._make_client()
        client.graphiti.search = AsyncMock(
            return_value=[FakeSearchResult(fact="Annex B contains pricing tables")]
        )
        result = await get_crossref_patterns(client, "sender_123")
        assert len(result) == 1
        assert result[0]["pattern"] == "Annex B contains pricing tables"


# ---------------------------------------------------------------------------
# Writer functions — verify episode args
# ---------------------------------------------------------------------------


class TestKGWriter:
    def _make_client(self) -> AsyncMock:
        client = AsyncMock()
        client.tenant_id = "test_tenant"
        client.graphiti.add_episode = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_ingest_extraction_correction(self):
        client = self._make_client()
        correction = {
            "field": "product_number",
            "wrong_value": "WG-4420",
            "correct_value": "WG-4420-BLK",
            "context": "Acme Corp tender, page 5",
        }
        with patch("src.knowledge.kg_writer.EpisodeType", create=True) as mock_ep:
            mock_ep.text = "text"
            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"graphiti_core": type("mod", (), {"EpisodeType": mock_ep})()},
            ):
                await ingest_extraction_correction(client, correction)

        client.graphiti.add_episode.assert_called_once()
        call_kwargs = client.graphiti.add_episode.call_args
        body = call_kwargs.kwargs["episode_body"]
        assert "WG-4420" in body
        assert "WG-4420-BLK" in body
        assert "product_number" in body
        assert call_kwargs.kwargs["source_description"] == "HITL_correction"

    @pytest.mark.asyncio
    async def test_ingest_product_resolution(self):
        client = self._make_client()
        with patch.dict(
            "sys.modules",
            {
                "graphiti_core": type(
                    "mod", (), {"EpisodeType": type("E", (), {"text": "text"})()}
                )()
            },
        ):
            await ingest_product_resolution(
                client, "Widget X", "SKU-123", "acme_corp"
            )

        client.graphiti.add_episode.assert_called_once()
        body = client.graphiti.add_episode.call_args.kwargs["episode_body"]
        assert "Widget X" in body
        assert "SKU-123" in body
        assert "acme_corp" in body

    @pytest.mark.asyncio
    async def test_ingest_sender_pattern(self):
        client = self._make_client()
        with patch.dict(
            "sys.modules",
            {
                "graphiti_core": type(
                    "mod", (), {"EpisodeType": type("E", (), {"text": "text"})()}
                )()
            },
        ):
            await ingest_sender_pattern(
                client, "acme_corp", "Specs always in Section 4"
            )

        client.graphiti.add_episode.assert_called_once()
        body = client.graphiti.add_episode.call_args.kwargs["episode_body"]
        assert "acme_corp" in body
        assert "Section 4" in body
        assert (
            client.graphiti.add_episode.call_args.kwargs["source_description"]
            == "sender_pattern"
        )
