"""Knowledge graph client wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core import Graphiti


class TenantKGClient:
    """Tenant-scoped Graphiti knowledge graph client."""

    def __init__(
        self,
        tenant_id: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "password",
    ):
        self.tenant_id = tenant_id
        # Import at runtime so graphiti-core is optional
        from graphiti_core import Graphiti

        self.graphiti: Graphiti = Graphiti(
            uri=neo4j_uri, user=neo4j_user, password=neo4j_pass
        )

    async def initialize(self) -> None:
        """Call once at startup to build indices."""
        await self.graphiti.build_indices_and_constraints()

    async def close(self) -> None:
        await self.graphiti.close()


class NullKGClient:
    """No-op KG client for when KG is not configured."""

    def __init__(self, tenant_id: str = "null"):
        self.tenant_id = tenant_id

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass
