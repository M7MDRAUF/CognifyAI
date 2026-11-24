"""Lexical retrieval via PostgreSQL full-text search (SDD section 9.2)."""

import uuid

from app.repositories.chunk_repository import ChunkRepository


class LexicalSearchService:
    def __init__(self, chunks: ChunkRepository) -> None:
        self._chunks = chunks

    async def search(
        self, project_id: uuid.UUID, query: str, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        cleaned = query.strip()
        if cleaned == "":
            return []
        return await self._chunks.lexical_search(project_id, cleaned, limit)
