"""Retriever agent (FR-AGENT-002).

Executes hybrid retrieval for each planned query, pools and deduplicates the
candidates, then reranks to the final context. Uses the retrieval services
(SDD: agents never touch Qdrant directly).
"""

import uuid
from collections.abc import Sequence
from dataclasses import replace

from app.domain.enums import RetrievalMode
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.services.retrieval_types import RetrievedChunk


class RetrieverAgent:
    def __init__(self, retrieval: RetrievalService, reranking: RerankingService) -> None:
        self._retrieval = retrieval
        self._reranking = reranking

    async def gather(
        self,
        project_id: uuid.UUID,
        queries: Sequence[str],
        mode: RetrievalMode,
        *,
        round_index: int = 0,
    ) -> list[RetrievedChunk]:
        """The deduplicated candidate pool for one retrieval round, **unreranked**.

        Split out of `retrieve` so an expansion can union round one's *candidates* with round
        two's and rerank the union once (C-10, ADR-006). Reranking each round separately and
        concatenating the results delivered up to 2K chunks downstream, and a round-two chunk
        had never been scored on the same scale as a round-one chunk.

        Deduplication is by `chunk_id` and keeps the higher `hybrid_score`: the same chunk found
        by two queries is one piece of evidence, and the better retrieval of it is the one to
        keep.
        """
        pooled: dict[uuid.UUID, RetrievedChunk] = {}
        for query in queries:
            for candidate in await self._retrieval.retrieve(project_id, query, mode):
                existing = pooled.get(candidate.chunk_id)
                if existing is None or candidate.hybrid_score > existing.hybrid_score:
                    pooled[candidate.chunk_id] = replace(candidate, round_index=round_index)
        merged = sorted(pooled.values(), key=lambda chunk: chunk.hybrid_score, reverse=True)
        return [replace(chunk, round_rank=rank) for rank, chunk in enumerate(merged)]

    async def rerank(
        self, rerank_question: str, candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Rerank a candidate set and cut it to the one global top-K."""
        return await self._reranking.rerank(rerank_question, candidates)

    async def retrieve(
        self,
        project_id: uuid.UUID,
        queries: Sequence[str],
        mode: RetrievalMode,
        rerank_question: str,
        *,
        round_index: int = 0,
    ) -> list[RetrievedChunk]:
        """Gather and rerank in one step, for the single-round case."""
        candidates = await self.gather(project_id, queries, mode, round_index=round_index)
        return await self.rerank(rerank_question, candidates)
