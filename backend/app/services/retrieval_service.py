"""Hybrid retrieval: semantic + lexical, normalized and merged (SDD section 9).

Defaults (FR-RET): semantic top-k 30, lexical top-k 30, merged pool 30,
weights semantic 0.65 / lexical 0.35. Score normalization, merge, and
deduplication are pure functions so they can be unit-tested exhaustively.
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.errors import RetrievalFailedError
from app.domain.enums import RetrievalMode
from app.integrations.qdrant.client import QdrantService
from app.integrations.qdrant.collections import SemanticHit
from app.repositories.chunk_repository import ChunkRepository
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.retrieval_types import EvidenceSource, RetrievedChunk, ScoredChunk

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalConfig",
    "RetrievalMode",
    "RetrievalService",
    "merge_candidates",
    "normalize_scores",
]

DEFAULT_SEMANTIC_TOP_K = 30
DEFAULT_LEXICAL_TOP_K = 30
DEFAULT_MERGED_LIMIT = 30
SEMANTIC_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35


# Past this fraction of candidates lost, the stage fails rather than answering from what is
# left. Chosen as "most of the evidence is gone": below it a few stale vectors degrade an answer,
# above it the answer is being built from a different corpus than the one that was retrieved.
_HYDRATION_FAILURE_THRESHOLD = 0.5

# Enough ids to diagnose which document went missing, bounded so a wholesale loss does not put a
# thousand identifiers on one log line.
_MAX_LOGGED_DROPPED_IDS = 10


@dataclass(frozen=True)
class HydrationReport:
    """What the hydration step asked for, got, lost, and left out.

    Carried on the service rather than returned, for the same reason the reranker's usage is:
    every caller of ``retrieve`` wants chunks, and widening the return type would ripple through
    the retriever and the orchestrator to serve one observer.

    ``excluded`` counts candidates whose document is failed or processing (RMD6-T02): they are
    not the document's current version, so they are left out on purpose and are not a loss. The
    drop rate is measured over the rest, the candidates that could have been evidence.
    """

    requested: int = 0
    hydrated: int = 0
    dropped: int = 0
    excluded: int = 0

    @property
    def drop_rate(self) -> float:
        eligible = self.requested - self.excluded
        return 0.0 if eligible <= 0 else self.dropped / eligible

    def exceeds_failure_threshold(self, threshold: float) -> bool:
        return self.requested > 0 and self.drop_rate > threshold


@dataclass(frozen=True)
class RetrievalConfig:
    semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K
    lexical_top_k: int = DEFAULT_LEXICAL_TOP_K
    merged_limit: int = DEFAULT_MERGED_LIMIT
    semantic_weight: float = SEMANTIC_WEIGHT
    lexical_weight: float = LEXICAL_WEIGHT


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize to 0..1; identical or empty inputs map to 1.0."""
    if not scores:
        return []
    lowest = min(scores)
    highest = max(scores)
    spread = highest - lowest
    if spread == 0:
        return [1.0 for _ in scores]
    return [(score - lowest) / spread for score in scores]


def merge_candidates(
    semantic: Sequence[tuple[uuid.UUID, float]],
    lexical: Sequence[tuple[uuid.UUID, float]],
    config: RetrievalConfig,
) -> list[ScoredChunk]:
    """Normalize each list, combine by weighted sum, dedupe by chunk_id."""
    semantic_norm = dict(
        zip(
            (chunk_id for chunk_id, _ in semantic),
            normalize_scores([score for _, score in semantic]),
            strict=True,
        )
    )
    lexical_norm = dict(
        zip(
            (chunk_id for chunk_id, _ in lexical),
            normalize_scores([score for _, score in lexical]),
            strict=True,
        )
    )
    raw_semantic = dict(semantic)
    raw_lexical = dict(lexical)
    merged: list[ScoredChunk] = []
    for chunk_id in {*semantic_norm, *lexical_norm}:
        sem = semantic_norm.get(chunk_id)
        lex = lexical_norm.get(chunk_id)
        hybrid = config.semantic_weight * (sem or 0.0) + config.lexical_weight * (lex or 0.0)
        if sem is not None and lex is not None:
            source = EvidenceSource.BOTH
        elif sem is not None:
            source = EvidenceSource.SEMANTIC
        else:
            source = EvidenceSource.LEXICAL
        merged.append(
            ScoredChunk(
                chunk_id=chunk_id,
                semantic_score=raw_semantic.get(chunk_id),
                lexical_score=raw_lexical.get(chunk_id),
                hybrid_score=hybrid,
                source=source,
            )
        )
    merged.sort(key=lambda candidate: candidate.hybrid_score, reverse=True)
    return merged[: config.merged_limit]


class RetrievalService:
    def __init__(
        self,
        qdrant: QdrantService,
        embeddings: EmbeddingService,
        lexical: LexicalSearchService,
        chunks: ChunkRepository,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._embeddings = embeddings
        self._lexical = lexical
        self._chunks = chunks
        # Counts from the most recent `retrieve`, reset per call by `_hydrate`.
        self.last_hydration = HydrationReport()
        self._config = config or RetrievalConfig()

    async def retrieve(
        self,
        project_id: uuid.UUID,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> list[RetrievedChunk]:
        hits = await self._semantic(project_id, query, mode)
        semantic = [(hit.chunk_id, hit.score) for hit in hits]
        lexical = await self._lexical_hits(project_id, query, mode)
        merged = merge_candidates(semantic, lexical, self._config)
        hit_documents = {hit.chunk_id: hit.document_id for hit in hits if hit.document_id}
        return await self._hydrate(merged, hit_documents)

    async def _semantic(
        self, project_id: uuid.UUID, query: str, mode: RetrievalMode
    ) -> list[SemanticHit]:
        if mode is RetrievalMode.LEXICAL:
            return []
        vector = await self._embeddings.embed_query(query)
        return await self._qdrant.search(project_id, vector, self._config.semantic_top_k)

    async def _lexical_hits(
        self, project_id: uuid.UUID, query: str, mode: RetrievalMode
    ) -> list[tuple[uuid.UUID, float]]:
        if mode is RetrievalMode.SEMANTIC:
            return []
        return await self._lexical.search(project_id, query, self._config.lexical_top_k)

    async def _hydrate(
        self,
        merged: Sequence[ScoredChunk],
        hit_documents: dict[uuid.UUID, uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        """Turn scored ids into full chunks, and **count what did not survive** (N-009).

        A chunk present in Qdrant but absent from PostgreSQL was skipped with a bare ``continue``
        and the ranks renumbered densely over the survivors, so the gap was erased twice: the row
        vanished and the rank sequence closed over it, leaving nothing to notice. That happens
        for real -- a deleted document whose vectors outlive its rows, a partially-failed
        ingestion -- and it silently shrinks the evidence an answer is built from.

        A drop is a *data integrity* signal, not a retrieval preference, so past a threshold the
        stage fails rather than quietly answering from whatever is left.

        A candidate whose document is failed or processing is not hydrated either (RMD6-T02,
        RE2E-DEF-035), and is counted as excluded, not dropped: its document is known, by its
        row or by the document its point names (``hit_documents``), and is not ready. The points
        of an ingestion killed before it committed are such candidates (OBS-RMD-10).
        """
        requested = len(merged)
        candidate_ids = [candidate.chunk_id for candidate in merged]
        rows = await self._chunks.get_many_with_filename(candidate_ids)
        missing = [chunk_id for chunk_id in candidate_ids if chunk_id not in rows]
        not_ready = await self._chunks.not_ready_candidates(missing, hit_documents or {})
        hydrated: list[RetrievedChunk] = []
        dropped: list[uuid.UUID] = []
        excluded = 0
        rank = 0
        for candidate in merged:
            row = rows.get(candidate.chunk_id)
            if row is None:
                if candidate.chunk_id in not_ready:
                    excluded += 1
                else:
                    dropped.append(candidate.chunk_id)
                continue
            chunk, filename = row
            hydrated.append(
                RetrievedChunk(
                    chunk_id=candidate.chunk_id,
                    document_id=chunk.document_id,
                    source_filename=filename,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    semantic_score=candidate.semantic_score,
                    lexical_score=candidate.lexical_score,
                    hybrid_score=candidate.hybrid_score,
                    rerank_score=None,
                    rank=rank,
                    source=candidate.source,
                )
            )
            rank += 1
        self.last_hydration = HydrationReport(
            requested=requested, hydrated=len(hydrated), dropped=len(dropped), excluded=excluded
        )
        # The loss is counted over the candidates that could have been evidence, as the
        # threshold is; the excluded ones are named apart (RMD6-T02).
        eligible = requested - excluded
        left_out = (
            f" ({excluded} more were left out: their document is failed or processing)"
            if excluded
            else ""
        )
        if dropped:
            # The ids are logged, not the text: a chunk id is not a credential (the log
            # formatter's UUID exemption is explicit about that) and it is the only thing that
            # makes the loss diagnosable against the corpus.
            logger.warning(
                "Hydration dropped %d of %d candidates absent from PostgreSQL%s: %s",
                len(dropped),
                eligible,
                left_out,
                ", ".join(str(chunk_id) for chunk_id in dropped[:_MAX_LOGGED_DROPPED_IDS]),
                extra={"event_type": "hydration_drop"},
            )
        if self.last_hydration.exceeds_failure_threshold(_HYDRATION_FAILURE_THRESHOLD):
            msg = (
                f"Retrieval hydration lost {len(dropped)} of {eligible} candidates{left_out}, "
                f"past the {_HYDRATION_FAILURE_THRESHOLD:.0%} threshold. Answering from the "
                "remainder would silently narrow the evidence."
            )
            raise RetrievalFailedError(msg)
        return hydrated
