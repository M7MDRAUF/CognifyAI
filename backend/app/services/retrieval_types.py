"""Shared retrieval value objects (FR-RET-003)."""

import uuid
from dataclasses import dataclass

from app.domain.enums import EvidenceSource

__all__ = ["EvidenceSource", "RetrievedChunk", "ScoredChunk"]


@dataclass(frozen=True)
class ScoredChunk:
    """A merged hybrid-retrieval candidate before hydration/reranking."""

    chunk_id: uuid.UUID
    semantic_score: float | None
    lexical_score: float | None
    hybrid_score: float
    source: EvidenceSource


@dataclass(frozen=True)
class RetrievedChunk:
    """A final, hydrated piece of evidence with text and citation metadata."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_filename: str
    text: str
    page_number: int | None
    section_title: str | None
    semantic_score: float | None
    lexical_score: float | None
    hybrid_score: float
    rerank_score: float | None
    # Dense, unique, global over the final reranked set: 0..K-1, no gap and no collision.
    rank: int
    source: EvidenceSource
    # Which retrieval round found this chunk (0 = initial, 1 = expansion), and its rank *within*
    # that round. Kept for diagnostics only and never used for ordering -- the collision N-004
    # describes is exactly what happens when a round-local rank is mistaken for a global one.
    round_index: int = 0
    round_rank: int | None = None
