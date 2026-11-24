"""Authority-aware final ordering for pharmaceutical runs (REQ-RET-08; ADR-006).

The reranker's full ranking is ordered by ``rerank_score`` (descending; unscored behind every
scored chunk, as in ADR-006), then by the document's authority tier for the inquiry's intent,
then by ``hybrid_score`` (descending), then by ``chunk_id`` (ascending), and cut to the final
context size with dense ranks ``0..K-1``. The tier only breaks ties of the reranker score: a
higher-scored reference document is never pushed below a lower-scored procedure.

For procedure, incident and readiness questions the tiers are: a controlled site procedure or
work instruction (0), another site document such as a form (1), a vendor manual (2), a regulation
or other reference (3), and a document without valid controlled metadata (4).
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import (
    ControlledDocumentType,
    ControlStatus,
    LifecycleStatus,
    PharmaQueryIntent,
    SourceAuthority,
)
from app.services.retrieval_types import RetrievedChunk

PROCEDURE_TYPES = frozenset({ControlledDocumentType.SOP, ControlledDocumentType.WORK_INSTRUCTION})
UNSCORED = -1.0
UNKNOWN_TIER = 4

# Intent -> source authority -> tier. The same order serves every current intent; it is a table
# so a later intent can state its own, and so the rule a run used is named in its record.
AUTHORITY_TIERS: Mapping[PharmaQueryIntent, Mapping[SourceAuthority, int]] = {
    intent: {
        SourceAuthority.SITE: 1,
        SourceAuthority.VENDOR: 2,
        SourceAuthority.REGULATOR: 3,
        SourceAuthority.OTHER: 3,
    }
    for intent in PharmaQueryIntent
}


def authority_tier(info: ControlledDocumentInfo | None, intent: PharmaQueryIntent) -> int:
    if info is None or info.lifecycle_status is LifecycleStatus.UNKNOWN:
        return UNKNOWN_TIER
    if info.source_authority is None:
        return UNKNOWN_TIER
    if (
        info.source_authority is SourceAuthority.SITE
        and info.document_type in PROCEDURE_TYPES
        and info.control_status is ControlStatus.CONTROLLED
    ):
        return 0
    return AUTHORITY_TIERS[intent][info.source_authority]


def authority_order(
    ranking: Sequence[RetrievedChunk],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    intent: PharmaQueryIntent,
    k: int,
) -> list[RetrievedChunk]:
    """The final context: ``ranking`` reordered as above and cut to ``k``, ranks ``0..K-1``."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)

    def key(chunk: RetrievedChunk) -> tuple[float, int, float, str]:
        score = chunk.rerank_score if chunk.rerank_score is not None else UNSCORED
        tier = authority_tier(infos.get(chunk.document_id), intent)
        return (-score, tier, -chunk.hybrid_score, str(chunk.chunk_id))

    ordered = sorted(ranking, key=key)[:k]
    return [replace(chunk, rank=rank) for rank, chunk in enumerate(ordered)]
