"""The database recorder of a pharmaceutical run (PDP8-T03; REQ-PER-02 to REQ-PER-04).

``DatabaseRecorder`` writes a ``PharmaRunOutcome`` into the run's ``pharma_run_records`` row,
which ``QueryService.create`` opened with the request's input. It writes through the workflow's
own session and does not commit: the workflow's next commit (``_persist_answer``) stores the
record and the answer together, and a run that fails before it rolls both back.

The evidence snapshot is a copy (REQ-PER-03): each candidate's anchor values, the SHA-256 of its
text, a bounded and redacted excerpt, its scores and rank, and whether it reached the final
context or why version resolution excluded it. Nothing in it is a reference the documents can
change later.
"""

import hashlib
import uuid
from typing import Protocol

from app.agents.prompts import flatten
from app.core.logging import bounded_excerpt
from app.pharma.response import ABSTAINING
from app.pharma.workflow import ConsideredPassage, PharmaRunOutcome
from app.schemas.pharma import MAX_EXCERPT_LENGTH

EXCERPT_LIMIT = MAX_EXCERPT_LENGTH
# The flattened text a snapshot excerpt is cut from; anything longer is cut by the excerpt limit.
_FLATTEN_LIMIT = 4 * MAX_EXCERPT_LENGTH


class RunRecordWriter(Protocol):
    async def save_outcome(
        self,
        query_run_id: uuid.UUID,
        *,
        rule_version: str,
        evidence_snapshot: list[dict[str, object]],
        sufficiency: dict[str, object],
        conflicts: list[dict[str, object]],
        policy: dict[str, object],
        escalation: dict[str, object],
        response: dict[str, object],
    ) -> None: ...


def _snapshot_entry(passage: ConsideredPassage) -> dict[str, object]:
    chunk, anchor = passage.chunk, passage.anchor
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "chunk_index": anchor.chunk_index,
        "document_number": anchor.document_number,
        "revision": anchor.revision,
        "title": anchor.title,
        "section": anchor.section,
        "page": anchor.page,
        "lifecycle_status": anchor.lifecycle_status.value,
        "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        "excerpt": bounded_excerpt(flatten(chunk.text, _FLATTEN_LIMIT), EXCERPT_LIMIT),
        "semantic_score": chunk.semantic_score,
        "lexical_score": chunk.lexical_score,
        "hybrid_score": chunk.hybrid_score,
        "rerank_score": chunk.rerank_score,
        "rank": chunk.rank,
        "included": passage.in_context,
        "exclusion_reason": passage.exclusion.value if passage.exclusion is not None else None,
    }


def snapshot_of(outcome: PharmaRunOutcome) -> list[dict[str, object]]:
    """Every candidate the run considered, in its persisted order (REQ-PER-02)."""
    return [_snapshot_entry(passage) for passage in outcome.considered]


class DatabaseRecorder:
    def __init__(self, records: RunRecordWriter) -> None:
        self._records = records

    async def record(self, outcome: PharmaRunOutcome) -> None:
        assessment = outcome.assessment
        response = outcome.response
        await self._records.save_outcome(
            outcome.query_run_id,
            rule_version=assessment.rule_version,
            evidence_snapshot=snapshot_of(outcome),
            sufficiency={
                "sufficiency": assessment.sufficiency.value,
                "rule_version": assessment.rule_version,
                "anchors": list(assessment.anchors),
                "covered_anchors": list(assessment.covered_anchors),
                "question_terms": list(assessment.question_terms),
                "best_coverage": assessment.best_coverage,
                "eligible_count": assessment.eligible_count,
                "effective_count": assessment.effective_count,
                "unknown_only": assessment.unknown_only,
                "abstained": outcome.abstained,
            },
            conflicts=[
                conflict.model_dump(mode="json")
                for conflict in response.conflicts_or_ambiguities.deterministic
            ],
            policy={
                "findings": [
                    {"code": f.code.value, "field": f.field, "sentence_sha256": f.sentence_sha256}
                    for f in outcome.findings
                ],
                "notices": [notice.model_dump(mode="json") for notice in outcome.notices],
                # PLV4-T03: the layered check's trace, present when something was removed, so a
                # clean run's record is as before.
                **({"trace": outcome.trace} if outcome.trace else {}),
            },
            escalation=response.required_human_escalation.model_dump(mode="json"),
            response={
                "sections": response.model_dump(mode="json"),
                "answer_markdown": outcome.answer_markdown,
                "banner": outcome.banner,
                # What the result shows (RMD3-T01): every abstention kind is one, while the
                # sufficiency field above keeps the sufficiency rule's own abstention.
                "abstained": outcome.disposition in ABSTAINING,
                "disposition": outcome.disposition.value,
                "disposition_gate": outcome.disposition_gate.value,
                "signals": list(outcome.signals.matched),
            },
        )
