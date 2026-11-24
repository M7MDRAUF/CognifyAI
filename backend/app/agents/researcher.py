"""Research agent (FR-AGENT-003)."""

import logging
from collections.abc import Sequence

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import SAFETY_RULES, format_evidence, format_question
from app.integrations.llm.base import ModelRole
from app.schemas.agents import Finding, ResearchOutput
from app.services.retrieval_types import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are the research agent. Synthesize the retrieved evidence into key findings grouped by "
    "theme, and list any contradictions between sources. Reference supporting chunk ids.\n"
    + SAFETY_RULES
)


class ResearchAgent:
    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner
        # How many invented chunk references the last run removed (N-010). Read by the
        # orchestrator's trace; a rate is only meaningful attributed to a run and an attempt.
        self.last_hallucinated_references = 0

    async def run(
        self,
        normalized_question: str,
        evidence: Sequence[RetrievedChunk],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[ResearchOutput]:
        user = (
            f"Question: {format_question(normalized_question)}\n\n"
            f"Evidence:\n{format_evidence(evidence)}\n\n"
            'Return JSON: {"key_findings": [{"statement": str, "theme": str, '
            '"supporting_chunk_ids": [uuid]}], "themes": [str], '
            '"contradictions": [{"description": str, "chunk_ids": [uuid]}]}'
        )
        result = await self._runner.run(_SYSTEM, user, ResearchOutput, role=role)
        result.output.key_findings, hallucinated = drop_unknown_refs(
            result.output.key_findings, evidence
        )
        if hallucinated:
            logger.warning(
                "Researcher cited %d chunk id(s) that were never retrieved; they were removed "
                "before persistence",
                hallucinated,
                extra={"event_type": "hallucinated_reference"},
            )
        self.last_hallucinated_references = hallucinated
        return result


def drop_unknown_refs(
    findings: list[Finding], evidence: Sequence[RetrievedChunk]
) -> tuple[list[Finding], int]:
    """Remove citations to chunks that were never retrieved, and **count them** (N-010).

    The filter itself is correct and stays: a model-invented chunk id must not reach the
    citation table. What was wrong is that it was silent. The filtered output was persisted as
    the model's own, so a run where the researcher invented half its references was
    indistinguishable from one where it invented none, and the rate could not be measured
    because nothing counted it.

    Returns the count alongside the findings rather than logging it here: the caller knows the
    agent, the run and the attempt, and a rate is only meaningful attributed to those.
    """
    known = {item.chunk_id for item in evidence}
    dropped = 0
    cleaned: list[Finding] = []
    for finding in findings:
        kept = [chunk_id for chunk_id in finding.supporting_chunk_ids if chunk_id in known]
        dropped += len(finding.supporting_chunk_ids) - len(kept)
        cleaned.append(finding.model_copy(update={"supporting_chunk_ids": kept}))
    return cleaned, dropped
