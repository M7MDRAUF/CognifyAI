"""Critic agent (FR-AGENT-004)."""

from collections.abc import Sequence

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import SAFETY_RULES, compact_json, format_evidence, format_question
from app.integrations.llm.base import ModelRole
from app.schemas.agents import CriticOutput, Finding
from app.services.retrieval_types import RetrievedChunk

_SYSTEM = (
    "You are the critic agent. Check whether each finding is supported by the evidence. List "
    "supported claims with their chunk ids, list unsupported claims, estimate hallucination risk "
    "(0..1), and decide whether more retrieval is needed (with expansion queries if so).\n"
    + SAFETY_RULES
)


class CriticAgent:
    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        findings: Sequence[Finding],
        evidence: Sequence[RetrievedChunk],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[CriticOutput]:
        findings_json = compact_json([finding.model_dump(mode="json") for finding in findings])
        user = (
            f"Question: {format_question(normalized_question)}\n\n"
            f"Findings:\n{findings_json}\n\n"
            f"Evidence:\n{format_evidence(evidence)}\n\n"
            'Return JSON: {"supported_claims": [{"claim": str, "chunk_ids": [uuid]}], '
            '"unsupported_claims": [str], "hallucination_risk": float, '
            '"needs_more_retrieval": bool, "expansion_queries": [str]}'
        )
        return await self._runner.run(_SYSTEM, user, CriticOutput, role=role)
