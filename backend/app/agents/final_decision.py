"""Final decision agent (FR-AGENT-007)."""

from collections.abc import Sequence

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import SAFETY_RULES, compact_json, format_evidence, format_question
from app.domain.enums import ResponseStyle
from app.integrations.llm.base import ModelRole
from app.schemas.agents import CriticOutput, FinalDecisionOutput, Finding, RiskOutput
from app.services.retrieval_types import RetrievedChunk

_SYSTEM = (
    "You are the final decision agent. Using only the evidence, write a clear recommendation in "
    "Markdown, cite each supporting passage by its key in square brackets (the key= on its "
    "evidence line, for example [C1]) and give that key as the citation_key of its citations "
    "entry, surface contradictory evidence, state uncertainty, and list next steps. Do not "
    "present unsupported claims as facts. When the question can refer to more than one activity, "
    "item or value in the evidence, say that it is ambiguous and name each reading first, then "
    "answer each reading from the evidence or ask which one is meant.\n" + SAFETY_RULES
)


class FinalDecisionAgent:
    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        response_style: ResponseStyle,
        findings: Sequence[Finding],
        critic: CriticOutput,
        risk: RiskOutput,
        evidence: Sequence[RetrievedChunk],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[FinalDecisionOutput]:
        findings_json = compact_json([finding.model_dump(mode="json") for finding in findings])
        user = (
            f"Question: {format_question(normalized_question)}\n"
            f"Response style: {response_style.value}\n\n"
            f"Findings:\n{findings_json}\n\n"
            f"Critic notes: {compact_json(critic.model_dump(mode='json'))}\n\n"
            f"Risks: {compact_json(risk.model_dump(mode='json'))}\n\n"
            f"Evidence:\n{format_evidence(evidence, keyed=True)}\n\n"
            'Return JSON: {"recommendation": str, "answer_markdown": str, '
            '"citations": [{"citation_key": str, "chunk_id": uuid, "snippet": str}], '
            '"contradictory_evidence": [str], "uncertainty": str, "next_steps": [str]}'
        )
        return await self._runner.run(_SYSTEM, user, FinalDecisionOutput, role=role)
