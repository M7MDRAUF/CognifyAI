"""Risk agent (FR-AGENT-005)."""

from collections.abc import Sequence

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import SAFETY_RULES, compact_json, format_label_list, format_question
from app.integrations.llm.base import ModelRole
from app.schemas.agents import Finding, RiskOutput

_SYSTEM = (
    "You are the risk agent. Identify the risks implied by the findings, each in one category: "
    "implementation, ethical, privacy, data_quality, operational or academic. Rate each by "
    "severity and likelihood (low/medium/high) and suggest a mitigation.\n" + SAFETY_RULES
)


class RiskAgent:
    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        findings: Sequence[Finding],
        risk_focus: Sequence[str],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[RiskOutput]:
        findings_json = compact_json([finding.model_dump(mode="json") for finding in findings])
        # `risk_focus` is `PlannerOutput.risk_focus`: model output, joined onto a line of the
        # instruction region, so one entry carrying a line terminator would write a line there.
        focus = format_label_list(risk_focus, empty="(none specified)")
        user = (
            f"Question: {format_question(normalized_question)}\n"
            f"Risk focus: {focus}\n\n"
            f"Findings:\n{findings_json}\n\n"
            'Return JSON: {"risks": [{"category": '
            '"implementation|ethical|privacy|data_quality|operational|academic", '
            '"description": str, "severity": "low|medium|high", '
            '"likelihood": "low|medium|high", "mitigation": str}]}'
        )
        return await self._runner.run(_SYSTEM, user, RiskOutput, role=role)
