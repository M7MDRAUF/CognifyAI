"""The pharmaceutical planner, critic, risk and decision agents (PDP7-T01; REQ-AGT-01 to 05).

Each runs on the baseline ``JsonAgentRunner``, so the validate, repair and strict-retry ladder,
metering and the "addresses the schema" test are the General agents'. The researcher and the
evaluator are the baseline agents, unchanged; so is every General prompt (REQ-AGT-08).

User prompts follow the baseline's containment: request text is flattened onto labelled lines,
model text travels as one-line JSON (``compact_json``), incident text and evidence are fenced.
"""

import uuid
from collections.abc import Mapping, Sequence

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import compact_json, fence, flatten, format_label_list, format_question
from app.integrations.llm.base import ModelRole
from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import EvidenceSufficiency, PharmaQueryIntent
from app.pharma.prompts import (
    CRITIC_SYSTEM,
    DECISION_SYSTEM,
    PLANNER_SYSTEM,
    RISK_SYSTEM,
    format_incident,
    format_pharma_evidence,
)
from app.schemas.agents import Finding
from app.schemas.pharma import (
    IncidentInput,
    PharmaCriticOutput,
    PharmaDecisionOutput,
    PharmaFilters,
    PharmaPlannerOutput,
    PharmaRiskOutput,
)
from app.schemas.queries import MAX_QUESTION_LENGTH
from app.services.retrieval_types import RetrievedChunk

_PLANNER_SHAPE = (
    'Return JSON: {"normalized_question": str, "sub_questions": [str], '
    '"search_queries": [str], "missing_facts": [str], "escalation_considerations": [str]}'
)
_CRITIC_SHAPE = (
    'Return JSON: {"unsupported_claims": [str], "wrong_version_use": [str], '
    '"missing_prerequisites": [str], "citation_errors": [str], '
    '"needs_more_retrieval": bool, "expansion_queries": [str]}'
)
_RISK_SHAPE = (
    'Return JSON: {"flags": [{"category": "operational|product_quality|equipment|'
    'documentation|data_integrity|escalation", "severity": "low|medium|high", '
    '"description": str, "chunk_ids": [the key of a passage\'s label line, such as E1]}]}'
)
_DECISION_SHAPE = (
    'Return JSON: {"summary": str, "points": [{"text": str, "citation_keys": [str], '
    '"action_category": "none|batch_disposition|deviation_closure|deviation_classification|'
    "capa_closure|root_cause|compliance_declaration|procedure_override|equipment_command|"
    'medical_instruction|qualified_person|review_waiver"}], "missing_information": [str], '
    '"investigation_questions": [str], "uncertainty": [str], '
    '"citations": [{"citation_key": the key of a passage\'s label line, such as E1, '
    '"snippet": str}], '
    '"answerability": one of "full", "partial", "none" or "conflict", '
    '"escalate_to": ["process_owner|document_control|quality_assurance|subject_matter_expert"]}'
)


# RMD3-T03 (RPH-25): the critic's request for more retrieval, which no step runs, is not passed to
# the decision, where it leaned the answer towards abstention.
_UNRUN = {"needs_more_retrieval", "expansion_queries"}


def _filters_line(filters: PharmaFilters) -> str:
    values = [
        *filters.equipment_ids,
        *filters.areas,
        *filters.processes,
        *(kind.value for kind in filters.document_types),
    ]
    return format_label_list(values, empty="(none)")


def _findings(findings: Sequence[Finding]) -> str:
    return compact_json([finding.model_dump(mode="json") for finding in findings])


class PharmaPlannerAgent:
    """REQ-AGT-01: decomposes the question; sees no evidence."""

    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        question: str,
        intent: PharmaQueryIntent,
        incident: IncidentInput | None,
        filters: PharmaFilters,
    ) -> AgentResult[PharmaPlannerOutput]:
        # The filter values are free text a person typed, so they are fenced as the incident is
        # (REQ-SEC-02, PH-DEF-035).
        user = (
            f"Question: {flatten(question, MAX_QUESTION_LENGTH)}\n"
            f"Intent: {intent.value}\n"
            f"Filters:\n{fence('filters', _filters_line(filters))}\n\n"
            f"Incident:\n{format_incident(incident)}\n\n"
            f"{_PLANNER_SHAPE}"
        )
        result = await self._runner.run(
            PLANNER_SYSTEM, user, PharmaPlannerOutput, role=ModelRole.SMALL
        )
        if not result.output.search_queries:
            # A plan with no query falls back to the question, as the General planner does.
            result.output.search_queries = [result.output.normalized_question or question]
        return result


class PharmaCriticAgent:
    """REQ-AGT-03: unsupported claims, wrong-version use, missing prerequisites, citations."""

    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        findings: Sequence[Finding],
        evidence: Sequence[RetrievedChunk],
        infos: Mapping[uuid.UUID, ControlledDocumentInfo],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[PharmaCriticOutput]:
        user = (
            f"Question: {format_question(normalized_question)}\n\n"
            f"Findings:\n{_findings(findings)}\n\n"
            f"Evidence:\n{format_pharma_evidence(evidence, infos)}\n\n"
            f"{_CRITIC_SHAPE}"
        )
        return await self._runner.run(CRITIC_SYSTEM, user, PharmaCriticOutput, role=role)


class PharmaRiskAgent:
    """REQ-AGT-04: flags with a category, a severity and supporting chunk ids in a field."""

    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        incident: IncidentInput | None,
        findings: Sequence[Finding],
        evidence: Sequence[RetrievedChunk],
        infos: Mapping[uuid.UUID, ControlledDocumentInfo],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[PharmaRiskOutput]:
        user = (
            f"Question: {format_question(normalized_question)}\n\n"
            f"Incident:\n{format_incident(incident)}\n\n"
            f"Findings:\n{_findings(findings)}\n\n"
            f"Evidence:\n{format_pharma_evidence(evidence, infos)}\n\n"
            f"{_RISK_SHAPE}"
        )
        return await self._runner.run(RISK_SYSTEM, user, PharmaRiskOutput, role=role)


class PharmaDecisionAgent:
    """REQ-AGT-05: the response's model-written parts as structured fields."""

    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        normalized_question: str,
        intent: PharmaQueryIntent,
        incident: IncidentInput | None,
        sufficiency: EvidenceSufficiency,
        findings: Sequence[Finding],
        critic: PharmaCriticOutput,
        risk: PharmaRiskOutput,
        evidence: Sequence[RetrievedChunk],
        infos: Mapping[uuid.UUID, ControlledDocumentInfo],
        *,
        role: ModelRole = ModelRole.STRONG,
    ) -> AgentResult[PharmaDecisionOutput]:
        user = (
            f"Question: {format_question(normalized_question)}\n"
            f"Intent: {intent.value}\n"
            f"Evidence sufficiency: {sufficiency.value}\n\n"
            f"Incident:\n{format_incident(incident)}\n\n"
            f"Findings:\n{_findings(findings)}\n\n"
            f"Critic notes: {compact_json(critic.model_dump(mode='json', exclude=_UNRUN))}\n\n"
            f"Risk flags: {compact_json(risk.model_dump(mode='json'))}\n\n"
            f"Evidence:\n{format_pharma_evidence(evidence, infos)}\n\n"
            f"{_DECISION_SHAPE}"
        )
        return await self._runner.run(DECISION_SYSTEM, user, PharmaDecisionOutput, role=role)
