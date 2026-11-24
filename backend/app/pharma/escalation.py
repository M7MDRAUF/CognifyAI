"""Deterministic human escalation (REQ-ESC-01 to REQ-ESC-07).

The rules decide whether a response requires human escalation, to which roles and why; the
analysis model can add reasons but never remove a rule's reason or role (REQ-ESC-07). Rules,
in order:

* REQ-ESC-01: an operational incident: quality assurance and the process owner;
* REQ-ESC-02: insufficient or partial evidence: quality assurance and a subject-matter expert;
* REQ-ESC-03: any conflict: document control and quality assurance;
* REQ-ESC-04: a high-severity risk flag, a product-quality or data-integrity flag, or a
  quality-relevant signal: quality assurance;
* REQ-ESC-05: any removal by the regulatory-boundary check: quality assurance;
* REQ-ESC-06: suspected instruction text in the evidence or the input: document control.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.enums import RiskLevel
from app.pharma.catalogue import ESCALATION_RULES, MODEL_ROLE_TEXTS
from app.pharma.enums import (
    EscalationRole,
    EvidenceSufficiency,
    PharmaQueryIntent,
    PharmaRiskCategory,
)
from app.pharma.signals import RiskSignals
from app.schemas.pharma import EscalationReason, HumanEscalationRead, RiskFlag

# Model-suggested reasons carry REQ-ESC-07's code with source "model".
MODEL_REASON_CODE = "REQ-ESC-07"
_QUALITY_CATEGORIES = frozenset(
    {PharmaRiskCategory.PRODUCT_QUALITY, PharmaRiskCategory.DATA_INTEGRITY}
)


@dataclass(frozen=True)
class EscalationInputs:
    intent: PharmaQueryIntent
    sufficiency: EvidenceSufficiency
    conflict_count: int
    risk_flags: Sequence[RiskFlag]
    policy_removals: int
    signals: RiskSignals


def _fired(inputs: EscalationInputs) -> list[str]:
    fired: list[str] = []
    if inputs.intent is PharmaQueryIntent.OPERATIONAL_INCIDENT:
        fired.append("REQ-ESC-01")
    if inputs.sufficiency in {EvidenceSufficiency.INSUFFICIENT, EvidenceSufficiency.PARTIAL}:
        fired.append("REQ-ESC-02")
    if inputs.conflict_count > 0:
        fired.append("REQ-ESC-03")
    if (
        any(
            flag.severity is RiskLevel.HIGH or flag.category in _QUALITY_CATEGORIES
            for flag in inputs.risk_flags
        )
        or inputs.signals.quality_relevant
    ):
        fired.append("REQ-ESC-04")
    if inputs.policy_removals > 0:
        fired.append("REQ-ESC-05")
    if inputs.signals.instruction_text:
        fired.append("REQ-ESC-06")
    if inputs.intent is PharmaQueryIntent.READINESS_REVIEW and inputs.sufficiency in {
        EvidenceSufficiency.INSUFFICIENT,
        EvidenceSufficiency.PARTIAL,
    }:
        fired.append("REQ-ESC-08")
    return fired


def decide_escalation(inputs: EscalationInputs) -> HumanEscalationRead:
    roles: list[EscalationRole] = []
    reasons: list[EscalationReason] = []
    for code in _fired(inputs):
        rule = ESCALATION_RULES[code]
        roles.extend(role for role in rule.roles if role not in roles)
        reasons.append(EscalationReason(code=code, text=rule.text, source="rule"))
    return HumanEscalationRead(required=bool(reasons), roles=roles, reasons=reasons)


def with_model_roles(
    decided: HumanEscalationRead, roles: Sequence[EscalationRole]
) -> HumanEscalationRead:
    """``decided`` with each role the analysis model named that it lacks, each under a fixed
    reason of REQ-ESC-07's code (PLV9-T03, PLV-D-033): the model can add a role, never remove
    one, and a role it adds makes escalation required."""
    added = [role for role in dict.fromkeys(roles) if role not in decided.roles]
    if not added:
        return decided
    merged = decided.model_copy(deep=True)
    merged.required = True
    for role in added:
        merged.roles.append(role)
        merged.reasons.append(
            EscalationReason(code=MODEL_REASON_CODE, text=MODEL_ROLE_TEXTS[role], source="model")
        )
    return merged


def merge_model_reasons(
    decided: HumanEscalationRead, model_flags: Sequence[str]
) -> HumanEscalationRead:
    """The rules' escalation with the model's reasons appended after them (REQ-ESC-07).

    The rules' reasons and roles are copied first and never filtered, so no model input can
    remove one. A model reason alone makes escalation required; REQ-ESC-07 names no role, so
    quality assurance, which reviews every output (the banner), receives it."""
    merged = decided.model_copy(deep=True)
    seen = {reason.text.strip().casefold() for reason in merged.reasons}
    for text in model_flags:
        cleaned = " ".join(text.split())
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        merged.reasons.append(
            EscalationReason(code=MODEL_REASON_CODE, text=cleaned, source="model")
        )
    if any(reason.source == "model" for reason in merged.reasons):
        merged.required = True
        if EscalationRole.QUALITY_ASSURANCE not in merged.roles:
            merged.roles.append(EscalationRole.QUALITY_ASSURANCE)
    return merged
