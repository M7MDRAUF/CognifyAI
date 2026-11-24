"""Codes and fixed texts the policy engine, the escalation rules, the response renderer and the
client all read from one place (PDP3-T03).

The wording follows PHARMA_DOMAIN_INTENDED_USE_AND_BOUNDARIES.md (sections 6 to 8) and
PHARMA_DOMAIN_REQUIREMENTS.md (sections 8 to 10). None of it claims compliance, validation or
approval: the prototype claims none (intended use, section 8).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from app.pharma.enums import EscalationRole, PolicyCode


@dataclass(frozen=True)
class PolicyRule:
    code: PolicyCode
    title: str
    # True for the codes checked on the response's structure (escalation, abstention,
    # disclosure) rather than on its sentences.
    structural: bool
    notice: str


@dataclass(frozen=True)
class EscalationRule:
    code: str
    roles: tuple[EscalationRole, ...]
    text: str


_QA = EscalationRole.QUALITY_ASSURANCE

POLICY_RULES: Mapping[PolicyCode, PolicyRule] = MappingProxyType(
    {
        PolicyCode.ROOT_CAUSE: PolicyRule(
            PolicyCode.ROOT_CAUSE,
            "Final root-cause claim",
            False,
            "A statement asserting a final root cause was removed; the root cause is determined by "
            "the site's investigation.",
        ),
        PolicyCode.BATCH_DISPOSITION: PolicyRule(
            PolicyCode.BATCH_DISPOSITION,
            "Batch release or disposition advice",
            False,
            "A statement about releasing or dispositioning a batch was removed; batch disposition "
            "is the quality unit's decision.",
        ),
        PolicyCode.APPROVAL_LANGUAGE: PolicyRule(
            PolicyCode.APPROVAL_LANGUAGE,
            "Approval or closure language",
            False,
            "A statement approving or closing an SOP, a procedure change, a deviation or a CAPA "
            "was removed; approval belongs to the site's quality process.",
        ),
        PolicyCode.AUTONOMOUS_DECISION: PolicyRule(
            PolicyCode.AUTONOMOUS_DECISION,
            "Autonomous regulated decision",
            False,
            "A statement taking a regulated decision, or advising against involving the quality "
            "unit, was removed.",
        ),
        PolicyCode.PROCEDURE_OVERRIDE: PolicyRule(
            PolicyCode.PROCEDURE_OVERRIDE,
            "Overriding an approved procedure",
            False,
            "A statement advising to skip, bypass or deviate from an approved procedure step was "
            "removed.",
        ),
        PolicyCode.HIGH_RISK_WITHOUT_ESCALATION: PolicyRule(
            PolicyCode.HIGH_RISK_WITHOUT_ESCALATION,
            "High risk without escalation",
            True,
            "A high-severity risk was identified, so human escalation was made required.",
        ),
        PolicyCode.INSUFFICIENT_EVIDENCE: PolicyRule(
            PolicyCode.INSUFFICIENT_EVIDENCE,
            "Fluent output on insufficient evidence",
            True,
            "The controlled evidence was insufficient, so the response states what is missing "
            "instead of answering.",
        ),
        PolicyCode.UNDISCLOSED_NON_EFFECTIVE: PolicyRule(
            PolicyCode.UNDISCLOSED_NON_EFFECTIVE,
            "Superseded, retired, draft or unknown-status material without disclosure",
            True,
            "Evidence from a document that is not Effective, or whose status is unknown, is "
            "disclosed under Conflicts or Ambiguities.",
        ),
        PolicyCode.MEDICAL_ADVICE: PolicyRule(
            PolicyCode.MEDICAL_ADVICE,
            "Medical, patient-specific or prescribing advice",
            False,
            "A statement giving medical, patient-specific or prescribing advice was removed.",
        ),
        PolicyCode.COMPLIANCE_CLAIM: PolicyRule(
            PolicyCode.COMPLIANCE_CLAIM,
            "Compliance, validation or approval claim",
            False,
            "A statement claiming regulatory compliance, validation or approval for the system or "
            "its output was removed.",
        ),
    }
)

ESCALATION_RULES: Mapping[str, EscalationRule] = MappingProxyType(
    {
        "REQ-ESC-01": EscalationRule(
            "REQ-ESC-01",
            (_QA, EscalationRole.PROCESS_OWNER),
            "An operational incident is reviewed by quality assurance and the process owner.",
        ),
        "REQ-ESC-02": EscalationRule(
            "REQ-ESC-02",
            (_QA, EscalationRole.SUBJECT_MATTER_EXPERT),
            "The controlled evidence is insufficient or partial; quality assurance and a "
            "subject-matter expert must review what is missing.",
        ),
        "REQ-ESC-03": EscalationRule(
            "REQ-ESC-03",
            (EscalationRole.DOCUMENT_CONTROL, _QA),
            "Documents conflict or a non-Effective version matched; document control and quality "
            "assurance must resolve which version applies.",
        ),
        "REQ-ESC-04": EscalationRule(
            "REQ-ESC-04",
            (_QA,),
            "A high-severity, product-quality or data-integrity risk was identified.",
        ),
        "REQ-ESC-05": EscalationRule(
            "REQ-ESC-05",
            (_QA,),
            "The regulatory-boundary check removed text from this response.",
        ),
        "REQ-ESC-06": EscalationRule(
            "REQ-ESC-06",
            (EscalationRole.DOCUMENT_CONTROL,),
            "Retrieved evidence or the input contains text that reads as an instruction to the "
            "system; document control should review the source.",
        ),
        "REQ-ESC-07": EscalationRule(
            "REQ-ESC-07",
            (),
            "Reasons suggested by the analysis model are listed after the rules' reasons; the "
            "model can add reasons but never remove one.",
        ),
        # PLV9-T03 (PLV-D-033, PLV-DEF-024): the process owner confirms whether a process is ready.
        "REQ-ESC-08": EscalationRule(
            "REQ-ESC-08",
            (EscalationRole.PROCESS_OWNER,),
            "A readiness review that the controlled evidence does not settle is confirmed by the "
            "process owner.",
        ),
    }
)

# The reason shown for a role the analysis model names for review (PLV9-T03, PLV-D-033): one fixed
# text per role, so the escalation section holds no model text for it (REQ-ESC-07's merge rule:
# the model can add a role, never remove one).
MODEL_ROLE_TEXTS: Mapping[EscalationRole, str] = MappingProxyType(
    {
        EscalationRole.QUALITY_ASSURANCE: (
            "The analysis model asked quality assurance to review this answer."
        ),
        EscalationRole.PROCESS_OWNER: (
            "The analysis model asked the process owner to review this answer."
        ),
        EscalationRole.SUBJECT_MATTER_EXPERT: (
            "The analysis model asked a subject-matter expert to review this answer."
        ),
        EscalationRole.DOCUMENT_CONTROL: (
            "The analysis model asked document control to review this answer."
        ),
    }
)

# (PharmaResponse field, heading), in the intended use's order (section 6).
RESPONSE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("incident_or_question_summary", "Incident or Question Summary"),
    ("relevant_procedures", "Relevant Procedures"),
    ("evidence_and_citations", "Evidence and Citations"),
    ("quality_and_operational_risks", "Quality and Operational Risks"),
    ("conflicts_or_ambiguities", "Conflicts or Ambiguities"),
    ("missing_information", "Missing Information"),
    ("required_human_escalation", "Required Human Escalation"),
    (
        "suggested_investigation_questions_or_next_steps",
        "Suggested Investigation Questions or Next Steps",
    ),
    ("limitations_and_uncertainty", "Limitations and Uncertainty"),
)

HUMAN_REVIEW_BANNER = (
    "Decision support for qualified personnel. This research prototype does not approve "
    "procedures, release batches, determine root causes or replace the site's quality process; "
    "a qualified person reviews every output."
)

PROTOTYPE_LIMITATION = (
    "This is a research prototype evaluated only on synthetic and public documents. It is not "
    "validated for regulated use, and its output is not a controlled record."
)

NONE_IDENTIFIED = "None identified."
