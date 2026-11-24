"""The pharmaceutical profile's API contracts (PDP3-T02; moved here under DV-PH-14 and DV-PH-16).

Two kinds of model, with the baseline's two validation policies:

* **API inputs** (``IncidentInput``, ``PharmaFilters``, ``PharmaQueryInput``) reject what cannot be
  stored or shown, like ``QueryCreate``: over-long values, control characters, unknown keys.
* **Read models** (``ControlledDocumentRead`` to ``PharmaRunRead``) describe what the API returns.
  Every pharmaceutical answer is the nine-section ``PharmaResponse``, rendered by the server from
  policy-checked fields (REQ-RSP-01, REQ-RSP-02).

This module imports no agent schema, so ``app.schemas.queries`` can use it: the agent schemas
import ``app.schemas.queries``, and the agent outputs live in ``app.schemas.pharma``, which
re-exports every public name of this module.
"""

import re
import uuid
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from app.domain.enums import RiskLevel
from app.pharma.enums import (
    ConflictKind,
    ControlledDocumentType,
    ControlStatus,
    DataClassification,
    DispositionGate,
    EscalationRole,
    EvidenceSufficiency,
    ExclusionReason,
    LifecycleStatus,
    MetadataStatus,
    PharmaDisposition,
    PharmaQueryIntent,
    PharmaRiskCategory,
    PolicyCode,
    SourceAuthority,
)
from app.schemas._text import is_visually_blank, reject_control_characters

# The incident form's bounds (intended use, section 5). Observations are the one field a person
# writes at length; the rest are labels.
MAX_INCIDENT_SUMMARY_LENGTH = 500
MAX_INCIDENT_FIELD_LENGTH = 2_000
MAX_INCIDENT_LABEL_LENGTH = 200
MAX_FILTER_VALUES = 8
MAX_FILTER_VALUE_LENGTH = 100

# The anchor pattern of the sufficiency rule (requirements, section 15) and the header contract:
# one to five capitals, an optional hyphen, one to five digits, an optional capital suffix.
EQUIPMENT_ID_PATTERN = r"^[A-Z]{1,5}-?[0-9]{1,5}[A-Z]?$"
_EQUIPMENT_ID = re.compile(EQUIPMENT_ID_PATTERN)


def _checked_label(value: str, field: str) -> str:
    reject_control_characters(value, field=field)
    return value.strip()


# --- API inputs ---------------------------------------------------------------------------------


class IncidentInput(BaseModel):
    """A structured operational incident (UC-PH-02). The deviation itself is handled through the
    site's own process; these fields only steer retrieval and the questions asked."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=MAX_INCIDENT_SUMMARY_LENGTH)
    area: str = Field(default="", max_length=MAX_INCIDENT_LABEL_LENGTH)
    equipment_id: str = Field(default="", max_length=MAX_INCIDENT_LABEL_LENGTH)
    process_step: str = Field(default="", max_length=MAX_INCIDENT_LABEL_LENGTH)
    product_family: str = Field(default="", max_length=MAX_INCIDENT_LABEL_LENGTH)
    observations: str = Field(default="", max_length=MAX_INCIDENT_FIELD_LENGTH)
    immediate_actions: str = Field(default="", max_length=MAX_INCIDENT_FIELD_LENGTH)

    @field_validator(
        "summary",
        "area",
        "equipment_id",
        "process_step",
        "product_family",
        "observations",
        "immediate_actions",
    )
    @classmethod
    def _no_control_characters(cls, value: str, info: ValidationInfo) -> str:
        return _checked_label(value, f"Incident {info.field_name}")

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        if value == "" or is_visually_blank(value):
            msg = "Incident summary must not be empty."
            raise ValueError(msg)
        return value

    @field_validator("equipment_id")
    @classmethod
    def _equipment_form(cls, value: str) -> str:
        if value and not _matches_equipment(value):
            msg = "Equipment identifier must look like TP-201 (capitals, optional hyphen, digits)."
            raise ValueError(msg)
        return value


def _matches_equipment(value: str) -> bool:
    return _EQUIPMENT_ID.fullmatch(value) is not None


class PharmaFilters(BaseModel):
    """Optional retrieval filters (REQ-RET-02). A document stating no value for a filtered field is
    kept; one stating a different value is excluded."""

    model_config = ConfigDict(extra="forbid")

    equipment_ids: list[str] = Field(default_factory=list[str], max_length=MAX_FILTER_VALUES)
    areas: list[str] = Field(default_factory=list[str], max_length=MAX_FILTER_VALUES)
    processes: list[str] = Field(default_factory=list[str], max_length=MAX_FILTER_VALUES)
    document_types: list[ControlledDocumentType] = Field(
        default_factory=list[ControlledDocumentType], max_length=len(ControlledDocumentType)
    )

    @field_validator("equipment_ids")
    @classmethod
    def _equipment_forms(cls, values: list[str]) -> list[str]:
        for value in values:
            if not _matches_equipment(value):
                msg = f"Equipment identifier {value!r} must look like TP-201."
                raise ValueError(msg)
        return values

    @field_validator("areas", "processes")
    @classmethod
    def _labels(cls, values: list[str]) -> list[str]:
        checked: list[str] = []
        for value in values:
            label = _checked_label(value, "Filter value")
            if not label or len(label) > MAX_FILTER_VALUE_LENGTH:
                msg = f"Filter values must be 1 to {MAX_FILTER_VALUE_LENGTH} characters."
                raise ValueError(msg)
            checked.append(label)
        return checked


class PharmaQueryInput(BaseModel):
    """The ``pharma`` object of a query on a pharmaceutical project (REQ-API-03)."""

    model_config = ConfigDict(extra="forbid")

    intent: PharmaQueryIntent = PharmaQueryIntent.PROCEDURE_QUESTION
    incident: IncidentInput | None = None
    filters: PharmaFilters = Field(default_factory=PharmaFilters)

    @model_validator(mode="after")
    def _incident_matches_intent(self) -> Self:
        is_incident = self.intent is PharmaQueryIntent.OPERATIONAL_INCIDENT
        if is_incident and self.incident is None:
            msg = "An operational incident needs its incident description."
            raise ValueError(msg)
        if not is_incident and self.incident is not None:
            msg = "An incident description is accepted only with the operational_incident intent."
            raise ValueError(msg)
        return self


MAX_EXCERPT_LENGTH = 600

UNKNOWN = "unknown"

RESPONSE_SECTION_FIELDS: tuple[str, ...] = (
    "incident_or_question_summary",
    "relevant_procedures",
    "evidence_and_citations",
    "quality_and_operational_risks",
    "conflicts_or_ambiguities",
    "missing_information",
    "required_human_escalation",
    "suggested_investigation_questions_or_next_steps",
    "limitations_and_uncertainty",
)


# --- Read models --------------------------------------------------------------------------------


class ControlledDocumentRead(BaseModel):
    """A document's controlled metadata, read-only (REQ-DOC-08). Absent or invalid values stay
    ``None``; the issues say why."""

    model_config = ConfigDict(from_attributes=True)

    document_number: str | None = None
    title: str | None = None
    document_type: ControlledDocumentType | None = None
    revision: str | None = None
    effective_date: date | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.UNKNOWN
    supersedes_number: str | None = None
    supersedes_revision: str | None = None
    site: str | None = None
    area: str | None = None
    equipment: list[str] = Field(default_factory=list[str])
    process: str | None = None
    product_family: str | None = None
    source_authority: SourceAuthority | None = None
    control_status: ControlStatus | None = None
    data_classification: DataClassification | None = None
    metadata_status: MetadataStatus = MetadataStatus.MISSING
    metadata_issues: list[str] = Field(default_factory=list[str])


class EvidenceAnchor(BaseModel):
    """Where a cited passage comes from, built from stored rows only (REQ-CHK-04, REQ-SEC-03)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    document_number: str = UNKNOWN
    revision: str = UNKNOWN
    title: str = UNKNOWN
    section: str = UNKNOWN
    page: int | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.UNKNOWN


class EvidenceItem(BaseModel):
    citation_key: str
    anchor: EvidenceAnchor
    excerpt: str = Field(default="", max_length=MAX_EXCERPT_LENGTH)


class ExcludedEvidenceRead(BaseModel):
    """A candidate version resolution kept out of the agents' context, disclosed (REQ-RET-06)."""

    anchor: EvidenceAnchor
    reason: ExclusionReason


class ConflictRead(BaseModel):
    kind: ConflictKind
    description: str
    document_numbers: list[str] = Field(default_factory=list[str])
    citation_keys: list[str] = Field(default_factory=list[str])


class ConflictsSection(BaseModel):
    """REQ-CON-02: the deterministic list and the model's reported contradictions, never merged."""

    deterministic: list[ConflictRead] = Field(default_factory=list[ConflictRead])
    model_reported: list[str] = Field(default_factory=list[str])


class RiskFlag(BaseModel):
    category: PharmaRiskCategory
    severity: RiskLevel
    description: str
    citation_keys: list[str] = Field(default_factory=list[str])


class EscalationReason(BaseModel):
    code: str
    text: str
    source: Literal["rule", "model"]


class HumanEscalationRead(BaseModel):
    required: bool
    roles: list[EscalationRole] = Field(default_factory=list[EscalationRole])
    reasons: list[EscalationReason] = Field(default_factory=list[EscalationReason])


class PolicyNotice(BaseModel):
    """What the regulatory-boundary check changed, shown with the result (REQ-POL-11)."""

    code: PolicyCode
    field: str
    action: Literal[
        "sentence_removed", "escalation_forced", "abstention_enforced", "disclosure_added"
    ]
    notice: str


class RelevantProcedure(BaseModel):
    document_number: str
    revision: str
    title: str
    lifecycle_status: LifecycleStatus
    sections: list[str] = Field(default_factory=list[str])
    summary: str = ""
    citation_keys: list[str] = Field(default_factory=list[str])


class PharmaResponse(BaseModel):
    """The nine sections of the intended use, in order, each present even when empty.

    REQ-RSP-01. ``RESPONSE_SECTION_FIELDS`` lists them, and a test pins this model to it.
    """

    incident_or_question_summary: str = ""
    relevant_procedures: list[RelevantProcedure] = Field(default_factory=list[RelevantProcedure])
    evidence_and_citations: list[EvidenceItem] = Field(default_factory=list[EvidenceItem])
    quality_and_operational_risks: list[RiskFlag] = Field(default_factory=list[RiskFlag])
    conflicts_or_ambiguities: ConflictsSection = Field(default_factory=ConflictsSection)
    missing_information: list[str] = Field(default_factory=list[str])
    required_human_escalation: HumanEscalationRead = Field(
        default_factory=lambda: HumanEscalationRead(required=True)
    )
    suggested_investigation_questions_or_next_steps: list[str] = Field(default_factory=list[str])
    limitations_and_uncertainty: list[str] = Field(default_factory=list[str])


class PharmaRunRead(BaseModel):
    """The ``pharma`` section of a run's detail (REQ-API-04).

    Read from the policy-checked run record, never from the model's agent rows (REQ-API-05).
    """

    intent: PharmaQueryIntent
    incident: IncidentInput | None = None
    filters: PharmaFilters = Field(default_factory=PharmaFilters)
    # The retrieval's own classification of the evidence (REQ-SUF-01).
    sufficiency: EvidenceSufficiency
    # Whether the result gives no procedure answer, whichever gate decided it (RMD3-T01).
    abstained: bool
    # What the result is and the gate that decided it (RMD3-T01); None for a record written
    # before they were stored.
    disposition: PharmaDisposition | None = None
    disposition_gate: DispositionGate | None = None
    rule_version: str
    banner: str
    response: PharmaResponse
    excluded_evidence: list[ExcludedEvidenceRead] = Field(
        default_factory=list[ExcludedEvidenceRead]
    )
    policy_notices: list[PolicyNotice] = Field(default_factory=list[PolicyNotice])
