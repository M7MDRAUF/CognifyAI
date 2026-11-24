"""Enumerations of the pharmaceutical profile (PHARMA_DOMAIN_REQUIREMENTS.md, sections 2 to 9).

Values are lower snake case because they are stored in text columns and parsed by the client's
zod enums; ``PolicyCode`` is the exception, keeping the catalogue's ``PH-POL-nn`` codes as its
values so a recorded finding reads the same in the database, the API and the documents.
"""

from enum import StrEnum


class LifecycleStatus(StrEnum):
    """REQ-DOC-03. Only ``EFFECTIVE`` documents are eligible on their own; ``UNKNOWN`` is usable but
    labelled and never sufficient alone (REQ-RET-07)."""

    DRAFT = "draft"
    EFFECTIVE = "effective"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    UNKNOWN = "unknown"


class ControlledDocumentType(StrEnum):
    SOP = "sop"
    WORK_INSTRUCTION = "work_instruction"
    EQUIPMENT_MANUAL = "equipment_manual"
    FORM = "form"
    REGULATION = "regulation"
    GUIDANCE = "guidance"
    OTHER = "other"


class SourceAuthority(StrEnum):
    """Who issued the document. Site procedures outrank reference documents when reranker scores tie
    for procedure and incident questions (REQ-RET-08)."""

    SITE = "site"
    VENDOR = "vendor"
    REGULATOR = "regulator"
    OTHER = "other"


class ControlStatus(StrEnum):
    CONTROLLED = "controlled"
    UNCONTROLLED = "uncontrolled"
    REFERENCE = "reference"


class DataClassification(StrEnum):
    """Intended use, section 4: the profile admits synthetic and public material only, so no value
    exists for real, proprietary or confidential documents."""

    SYNTHETIC = "synthetic"
    PUBLIC = "public"


class MetadataStatus(StrEnum):
    """REQ-DOC-02 and REQ-DOC-04: ``MISSING`` when there is no header block (or the file type has
    none), ``INVALID`` when a header value failed validation."""

    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


class PharmaQueryIntent(StrEnum):
    PROCEDURE_QUESTION = "procedure_question"
    OPERATIONAL_INCIDENT = "operational_incident"
    READINESS_REVIEW = "readiness_review"


class EvidenceSufficiency(StrEnum):
    """The classes of the pre-registered sufficiency rule, version 1 (requirements, section 15)."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class PharmaDisposition(StrEnum):
    """What a pharmaceutical result is (RMD3-T01, RE2E-DEF-034), set by the gate that decided it
    (``DispositionGate``). ``OUT_OF_SCOPE``, ``CONFLICT_REQUIRES_REVIEW`` and ``SYSTEM_FAILURE``
    are reserved: no gate gives them yet, and a run that fails stores no result."""

    ANSWER = "answer"
    PARTIAL_ANSWER = "partial_answer"
    ABSTAIN_INSUFFICIENT_EVIDENCE = "abstain_insufficient_evidence"
    REFUSE_PROHIBITED_DECISION = "refuse_prohibited_decision"
    OUT_OF_SCOPE = "out_of_scope"
    CONFLICT_REQUIRES_REVIEW = "conflict_requires_review"
    SYSTEM_FAILURE = "system_failure"


class DispositionGate(StrEnum):
    """The gate that decided a result's disposition: the retrieval's sufficiency rule, the
    decision agent's reading of the passages, the grounding gate, or the regulatory-boundary
    check."""

    SUFFICIENCY = "sufficiency"
    DECISION = "decision"
    GROUNDING = "grounding"
    POLICY = "policy"


class PharmaRiskCategory(StrEnum):
    """REQ-AGT-04."""

    OPERATIONAL = "operational"
    PRODUCT_QUALITY = "product_quality"
    EQUIPMENT = "equipment"
    DOCUMENTATION = "documentation"
    DATA_INTEGRITY = "data_integrity"
    ESCALATION = "escalation"


class EscalationRole(StrEnum):
    """The roles REQ-ESC-01 to REQ-ESC-07 name."""

    QUALITY_ASSURANCE = "quality_assurance"
    PROCESS_OWNER = "process_owner"
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"
    DOCUMENT_CONTROL = "document_control"


class ConflictKind(StrEnum):
    """The deterministic conflict kinds of REQ-CON-01, in the requirement's order."""

    SUPERSEDED_MATCH = "superseded_match"
    RETIRED_MATCH = "retired_match"
    DRAFT_MATCH = "draft_match"
    UNKNOWN_STATUS = "unknown_status"
    MULTIPLE_EFFECTIVE_REVISIONS = "multiple_effective_revisions"
    SUPERSEDED_MARKED_EFFECTIVE = "superseded_marked_effective"
    OVERLAPPING_EFFECTIVE_SCOPE = "overlapping_effective_scope"
    PARAMETER_DISAGREEMENT = "parameter_disagreement"


class ExclusionReason(StrEnum):
    """Why version resolution kept a candidate out of the agents' context.

    REQ-RET-02 to REQ-RET-05; every exclusion carries exactly one reason.
    """

    SUPERSEDED = "superseded"
    RETIRED = "retired"
    DRAFT = "draft"
    LOWER_REVISION = "lower_revision"
    SUPERSEDED_BY_EFFECTIVE = "superseded_by_effective"
    FILTER_MISMATCH = "filter_mismatch"


class PolicyCode(StrEnum):
    """The regulatory-boundary codes PH-POL-01 to PH-POL-10 (requirements section 8)."""

    ROOT_CAUSE = "PH-POL-01"
    BATCH_DISPOSITION = "PH-POL-02"
    APPROVAL_LANGUAGE = "PH-POL-03"
    AUTONOMOUS_DECISION = "PH-POL-04"
    PROCEDURE_OVERRIDE = "PH-POL-05"
    HIGH_RISK_WITHOUT_ESCALATION = "PH-POL-06"
    INSUFFICIENT_EVIDENCE = "PH-POL-07"
    UNDISCLOSED_NON_EFFECTIVE = "PH-POL-08"
    MEDICAL_ADVICE = "PH-POL-09"
    COMPLIANCE_CLAIM = "PH-POL-10"
