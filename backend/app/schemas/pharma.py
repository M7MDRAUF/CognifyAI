"""Pydantic contracts of the pharmaceutical manufacturing profile (PDP3-T02).

The **agent outputs** (``PharmaPlannerOutput`` to ``PharmaDecisionOutput``) follow
``app.schemas.agents``: unknown keys are ignored and over-long model text is clamped rather than
rejected, because a rejection spends repair calls. Chunk ids appear only in list fields, never in
prose (AUD-03, AUD-05).

The API inputs and the read models are defined in ``app.schemas.pharma_api``, which imports no
agent schema so that ``app.schemas.queries`` can use them (DV-PH-14, DV-PH-16), and re-exported
here unchanged; importers of either kind may keep importing from this module.
"""

import logging
import re
import uuid
from enum import StrEnum
from typing import Annotated, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.domain.enums import RiskLevel
from app.pharma.enums import EscalationRole, PharmaRiskCategory
from app.schemas.agents import MAX_AGENT_LIST_ITEMS
from app.schemas.pharma_api import (
    EQUIPMENT_ID_PATTERN,
    MAX_EXCERPT_LENGTH,
    MAX_FILTER_VALUE_LENGTH,
    MAX_FILTER_VALUES,
    MAX_INCIDENT_FIELD_LENGTH,
    MAX_INCIDENT_LABEL_LENGTH,
    MAX_INCIDENT_SUMMARY_LENGTH,
    RESPONSE_SECTION_FIELDS,
    UNKNOWN,
    ConflictRead,
    ConflictsSection,
    ControlledDocumentRead,
    EscalationReason,
    EvidenceAnchor,
    EvidenceItem,
    ExcludedEvidenceRead,
    HumanEscalationRead,
    IncidentInput,
    PharmaFilters,
    PharmaQueryInput,
    PharmaResponse,
    PharmaRunRead,
    PolicyNotice,
    RelevantProcedure,
    RiskFlag,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_CATEGORIES",
    "EQUIPMENT_ID_PATTERN",
    "MAX_EXCERPT_LENGTH",
    "MAX_FILTER_VALUES",
    "MAX_FILTER_VALUE_LENGTH",
    "MAX_INCIDENT_FIELD_LENGTH",
    "MAX_INCIDENT_LABEL_LENGTH",
    "MAX_INCIDENT_SUMMARY_LENGTH",
    "MAX_PHARMA_AGENT_TEXT",
    "RESPONSE_SECTION_FIELDS",
    "UNKNOWN",
    "ActionCategory",
    "Answerability",
    "ConflictRead",
    "ConflictsSection",
    "ControlledDocumentRead",
    "DecisionCitation",
    "DecisionPoint",
    "EscalationReason",
    "EvidenceAnchor",
    "EvidenceItem",
    "ExcludedEvidenceRead",
    "HumanEscalationRead",
    "IncidentInput",
    "PharmaCriticOutput",
    "PharmaDecisionOutput",
    "PharmaFilters",
    "PharmaPlannerOutput",
    "PharmaQueryInput",
    "PharmaResponse",
    "PharmaRiskFlagDraft",
    "PharmaRiskOutput",
    "PharmaRunRead",
    "PolicyNotice",
    "RelevantProcedure",
    "RiskFlag",
]

# Agent text items are sentences, not documents; the same reasoning as MAX_FINDING_STATEMENT_LENGTH.
MAX_PHARMA_AGENT_TEXT = 2_000


# --- Agent outputs ------------------------------------------------------------------------------


def _clamp_text(value: object, limit: int, field: str) -> object:
    """Truncate an over-long model string instead of rejecting it, for the reason
    ``app.schemas.agents`` gives: a rejection spends a repair call and a strict retry, then fails
    the run. Kept here rather than imported, so that module's private helpers stay private."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    logger.warning(
        "Clamped agent output field %r from %d to %d characters", field, len(value), limit
    )
    return value[:limit]


def _clamp_items(value: object, limit: int, field: str) -> object:
    if not isinstance(value, list):
        return value
    items = cast("list[object]", value)
    if len(items) <= limit:
        return items
    logger.warning("Clamped agent output field %r from %d to %d items", field, len(items), limit)
    return items[:limit]


def _clamp_text_items(value: object, field: str) -> object:
    clamped = _clamp_items(value, MAX_AGENT_LIST_ITEMS, field)
    if isinstance(clamped, list):
        items = cast("list[object]", clamped)
        return [_clamp_text(item, MAX_PHARMA_AGENT_TEXT, field) for item in items]
    return clamped


def _listed(value: object) -> object:
    """RE2E-DEF-009 (CT-RMD-03): a list field the model wrote as one string holds it as a one-item
    list, and one written as ``null`` is empty; any other value is left for validation to judge.
    ``deepseek-flash`` wrote ``uncertainty`` as one string in 56 of 82 decision replies (B-009)."""
    if isinstance(value, str):
        return [value]
    return [] if value is None else value


# A passage's key on its label line (prompt version 4, CT-RMD-03): `E` and its position.
PASSAGE_KEY = re.compile(r"E[0-9]{1,3}")
PassageKey = Annotated[str, StringConstraints(pattern=r"^E[0-9]{1,3}$")]


def _flag_reference(item: object) -> object:
    """A risk flag's reference with a passage key in its label form; a chunk id is left as it
    is."""
    if isinstance(item, str) and PASSAGE_KEY.fullmatch(item.strip().strip("[]").strip().upper()):
        return item.strip().strip("[]").strip().upper()
    return item


def _is_reference(item: object) -> bool:
    """Whether a risk flag's reference is a passage key or a chunk id."""
    if isinstance(item, uuid.UUID):
        return True
    if not isinstance(item, str):
        return False
    if PASSAGE_KEY.fullmatch(item):
        return True
    try:
        uuid.UUID(item)
    except ValueError:
        return False
    return True


class PharmaPlannerOutput(BaseModel):
    """REQ-AGT-01: the planner decomposes the question and lists missing facts and escalation
    considerations. It sees no evidence and gets no evidence rule (AUD-04)."""

    normalized_question: str = ""
    sub_questions: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    search_queries: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    missing_facts: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    escalation_considerations: list[str] = Field(
        default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS
    )

    @field_validator("normalized_question", mode="before")
    @classmethod
    def _clamp_question(cls, value: object) -> object:
        return _clamp_text(value, MAX_PHARMA_AGENT_TEXT, "PharmaPlannerOutput.normalized_question")

    @field_validator(
        "sub_questions",
        "search_queries",
        "missing_facts",
        "escalation_considerations",
        mode="before",
    )
    @classmethod
    def _clamp_lists(cls, value: object) -> object:
        return _clamp_text_items(value, "PharmaPlannerOutput list")


class PharmaCriticOutput(BaseModel):
    """REQ-AGT-03: unsupported claims, wrong-version use, missing prerequisites, citation errors."""

    unsupported_claims: list[str] = Field(
        default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS
    )
    wrong_version_use: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    missing_prerequisites: list[str] = Field(
        default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS
    )
    citation_errors: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    needs_more_retrieval: bool = False
    expansion_queries: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)

    @field_validator(
        "unsupported_claims",
        "wrong_version_use",
        "missing_prerequisites",
        "citation_errors",
        "expansion_queries",
        mode="before",
    )
    @classmethod
    def _clamp_lists(cls, value: object) -> object:
        return _clamp_text_items(value, "PharmaCriticOutput list")


class PharmaRiskFlagDraft(BaseModel):
    category: PharmaRiskCategory
    severity: RiskLevel
    description: str
    # The passages a flag rests on: each passage's key on its label line, or its chunk id
    # (RMD-DEF-002). The workflow resolves a key to its chunk; an entry that is neither is dropped,
    # never the whole output.
    chunk_ids: list[uuid.UUID | PassageKey] = Field(
        default_factory=list[uuid.UUID | PassageKey], max_length=16
    )

    @field_validator("description", mode="before")
    @classmethod
    def _clamp_description(cls, value: object) -> object:
        return _clamp_text(value, MAX_PHARMA_AGENT_TEXT, "PharmaRiskFlagDraft.description")

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def _clamp_refs(cls, value: object) -> object:
        listed = _listed(value)
        if isinstance(listed, list):
            # Round 3 (AC-RMD-33 N5): a key written as a marker, with spaces or in lower case
            # ("[E2]", " E2", "e2") is the key, once.
            items: list[object] = []
            for item in cast("list[object]", listed):
                reference = _flag_reference(item)
                if reference not in items:
                    items.append(reference)
            usable = [item for item in items if _is_reference(item)]
            if len(usable) < len(items):
                logger.warning(
                    "Dropped %d risk-flag reference(s) that name no passage key or chunk id",
                    len(items) - len(usable),
                )
            listed = usable
        return _clamp_items(listed, 16, "PharmaRiskFlagDraft.chunk_ids")


class PharmaRiskOutput(BaseModel):
    """REQ-AGT-04: flags with a category, a severity and their supporting chunk ids in a field."""

    flags: list[PharmaRiskFlagDraft] = Field(default_factory=list[PharmaRiskFlagDraft])

    @field_validator("flags", mode="before")
    @classmethod
    def _clamp_flags(cls, value: object) -> object:
        return _clamp_items(value, MAX_AGENT_LIST_ITEMS, "PharmaRiskOutput.flags")


# The regulated act a decision point would take if read as a decision (PLV4-T03): "none" for
# information, deferral, a question or a report; otherwise the class of the taxonomy
# (docs/pharma-live/policy/action-taxonomy.md). "unspecified" is what an unknown value reads as,
# and is treated as a taking.
ActionCategory = Literal[
    "none",
    "batch_disposition",
    "deviation_closure",
    "deviation_classification",
    "capa_closure",
    "root_cause",
    "compliance_declaration",
    "procedure_override",
    "equipment_command",
    "medical_instruction",
    "qualified_person",
    "review_waiver",
    "unspecified",
]
ACTION_CATEGORIES: tuple[str, ...] = get_args(ActionCategory)
MAX_POINT_KEYS = 16


class DecisionPoint(BaseModel):
    """One structured point of the decision output v2 (PLV4-T03): its text, the citation keys it
    rests on, and the act it would take."""

    text: str = ""
    citation_keys: list[str] = Field(default_factory=list[str], max_length=MAX_POINT_KEYS)
    action_category: ActionCategory = "none"

    @field_validator("text", mode="before")
    @classmethod
    def _clamp_point(cls, value: object) -> object:
        return _clamp_text(value, MAX_PHARMA_AGENT_TEXT, "DecisionPoint.text")

    @field_validator("citation_keys", mode="before")
    @classmethod
    def _clamp_keys(cls, value: object) -> object:
        listed = _listed(value)
        if isinstance(listed, list):
            listed = _split_keys(cast("list[object]", listed))
        keys = _clamp_items(listed, MAX_POINT_KEYS, "DecisionPoint.citation_keys")
        if not isinstance(keys, list):
            return keys
        return [_written_key(key) for key in cast("list[object]", keys)]

    @field_validator("action_category", mode="before")
    @classmethod
    def _known_category(cls, value: object) -> object:
        return value if value in ACTION_CATEGORIES else "unspecified"

    def display_text(self) -> str:
        """The point as it is shown: its text, then the marker of each citation key the text does
        not already carry, in any marker form ("[E5, E8]", "[e1]"; round 3, AC-RMD-33 N11)."""
        carried = {
            label_key(key)
            for marker in _TEXT_MARKER.finditer(self.text)
            for key in marker.group(1).split(",")
        }
        markers = [f"[{key}]" for key in self.citation_keys if key not in carried]
        return " ".join([self.text.strip(), *markers]).strip()


class DecisionCitation(BaseModel):
    """A citation as the decision agent writes it (prompt version 4, CT-RMD-03): the key of the
    passage's label line and a snippet. The workflow takes the chunk from the key, so a chunk id the
    model wrote is kept only when it is one, and is never what the citation resolves by."""

    citation_key: str
    chunk_id: uuid.UUID | None = None
    snippet: str = ""

    @field_validator("citation_key", mode="before")
    @classmethod
    def _as_label_key(cls, value: object) -> object:
        return _written_key(value)

    @field_validator("chunk_id", mode="before")
    @classmethod
    def _unusable_id_is_none(cls, value: object) -> object:
        if value is None or isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return None


class Answerability(StrEnum):
    """What the passages say to the question (RMD3-T03, RE2E-DEF-032): all of it, part of it,
    different values in two or more Effective passages, or none of it."""

    FULL = "full"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    NONE = "none"


class PharmaDecisionOutput(BaseModel):
    """REQ-AGT-05: the response's model-written parts as structured fields. Prose cites passages by
    their citation key (``[E1]``); chunk ids appear only in ``citations``. Version 2 (PLV4-T03) adds
    ``points``, structured procedure points with their keys and action category; an output without
    them validates as before and its ``procedure_points`` are used."""

    summary: str = ""
    points: list[DecisionPoint] = Field(
        default_factory=list[DecisionPoint], max_length=MAX_AGENT_LIST_ITEMS
    )
    procedure_points: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    missing_information: list[str] = Field(
        default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS
    )
    investigation_questions: list[str] = Field(
        default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS
    )
    uncertainty: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    # Version 4 (RMD2-T03, CT-RMD-03): each citation names its passage by the key on the passage's
    # label line; the workflow resolves the chunk from the key (`resolve_citations`).
    citations: list[DecisionCitation] = Field(default_factory=list[DecisionCitation])
    # Version 3 (PLV9-T03, PLV-D-033): whether the passages state what the question asks, and the
    # roles the agent names for review; an output without them answers and names no role, as
    # before. The model can only add an abstention or a role.
    answers_question: bool = True
    escalate_to: list[EscalationRole] = Field(default_factory=list[EscalationRole])
    # Version 5 (RMD3-T03, RE2E-DEF-032): what the passages say to the question. An output that
    # states only answers_question (version 4 or earlier) reads false as none; answers_question
    # is then the answerability's reading, so its readers are unchanged.
    answerability: Answerability = Answerability.FULL
    # Citation entries the model wrote that name no key (not an object, or a key that is not a
    # string): dropped here and counted, never the whole output (CT-RMD-03).
    _dropped_citations: int = PrivateAttr(default=0)
    # Items past a list's limit, cut by the validators below and counted so the response can say
    # so (RMD-DEF-023).
    _cut_items: int = PrivateAttr(default=0)

    @model_validator(mode="wrap")
    @classmethod
    def _drop_unkeyed_citations(
        cls, data: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        dropped = 0
        if isinstance(data, dict):
            payload = dict(cast("dict[str, object]", data))
            entries = payload.get("citations", [])
            if entries is None:
                none: list[object] = []
                payload["citations"] = none
            elif isinstance(entries, (str, dict)):
                # RMD-DEF-025: one citation written alone is that one citation.
                alone: list[object] = [entries]
                entries = alone
            if isinstance(entries, list):
                items = [_as_entry(item) for item in _split_keys(cast("list[object]", entries))]
                kept: list[object] = [item for item in items if _is_keyed(item)]
                dropped = len(items) - len(kept)
                payload["citations"] = kept
            if "answerability" not in payload and payload.get("answers_question") is False:
                payload["answerability"] = Answerability.NONE.value
            lengths = [
                len(cast("list[object]", listed))
                for name in _CUT_LISTS
                if isinstance(listed := payload.get(name), list)
            ]
            cut = sum(n - MAX_AGENT_LIST_ITEMS for n in lengths if n > MAX_AGENT_LIST_ITEMS)
            data = payload
        else:
            cut = 0
        output = handler(data)
        output._dropped_citations = dropped
        output._cut_items = cut
        output.answers_question = output.answerability is not Answerability.NONE
        return output

    @property
    def dropped_citations(self) -> int:
        return self._dropped_citations

    @property
    def cut_items(self) -> int:
        return self._cut_items

    @field_validator("escalate_to", mode="before")
    @classmethod
    def _known_roles(cls, value: object) -> object:
        """Only the roles the escalation rules know, each once; anything else is dropped. A role
        written as one string is that one role (CT-RMD-03)."""
        value = _listed(value)
        if not isinstance(value, list):
            return []
        known = {role.value for role in EscalationRole}
        items = cast("list[object]", value)
        return list(dict.fromkeys(str(item) for item in items if str(item) in known))

    @field_validator("summary", mode="before")
    @classmethod
    def _clamp_summary(cls, value: object) -> object:
        return _clamp_text(value, MAX_PHARMA_AGENT_TEXT, "PharmaDecisionOutput.summary")

    @field_validator("points", mode="before")
    @classmethod
    def _clamp_points(cls, value: object) -> object:
        return _clamp_items(_listed(value), MAX_AGENT_LIST_ITEMS, "PharmaDecisionOutput.points")

    @field_validator(
        "procedure_points",
        "missing_information",
        "investigation_questions",
        "uncertainty",
        mode="before",
    )
    @classmethod
    def _clamp_lists(cls, value: object) -> object:
        return _clamp_text_items(_listed(value), "PharmaDecisionOutput list")


_CUT_LISTS = (
    "points",
    "procedure_points",
    "missing_information",
    "investigation_questions",
    "uncertainty",
)


def label_key(written: str) -> str:
    """RMD-DEF-025: a passage key as its label line writes it (``E1``), from the key as the model
    wrote it: without its marker's brackets, the spaces around it or lower case."""
    return written.strip().strip("[]").strip().upper()


# A text's marker of one passage key or several ("[E1]", "[E5, E8]", "[ e1 ]").
_TEXT_MARKER = re.compile(r"\[\s*([Ee]\d{1,3}(?:\s*,\s*[Ee]\d{1,3})*)\s*\]")
_KEY_LIST = re.compile(r"\[?\s*[Ee]\d{1,3}(?:\s*[,;\s]\s*[Ee]\d{1,3})+\s*\]?")
_KEY_SEPARATOR = re.compile(r"[\s,;]+")


def _split_keys(items: list[object]) -> list[object]:
    """Round 3 (AC-RMD-33 N4): a key list the model wrote as one string ("E1, E2", "[E1, E3]") is
    those keys; any other string is left whole."""
    split: list[object] = []
    for item in items:
        if isinstance(item, str) and _KEY_LIST.fullmatch(item.strip()):
            split.extend(key for key in _KEY_SEPARATOR.split(item.strip().strip("[]")) if key)
        else:
            split.append(item)
    return split


def _written_key(value: object) -> object:
    return label_key(value) if isinstance(value, str) else value


def _as_entry(entry: object) -> object:
    """A citation the model wrote as its key alone (``"E1"``, RMD-DEF-025) is that key's
    citation."""
    return {"citation_key": entry} if isinstance(entry, str) else entry


def _is_keyed(entry: object) -> bool:
    """Whether a citation entry names a passage key at all (an object whose key is a string)."""
    if isinstance(entry, DecisionCitation):
        return True
    if not isinstance(entry, dict):
        return False
    return isinstance(cast("dict[str, object]", entry).get("citation_key"), str)
