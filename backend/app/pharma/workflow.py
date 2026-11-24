"""The pharmaceutical workflow: ``PharmaOrchestrator`` (PDP7-T05; ADR-001, PH-D-010 to PH-D-012).

A subclass of the baseline ``Orchestrator`` that overrides only ``_run_workflow``. Persistence,
metering, events, deadlines, cancellation and failure handling are the baseline's own helpers
(``_step``, ``_supply``, ``_persist_evidence``, ``_persist_answer``, ``_persist_citations``,
``_evaluate``, ``_finish``, and ``_fail`` through ``run``), so a pharmaceutical run is recorded,
metered and stopped exactly as a General one (REQ-SEC-07). The order:

1. the pharmaceutical planner, which sees no evidence;
2. retrieval: the incident's deterministic queries and the planner's, gathered by the baseline
   hybrid pipeline (REQ-RET-01, REQ-AGT-07); version resolution on the candidates (REQ-RET-02 to
   05); the baseline reranker over the eligible candidates only (REQ-RET-03); authority ordering
   (REQ-RET-08); then sufficiency and conflicts on the final context;
3. insufficient evidence: the abstention, with no research, critic, risk, decision or evaluator
   call (REQ-SUF-02, PH-D-010); otherwise the baseline researcher, on copies of the passages
   labelled from the document register (PH-D-028), then the pharmaceutical critic, risk and
   decision agents;
4. escalation by the rules with the model's reasons appended, the nine-section response, the
   regulatory-boundary check (fail closed), the recorder, then the answer, its citations and the
   baseline evaluator.

Only the baseline event types are emitted; ``retrieval_completed`` gains ``eligible``,
``excluded`` and ``sufficiency`` (REQ-SSE-02). A critic's request for more retrieval stays on its
agent row and is not executed: a second round would need its own version resolution, and the
workflow's sequence has none (PH-D-029).
"""

import logging
import re
import time
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Final, Protocol

from app.agents.base import JsonAgentRunner
from app.agents.orchestrator import Orchestrator, relevance_of
from app.agents.premise_guard import claims_unseen_evidence
from app.agents.prompts import flatten
from app.core.errors import CognifyError
from app.domain.enums import (
    AgentName,
    AgentRunStatus,
    AnswerModel,
    EventType,
    RetrievalMode,
    SelectionState,
)
from app.domain.models import Chunk, Citation, ControlledDocument, QueryRun
from app.integrations.llm.base import ChatProvider, TokenUsage
from app.integrations.llm.profiles import CallSite, RoutingProfile, resolve
from app.pharma.action_classifier import (
    CLASSIFIER_COMPONENT,
    UNAVAILABLE_REASON_TEXT,
    Classification,
    apply_classification,
    classifier_record,
    classifier_texts,
    classify_actions,
    with_unavailable_escalation,
)
from app.pharma.actions import ActionClass, detect_actions
from app.pharma.agents import (
    PharmaCriticAgent,
    PharmaDecisionAgent,
    PharmaPlannerAgent,
    PharmaRiskAgent,
)
from app.pharma.anchors import ControlledDocumentInfo, anchor_for, info_from_row
from app.pharma.catalogue import HUMAN_REVIEW_BANNER
from app.pharma.conflicts import detect_conflicts
from app.pharma.enums import (
    DataClassification,
    DispositionGate,
    EvidenceSufficiency,
    ExclusionReason,
    PharmaDisposition,
    PharmaQueryIntent,
    PharmaRiskCategory,
)
from app.pharma.escalation import (
    EscalationInputs,
    decide_escalation,
    merge_model_reasons,
    with_model_roles,
)
from app.pharma.grounding import (
    SUPPORT_THRESHOLD,
    ClaimCheck,
    ClaimSource,
    GroundingResult,
    PassageLabel,
    check_grounding,
)
from app.pharma.metadata import validate_document_number, validate_revision
from app.pharma.policy import (
    PolicyCheckError,
    PolicyContext,
    PolicyFinding,
    PolicyResult,
    apply_policy,
    check_decision,
    check_fields,
    check_rendered,
    layered_violations,
    needs_abstention,
    quoted_texts,
    split_sentences,
    trace_records,
    with_abstention,
    with_removals,
    withhold,
    withhold_displayed,
)
from app.pharma.prompts import citation_key_map, section_number
from app.pharma.response import (
    GROUNDING_WITHHELD_SUMMARY,
    TRUSTED_TEXTS,
    ResponseInputs,
    abstention_response,
    assemble_response,
    cited_keys,
    disposition_of,
    missing_from,
    point_texts,
    policy_abstention,
    render_markdown,
    with_rule,
)
from app.pharma.retrieval_tuning import ACTIVE, refine_context, tuned_queries, tuned_resolution
from app.pharma.signals import RiskSignals, extract_signals
from app.pharma.sufficiency import SufficiencyAssessment, classify_sufficiency_v2
from app.pharma.versions import ExcludedCandidate, VersionResolution, resolve_versions
from app.repositories.query_repository import QueryRepository
from app.schemas.agents import CitationDraft, FinalDecisionOutput
from app.schemas.pharma import (
    MAX_EXCERPT_LENGTH,
    UNKNOWN,
    Answerability,
    ConflictRead,
    EvidenceAnchor,
    EvidenceItem,
    ExcludedEvidenceRead,
    HumanEscalationRead,
    IncidentInput,
    PharmaDecisionOutput,
    PharmaFilters,
    PharmaPlannerOutput,
    PharmaQueryInput,
    PharmaResponse,
    PolicyNotice,
    RiskFlag,
    label_key,
)
from app.services.answer_hygiene import drop_planted_copies, notice_for, strip_context_lines
from app.services.citation_service import UNLABELLED_SOURCE, validate_citations
from app.services.event_emitter import EventEmitter
from app.services.interfaces import SupportsCommit
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.services.retrieval_types import RetrievedChunk
from app.utils.ids import new_uuid
from app.utils.time import utc_now

# REQ-AGT-07 and threat T-17: an incident adds at most three bounded queries.
MAX_INCIDENT_QUERY_CHARS = 200
MAX_SUMMARY_LINE = 500
_RAW_ID = re.compile(
    r"\[?\s*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\s*\]?"
)
# A document number named in prose: the register format with at least two segments after the
# prefix ("SOP-SYN-MFG-010"), so an equipment identifier ("TP-201") is not one.
_NAMED_NUMBER = re.compile(r"\b[A-Z]{2,5}(?:-[A-Z0-9]{1,6}){2,5}\b")
# The same shape in any case, read after the hyphen variants below become "-" (PH-DEF-036). A match
# counts as a document number when it is upper case with two or more segments, holds a digit in two
# or more segments, or starts with a document-type prefix; so "out-of-limit" and "TP-201" do not,
# and "sop-syn-zzz-999" and "SOP-999" do.
_ANY_CASE_NUMBER = re.compile(r"\b[A-Za-z]{2,5}(?:-[A-Za-z0-9]{1,6}){1,5}\b")
_HYPHENS = re.compile("[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]")
DOCUMENT_PREFIXES: Final = frozenset({"SOP", "WI", "MAN", "FRM", "LOG", "POL", "SPEC", "STD"})
# A revision named right after a document number ("SOP-SYN-MFG-010 revision 07", ", rev 07").
# Read with ``match(text, pos)``, which anchors at ``pos``; a "^" would only match at position 0.
_NAMED_REVISION = re.compile(r"\s*,?\s*(?:revision|rev\.?)\s+([A-Za-z0-9]{1,4})\b", re.IGNORECASE)

logger = logging.getLogger(__name__)

# The structured log events of the deterministic steps (PDP13-T03). Each call writes its
# `event_type` as a literal, which `app.core.logging.KNOWN_EVENT_TYPES` lists, and builds its
# `pharma` field with `step_fields`, which refuses any value that could hold document or model text.
PHARMA_STEP_EVENTS: Final = frozenset(
    {
        "pharma_versions_resolved",
        "pharma_sufficiency_classified",
        "pharma_conflicts_detected",
        "pharma_signals_extracted",
        "pharma_abstained",
        "pharma_unknown_references_removed",
        "pharma_policy_checked",
        "pharma_escalation_decided",
    }
)
# A code: one identifier-like token of at most 80 characters, never a raw chunk id.
_LOG_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}")
_LOG_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

type LogValue = int | bool | str | Enum | Sequence[str | Enum]


def _log_code(value: object) -> str:
    text = value.value if isinstance(value, Enum) else value
    if not isinstance(text, str) or not _LOG_CODE.fullmatch(text) or _LOG_UUID.fullmatch(text):
        msg = "a pharmaceutical step event carries codes and counts only"
        raise TypeError(msg)
    return text


def step_fields(**fields: LogValue) -> dict[str, int | bool | str | list[str]]:
    """The `pharma` field of a step event: integers, booleans, enumeration values and short codes.
    Anything that could carry text is refused before the record is made, so no log line holds
    document or model text."""
    clean: dict[str, int | bool | str | list[str]] = {}
    for name, value in fields.items():
        if isinstance(value, bool | int) and not isinstance(value, Enum):
            clean[name] = value
        elif isinstance(value, str | Enum):
            clean[name] = _log_code(value)
        elif isinstance(value, list | tuple):
            clean[name] = [_log_code(item) for item in value]
        else:
            msg = "a pharmaceutical step event carries codes and counts only"
            raise TypeError(msg)
    return clean


@dataclass(frozen=True)
class PharmaRunInput:
    """The pharmaceutical part of a run's request (REQ-API-03)."""

    intent: PharmaQueryIntent = PharmaQueryIntent.PROCEDURE_QUESTION
    incident: IncidentInput | None = None
    filters: PharmaFilters = field(default_factory=PharmaFilters)

    @classmethod
    def from_query(cls, query: PharmaQueryInput) -> "PharmaRunInput":
        return cls(intent=query.intent, incident=query.incident, filters=query.filters)


@dataclass(frozen=True)
class ConsideredPassage:
    """One candidate the run considered (REQ-PER-02): the passage as retrieved (scores and its
    dense rank in the persisted order), its anchor from stored rows, whether it reached the final
    context, and the reason version resolution excluded it, if it did."""

    chunk: RetrievedChunk
    anchor: EvidenceAnchor
    in_context: bool
    exclusion: ExclusionReason | None = None


@dataclass(frozen=True)
class PharmaRunOutcome:
    """What the workflow hands its recorder: the checked response and everything it was made from
    (REQ-SUF-04, REQ-POL-11). Never the model's unchecked text."""

    query_run_id: uuid.UUID
    run_input: PharmaRunInput
    assessment: SufficiencyAssessment
    # The sufficiency rule's abstention (REQ-SUF-02), which the evaluations count; the result's
    # own disposition, every abstention kind included, is ``disposition`` (RMD3-T01).
    abstained: bool
    response: PharmaResponse
    answer_markdown: str
    excluded: list[ExcludedEvidenceRead]
    findings: list[PolicyFinding]
    notices: list[PolicyNotice]
    signals: RiskSignals
    disposition: PharmaDisposition
    disposition_gate: DispositionGate
    # Every candidate considered, in the persisted order, so the record keeps what the run saw
    # after its documents change (REQ-PER-02, REQ-PER-03).
    considered: list[ConsideredPassage] = field(default_factory=list[ConsideredPassage])
    banner: str = HUMAN_REVIEW_BANNER
    # The policy trace of every layer's removals (PLV4-T03, ``policy.trace_records``): codes,
    # classes, layers, dispositions and hashes, never the removed text.
    trace: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    @property
    def rule_version(self) -> str:
        return self.assessment.rule_version


class PharmaRunRecorder(Protocol):
    async def record(self, outcome: PharmaRunOutcome) -> None: ...


class InMemoryRecorder:
    """Keeps every outcome in memory; the database recorder is PDP8-T03's."""

    def __init__(self) -> None:
        self.outcomes: list[PharmaRunOutcome] = []

    async def record(self, outcome: PharmaRunOutcome) -> None:
        self.outcomes.append(outcome)


class PharmaMetadataLoader(Protocol):
    """The stored rows the workflow reads: a project's controlled documents and chunk rows by id.
    ``PharmaRepository`` and ``ChunkRepository`` each provide one method."""

    async def controlled_for_project(
        self, project_id: uuid.UUID
    ) -> Sequence[ControlledDocument]: ...

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> Mapping[uuid.UUID, Chunk]: ...


class _ControlledSource(Protocol):
    async def controlled_for_project(self, project_id: uuid.UUID) -> list[ControlledDocument]: ...


class _ChunkSource(Protocol):
    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Chunk]: ...


class RepositoryMetadataLoader:
    """``PharmaMetadataLoader`` over the two repositories."""

    def __init__(self, controlled: _ControlledSource, chunks: _ChunkSource) -> None:
        self._controlled = controlled
        self._chunks = chunks

    async def controlled_for_project(self, project_id: uuid.UUID) -> Sequence[ControlledDocument]:
        return await self._controlled.controlled_for_project(project_id)

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> Mapping[uuid.UUID, Chunk]:
        return await self._chunks.get_many(chunk_ids)


def incident_queries(run_input: PharmaRunInput) -> list[str]:
    """REQ-AGT-07: the incident's equipment identifier, process step and summary, each as one
    bounded query, in that order, without repeats."""
    incident = run_input.incident
    if incident is None:
        return []
    queries: list[str] = []
    for value in (incident.equipment_id, incident.process_step, incident.summary):
        query = flatten(value, MAX_INCIDENT_QUERY_CHARS).strip()
        if query and query.casefold() not in {known.casefold() for known in queries}:
            queries.append(query)
    return queries


def retrieval_queries(
    run_input: PharmaRunInput, plan: PharmaPlannerOutput, question: str
) -> list[str]:
    """The incident's queries, then the planner's (bounded by its schema), without repeats."""
    queries = incident_queries(run_input)
    for planned in plan.search_queries:
        query = flatten(planned, MAX_INCIDENT_QUERY_CHARS).strip()
        if query and query.casefold() not in {known.casefold() for known in queries}:
            queries.append(query)
    return queries or [question]


def resolve_citations(
    decision: PharmaDecisionOutput, keys: Mapping[str, uuid.UUID]
) -> tuple[list[CitationDraft], int]:
    """The decision's citations with their chunks taken from their keys (CT-RMD-03), and how many
    were dropped: those whose key no label line holds, and those the schema dropped for naming no
    key at all. A chunk id the model wrote is never what a citation resolves by (prompt version 4
    asks for none), so a citation cannot point at a passage its key does not name. A key a point or
    a marker in the prose names is a citation too when the list left it out (RMD-DEF-025): the
    point resting on it would otherwise be shown as citing nothing."""
    resolved: list[CitationDraft] = []
    unresolved = 0
    for citation in decision.citations:
        chunk = keys.get(citation.citation_key)
        if chunk is None:
            unresolved += 1
            continue
        resolved.append(
            CitationDraft(
                citation_key=citation.citation_key, chunk_id=chunk, snippet=citation.snippet
            )
        )
    prose = [
        decision.summary,
        *(point.text for point in decision.points),
        *decision.procedure_points,
        *decision.missing_information,
        *decision.investigation_questions,
        *decision.uncertainty,
    ]
    named = [key for point in decision.points for key in point.citation_keys]
    named += [label_key(key) for text in prose for key in cited_keys(text)]
    listed = {draft.citation_key for draft in resolved}
    for key in dict.fromkeys(named):
        chunk = keys.get(key)
        if chunk is not None and key not in listed:
            resolved.append(CitationDraft(citation_key=key, chunk_id=chunk, snippet=""))
    return resolved, unresolved + decision.dropped_citations


def resolve_references(
    references: Sequence[uuid.UUID | str], keys: Mapping[str, uuid.UUID]
) -> list[uuid.UUID]:
    """A risk flag's passages (RMD-DEF-002): a key resolves to its passage's chunk, a chunk id is
    kept, and a key no label line holds is dropped."""
    chunks: list[uuid.UUID] = []
    for reference in references:
        chunk = keys.get(reference) if isinstance(reference, str) else reference
        if chunk is not None:
            chunks.append(chunk)
    return chunks


# RMD-DEF-020: a request for information frames the act it names ("Steps to restart TP-201 after a
# jam", "How to start up the line", "Explain how to release a batch"): it asks how, not for the act.
_INFORMATION_FRAME = re.compile(
    r"^(?:please )?(?:(?:what are |list |give me |show me )?(?:the )?(?:steps|instructions|"
    # Round 3 (RMD-DEF-020): "Prerequisites to release lot X", "Checklist for releasing lot X".
    r"procedure|process|guidance|requirements|prerequisites|preconditions|checklist|criteria|"
    r"conditions)(?: to| for| before)\b|how (?:to|do|does|should|can|is|are|"
    r"would)\b|(?:explain|describe|outline|summari[sz]e|tell me|show me) (?:how|what|when|which|"
    r"the)\b|"
    # Round 2 (RMD-DEF-020): the criteria or requirements of an act are information about it
    # ("Release criteria for lot SYN-L-314", "I need the release requirements for lot ...").
    r"(?:the )?(?:release|disposition|closure|approval|shipment) (?:criteria|requirements|"
    r"conditions|checklist|steps|prerequisites)\b|i (?:need|want|would like) (?:to (?:know|see|"
    r"understand) )?(?:the )?(?:[\w-]+ )?(?:criteria|requirements|conditions|steps|procedure|"
    r"prerequisites|information|details)\b)",
    re.IGNORECASE,
)
# RMD-DEF-102: a request attached to another ("Summarize the investigation and close deviation
# DEV-SYN-045", "Explain how to release a batch, then release batch RE2E-B-01") is read on its own.
# Round 3: after a comma too ("Summarize the investigation, close DEV-SYN-045"), and "declare".
_ATTACHED_REQUEST = re.compile(
    r",\s*(?:and\s+)?then\s+|;\s*|(?:,\s*|\s+and\s+)(?=(?:then\s+)?(?:release|reject|ship|close|"
    r"approve|sign|waive|skip|bypass|certify|disposition|scrap|destroy|move|clear|declare)\b)",
    re.IGNORECASE,
)
# A question about how to take, or whether to take, the act on a named batch or record asks for the
# decision on it ("How can I release batch RE2E-B-01 today?", "Should batch RE2E-B-01 be
# released?").
_HOW_I = re.compile(
    r"^(?:so |and |then )?how (?:can|could|do|should|may) (?:i|we)\s+", re.IGNORECASE
)
# A named item holds a letter: a date ("2026-10-27") names none (round 3, AC-RMD-31 N12).
_NAMED_ITEM = re.compile(
    r"(?<![\w-])(?:(?=[a-z0-9-]*[a-z])[a-z0-9]+-[a-z0-9]+-[a-z0-9-]*\d[a-z0-9-]*|"
    r"(?:dev|ncr|capa)-[\w-]*\d[\w-]*)",
    re.IGNORECASE,
)
_SHOULD_IT = re.compile(
    r"^(?:so |and |then )?(should|can|could|may|must|shall) (?!i\b|we\b)(.+?) be (.+?)\s*\?\s*$",
    re.IGNORECASE,
)
# Round 3 (AC-RMD-31 N1, RMD-DEF-102): a yes-no question whose subject is the named item ("Can
# batch X ship today?", "Is batch X releasable?") asks the statement with the verb after it ("batch
# X can ship today"); "is" and "are" only with a judgement of the item.
_ASKED_OF_ITEM = re.compile(
    r"^(?:so |and |then )?(should|can|could|may|must|shall|will|is|are) ((?:the |this |that )?"
    r"(?:batch|lot|deviation|record|capa)?\s*)$",
    re.IGNORECASE,
)
_JUDGEMENT = re.compile(
    r"^\s*(?:now |therefore |already )?(?:releasable|shippable|ready|fit|safe|suitable|acceptable|"
    r"good|ok|okay|cleared|clear)\b",
    re.IGNORECASE,
)
# "Decide whether batch X should be released" asks for the decision itself.
_DECIDE_WHETHER = re.compile(
    r"^(?:please |can you |could you )?(?:decide|determine|judge) (?:whether|if)\s+",
    re.IGNORECASE,
)
# A yes-no question asking whether the user may take the act asks the assistant to decide it ("Can
# I release batch SYN-B-734 today?"): what follows the frame is read as the act requested.
_PERMISSION_QUESTION = re.compile(
    r"^(?:so |and |then )?(?:(?:can|could|may|might|should|shall) (?:i|we)|(?:is|would) it be "
    r"(?:ok|okay|fine|safe|acceptable|alright|allowed)(?: for (?:me|us))? to|is it (?:ok|okay|"
    r"fine|safe|acceptable|alright|allowed)(?: for (?:me|us))? to|am i allowed to|are we allowed "
    r"to)\s+",
    re.IGNORECASE,
)


def requests_regulated_decision(question: str) -> bool:
    """Whether the question asks the assistant to take a regulated decision, read by the
    regulatory-boundary detector (``actions.detect_actions``) as a critical act of its own
    (RMD3-T03): a request is refused, a question about such a decision is answered as any other.
    Read sentence by sentence (RMD-DEF-020): a request for information about the act is no request
    for it; a question whether the user may take it is one; and an equipment command is not a
    regulated decision (the assistant runs no equipment, and the answer's own boundary still
    removes one), so it refuses nothing."""
    for sentence in split_sentences(question):
        for part in _ATTACHED_REQUEST.split(sentence.strip()):
            if _requests(part.strip()):
                return True
    return False


def _requests(text: str) -> bool:
    """One request of a question: ``requests_regulated_decision``'s reading of it."""
    if not text:
        return False
    item = _NAMED_ITEM.search(text)
    named = item is not None
    decide = _DECIDE_WHETHER.match(text)
    if decide is not None:
        return _decision(text[decide.end() :].rstrip(" ?."))
    permission = _PERMISSION_QUESTION.match(text) or (_HOW_I.match(text) if named else None)
    if permission is not None:
        return _decision(text[permission.end() :].rstrip(" ?"))
    should = _SHOULD_IT.match(text)
    if should is not None and named:
        return _decision(f"{should.group(2)} {should.group(1)} be {should.group(3)}")
    if item is not None and text.rstrip().endswith("?"):
        asked = _ASKED_OF_ITEM.match(text[: item.start()])
        rest = text[item.end() :].rstrip(" ?")
        if asked is not None and (
            asked.group(1).lower() not in {"is", "are"} or _JUDGEMENT.match(rest) is not None
        ):
            return _decision(f"{asked.group(2)}{item.group()} {asked.group(1)}{rest}")
    if _INFORMATION_FRAME.match(text) is not None:
        return False
    return _decision(text)


def _decision(text: str) -> bool:
    """Whether ``text`` takes a critical act other than an equipment command (the assistant runs
    no equipment, and the answer's own boundary still removes one; round 3, RMD-DEF-020)."""
    return any(
        finding.critical and finding.action_class is not ActionClass.EQUIPMENT_COMMAND
        for finding in detect_actions(text)
    )


def register_label(chunk: RetrievedChunk, info: ControlledDocumentInfo | None) -> str:
    """A passage's label from validated register values only (PH-D-027, PH-D-028)."""
    number = revision = status = UNKNOWN
    if info is not None and info.document_id == chunk.document_id:
        if info.document_number and validate_document_number(info.document_number):
            number = info.document_number
        if info.revision and validate_revision(info.revision):
            revision = info.revision
        status = info.lifecycle_status.value
    section = section_number(chunk.section_title)
    return f"{number} revision {revision}, {status}, section {section}"


def retrieval_counts(
    resolution: VersionResolution[RetrievedChunk], assessment: SufficiencyAssessment
) -> dict[str, object]:
    """REQ-SSE-02, PH-D-011: the keys a pharmaceutical run adds to ``retrieval_completed``."""
    return {
        "eligible": len(resolution.eligible),
        "excluded": len(resolution.excluded),
        "sufficiency": assessment.sufficiency.value,
    }


type _Entry = tuple[RetrievedChunk, SelectionState, ExclusionReason | None]


def considered_order(
    context: Sequence[RetrievedChunk],
    ranking: Sequence[RetrievedChunk],
    excluded: Sequence[ExcludedCandidate[RetrievedChunk]],
) -> list[_Entry]:
    """Every gathered chunk once, with one dense rank: the final context in its authority order
    (``selected``), the rest of the reranker's ordering (``reranked``, or ``candidate`` when it
    was never scored), then the excluded candidates, which never reached the reranker
    (``candidate``, with their reason)."""
    chosen = {chunk.chunk_id for chunk in context}
    entries: list[_Entry] = [
        *((chunk, SelectionState.SELECTED, None) for chunk in context),
        *(
            (
                chunk,
                SelectionState.RERANKED
                if chunk.rerank_score is not None
                else SelectionState.CANDIDATE,
                None,
            )
            for chunk in ranking
            if chunk.chunk_id not in chosen
        ),
        *((item.chunk, SelectionState.CANDIDATE, item.reason) for item in excluded),
    ]
    return [
        (replace(chunk, rank=rank), state, reason)
        for rank, (chunk, state, reason) in enumerate(entries)
    ]


@dataclass(frozen=True)
class _Retrieved:
    context: list[RetrievedChunk]
    resolution: VersionResolution[RetrievedChunk]
    assessment: SufficiencyAssessment
    conflicts: list[ConflictRead]
    considered: list[_Entry]


class PharmaOrchestrator(Orchestrator):
    def __init__(
        self,
        session: SupportsCommit,
        queries: QueryRepository,
        emitter: EventEmitter,
        provider: ChatProvider,
        retrieval: RetrievalService,
        reranking: RerankingService,
        *,
        pharma_input: PharmaRunInput,
        recorder: PharmaRunRecorder,
        infos_loader: PharmaMetadataLoader,
        rerank_expanded_union: bool = True,
        action_classifier: JsonAgentRunner | None = None,
    ) -> None:
        """``action_classifier`` runs the secondary detector of regulated acts (PLV4-T04) beside
        the deterministic layers; the production builder (``jobs.build_pharma_orchestrator``)
        always passes one. Without it the deterministic layers decide alone, as in the suites
        whose component router answers only the known agents."""
        super().__init__(
            session,
            queries,
            emitter,
            provider,
            retrieval,
            reranking,
            rerank_expanded_union=rerank_expanded_union,
        )
        self._pharma_input = pharma_input
        self._recorder = recorder
        self._infos_loader = infos_loader
        self._pharma_planner = PharmaPlannerAgent(JsonAgentRunner(provider, "planner"))
        self._pharma_critic = PharmaCriticAgent(JsonAgentRunner(provider, "critic"))
        self._pharma_risk = PharmaRiskAgent(JsonAgentRunner(provider, "risk"))
        self._pharma_decision = PharmaDecisionAgent(JsonAgentRunner(provider, "final_decision"))
        self._action_classifier = action_classifier
        # The question the analysis agents and the reranker read: the planner's restatement, set
        # when the plan arrives; never displayed (PH-DEF-026).
        self._analysis_question = ""
        # Each citation key's policy-checked excerpt, for the citation rows (PH-DEF-026).
        self._checked_excerpts: dict[str, str] = {}

    async def _run_workflow(
        self, query_run: QueryRun, domain: str, objective: str, started: float
    ) -> None:
        run_input = self._pharma_input
        profile = resolve(query_run.routing_profile, AnswerModel(query_run.answer_model))

        await self._emitter.emit(EventType.PLANNING_STARTED, "Planning the analysis.")
        plan = await self._step(
            AgentName.PLANNER,
            {"question": query_run.question, "intent": run_input.intent.value},
            self._pharma_planner.run(
                query_run.question, run_input.intent, run_input.incident, run_input.filters
            ),
        )
        normalized = plan.normalized_question or query_run.question
        # FR-002, as the General workflow: the planner has seen no evidence.
        self._analysis_question = (
            query_run.question if claims_unseen_evidence(normalized) else normalized
        )
        # PH-DEF-026: the planner's restatement is model text the regulatory-boundary check never
        # reads, so it is only the analysis agents' input; the run shows the question as asked.
        query_run.normalized_question = query_run.question

        rows = await self._infos_loader.controlled_for_project(query_run.project_id)
        infos = {row.document_id: info_from_row(row) for row in rows}
        references = _UnknownReferences(
            _known_references(infos, query_run.question, run_input.incident)
        )
        considerations = _checked_model_text(plan.escalation_considerations, references)
        classes = {row.document_id: row.data_classification for row in rows}
        retrieved = await self._pharma_retrieve(
            query_run, retrieval_queries(run_input, plan, query_run.question), infos
        )
        context = retrieved.context
        excluded = retrieved.resolution.excluded
        chunk_rows = await self._infos_loader.get_many(
            [chunk.chunk_id for chunk, _, _ in retrieved.considered]
        )
        signals = extract_signals(
            query_run.question, run_input.incident, context, filters=run_input.filters
        )
        logger.info(
            "Pharmaceutical signals extracted",
            extra={
                "event_type": "pharma_signals_extracted",
                "pharma": step_fields(
                    instruction_text=signals.instruction_text,
                    quality_relevant=signals.quality_relevant,
                    matched=len(signals.matched),
                ),
            },
        )
        escalation_inputs = EscalationInputs(
            intent=run_input.intent,
            sufficiency=retrieved.assessment.sufficiency,
            conflict_count=len(retrieved.conflicts),
            risk_flags=(),
            policy_removals=0,
            signals=signals,
        )
        excluded_read = _excluded(excluded, chunk_rows, infos)
        considered = [
            ConsideredPassage(
                chunk=chunk,
                anchor=anchor_for(chunk_rows[chunk.chunk_id], infos.get(chunk.document_id)),
                in_context=state is SelectionState.SELECTED,
                exclusion=reason,
            )
            for chunk, state, reason in retrieved.considered
            if chunk.chunk_id in chunk_rows
        ]
        used = _classifications(context, classes)

        if retrieved.assessment.sufficiency is EvidenceSufficiency.INSUFFICIENT:
            logger.info(
                "Pharmaceutical run abstained",
                extra={
                    "event_type": "pharma_abstained",
                    "pharma": step_fields(
                        sufficiency=retrieved.assessment.sufficiency,
                        context=len(context),
                        excluded=len(excluded),
                    ),
                },
            )
            escalation = merge_model_reasons(decide_escalation(escalation_inputs), considerations)
            inputs = ResponseInputs(
                question=query_run.question,
                intent=run_input.intent,
                incident=run_input.incident,
                assessment=retrieved.assessment,
                escalation=escalation,
                conflicts=retrieved.conflicts,
                classifications=used,
            )
            requested = requests_regulated_decision(query_run.question)
            await self._conclude(
                query_run,
                inputs,
                abstention_response(inputs, request=requested),
                escalation_inputs=escalation_inputs,
                excluded=excluded_read,
                considered=considered,
                signals=signals,
                requested=requested,
                started=started,
                # REQ-PER-05 (PH-DEF-032): no answer model ran, so the run keeps the provider and
                # model it was created with.
                provenance=(query_run.resolved_provider or "", query_run.resolved_model_name or ""),
            )
            return

        await self._emitter.emit(EventType.RESEARCH_STARTED, "Synthesizing evidence.")
        await self._supply(query_run.id, context)
        labelled = [
            replace(c, source_filename=register_label(c, infos.get(c.document_id))) for c in context
        ]
        research = await self._step(
            AgentName.RESEARCHER,
            {
                "normalized_question": self._analysis_question,
                "supplied_chunk_ids": [str(item.chunk_id) for item in context],
            },
            self._researcher.run(
                self._analysis_question,
                labelled,
                role=profile.role_for(CallSite.RESEARCHER),
            ),
        )

        await self._emitter.emit(EventType.CRITIC_STARTED, "Checking claim support.")
        critic = await self._step(
            AgentName.CRITIC,
            {"findings": len(research.key_findings)},
            self._pharma_critic.run(
                self._analysis_question,
                research.key_findings,
                context,
                infos,
                role=profile.role_for(CallSite.CRITIC),
            ),
        )

        await self._emitter.emit(EventType.RISK_STARTED, "Assessing risks.")
        risk = await self._step(
            AgentName.RISK,
            {"findings": len(research.key_findings)},
            self._pharma_risk.run(
                self._analysis_question,
                run_input.incident,
                research.key_findings,
                context,
                infos,
                role=profile.role_for(CallSite.RISK),
            ),
        )

        await self._emitter.emit(EventType.ANSWER_STARTED, "Composing the decision.")
        await self._supply(query_run.id, context)
        decision = await self._step(
            AgentName.FINAL_DECISION,
            {
                "sufficiency": retrieved.assessment.sufficiency.value,
                "answer_model": query_run.answer_model.value,
            },
            self._pharma_decision.run(
                self._analysis_question,
                run_input.intent,
                run_input.incident,
                retrieved.assessment.sufficiency,
                research.key_findings,
                critic,
                risk,
                context,
                infos,
                role=profile.role_for(CallSite.FINAL_DECISION),
            ),
        )
        provider, model = self._last_provider, self._last_model
        # PH-DEF-036: the risk descriptions and the researcher's contradictions are checked for
        # raw identifiers and invented references too, before the decision, whose uncertainty
        # discloses every removal.
        contradictions = _checked_model_text(
            [c.description for c in research.contradictions], references
        )
        descriptions = [
            references.keep(_RAW_ID.sub(UNLABELLED_SOURCE, draft.description))
            for draft in risk.flags
        ]
        decision = _conceal_ids(decision)
        decision = _drop_unknown_references(decision, references)
        # PLV4-T03: the structured layer removes a point that declares or takes a regulated act
        # before the response is assembled; its findings join the response check's.
        decision, structured = check_decision(decision)
        # PLV4-T04: the secondary detector reads the same model-written text and can only add
        # removals; its findings join the structured layer's.
        # RMD-DEF-034: the planner's escalation considerations become reasons shown to the user,
        # so the detector reads them too.
        classification = await self._classify(
            {
                **classifier_texts(decision),
                **{f"contradictions[{i}]": text for i, text in enumerate(contradictions)},
                **{f"risk_flags[{i}].description": text for i, text in enumerate(descriptions)},
                **{
                    f"escalation_considerations[{i}]": text for i, text in enumerate(considerations)
                },
            },
            [chunk.text for chunk in context],
        )
        if classification is not None:
            texts = classification.texts
            decision = apply_classification(decision, texts)
            contradictions = [
                kept
                for i in range(len(contradictions))
                if (kept := texts[f"contradictions[{i}]"]).strip()
            ]
            descriptions = [texts[f"risk_flags[{i}].description"] for i in range(len(descriptions))]
            considerations = [
                kept
                for i, text in enumerate(considerations)
                if (kept := texts.get(f"escalation_considerations[{i}]", text)).strip()
            ]
            structured = [*structured, *classification.findings]
        keys = citation_key_map(context)
        flagged = [resolve_references(draft.chunk_ids, keys) for draft in risk.flags]
        # Only a flag still shown adds its passages (round 3, RMD-DEF-160): a removed one's passage
        # is neither evidence nor a citation.
        shown = [c for f, d in zip(flagged, descriptions, strict=True) if d.strip() for c in f]
        evidence_items, drafts = self._evidence_items(
            decision, context, chunk_rows, infos, keys, flagged=shown
        )
        key_of = {draft.chunk_id: draft.citation_key for draft in drafts}
        flags = [
            RiskFlag(
                category=draft.category,
                severity=draft.severity,
                description=description,
                citation_keys=[key_of[c] for c in chunks if c in key_of],
            )
            for draft, description, chunks in zip(risk.flags, descriptions, flagged, strict=True)
            if description.strip()
        ]
        model_reasons = [
            *considerations,
            *(f.description for f in flags if f.category is PharmaRiskCategory.ESCALATION),
        ]
        # PLV9-T03 (PLV-D-033): a decision that says the passages do not answer the question is
        # escalated as insufficient evidence and shows the decision abstention; the roles it
        # names join the rules' roles. The model can only add an abstention, a review or a role:
        # since RMD3-T03 a partial answer is escalated as partial evidence (REQ-ESC-02), and a
        # conflict as a conflict (REQ-ESC-03, document control and quality assurance).
        answerability = decision.answerability
        # RMD3-T03: a request to take a regulated decision is refused whatever the decision says.
        requested = requests_regulated_decision(query_run.question)
        decided = requested or answerability is Answerability.NONE
        sufficiency = escalation_inputs.sufficiency
        if decided:
            sufficiency = EvidenceSufficiency.INSUFFICIENT
        elif (
            answerability is Answerability.PARTIAL and sufficiency is EvidenceSufficiency.SUFFICIENT
        ):
            sufficiency = EvidenceSufficiency.PARTIAL
        escalation_inputs = replace(
            escalation_inputs,
            risk_flags=flags,
            sufficiency=sufficiency,
            conflict_count=escalation_inputs.conflict_count
            + int(answerability is Answerability.CONFLICT),
        )
        escalation = with_model_roles(
            merge_model_reasons(decide_escalation(escalation_inputs), model_reasons),
            decision.escalate_to,
        )
        inputs = ResponseInputs(
            question=query_run.question,
            intent=run_input.intent,
            incident=run_input.incident,
            assessment=retrieved.assessment,
            escalation=escalation,
            decision=decision,
            evidence=evidence_items,
            conflicts=retrieved.conflicts,
            model_contradictions=contradictions,
            risk_flags=flags,
            classifications=used,
        )
        await self._conclude(
            query_run,
            inputs,
            abstention_response(inputs, decided=True, request=requested)
            if decided
            else assemble_response(inputs),
            escalation_inputs=escalation_inputs,
            excluded=excluded_read,
            considered=considered,
            signals=signals,
            started=started,
            citations=drafts,
            context=context,
            provenance=(provider, model),
            profile=profile,
            structured=structured,
            classification=classification,
            decided=decided,
            answerability=answerability,
            requested=requested,
        )

    async def _classify(
        self, texts: dict[str, str], passages: Sequence[str] = ()
    ) -> Classification | None:
        """The secondary detector over the run's model-written text (PLV4-T04), metered as any
        model call is (its failure as a charged failure); ``None`` without a detector. The
        context's ``passages`` let a flag on a sentence they state word for word be kept off it
        (RE2E-DEF-033)."""
        if self._action_classifier is None:
            return None
        started = time.monotonic()
        classification = await classify_actions(self._action_classifier, texts, passages=passages)
        if classification.provider:
            usage = classification.usage
            self._usage = TokenUsage(
                prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens + usage.completion_tokens,
            )
            await self._record_llm_call(
                CLASSIFIER_COMPONENT,
                usage,
                started,
                provider=classification.provider,
                model=classification.model,
                rung=max(classification.rungs, 1),
            )
        if isinstance(classification.failure, CognifyError):
            await self._record_charged_failure(
                CLASSIFIER_COMPONENT, classification.failure, started
            )
        return classification

    async def _pharma_retrieve(
        self,
        query_run: QueryRun,
        queries: Sequence[str],
        infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    ) -> _Retrieved:
        """Round 0 only: gather, resolve versions, rerank the eligible, order by authority, then
        classify sufficiency and detect conflicts, each step with the active retrieval tuning
        (``retrieval_tuning.ACTIVE``; PLV6-T03, PLV-D-019), which ``CURRENT`` leaves as it was.
        The retriever's row is opened before the work and records failure like any step (M-6,
        M-9); it holds the queries actually searched."""
        run_input = self._pharma_input
        queries, expansion = tuned_queries(queries, query_run.question, ACTIVE)
        await self._emitter.emit(
            EventType.RETRIEVAL_STARTED, "Retrieving evidence.", payload={"round": 0}
        )
        started = time.monotonic()
        agent_run_id = await self._open_agent_run(AgentName.RETRIEVER, {"queries": list(queries)})
        mode = RetrievalMode(query_run.retrieval_mode)
        try:
            self._candidates = await self._retriever.gather(
                query_run.project_id, queries, mode, round_index=0
            )
            resolution = tuned_resolution(
                self._candidates,
                resolve_versions(self._candidates, infos, run_input.filters),
                ACTIVE,
            )
            await self._emitter.emit(
                EventType.RERANKING_STARTED,
                "Reranking candidates.",
                payload={"round": 0, "candidates": len(resolution.eligible)},
            )
            reranked = await self._retriever.rerank(self._analysis_question, resolution.eligible)
            await self._meter_reranker(started)
            context = (
                refine_context(
                    self._reranking.last_ranking,
                    infos,
                    run_input.intent,
                    len(reranked),
                    ACTIVE,
                    question=query_run.question,
                )
                if reranked
                else []
            )
            considered = considered_order(
                context, self._reranking.last_ranking, resolution.excluded
            )
            await self._persist_evidence(
                query_run.id, [(chunk, state) for chunk, state, _ in considered]
            )
            kept = {chunk.chunk_id for chunk in context}
            in_context = VersionResolution(
                [chunk for chunk in resolution.eligible if chunk.chunk_id in kept],
                resolution.excluded,
                resolution.conflicts,
            )
            assessment = classify_sufficiency_v2(
                query_run.question, run_input.incident, context, infos
            )
            conflicts = [
                ConflictRead(
                    kind=found.kind,
                    description=found.description,
                    document_numbers=list(found.document_numbers),
                )
                for found in detect_conflicts(in_context, infos)
            ]
        except Exception as exc:
            await self._close_failed(AgentName.RETRIEVER, agent_run_id, started, exc)
            raise
        counts = retrieval_counts(resolution, assessment)
        logger.info(
            "Pharmaceutical versions resolved",
            extra={
                "event_type": "pharma_versions_resolved",
                "pharma": step_fields(
                    candidates=len(self._candidates),
                    eligible=len(resolution.eligible),
                    excluded=len(resolution.excluded),
                    reasons=sorted({item.reason.value for item in resolution.excluded}),
                    tuning=ACTIVE.name,
                    queries=len(queries),
                    expanded_terms=expansion.matched if expansion is not None else [],
                ),
            },
        )
        logger.info(
            "Pharmaceutical sufficiency classified",
            extra={
                "event_type": "pharma_sufficiency_classified",
                "pharma": step_fields(
                    sufficiency=assessment.sufficiency,
                    rule_version=assessment.rule_version,
                    context=len(context),
                ),
            },
        )
        logger.info(
            "Pharmaceutical conflicts detected",
            extra={
                "event_type": "pharma_conflicts_detected",
                "pharma": step_fields(
                    conflicts=len(conflicts),
                    kinds=sorted({conflict.kind.value for conflict in conflicts}),
                ),
            },
        )
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.COMPLETED,
            output_json={
                **self._retrieval_envelope(round_index=0, selected=context),
                **counts,
                "rule_version": assessment.rule_version,
            },
        )
        await self._emitter.emit(
            EventType.RETRIEVAL_COMPLETED,
            "Evidence selected.",
            payload={"round": 0, "count": len(context), **counts},
        )
        return _Retrieved(context, resolution, assessment, conflicts, considered)

    def _evidence_items(
        self,
        decision: PharmaDecisionOutput,
        context: Sequence[RetrievedChunk],
        chunk_rows: Mapping[uuid.UUID, Chunk],
        infos: Mapping[uuid.UUID, ControlledDocumentInfo],
        keys: Mapping[str, uuid.UUID],
        flagged: Sequence[uuid.UUID] = (),
    ) -> tuple[list[EvidenceItem], list[CitationDraft]]:
        """The decision's citations, resolved from their keys (CT-RMD-03), kept by the baseline
        rules (a chunk of the final context, a key the decision's text uses; REQ-SEC-03), each
        anchored from stored rows; then each passage of the final context a risk flag names that
        the decision did not cite (RMD-DEF-024), so the flag keeps the passage it rests on."""
        text = " ".join(
            [
                decision.summary,
                *point_texts(decision),
                *decision.missing_information,
                *decision.investigation_questions,
                *decision.uncertainty,
            ]
        )
        resolved, dropped = resolve_citations(decision, keys)
        if dropped:
            logger.warning("Dropped %d decision citation(s) that name no passage key", dropped)
        drafts, self._citation_report = validate_citations(resolved, context, text)
        by_id = {chunk.chunk_id: chunk for chunk in context}
        items: list[EvidenceItem] = []
        kept: list[CitationDraft] = []
        for draft in drafts:
            row = chunk_rows.get(draft.chunk_id)
            if row is None:
                continue
            chunk = by_id[draft.chunk_id]
            items.append(
                EvidenceItem(
                    citation_key=draft.citation_key,
                    anchor=anchor_for(row, infos.get(chunk.document_id)),
                    excerpt=flatten(draft.snippet, MAX_EXCERPT_LENGTH),
                )
            )
            kept.append(draft)
        key_by_chunk = {chunk_id: key for key, chunk_id in keys.items()}
        cited = {draft.chunk_id for draft in kept}
        for chunk_id in dict.fromkeys(flagged):
            row = chunk_rows.get(chunk_id)
            key = key_by_chunk.get(chunk_id)
            if row is None or key is None or chunk_id in cited or chunk_id not in by_id:
                continue
            items.append(
                EvidenceItem(
                    citation_key=key,
                    anchor=anchor_for(row, infos.get(by_id[chunk_id].document_id)),
                    excerpt="",
                )
            )
            kept.append(CitationDraft(citation_key=key, chunk_id=chunk_id, snippet=""))
            cited.add(chunk_id)
        return items, kept

    async def _conclude(
        self,
        query_run: QueryRun,
        inputs: ResponseInputs,
        response: PharmaResponse,
        *,
        escalation_inputs: EscalationInputs,
        excluded: list[ExcludedEvidenceRead],
        considered: list[ConsideredPassage],
        signals: RiskSignals,
        started: float,
        citations: Sequence[CitationDraft] = (),
        context: Sequence[RetrievedChunk] = (),
        provenance: tuple[str, str] = ("", ""),
        profile: RoutingProfile | None = None,
        structured: Sequence[PolicyFinding] = (),
        classification: Classification | None = None,
        decided: bool = False,
        answerability: Answerability = Answerability.FULL,
        requested: bool = False,
    ) -> None:
        """Check, record, persist and finish. Nothing reaches the recorder or the database before
        the regulatory-boundary check has passed; a check that raises fails the run (fail
        closed, REQ-POL-12, REQ-NFR-02).

        The check is layered (PLV4-T03): the structured layer's point removals (``structured``),
        the response check, the policy abstention when a critical removal leaves no supported
        procedure statement, the display layer over every displayed text, and the rendered
        layer over the answer Markdown; the result is rechecked, and the trace of every layer's
        removals is recorded, never the removed text.

        Before the check, on an answered run, the grounding gate (PLV7-T03, PLV-D-023) withholds
        each claim its cited passages do not support, or gives the grounding abstention when no
        supported claim remains; its trace record holds counts and digests only. An unsupported
        claim the policy's engines flag is left to the response check, so a regulated act keeps
        its code, its trace and its escalation whichever gate withholds it.

        A decision abstention (``decided``, PLV9-T03) holds no claim, so the grounding gate does
        not run on it; the policy layers check it as they check any response.

        First of all, the response's model-written texts lose the evidence context lines and the
        planted instructions the model copied into them (RMD8-T02, DV-RMD-56), so every gate and
        layer reads the filtered text."""
        response, context_lines, quoted_instructions = _without_copied_context(response, context)
        abstained = profile is None
        grounded = None if abstained or decided else _ground(inputs, response, citations, context)
        if grounded is not None:
            response = grounded.response
        policy_context = PolicyContext(
            sufficiency=inputs.assessment.sufficiency,
            abstention=abstention_response(inputs),
        )
        result = apply_policy(response, policy_context)
        if grounded is not None and grounded.action != GROUNDING_ABSTENTION:
            # RMD-DEF-015: a sentence the gate left to the response check because it takes an act
            # is read again as that check left it. A part it kept (a leading clause, RMD4-T02's
            # cuts) that its passages do not hold is withheld as the gate withholds any other.
            again = _ground(inputs, result.response, citations, context)
            if again.action == CLAIMS_WITHHELD:
                result = replace(result, response=again.response)
                grounded = replace(
                    grounded,
                    response=again.response,
                    action=CLAIMS_WITHHELD,
                    withheld=grounded.withheld + again.withheld,
                )
        if grounded is not None and grounded.action == GROUNDING_ABSTENTION:
            # The check has read the drafted claims; its removals among them stand, and the
            # grounding abstention, checked in turn, is what is displayed.
            drafted = [f for f in result.findings if f.field in grounded.claim_fields]
            result = with_removals(
                apply_policy(abstention_response(inputs, grounding=True), policy_context), drafted
            )
        # DV-RMD-19 and DV-RMD-23 (RMD3-T03): an abstention decided before this check (by the
        # sufficiency rule or the decision) never held a procedure statement, so a removal (from
        # its discarded draft, or from model text it displays such as an escalation reason) is
        # recorded and escalated but does not replace it with the policy abstention.
        result = with_removals(result, structured)
        refused = not (abstained or decided) and needs_abstention(result)
        if refused:
            result = with_abstention(result, policy_abstention(result.response))
        checked = result.response
        if (
            inputs.assessment.sufficiency is EvidenceSufficiency.PARTIAL
            and not checked.missing_information
        ):
            # REQ-SUF-03 (PH-DEF-028): the check removed every model-stated item, so the rule's
            # own statement of what is missing takes their place.
            checked.missing_information = missing_from(inputs.assessment)
        result = withhold_displayed(result)
        rendered = check_rendered(render_markdown(result.response), _masked(result.response))[1]
        result = with_removals(result, rendered)
        checked = result.response
        # PH-DEF-026: a citation row shows the checked excerpt, never the model's quote or the
        # raw passage the citation rules fall back to.
        self._checked_excerpts = {
            item.citation_key: item.excerpt for item in checked.evidence_and_citations
        }
        if result.findings:
            # REQ-ESC-05: any removal escalates to quality assurance, whatever its code.
            removal = decide_escalation(
                replace(escalation_inputs, policy_removals=len(result.findings))
            )
            checked.required_human_escalation = _with_rule_reasons(
                checked.required_human_escalation, removal
            )
        if classification is not None and not classification.available:
            # PLV4-T04: the deterministic result stands, and quality assurance reviews it.
            checked.required_human_escalation = with_unavailable_escalation(
                checked.required_human_escalation
            )
        logger.info(
            "Pharmaceutical policy checked",
            extra={
                "event_type": "pharma_policy_checked",
                "pharma": step_fields(
                    findings=len(result.findings),
                    codes=sorted({finding.code.value for finding in result.findings}),
                    layers=sorted({finding.layer for finding in result.findings}),
                    notices=len(result.notices),
                    forced_escalation=result.forced_escalation,
                    classifier=(
                        "absent"
                        if classification is None
                        else str(classifier_record(classification)["action"])
                    ),
                    grounding_claims=0 if grounded is None else len(grounded.result.claims),
                    grounding_unsupported=(
                        0 if grounded is None else len(grounded.result.unsupported)
                    ),
                    grounding_action=GROUNDING_NOT_RUN if grounded is None else grounded.action,
                    context_lines=context_lines,
                    quoted_instructions=quoted_instructions,
                ),
            },
        )
        escalation = checked.required_human_escalation
        logger.info(
            "Pharmaceutical escalation decided",
            extra={
                "event_type": "pharma_escalation_decided",
                "pharma": step_fields(
                    required=escalation.required,
                    roles=[role.value for role in escalation.roles],
                    reasons=[reason.code for reason in escalation.reasons],
                ),
            },
        )
        markdown = _rechecked(result, rendered)
        disposition, gate = disposition_of(
            inputs.assessment.sufficiency,
            sufficiency_abstained=abstained,
            decided=decided,
            grounding_abstained=grounded is not None and grounded.action == GROUNDING_ABSTENTION,
            refused=refused,
            answerability=answerability,
            requested=requested,
        )
        await self._recorder.record(
            PharmaRunOutcome(
                query_run_id=query_run.id,
                run_input=self._pharma_input,
                assessment=inputs.assessment,
                abstained=abstained,
                response=checked,
                answer_markdown=markdown,
                excluded=excluded,
                findings=result.findings,
                notices=result.notices,
                signals=signals,
                disposition=disposition,
                disposition_gate=gate,
                considered=considered,
                trace=[
                    *([] if grounded is None else grounded.trace()),
                    *trace_records(result.findings),
                    *([] if classification is None else [classifier_record(classification)]),
                ],
            )
        )
        answer = FinalDecisionOutput(
            recommendation=flatten(checked.incident_or_question_summary, MAX_SUMMARY_LINE),
            answer_markdown=markdown,
            citations=list(citations),
            contradictory_evidence=list(checked.conflicts_or_ambiguities.model_reported),
            uncertainty=" ".join(checked.limitations_and_uncertainty),
            next_steps=list(checked.suggested_investigation_questions_or_next_steps),
        )
        provider, model = provenance
        await self._persist_answer(query_run, answer, provider, model)
        if profile is not None:
            kept = await self._persist_citations(query_run, answer, context)
            await self._emitter.emit(EventType.EVALUATION_STARTED, "Scoring the answer.")
            await self._evaluate(
                query_run, answer, kept, context, profile.role_for(CallSite.EVALUATOR)
            )
        await self._finish(query_run, started)

    async def _persist_citations(
        self,
        query_run: QueryRun,
        decision: FinalDecisionOutput,
        evidence: Sequence[RetrievedChunk],
    ) -> list[CitationDraft]:
        """The baseline's citation rows with one change (PH-DEF-026): each snippet is the
        policy-checked excerpt of its key, so neither the model's quote nor the passage text the
        citation rules fall back to reaches the detail or the report unchecked."""
        valid, self._citation_report = validate_citations(
            decision.citations, evidence, decision.answer_markdown
        )
        valid = [
            draft.model_copy(update={"snippet": self._checked_excerpts.get(draft.citation_key, "")})
            for draft in valid
        ]
        now = utc_now()
        by_chunk = {item.chunk_id: item for item in evidence}
        rows = [
            Citation(
                id=new_uuid(),
                query_run_id=query_run.id,
                chunk_id=draft.chunk_id,
                document_id=by_chunk[draft.chunk_id].document_id,
                citation_key=draft.citation_key,
                snippet=draft.snippet,
                source_filename=by_chunk[draft.chunk_id].source_filename,
                page_number=by_chunk[draft.chunk_id].page_number,
                relevance_score=relevance_of(by_chunk[draft.chunk_id]),
                created_at=now,
            )
            for draft in valid
        ]
        await self._queries.add_citations(rows)
        await self._session.commit()
        return valid


def _masked(response: PharmaResponse) -> tuple[str, ...]:
    """What the rendered layer leaves unread: the system's own texts, and the document's own
    words the response and display layers read with the patterns (``policy.quoted_texts``)."""
    return (*TRUSTED_TEXTS, UNAVAILABLE_REASON_TEXT, *quoted_texts(response))


# --- The grounding gate (PLV7-T03, PLV-D-023) -----------------------------------------------------

GROUNDING_LAYER: Final = "grounding"
GROUNDING_NOT_RUN: Final = "not_run"
GROUNDING_PASSED: Final = "passed"
CLAIMS_WITHHELD: Final = "claims_withheld"
GROUNDING_ABSTENTION: Final = "abstention_enforced"
_SUMMARY_FIELD: Final = "incident_or_question_summary"


@dataclass(frozen=True)
class _Grounded:
    """The gate's outcome: the response the policy check reads (unsupported sentences withheld,
    those the policy's engines flag left for it), the claims checked, what the gate did, how many
    sentences it withheld itself, and the claims' field names, by which the policy's removals
    among them are found."""

    response: PharmaResponse
    result: GroundingResult
    action: str
    withheld: int
    claim_fields: frozenset[str]

    def trace(self) -> list[dict[str, object]]:
        """The gate's trace record when it withheld or abstained: counts, reason codes and
        digests, never a claim's text. A sentence left to the response check is that layer's
        record, under its code."""
        if self.action == GROUNDING_PASSED:
            return []
        return [
            {
                "layer": GROUNDING_LAYER,
                "action": self.action,
                **self.result.record(),
                "withheld": self.withheld,
                "support_threshold": SUPPORT_THRESHOLD,
            }
        ]


def claim_sources(response: PharmaResponse) -> list[ClaimSource]:
    """The evidence-bearing fields (methodology section 2): the summary, each procedure's summary
    with its citation keys and each risk description with its flag's keys."""
    return [
        ClaimSource(_SUMMARY_FIELD, response.incident_or_question_summary),
        *(
            ClaimSource(f"relevant_procedures[{i}].summary", p.summary, tuple(p.citation_keys))
            for i, p in enumerate(response.relevant_procedures)
        ),
        *(
            ClaimSource(
                f"quality_and_operational_risks[{i}].description",
                flag.description,
                tuple(flag.citation_keys),
            )
            for i, flag in enumerate(response.quality_and_operational_risks)
        ),
    ]


def _without_copied_context(
    response: PharmaResponse, context: Sequence[RetrievedChunk]
) -> tuple[PharmaResponse, int, int]:
    """The response without the evidence context lines and the planted instructions the model
    copied into its texts, the notice saying so joining its limitations; and the two counts.

    RMD8-T02 (RE2E-DEF-043, RE2E-DEF-044, DV-RMD-56). The model-written texts are the summary,
    the procedure summaries, the risk descriptions, the reported contradictions, the missing
    information, the next steps and the limitations. The deterministic conflicts are the
    application's. The excerpts quote the passages, so they lose only a planted sentence they
    copy (DV-RMD-62)."""
    passages = [chunk.text for chunk in context]
    lines = 0
    copies = 0

    def clean(text: str) -> str:
        nonlocal lines, copies
        stripped, removed_lines = strip_context_lines(text)
        cleaned, removed_copies = drop_planted_copies(stripped, passages)
        lines += removed_lines
        copies += removed_copies
        return cleaned

    def clean_all(texts: Sequence[str]) -> list[str]:
        return [cleaned for cleaned in map(clean, texts) if cleaned]

    def clean_excerpt(excerpt: str) -> str:
        # DV-RMD-62: an excerpt quotes its passage, so it keeps the passage's lines but not a
        # sentence planted in it (SC-P09's live excerpts quoted DS-P09's override).
        nonlocal copies
        cleaned_excerpt, removed = drop_planted_copies(excerpt, passages)
        copies += removed
        return cleaned_excerpt

    cleaned = response.model_copy(
        update={
            "incident_or_question_summary": clean(response.incident_or_question_summary),
            "evidence_and_citations": [
                item.model_copy(update={"excerpt": clean_excerpt(item.excerpt)})
                for item in response.evidence_and_citations
            ],
            "relevant_procedures": [
                procedure.model_copy(update={"summary": clean(procedure.summary)})
                for procedure in response.relevant_procedures
            ],
            "quality_and_operational_risks": [
                flag.model_copy(update={"description": clean(flag.description)})
                for flag in response.quality_and_operational_risks
            ],
            "conflicts_or_ambiguities": response.conflicts_or_ambiguities.model_copy(
                update={
                    "model_reported": clean_all(response.conflicts_or_ambiguities.model_reported)
                }
            ),
            "missing_information": clean_all(response.missing_information),
            "suggested_investigation_questions_or_next_steps": clean_all(
                response.suggested_investigation_questions_or_next_steps
            ),
            "limitations_and_uncertainty": clean_all(response.limitations_and_uncertainty),
        }
    )
    if not (lines or copies):
        return response, 0, 0
    cleaned.limitations_and_uncertainty.append(notice_for(lines, copies))
    return cleaned, lines, copies


def _ground(
    inputs: ResponseInputs,
    response: PharmaResponse,
    citations: Sequence[CitationDraft],
    context: Sequence[RetrievedChunk],
) -> _Grounded:
    """Each claim of the assembled response checked against the full text of the passages it
    cites (``grounding.check_grounding``): unchanged when every substantive claim is supported or
    each unsupported one is a sentence the policy's engines flag (``layered_violations``; the
    response check removes it under its code), the other unsupported sentences withheld when
    some claim is supported, the grounding abstention (which ``_conclude`` displays after the
    policy check has read the drafted claims) when none is."""
    sources = claim_sources(response)
    fields = frozenset(source.field for source in sources)
    texts = {chunk.chunk_id: chunk.text for chunk in context}
    passages = {d.citation_key: texts[d.chunk_id] for d in citations if d.chunk_id in texts}
    # DV-RMD-14: each cited passage's register label, from the stored anchors the response holds.
    labels = {
        item.citation_key: PassageLabel(
            item.anchor.document_number, item.anchor.revision, item.anchor.lifecycle_status.value
        )
        for item in response.evidence_and_citations
    }
    result = check_grounding(
        sources,
        passages,
        trusted=frozenset(TRUSTED_TEXTS),
        labels=labels,
        inflections=True,
        clauses=True,
        strict=True,
    )
    withheld = [claim for claim in result.unsupported if not layered_violations(claim.sentence)]
    if not any(claim.supported for claim in result.claims):
        return _Grounded(
            _withheld(response, withheld), result, GROUNDING_ABSTENTION, len(withheld), fields
        )
    if not withheld:
        return _Grounded(response, result, GROUNDING_PASSED, 0, fields)
    return _Grounded(_withheld(response, withheld), result, CLAIMS_WITHHELD, len(withheld), fields)


def _withheld(response: PharmaResponse, withheld: Sequence[ClaimCheck]) -> PharmaResponse:
    """``response`` without the ``withheld`` sentences: a risk flag left without a sentence is
    dropped, a summary left without one gives its fixed text, their number is disclosed and
    quality assurance and a subject-matter expert review it (REQ-ESC-02)."""
    unsupported = {(claim.field, claim.sentence) for claim in withheld}
    fields = {name for name, _ in unsupported}

    def kept(name: str, text: str) -> str:
        if name not in fields:
            return text
        return " ".join(
            sentence.strip()
            for sentence in split_sentences(text)
            if sentence.strip() and (name, sentence.strip()) not in unsupported
        )

    grounded = response.model_copy(deep=True)
    summary = kept(_SUMMARY_FIELD, response.incident_or_question_summary)
    grounded.incident_or_question_summary = summary or (
        GROUNDING_WITHHELD_SUMMARY if response.incident_or_question_summary.strip() else ""
    )
    for i, procedure in enumerate(grounded.relevant_procedures):
        procedure.summary = kept(f"relevant_procedures[{i}].summary", procedure.summary)
    flags: list[RiskFlag] = []
    for i, flag in enumerate(grounded.quality_and_operational_risks):
        flag.description = kept(f"quality_and_operational_risks[{i}].description", flag.description)
        if flag.description.strip():
            flags.append(flag)
    grounded.quality_and_operational_risks = flags
    grounded.limitations_and_uncertainty.append(
        f"{len(withheld)} statement(s) not supported by the passages they cite were withheld."
    )
    grounded.required_human_escalation = with_rule(grounded.required_human_escalation, "REQ-ESC-02")
    return grounded


def _rechecked(result: PolicyResult, rendered: Sequence[PolicyFinding]) -> str:
    """The answer Markdown of the final checked response, the rendered layer's lines withheld,
    after the recheck (PLV4-T03): no displayed text and no rendered line may violate a code, and
    rendering the final response may withhold no line the rendered layer had not already found.
    Anything else fails the run closed (REQ-POL-12)."""
    masked = _masked(result.response)
    markdown, again = check_rendered(render_markdown(result.response), masked)
    found = {(f.code, f.sentence_sha256) for f in rendered}
    if (
        any((f.code, f.sentence_sha256) not in found for f in again)
        or check_fields(result.response)
        or check_rendered(markdown, masked)[1]
    ):
        msg = "The rewritten answer failed its regulatory-boundary recheck; no answer is shown."
        raise PolicyCheckError(msg)
    return markdown


def _with_rule_reasons(
    escalation: HumanEscalationRead, decided: HumanEscalationRead
) -> HumanEscalationRead:
    """``escalation`` with every rule reason and role of ``decided`` it lacks (never removes)."""
    merged = escalation.model_copy(deep=True)
    merged.required = merged.required or decided.required
    codes = {reason.code for reason in merged.reasons}
    merged.reasons.extend(r.model_copy() for r in decided.reasons if r.code not in codes)
    merged.roles.extend(role for role in decided.roles if role not in merged.roles)
    return merged


def _conceal_ids(decision: PharmaDecisionOutput) -> PharmaDecisionOutput:
    """No raw chunk identifier reaches the reader through prose (L-2); ids live in citations."""

    def conceal(text: str) -> str:
        return _RAW_ID.sub(UNLABELLED_SOURCE, text)

    return decision.model_copy(
        update={
            "summary": conceal(decision.summary),
            "points": [p.model_copy(update={"text": conceal(p.text)}) for p in decision.points],
            "procedure_points": [conceal(t) for t in decision.procedure_points],
            "missing_information": [conceal(t) for t in decision.missing_information],
            "investigation_questions": [conceal(t) for t in decision.investigation_questions],
            "uncertainty": [conceal(t) for t in decision.uncertainty],
        }
    )


@dataclass(frozen=True)
class KnownReferences:
    """The document numbers and revisions prose may name: the project's controlled documents, and
    any the inquiry itself names (REQ-SEC-03)."""

    numbers: frozenset[str]
    revisions: frozenset[tuple[str, str]]
    prefixes: frozenset[str]


def references_in(text: str, prefixes: frozenset[str]) -> list[tuple[str, str | None]]:
    """Each document number ``text`` names, upper case, with the revision named right after it."""
    found: list[tuple[str, str | None]] = []
    plain = _HYPHENS.sub("-", unicodedata.normalize("NFKC", text))
    for match in _ANY_CASE_NUMBER.finditer(plain):
        token = match.group(0)
        segments = token.split("-")
        upper = token.upper()
        if not (
            (token == upper and len(segments) >= 3)
            or (len(segments) >= 3 and any(c.isdigit() for c in token))
            or segments[0].upper() in prefixes
        ):
            continue
        revision = _NAMED_REVISION.match(plain, match.end())
        found.append((upper, revision.group(1).upper() if revision else None))
    return found


def _known_references(
    infos: Mapping[uuid.UUID, ControlledDocumentInfo], question: str, incident: IncidentInput | None
) -> KnownReferences:
    numbers = {info.document_number.upper() for info in infos.values() if info.document_number}
    prefixes = DOCUMENT_PREFIXES | {number.split("-")[0] for number in numbers}
    revisions = {
        (info.document_number.upper(), info.revision.upper())
        for info in infos.values()
        if info.document_number and info.revision
    }
    fields = incident.model_dump().values() if incident is not None else ()
    inquiry = " ".join([question, *(str(value) for value in fields if value)])
    for number, revision in references_in(inquiry, frozenset(prefixes)):
        numbers.add(number)
        if revision is not None:
            revisions.add((number, revision))
    return KnownReferences(frozenset(numbers), frozenset(revisions), frozenset(prefixes))


class _UnknownReferences:
    """Removes each sentence naming a document number, or a revision of a known number, the
    project does not hold, and remembers what it removed (REQ-SEC-03, PH-DEF-020, PH-DEF-036)."""

    def __init__(self, known: KnownReferences) -> None:
        self._known = known
        self.removed: set[str] = set()

    def _unknown(self, sentence: str) -> set[str]:
        unknown: set[str] = set()
        for number, revision in references_in(sentence, self._known.prefixes):
            if number not in self._known.numbers:
                unknown.add(number)
            elif revision is not None and (number, revision) not in self._known.revisions:
                unknown.add(f"{number} revision {revision}")
        return unknown

    def keep(self, text: str) -> str:
        kept: list[str] = []
        for sentence in split_sentences(text):
            unknown = self._unknown(sentence)
            self.removed.update(unknown)
            if not unknown:
                kept.append(sentence)
        return " ".join(kept)

    def keep_all(self, items: Sequence[str]) -> list[str]:
        return [kept for kept in map(self.keep, items) if kept.strip()]


def _checked_model_text(items: Sequence[str], references: _UnknownReferences) -> list[str]:
    """Model text outside the decision: raw chunk ids concealed, sentences naming an invented
    reference removed, emptied items dropped (PH-DEF-036)."""
    checked = [references.keep(_RAW_ID.sub(UNLABELLED_SOURCE, item)) for item in items]
    return [item for item in checked if item.strip()]


def _drop_unknown_references(
    decision: PharmaDecisionOutput, references: _UnknownReferences
) -> PharmaDecisionOutput:
    """REQ-SEC-03: a sentence naming a document number the project does not hold, or a revision of
    a held number it does not hold, is removed with its citation marker, so an invented source is
    never shown as one, and every removal in any field ``references`` checked is disclosed under
    uncertainty (PH-DEF-020, PH-DEF-036). The other fields are checked before this call."""
    keep, keep_all = references.keep, references.keep_all
    points = [p.model_copy(update={"text": keep(p.text)}) for p in decision.points]
    checked = decision.model_copy(
        update={
            "summary": keep(decision.summary),
            "points": [p for p in points if p.text.strip()],
            "procedure_points": keep_all(decision.procedure_points),
            "missing_information": keep_all(decision.missing_information),
            "investigation_questions": keep_all(decision.investigation_questions),
            "uncertainty": keep_all(decision.uncertainty),
        }
    )
    removed = references.removed
    if not removed:
        return checked
    logger.info(
        "Pharmaceutical unknown references removed",
        extra={
            "event_type": "pharma_unknown_references_removed",
            "pharma": step_fields(removed=len(removed)),
        },
    )
    notes = [
        f"A statement naming {number}, which is not a document of this project, was removed."
        for number in sorted(removed)
    ]
    # RMD-DEF-023: the notes go after the model's own items and cut none of them.
    return checked.model_copy(update={"uncertainty": [*checked.uncertainty, *notes]})


def _excluded(
    excluded: Sequence[ExcludedCandidate[RetrievedChunk]],
    chunk_rows: Mapping[uuid.UUID, Chunk],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
) -> list[ExcludedEvidenceRead]:
    """The excluded candidates as the result view lists them. Their titles and section headings
    come from the documents and are displayed, so each is withheld whole when it states something
    the regulatory-boundary check removes (PH-DEF-027); the passages themselves never reach the
    answer."""
    findings: list[PolicyFinding] = []
    reads: list[ExcludedEvidenceRead] = []
    for item in excluded:
        if item.chunk.chunk_id not in chunk_rows:
            continue
        anchor = anchor_for(chunk_rows[item.chunk.chunk_id], infos.get(item.chunk.document_id))
        anchor = anchor.model_copy(
            update={
                "title": withhold("excluded_evidence.anchor.title", anchor.title, findings),
                "section": withhold("excluded_evidence.anchor.section", anchor.section, findings),
            }
        )
        reads.append(ExcludedEvidenceRead(anchor=anchor, reason=item.reason))
    return reads


def _classifications(
    context: Sequence[RetrievedChunk], classes: Mapping[uuid.UUID, DataClassification | None]
) -> list[DataClassification | None]:
    documents = dict.fromkeys(chunk.document_id for chunk in context)
    return [classes.get(document_id) for document_id in documents]
