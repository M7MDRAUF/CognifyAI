"""Query API schemas (FR-CHAT-001/003)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.enums import (
    AgentName,
    AgentRunStatus,
    AnswerModel,
    QueryRunStatus,
    ResponseStyle,
    RetrievalMode,
    RiskCategory,
    RiskLevel,
)
from app.schemas._text import is_visually_blank, reject_control_characters
from app.schemas.pharma_api import PharmaQueryInput, PharmaRunRead

# The question is re-sent, in full, as part of the prompt for every one of the 7-20 billed vendor
# chat completions a single accepted run drives (directly for the planner, and via
# `normalized_question`'s fallback for the rest), so an unbounded `question` is an unbounded
# per-call input-token bill chosen by the caller. 4000 characters is roughly 1000 tokens: far
# more than any decision question needs, and small enough that the prompt size stays dominated by
# the retrieved evidence rather than by the request body.
MAX_QUESTION_LENGTH = 4000


class QueryCreate(BaseModel):
    """``model_option_id`` (optional) picks a concrete provider+model pair by the ``id`` a
    ``GET /llm/config`` ``model_options[]`` entry carries. Precedence against ``answer_model``:

    - absent: legacy path. ``answer_model`` alone picks the tier on the default provider,
      bit-identical to the pre-selector behaviour.
    - present and valid/available: wins outright. It determines both provider and model, and
      its tier is written back into ``answer_model`` before persistence — even when it
      conflicts with the ``answer_model`` the caller sent (e.g. ``"fast"`` plus a
      powerful-tier option) — so the stored tier always matches what was actually selected.
    - present but unrecognised: rejected with ``invalid_model_option`` (400).
    - present, a real option, but its provider is currently unavailable: rejected with
      ``model_option_unavailable`` (409) rather than silently falling back to another model.

    ``pharma`` (optional) is the pharmaceutical workflow's input: the intent, an incident and
    filters. Absent on a pharmaceutical project, the run is a procedure question; present on a
    General project, it is rejected with ``pharma_input_not_allowed`` (400) before anything is
    created.
    """

    question: str = Field(max_length=MAX_QUESTION_LENGTH)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    response_style: ResponseStyle = ResponseStyle.DETAILED
    answer_model: AnswerModel = AnswerModel.POWERFUL
    model_option_id: str | None = None
    # The named profile this run routes by (SDD 10.3). Additive and optional; absent means
    # `balanced`, or the `answer_model` alias when one was sent. An unknown name is rejected
    # 400 rather than coerced -- see `QueryService.create`.
    routing_profile: str | None = None
    # The pharmaceutical workflow's input (REQ-API-03). Additive and optional: absent on a
    # pharmaceutical project means a procedure question; present on a General project it is
    # refused with `pharma_input_not_allowed` before anything is created.
    pharma: PharmaQueryInput | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        reject_control_characters(value, field="Question")
        stripped = value.strip()
        if stripped == "" or is_visually_blank(stripped):
            msg = "Question must not be empty."
            raise ValueError(msg)
        return stripped


class QueryRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    question: str
    normalized_question: str
    retrieval_mode: RetrievalMode
    response_style: ResponseStyle
    answer_model: AnswerModel
    status: QueryRunStatus
    answer_markdown: str | None
    error_message: str | None
    latency_ms: int
    resolved_provider: str | None = None
    resolved_model_name: str | None = None
    created_at: datetime


class QueryRunCancelResponse(BaseModel):
    """The 202 of ``POST /queries/{id}/cancel`` (RMD5-T03; RE2E-DEF-037).

    Every property ``QueryRunRead`` had, plus ``cancel_requested``: the run as read when the
    cancel was accepted, whose ``status`` usually still reads ``running``, and the flag that says
    a cancel is pending -- the 202 used to carry nothing that did. This process's task is
    cancelled before the database is read, so during an outage the run cannot be read at all;
    the body then holds only ``id``, ``status`` and ``cancel_requested``, which is why those are
    the only required properties. The route leaves unset properties out of the body.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: QueryRunStatus
    cancel_requested: bool
    project_id: uuid.UUID | None = None
    question: str | None = None
    normalized_question: str | None = None
    retrieval_mode: RetrievalMode | None = None
    response_style: ResponseStyle | None = None
    answer_model: AnswerModel | None = None
    answer_markdown: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    resolved_provider: str | None = None
    resolved_model_name: str | None = None
    created_at: datetime | None = None


class QueryRunSummary(BaseModel):
    """One row of the paged run list (CT-02): what a history row shows, and no answer text.

    ``answer_markdown`` made up 78% of the legacy list's bytes; a list never renders it. The
    detail route (``GET /queries/{id}``) still serves the whole run.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    question: str
    status: QueryRunStatus
    answer_model: AnswerModel
    resolved_provider: str | None = None
    resolved_model_name: str | None = None
    error_message: str | None
    created_at: datetime


class QueryRunSummaryPage(BaseModel):
    """A keyset page: its rows newest first, the cursor for the next page (``None`` on the last),
    and how many of the project's runs match the filter."""

    items: list[QueryRunSummary]
    next_cursor: str | None
    total: int


class CitationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    citation_key: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_filename: str
    snippet: str
    page_number: int | None
    relevance_score: float


class AgentRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_name: AgentName
    status: AgentRunStatus
    latency_ms: int | None
    error_message: str | None
    llm_provider: str | None = None
    llm_model: str | None = None
    # Additive, and safe for the frontend's non-strict `z.object`, which strips fields it does
    # not know. `failure_excerpt` is deliberately **not** exposed here: it is a diagnostic for an
    # operator reading the database, not something to put on an anonymous API surface.
    attempt_count: int = 1
    error_classification: str | None = None


# PH-D-030: shown in place of the evaluator's explanation on a pharmaceutical run, in the detail
# and the report. The explanation is model text the regulatory-boundary check never sees; the
# five scores are numbers and are shown as they are.
EVALUATION_EXPLANATION_WITHHELD = (
    "Not shown for a pharmaceutical run: the evaluator's explanation is model text that the "
    "regulatory-boundary check does not cover."
)


class EvaluationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    answer_relevance: float
    faithfulness: float
    citation_accuracy: float
    context_sufficiency: float
    hallucination_risk: float
    explanation: str


class RiskItemRead(BaseModel):
    """One Risk agent item as a reader sees it: the prose fields concealed (L-2)."""

    category: RiskCategory
    description: str
    severity: RiskLevel
    likelihood: RiskLevel
    mitigation: str


class QueryRunDetail(QueryRunRead):
    citations: list[CitationRead] = []
    agent_runs: list[AgentRunRead] = []
    evaluation: EvaluationRead | None = None
    # C-15: requested by the decision prompt, produced by the agent, and read by no consumer
    # until the report surfaced them -- the detail response is the other reader. `None` means the
    # decision step produced no readable output, which is not the same as an agent that stated
    # no uncertainty (""). Additive; the frontend's non-strict `z.object` strips unknown fields.
    uncertainty: str | None = None
    next_steps: list[str] = []
    # FRP2-T02 (PRD FR-CHAT-003): the answer screen shows the risks and the contradictory evidence
    # beside the answer, and before this they reached a reader only through a report -- which a
    # late-failed run cannot have. Both are read from the agents' recorded outputs and concealed
    # like the two fields above; an empty list when the step produced nothing readable. Additive.
    risks: list[RiskItemRead] = []
    contradictory_evidence: list[str] = []
    # RMD8-T01 (RE2E-DEF-031): how many citation markers naming no returned source were removed
    # from the texts above and the answer, counted from the decision's trace against the stored
    # citations (there is no column for it, DV-RMD-52). Additive; 0 for a pharmaceutical run.
    unresolved_citation_markers: int = 0
    # RMD8-T02 (RE2E-DEF-043, RE2E-DEF-044): how many evidence context lines and quoted planted
    # instructions were left out of the answer and the texts above, counted on read from the
    # decision's trace as the field above is (DV-RMD-56). Additive; 0 for a pharmaceutical run,
    # whose response states its removals in its limitations.
    hygiene_removals: int = 0
    # REQ-API-04, REQ-API-05: a pharmaceutical run's section, read from its policy-checked record;
    # `None` for a General run and for a pharmaceutical run with no checked result yet. For a
    # pharmaceutical run the four fields above come from the same record, never from the model's
    # agent rows (PH-DEF-005); its risks are the response's, whose categories are its own.
    pharma: PharmaRunRead | None = None
