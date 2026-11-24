"""Agent input/output schemas (SDD section 8, FR-AGENT-001..007).

Every agent returns JSON matching one of these models. Fields use explicit
nested models — never free-form dicts — so validation is meaningful and the
whole layer stays free of ``Any``.
"""

import logging
import uuid
from typing import cast

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app.domain.enums import RiskCategory, RiskLevel
from app.schemas.common import Score
from app.schemas.queries import MAX_QUESTION_LENGTH

logger = logging.getLogger(__name__)

# Ceiling on the model-produced question that every agent after the planner is handed.
#
# `QueryCreate.question` caps the *request*, on the stated grounds that the question is re-sent
# on each of the 7-20 billed completions a run drives. That is a proxy for the invariant rather
# than the invariant: `Orchestrator._run_workflow` sets
# `query_run.normalized_question = plan.normalized_question or query_run.question` and re-sends
# *that* to the researcher, critic, risk, final-decision and evaluator agents and to the
# reranker. It is model output, so the request cap reaches it only when the planner returns an
# empty string.
#
# Derived from the request cap rather than invented, so the two cannot drift. The factor of two
# is headroom for a normalization that legitimately grows -- resolving a pronoun, folding in the
# project's domain and objective -- while still placing the ceiling far outside anything a
# well-behaved planner produces: it would take a planner doubling the longest question the API
# will even accept to reach it.
MAX_NORMALIZED_QUESTION_LENGTH = 2 * MAX_QUESTION_LENGTH

# Ceiling on the model-produced lists whose length decides how much work an accepted run does.
MAX_AGENT_LIST_ITEMS = 8

# The report body. Derived from the request cap like its siblings rather than invented: sixteen
# times the longest question the API will accept is far outside any decision report a
# well-behaved model writes, while still bounding the one agent output field that had no ceiling
# at all. It is the field the report renderer scans line by line, and an unbounded value there
# is an unbounded amount of scanning on the request thread -- which is how a quadratic in that
# scanner became a 90-second stall reachable from an anonymous endpoint. The quadratic is fixed;
# this is the bound that keeps the next one from mattering as much.
MAX_ANSWER_MARKDOWN_LENGTH = 16 * MAX_QUESTION_LENGTH


def _clamp_text(value: object, limit: int, field: str) -> object:
    """Truncate an over-long model string rather than rejecting it.

    Rejecting is the expensive option, which is not obvious until you follow it through
    `JsonAgentRunner.run`: a schema violation triggers a repair call whose prompt *embeds the
    offending payload verbatim*, then a stricter retry, and then
    `AgentOutputValidationError`, which fails the entire run. A ceiling that exists to bound
    spend would therefore spend two further calls -- one of them larger than the call that
    overran -- in order to produce nothing for the user. Clamping bounds the spend on the first
    response and lets the run finish.

    Non-string input is passed through untouched so that ordinary type validation, and the repair
    ladder behind it, still sees it.
    """
    if not isinstance(value, str) or len(value) <= limit:
        return value
    logger.warning(
        "Clamped agent output field %r from %d to %d characters", field, len(value), limit
    )
    return value[:limit]


def _clamp_items(value: object, limit: int, field: str) -> object:
    """The list counterpart of ``_clamp_text``, for the same reason."""
    if not isinstance(value, list):
        return value
    items = cast("list[object]", value)
    if len(items) <= limit:
        return items
    logger.warning("Clamped agent output field %r from %d to %d items", field, len(items), limit)
    return items[:limit]


# --- Planner (FR-AGENT-001) ---
class RetrievalPlan(BaseModel):
    # May be empty; the planner agent backfills the normalized question as a
    # fallback query so retrieval always has at least one query to run.
    #
    # Bounded like `PlannerOutput.sub_questions` below, and for a reason the sibling field does
    # not have: `RetrieverAgent.retrieve` iterates every entry, and each one costs a billed
    # embedding call plus a Qdrant search plus a hydration SELECT. Unbounded, the retrieval
    # fan-out of an accepted run is decided by model output rather than by anything in source.
    #
    # The `max_length` stays alongside the clamp below and is not redundant with it:
    # `JsonAgentRunner.run` passes `model_json_schema()` to `generate_json`, so this is the
    # `maxItems` by which a structured-output provider is *asked* to stay inside the limit. The
    # clamp is what happens when it does not.
    queries: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    semantic_top_k: int = 30
    lexical_top_k: int = 30

    @field_validator("queries", mode="before")
    @classmethod
    def _clamp_queries(cls, value: object) -> object:
        return _clamp_items(value, MAX_AGENT_LIST_ITEMS, "retrieval_plan.queries")


class PlannerOutput(BaseModel):
    normalized_question: str = Field(max_length=MAX_NORMALIZED_QUESTION_LENGTH)
    sub_questions: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)
    retrieval_plan: RetrievalPlan
    required_evidence: list[str] = Field(default_factory=list[str])
    # Bounded for the same reason as `normalized_question`, and the sibling `6a5c794` missed:
    # `sub_questions` was capped and is never re-sent, while `risk_focus` was left uncapped and
    # is -- `Orchestrator._run_workflow` hands it to `RiskAgent.run`, which joins it into the
    # risk prompt.
    risk_focus: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)

    @field_validator("normalized_question", mode="before")
    @classmethod
    def _clamp_normalized_question(cls, value: object) -> object:
        return _clamp_text(value, MAX_NORMALIZED_QUESTION_LENGTH, "normalized_question")

    @field_validator("sub_questions", "risk_focus", mode="before")
    @classmethod
    def _clamp_planner_lists(cls, value: object) -> object:
        return _clamp_items(value, MAX_AGENT_LIST_ITEMS, "planner list")


# --- Researcher (FR-AGENT-003) ---
# Every field below is **re-sent**: a finding written by the researcher is carried into the
# critic's prompt, the risk agent's, the final decision's and the evaluator's. So an unbounded
# field is not one over-long value, it is one over-long value paid for four more times -- and
# each of those calls may itself spend three repair rungs and three retries (N-008).
#
# The limits are generous relative to any real statement and tight relative to a document: a
# finding is a sentence, not a chapter.
MAX_FINDING_STATEMENT_LENGTH = 2_000
MAX_THEME_LENGTH = 200
MAX_CONTRADICTION_LENGTH = 2_000
MAX_CHUNK_REFERENCES = 16


class Finding(BaseModel):
    statement: str = Field(max_length=MAX_FINDING_STATEMENT_LENGTH)
    theme: str = Field(default="", max_length=MAX_THEME_LENGTH)
    supporting_chunk_ids: list[uuid.UUID] = Field(
        default_factory=list[uuid.UUID], max_length=MAX_CHUNK_REFERENCES
    )

    @field_validator("statement", "theme", mode="before")
    @classmethod
    def _clamp(cls, value: object, info: ValidationInfo) -> object:
        # Clamped, not rejected, for the reason `docs/api.md` gives: a rejection spends a repair
        # call whose prompt embeds the offending payload, then a stricter retry, then fails the
        # run -- two further billed calls, one larger than the call that overran, to produce
        # nothing.
        limit = MAX_FINDING_STATEMENT_LENGTH if info.field_name == "statement" else MAX_THEME_LENGTH
        return _clamp_text(value, limit, f"Finding.{info.field_name}")

    @field_validator("supporting_chunk_ids", mode="before")
    @classmethod
    def _clamp_refs(cls, value: object) -> object:
        return _clamp_items(value, MAX_CHUNK_REFERENCES, "Finding.supporting_chunk_ids")


class Contradiction(BaseModel):
    description: str = Field(max_length=MAX_CONTRADICTION_LENGTH)
    chunk_ids: list[uuid.UUID] = Field(
        default_factory=list[uuid.UUID], max_length=MAX_CHUNK_REFERENCES
    )

    @field_validator("description", mode="before")
    @classmethod
    def _clamp(cls, value: object) -> object:
        return _clamp_text(value, MAX_CONTRADICTION_LENGTH, "Contradiction.description")

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def _clamp_refs(cls, value: object) -> object:
        return _clamp_items(value, MAX_CHUNK_REFERENCES, "Contradiction.chunk_ids")


class ResearchOutput(BaseModel):
    key_findings: list[Finding] = Field(default_factory=list["Finding"])
    themes: list[str] = Field(default_factory=list[str])
    contradictions: list[Contradiction] = Field(default_factory=list["Contradiction"])


# --- Critic (FR-AGENT-004) ---
class ClaimCheck(BaseModel):
    claim: str
    chunk_ids: list[uuid.UUID] = Field(default_factory=list[uuid.UUID])


class CriticOutput(BaseModel):
    supported_claims: list[ClaimCheck] = Field(default_factory=list["ClaimCheck"])
    unsupported_claims: list[str] = Field(default_factory=list[str])
    hallucination_risk: Score = 0.0
    needs_more_retrieval: bool = False
    # Bounded for the same reason as `RetrievalPlan.queries`: these drive the second retrieval
    # round (`Orchestrator._expand`) through the same per-query embedding call.
    expansion_queries: list[str] = Field(default_factory=list[str], max_length=MAX_AGENT_LIST_ITEMS)

    @field_validator("expansion_queries", mode="before")
    @classmethod
    def _clamp_expansion_queries(cls, value: object) -> object:
        return _clamp_items(value, MAX_AGENT_LIST_ITEMS, "expansion_queries")


# --- Risk (FR-AGENT-005) ---
class RiskItem(BaseModel):
    category: RiskCategory
    description: str
    severity: RiskLevel
    likelihood: RiskLevel
    mitigation: str = ""


class RiskOutput(BaseModel):
    risks: list[RiskItem] = Field(default_factory=list["RiskItem"])


# --- Final Decision (FR-AGENT-007) ---
class CitationDraft(BaseModel):
    citation_key: str
    chunk_id: uuid.UUID
    snippet: str = ""


class FinalDecisionOutput(BaseModel):
    recommendation: str
    answer_markdown: str

    @field_validator("answer_markdown", mode="before")
    @classmethod
    def _clamp_answer_markdown(cls, value: object) -> object:
        """The one free-text agent output field that had no ceiling; its five siblings do."""
        return _clamp_text(value, MAX_ANSWER_MARKDOWN_LENGTH, "answer_markdown")

    citations: list[CitationDraft] = Field(default_factory=list["CitationDraft"])
    contradictory_evidence: list[str] = Field(default_factory=list[str])
    uncertainty: str = ""
    next_steps: list[str] = Field(default_factory=list[str])


# --- Evaluator (FR-AGENT-006) ---
class EvaluatorOutput(BaseModel):
    answer_relevance: Score
    faithfulness: Score
    citation_accuracy: Score
    context_sufficiency: Score
    hallucination_risk: Score
    explanation: str = ""
