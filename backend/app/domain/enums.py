"""Domain enumerations shared across layers."""

from enum import StrEnum


class DomainProfile(StrEnum):
    """A project's domain profile, fixed when the project is created.

    ``GENERAL`` is the default and keeps every baseline behaviour. ``PHARMA_MANUFACTURING`` opts
    the project into controlled-document metadata, version-aware retrieval and the
    pharmaceutical workflow (``app.pharma``); nothing in the General path reads it.
    """

    GENERAL = "general"
    PHARMA_MANUFACTURING = "pharma_manufacturing"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"


class ResponseStyle(StrEnum):
    CONCISE = "concise"
    DETAILED = "detailed"


class AnswerModel(StrEnum):
    """Which configured text model synthesises the answer (SDD 10.3).

    ``FAST`` routes the synthesis agents to ``LLM_TEXT_MODEL_SMALL`` and
    ``POWERFUL`` to ``LLM_TEXT_MODEL_STRONG``. Named ``answer_model`` rather
    than ``model_*`` to stay clear of Pydantic's protected ``model_`` namespace.
    """

    FAST = "fast"
    POWERFUL = "powerful"


class QueryRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentName(StrEnum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    RISK = "risk"
    EVALUATOR = "evaluator"
    FINAL_DECISION = "final_decision"


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    PLANNING_STARTED = "planning_started"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RERANKING_STARTED = "reranking_started"
    RESEARCH_STARTED = "research_started"
    CRITIC_STARTED = "critic_started"
    RISK_STARTED = "risk_started"
    ANSWER_STARTED = "answer_started"
    EVALUATION_STARTED = "evaluation_started"
    ANSWER_COMPLETED = "answer_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"


class RiskCategory(StrEnum):
    IMPLEMENTATION = "implementation"
    ETHICAL = "ethical"
    PRIVACY = "privacy"
    DATA_QUALITY = "data_quality"
    OPERATIONAL = "operational"
    ACADEMIC = "academic"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionState(StrEnum):
    """How far a retrieved chunk got (M-8, SDD 8.5).

    Ordered: each value implies the ones before it. `selected_for_context` was a boolean written
    `True` for every row, so a candidate the reranker cut and a chunk the agents actually read
    were indistinguishable — and "what evidence did this answer see" was unanswerable from the
    persisted record.

    `SELECTED` and `SUPPLIED` differ deliberately. A run cancelled between the top-K cut and the
    next agent call has rows that were selected and never supplied; collapsing them would claim
    an agent saw evidence it never received.

    The state describes the run's **final** context. An expansion reranks the whole union and cuts
    it once (ADR-006), so a chunk the second cut drops returns to `RERANKED` -- or `CANDIDATE`, if
    the union rerank returned no score for it -- even though round one supplied it to the
    researcher. Which round first found a chunk is `round_index`, not the state.
    """

    CANDIDATE = "candidate"
    RERANKED = "reranked"
    SELECTED = "selected"
    SUPPLIED = "supplied"


class EvidenceSource(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    BOTH = "both"
