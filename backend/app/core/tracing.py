"""Versioned trace records (SDD 13.0.3, trace schema v1).

Two record types, and they answer different questions:

* ``LLMCallRecord`` — one per **provider HTTP attempt**. Not one per agent, not one per logical
  completion: the billed unit. An agent invocation may spend three repair rungs and each rung up
  to three retries, so a single agent step can emit nine of these. That factor of nine is exactly
  what a per-agent counter cannot see, and it is why the token estimate written by ``_finish``
  understated real spend.
* ``StepRecord`` — one per **component execution**, which includes the deterministic ones. The
  retriever and the orchestrator place no model call and so emit no ``LLMCallRecord``, but they
  do take time, do fail, and must appear in a trace that claims to describe the run.

**Every free-text field is redacted on construction, not on write.** A record that has been built
is already safe to serialise, log, persist or attach to a report. Redacting at the write site
would mean each of those four consumers has to remember, and the one that forgets is the leak.
``app.core.logging.redact_secrets`` is the single implementation; this module never writes its
own pattern.

**Required fields have no defaults.** A record missing ``provider`` fails validation rather than
serialising with a plausible blank, because a trace that silently records ``provider=""`` for a
billed call is worse than no trace: it is a spend record that looks complete.
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.errors import LLMFailureKind
from app.core.logging import bounded_excerpt

TRACE_SCHEMA_VERSION = 1

# The closed component set from SDD 8.3.1. Deliberately *not* `AgentName`: the whole point is
# that these five are the producers that are not agents, plus the agents themselves by name.
ComponentName = Literal[
    "orchestrator",
    "retriever",
    "reranker",
    "provider_adapter",
    "evaluation_runner",
    "worker",
    "planner",
    "researcher",
    "critic",
    "risk",
    "final_decision",
    "evaluator",
]

# What an attempt did, from the caller's point of view. `charged_failure` is the one worth
# naming: a 2xx the vendor billed whose body carried no usable answer. Collapsing it into
# `failure` is how usage from billed-but-unusable responses came to be discarded.
AttemptOutcome = Literal["success", "failure", "charged_failure"]

StepStatus = Literal["completed", "failed", "cancelled"]

_Redacted = Annotated[str, Field(min_length=0)]


class _TraceRecord(BaseModel):
    """Shared shape. Frozen, because a trace record that can be edited after the fact is not
    evidence; and extra-forbidden, so a typo in a field name fails loudly instead of vanishing
    into an ignored key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = TRACE_SCHEMA_VERSION
    correlation_id: str | None = None
    query_run_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None


class LLMCallRecord(_TraceRecord):
    """One provider HTTP attempt.

    ``attempt`` counts retries **within** one logical completion, and ``rung`` counts repair
    rungs within one agent invocation. Both are needed: ``rung=2, attempt=3`` is the ninth and
    most expensive request an agent can make, and a counter that reports only one of the two
    cannot distinguish it from the first.
    """

    record_type: Literal["llm_call"] = "llm_call"

    component: ComponentName
    provider: str
    model: str
    rung: int = Field(ge=1, le=3)
    attempt: int = Field(ge=1)
    outcome: AttemptOutcome
    failure_kind: LLMFailureKind | None = None

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    backoff_ms: int = Field(ge=0, default=0)

    # `None` means *no rate is published for this model*, and it is never 0.0. A missing rate
    # costed as zero is the one error that lets a budget be silently exceeded (ADR-004).
    cost_usd: float | None = None
    pricing_status: Literal["priced", "unpriced"] = "unpriced"

    detail: _Redacted = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @field_validator("provider", "model")
    @classmethod
    def _non_blank(cls, value: str, /) -> str:
        if not value.strip():
            msg = (
                "must name a provider and model; a billed attempt with a blank identity "
                "is not a usable spend record"
            )
            raise ValueError(msg)
        return value

    @field_validator("detail", mode="before")
    @classmethod
    def _redact(cls, value: object, /) -> object:
        return bounded_excerpt(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _failure_carries_a_kind(self) -> "LLMCallRecord":
        """A failed attempt must say *why*, and a successful one must not claim a reason.

        Without this, `outcome="failure", failure_kind=None` validates, and the retry
        classifier's own decision becomes unauditable after the fact -- the record shows a
        failure that no category explains.
        """
        if self.outcome == "success" and self.failure_kind is not None:
            msg = "a successful attempt must not carry a failure_kind"
            raise ValueError(msg)
        if self.outcome != "success" and self.failure_kind is None:
            msg = (
                "a failed or charged-failure attempt must carry the failure_kind that classified it"
            )
            raise ValueError(msg)
        return self


class StepRecord(_TraceRecord):
    """One component execution, model-calling or not.

    ``llm_calls`` is the number of ``LLMCallRecord`` entries this step produced, so a reader can
    tell a deterministic step (0) from a step whose single invocation cost nine requests.
    """

    record_type: Literal["step"] = "step"

    component: ComponentName
    status: StepStatus
    latency_ms: int = Field(ge=0)
    attempt_count: int = Field(ge=1, default=1)
    llm_calls: int = Field(ge=0, default=0)

    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)

    error_kind: str | None = None
    detail: _Redacted = ""

    @field_validator("detail", "error_kind", mode="before")
    @classmethod
    def _redact(cls, value: object, /) -> object:
        return bounded_excerpt(value) if isinstance(value, str) else value
