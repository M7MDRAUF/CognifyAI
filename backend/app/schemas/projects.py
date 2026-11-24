"""Project API schemas (FR-PROJ-001/002)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.enums import DomainProfile
from app.schemas._text import is_visually_blank, reject_control_characters
from app.schemas.queries import MAX_QUESTION_LENGTH

# `QueryCreate.question` is capped at `MAX_QUESTION_LENGTH` because the question is re-sent in
# full as part of the prompt for every one of the 7-20 billed vendor completions one accepted
# run drives. `domain` and `objective` are sent on exactly those terms and were measured by
# nothing: `PlannerAgent.run` interpolates both into the *same* user prompt as the question, and
# `JsonAgentRunner.run` re-sends that prompt on the repair rung and again on the strict rung. A
# project field therefore set the input size of a call whose question the same API had already
# refused to let past 4000 characters -- measured at 11,004,247 prompt characters from a legal
# 4000-character question, 2751x that cap, on one accepted query.
#
# Derived from that cap rather than invented, the way `MAX_NORMALIZED_QUESTION_LENGTH` is, so
# the two cannot drift: raise the request cap and these follow.
#
# A *label* is one phrase occupying one line. `domain` is exactly that in the planner prompt --
# "K-12 education technology" on a `Project domain: ` line -- and `name` is the same kind of
# thing in a project list. A sixteenth of the question cap is 250 characters: longer than any
# label a UI can show without eliding, and in the same range as the 255 at which
# `DocumentService` bounds the other stored display name in this system.
MAX_PROJECT_LABEL_LENGTH = MAX_QUESTION_LENGTH // 16

# A *paragraph*. `objective` states what the analysis is for and `description` what the project
# is; both are prose, so neither can wear the label bound. Half the question cap is 2000
# characters -- roughly 500 tokens, several paragraphs -- which is far more than a stated
# objective needs and still leaves the question the dominant caller-chosen term in the prompt
# that carries both. `description` reaches no prompt today; it is bounded here because it is the
# same unbounded `Text()` column returned in every `GET /projects` response, and because a field
# that is one edit away from an interpolation should not be the one field without a ceiling.
MAX_PROJECT_PARAGRAPH_LENGTH = MAX_QUESTION_LENGTH // 2

# The invariant the two constants exist to hold, asserted in `test_project_text_bounds.py`:
# `MAX_PROJECT_LABEL_LENGTH + MAX_PROJECT_PARAGRAPH_LENGTH <= MAX_QUESTION_LENGTH`. The project
# metadata a planner prompt carries can at most double the per-call input the question already
# buys, so it can never again be the term that decides the bill.


class ProjectCreate(BaseModel):
    """``domain_profile`` (optional) is ``general`` when omitted and is fixed at creation.
    ``pharma_manufacturing`` is rejected with ``domain_profile_disabled`` (400) while the
    deployment's ``PHARMA_PROFILE_ENABLED`` is off; a General project is unaffected.
    """

    name: str = Field(max_length=MAX_PROJECT_LABEL_LENGTH)
    description: str = Field(default="", max_length=MAX_PROJECT_PARAGRAPH_LENGTH)
    domain: str = Field(default="", max_length=MAX_PROJECT_LABEL_LENGTH)
    objective: str = Field(default="", max_length=MAX_PROJECT_PARAGRAPH_LENGTH)
    # REQ-PRF-01: optional, General when omitted; fixed at creation (no route changes it).
    domain_profile: DomainProfile = DomainProfile.GENERAL

    @field_validator("name", "description", "domain", "objective")
    @classmethod
    def text_must_be_storable(cls, value: str) -> str:
        """Every stored free-text field, not just the name: a control character in
        any of them fails the same INSERT and surfaces as a 500."""
        return reject_control_characters(value, field="Project text")

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "" or is_visually_blank(stripped):
            msg = "Project name must not be empty."
            raise ValueError(msg)
        return stripped


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    domain: str
    objective: str
    domain_profile: DomainProfile = DomainProfile.GENERAL
    created_at: datetime
    updated_at: datetime

    @field_validator("domain_profile", mode="before")
    @classmethod
    def unset_profile_is_general(cls, value: object) -> object:
        """A row read before its first flush (the ORM default applies at insert) is General,
        as every row written before migration 0012 reads (PH-D-024)."""
        return DomainProfile.GENERAL if value is None else value


class ProjectStats(BaseModel):
    document_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    last_query_at: datetime | None = None
    average_answer_quality: float | None = None
    average_citation_accuracy: float | None = None
    average_latency_ms: float | None = None


class ProjectDetail(ProjectRead):
    stats: ProjectStats
