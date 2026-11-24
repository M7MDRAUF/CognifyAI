"""Named routing profiles (C-16, SDD 10.3, ADR-005).

``answer_model`` is a tier -- ``fast`` or ``powerful`` -- and ``_synthesis_role`` mapped it onto
one ``ModelRole`` shared by the critic, the risk agent, the final-decision agent and the
evaluator. So ``answer_model=fast`` quietly put **all four** on the small model, while
``SDD.md`` §10.3, the section the code's own comment cited, placed all four on the stronger one.
The request field said "fast" and what it actually changed was which model reviewed the evidence
and wrote the answer.

A tier is also the wrong shape for an auditable system: it says nothing about *which* of the
seven call sites it affects, so "this run used the fast model" was not answerable from the
request.

A profile maps **every** call site to exactly one tier. There is no unspecified site that falls
back to whatever, which is what makes the model that will answer each step decidable before the
run starts.

**Unknown and unavailable profiles are refused, never substituted.** A silent fallback bills a
vendor the caller did not choose and writes a run whose recorded provenance contradicts its
request -- the same defect this module exists to fix, pointed the other way.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.domain.enums import AgentName, AnswerModel
from app.integrations.llm.base import ModelRole


class CallSite(StrEnum):
    """The seven places a model is actually called.

    Six agents plus the reranking service. The retriever and the orchestrator are deterministic
    and place no call of their own (ADR-001), so neither appears here -- a profile cannot route
    a call that does not exist.
    """

    PLANNER = "planner"
    RERANKER = "reranker"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    RISK = "risk"
    FINAL_DECISION = "final_decision"
    EVALUATOR = "evaluator"


# The agent enum has seven members and one of them -- RETRIEVER -- is not a call site. Mapping
# the six that are, explicitly, so a reader can see which one is missing and why.
AGENT_CALL_SITES: Final[Mapping[AgentName, CallSite]] = {
    AgentName.PLANNER: CallSite.PLANNER,
    AgentName.RESEARCHER: CallSite.RESEARCHER,
    AgentName.CRITIC: CallSite.CRITIC,
    AgentName.RISK: CallSite.RISK,
    AgentName.FINAL_DECISION: CallSite.FINAL_DECISION,
    AgentName.EVALUATOR: CallSite.EVALUATOR,
}


@dataclass(frozen=True)
class RoutingProfile:
    """One tier per call site. Every site, every time."""

    name: str
    sites: Mapping[CallSite, ModelRole]

    def role_for(self, site: CallSite) -> ModelRole:
        """The tier for ``site``.

        Raises rather than defaulting: a profile missing a site is a construction bug, and
        defaulting would reintroduce the "unspecified site falls back to whatever" behaviour
        this whole design exists to remove. ``validate`` below makes it unreachable in practice.
        """
        role = self.sites.get(site)
        if role is None:  # pragma: no cover - `validate` forbids it at import time
            msg = f"routing profile {self.name!r} does not map call site {site.value!r}"
            raise KeyError(msg)
        return role

    @property
    def tier(self) -> AnswerModel:
        """The ``answer_model`` tier this profile stands for (SDD 10.3.4, F-011, F-012).

        ``fast`` only when every call site is on the small model, as in ``flash-all``;
        ``powerful`` otherwise. It is what a stored row's ``answer_model`` says about a run that
        named this profile, and what an ``answer_model`` sent beside the profile must agree with.
        """
        if all(role is ModelRole.SMALL for role in self.sites.values()):
            return AnswerModel.FAST
        return AnswerModel.POWERFUL


def _profile(name: str, **sites: ModelRole) -> RoutingProfile:
    return RoutingProfile(name=name, sites={CallSite(key): role for key, role in sites.items()})


# SDD 10.3.2, normative. The planner and the reranker stay on the small model in every profile:
# the planner normalises a question and the reranker emits a score map, neither improves with a
# stronger model, and both run on every query.
FLASH_ALL: Final = _profile(
    "flash-all",
    planner=ModelRole.SMALL,
    reranker=ModelRole.SMALL,
    researcher=ModelRole.SMALL,
    critic=ModelRole.SMALL,
    risk=ModelRole.SMALL,
    final_decision=ModelRole.SMALL,
    evaluator=ModelRole.SMALL,
)

BALANCED: Final = _profile(
    "balanced",
    planner=ModelRole.SMALL,
    reranker=ModelRole.SMALL,
    researcher=ModelRole.STRONG,
    critic=ModelRole.STRONG,
    risk=ModelRole.STRONG,
    final_decision=ModelRole.STRONG,
    evaluator=ModelRole.SMALL,
)

# Differs from `balanced` only in the evaluator -- the one site where the cheap choice is also
# the least trustworthy. See ADR-003: the in-product evaluator is not a quality gate under any
# profile, and this does not make it one.
POWERFUL_SYNTHESIS: Final = _profile(
    "powerful-synthesis",
    planner=ModelRole.SMALL,
    reranker=ModelRole.SMALL,
    researcher=ModelRole.STRONG,
    critic=ModelRole.STRONG,
    risk=ModelRole.STRONG,
    final_decision=ModelRole.STRONG,
    evaluator=ModelRole.STRONG,
)

PROFILES: Final[Mapping[str, RoutingProfile]] = {
    profile.name: profile for profile in (FLASH_ALL, BALANCED, POWERFUL_SYNTHESIS)
}

DEFAULT_PROFILE_NAME: Final = BALANCED.name

# The deprecated alias, kept because removing an accepted request field is a breaking change and
# there is no reason to spend one here (SDD 10.3.4). `fast` and `powerful` are the only two tiers
# `answer_model` ever had, so the mapping is total.
ANSWER_MODEL_ALIASES: Final[Mapping[AnswerModel, str]] = {
    AnswerModel.FAST: FLASH_ALL.name,
    AnswerModel.POWERFUL: BALANCED.name,
}


def validate() -> None:
    """Every profile maps every call site. Called at import, so a gap is a startup failure.

    A profile with a missing site would resolve that site through ``role_for``'s ``KeyError`` at
    the worst possible moment -- mid-run, after earlier steps were already billed.
    """
    for profile in PROFILES.values():
        missing = set(CallSite) - set(profile.sites)
        if missing:
            names = ", ".join(sorted(site.value for site in missing))
            msg = f"routing profile {profile.name!r} does not map call site(s): {names}"
            raise ValueError(msg)


validate()


def resolve(name: str | None, answer_model: AnswerModel | None = None) -> RoutingProfile:
    """The profile for a request, or a ``KeyError`` naming what was asked for.

    Resolution order, from SDD 10.3.3:

    1. a named profile wins;
    2. otherwise the deprecated ``answer_model`` alias, if one was given;
    3. otherwise ``balanced``.

    An unknown name raises. It is **not** coerced to the default: a caller who asked for
    ``powerful-synthesis`` and silently got ``balanced`` would be handed an answer written by a
    model they did not choose, and a run whose provenance contradicts its own request.
    """
    if name is not None:
        profile = PROFILES.get(name)
        if profile is None:
            known = ", ".join(sorted(PROFILES))
            msg = f"unknown routing profile {name!r}; known profiles are: {known}"
            raise KeyError(msg)
        return profile
    if answer_model is not None:
        return PROFILES[ANSWER_MODEL_ALIASES[answer_model]]
    return PROFILES[DEFAULT_PROFILE_NAME]
