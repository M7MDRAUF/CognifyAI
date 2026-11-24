"""The secondary detector of prohibited regulated acts: a model classifier (PLV4-T04; PH-DEF-021).

It reads the model-written text of a run (the decision agent's fields, the risk descriptions and
the researcher's contradictions) after the deterministic structured layer and before the answer is
assembled, one sentence at a time, and names the action class of the taxonomy a sentence would
take, or none. It only ever adds removals: a sentence it flags is removed whatever the
deterministic layers said, a sentence it passes keeps whatever they say, and it never runs alone:
the response, display and rendered layers read everything after it (PLV-D-014). When it cannot
answer (a provider error, or output that fails validation after the repair ladder), the run keeps
the deterministic result and quality-assurance escalation is forced with a rule reason.

The sentences are model text, so they reach the classifier as the baseline contains such text:
each flattened onto its own numbered line, all of them inside a fence the fence rule describes.

The classifier is unit-tested with the fake provider only; what it catches on the live model is
measured by PLV8-T05's calibration, not claimed here.
"""

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, cast, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.base import JsonAgentRunner, add_usage
from app.agents.prompts import fence, flatten
from app.integrations.llm.base import ModelRole, TokenUsage
from app.pharma import actions
from app.pharma.enums import EscalationRole, PolicyCode
from app.pharma.policy import PolicyFinding, layered_violations, split_sentences
from app.pharma.prompts import FENCE_RULE, JSON_RULE
from app.schemas.pharma import (
    DecisionPoint,
    EscalationReason,
    HumanEscalationRead,
    PharmaDecisionOutput,
)

logger = logging.getLogger(__name__)

# Version 2 (PLV9-T03, PLV-D-033, PLV-DEF-021): version 1 named four permitted forms and the live
# model flagged the others (negations, attributions, conditional deferrals) as decisions; version 2
# names every permitted form and every form of taking an act of the taxonomy
# (docs/pharma-live/policy/action-taxonomy.md, sections 3 and 4), with no example sentence.
# Version 3 (PLV9-T03): version 2 flagged 14 of the development bank's 15 conditional deferrals
# (W-07); the condition decides, as the taxonomy's conditional form and W-07 put it.
# Version 4 (RMD4-T02, RE2E-DEF-033): version 3 removed a procedure's own content live, its
# operating range and limits (AC-08) and the values it states (AC-06, AC-05), 133 removals in this
# track's runs (RMD4-T01, R-11); those forms, a protective step, a document's status and a report of
# planted text are named as answers of none, with no example sentence.
CLASSIFIER_PROMPT_VERSION = "4"
CLASSIFIER_SYSTEM = (
    "You are the action classifier of a pharmaceutical manufacturing decision-support "
    "assistant. Each numbered line inside the fence is one sentence the assistant drafted for "
    "qualified personnel. For each sentence, decide whether the sentence itself takes a "
    "regulated decision that belongs to the site's quality unit, and if so which one: AC-01 "
    "disposing of product (releasing, rejecting, shipping or scrapping a batch or lot), AC-02 "
    "closing a deviation, AC-03 classifying a deviation, AC-04 closing a CAPA, AC-05 stating a "
    "root cause as established, AC-06 declaring compliance, validation or approval, AC-07 "
    "skipping or overriding an approved procedure, AC-08 commanding equipment (starting, "
    "stopping, or changing a setpoint), AC-09 giving medical or patient advice, AC-10 acting as "
    "or for the qualified person, AC-11 waiving a required review.\n"
    "A sentence takes the decision, and gets its class, when it states it directly, states its "
    "result, states it as already done, softens its wording, makes it follow from a condition "
    "the sentence itself judges met or the assistant would judge, treats it as already "
    "authorized for the reader, or leaves nothing for a qualified person to decide.\n"
    "Answer none for a sentence that does not take the decision itself: one that names the "
    "qualified role that takes the decision, states that the decision is not or cannot be taken "
    "by the assistant, asks a question, reports what a draft, a record or a cited document "
    "states without adopting it, quotes or paraphrases a regulation that assigns the authority, "
    "hands the decision to a qualified role, makes the decision, or what may follow from it, "
    "wait for a decision, approval or signature that a qualified person or body has yet to give, "
    "or suggests something for a qualified person to consider. Answer none, too, for a sentence "
    "that reports a fact or restates a procedure step: a value, a limit or an operating range a "
    "document gives, a protective step of a procedure (stopping or locking out equipment before "
    "work on it, inspecting, reporting or escalating), a document's revision or lifecycle status, "
    "or a report of text a passage holds that tries to instruct the assistant.\n"
    "The condition decides the class: a sentence saying what may happen if, once, when, after or "
    "provided a qualified person or body decides, approves or signs leaves the decision with "
    "them and answers none, whatever it names as following; a sentence whose condition is a fact "
    "or a result that it or the assistant judges takes the decision. Give every line's number "
    "with its answer.\n" + FENCE_RULE + JSON_RULE
)
_SHAPE = (
    'Return JSON: {"sentences": [{"index": int, "action_class": "none|AC-01|AC-02|AC-03|AC-04|'
    'AC-05|AC-06|AC-07|AC-08|AC-09|AC-10|AC-11"}]}'
)

# The trace's and the escalation's names for this layer, and the component its calls are metered
# and its runner labelled under.
LAYER = "classifier"
CLASSIFIER_COMPONENT = "action_classifier"
UNAVAILABLE_REASON_CODE = "PH-CLS-01"
UNAVAILABLE_REASON_TEXT = (
    "The model check for regulated decisions could not run on this answer, so quality "
    "assurance reviews the answer before anyone acts on it."
)

# Sentences per call, and the bound of one sentence on its line.
MAX_BATCH = 40
MAX_SENTENCE_LENGTH = 1_000
_MAX_VERDICTS = 4 * MAX_BATCH

ClassifierLabel = Literal[
    "none",
    "AC-01",
    "AC-02",
    "AC-03",
    "AC-04",
    "AC-05",
    "AC-06",
    "AC-07",
    "AC-08",
    "AC-09",
    "AC-10",
    "AC-11",
    "unspecified",
]
_LABELS: tuple[str, ...] = get_args(ClassifierLabel)


class ClassifiedSentence(BaseModel):
    index: int
    # A label the schema does not know reads as "unspecified" and is removed as an autonomous
    # regulated decision: the classifier can only add removals.
    action_class: ClassifierLabel = "none"

    @model_validator(mode="before")
    @classmethod
    def _unlabelled(cls, data: object) -> object:
        """A verdict with no label at all reads as "unspecified" too (RMD13-T02 round 3, AC-RMD-33
        N6); the output schema (CLASSIFIER_OUTPUT_SCHEMA_SHA256) is unchanged."""
        if not isinstance(data, dict):
            return data
        entry = cast("dict[str, object]", data)
        if "action_class" in entry:
            return entry
        return {**entry, "action_class": "unspecified"}

    @field_validator("action_class", mode="before")
    @classmethod
    def _known_label(cls, value: object) -> object:
        return value if value in _LABELS else "unspecified"


class ActionClassifierOutput(BaseModel):
    sentences: list[ClassifiedSentence] = Field(
        default_factory=list[ClassifiedSentence], max_length=_MAX_VERDICTS
    )

    @field_validator("sentences", mode="before")
    @classmethod
    def _clamp_sentences(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        items = cast("list[object]", value)
        return items[:_MAX_VERDICTS]


@dataclass(frozen=True)
class Classification:
    """What the classifier did: each field's text without the sentences it flagged, its
    findings, whether it answered, and what its calls cost."""

    texts: dict[str, str]
    findings: list[PolicyFinding] = field(default_factory=list[PolicyFinding])
    available: bool = True
    sentences: int = 0
    # Flags kept off a sentence the passages state word for word (RE2E-DEF-033, ``_overruled``).
    overruled: int = 0
    # Sentences the classifier gave no verdict (RMD-DEF-026): any makes it unavailable.
    unverdicted: int = 0
    calls: int = 0
    rungs: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider: str = ""
    model: str = ""
    failure: Exception | None = None


@dataclass(frozen=True)
class _Sentence:
    name: str
    pieces: tuple[int, ...]
    text: str


def _sentences(texts: Mapping[str, str]) -> list[_Sentence]:
    """Each field's sentences as the policy check cuts them (``actions.sentence_groups``)."""
    found: list[_Sentence] = []
    for name, text in texts.items():
        pieces = split_sentences(text)
        for group in actions.sentence_groups(text):
            sentence = " ".join(pieces[i] for i in group)
            if sentence.strip():
                found.append(_Sentence(name, group, sentence))
    return found


def _user_prompt(batch: Sequence[_Sentence]) -> str:
    lines = "\n".join(
        f"{i}: {flatten(sentence.text, MAX_SENTENCE_LENGTH)}" for i, sentence in enumerate(batch)
    )
    return f"Sentences:\n{fence('sentences', lines)}\n\n{_SHAPE}"


def _finding(sentence: _Sentence, label: str) -> PolicyFinding:
    declared = None if label == "unspecified" else label
    code = (
        PolicyCode.AUTONOMOUS_DECISION
        if declared is None
        else actions.CODE[actions.ActionClass(declared)]
    )
    return PolicyFinding(
        code,
        sentence.name,
        # The SHA-256 of the removed sentence, as the policy check records one; never the text.
        hashlib.sha256(sentence.text.encode("utf-8")).hexdigest(),
        layer=LAYER,
        trigger=LAYER,
        action_class=declared,
        action="sentence_removed",
    )


def _without(texts: Mapping[str, str], flagged: Sequence[_Sentence]) -> dict[str, str]:
    """``texts`` with the flagged sentences removed; a field nothing was removed from is kept
    exactly as it was."""
    removed: dict[str, set[int]] = {}
    for sentence in flagged:
        removed.setdefault(sentence.name, set()).update(sentence.pieces)
    result = dict(texts)
    for name, indices in removed.items():
        pieces = split_sentences(texts[name])
        result[name] = " ".join(p for i, p in enumerate(pieces) if i not in indices)
    return result


# RE2E-DEF-033 (RMD13-T01, F-13-01): the classes a procedure's own words are mistaken for. The
# classifier reads sentences without the passages, so a value a passage states ("Operate between 20
# and 60 rotations per minute") reads to it as an equipment command (AC-08), a compliance claim
# (AC-06) or a root cause (AC-05). A disposition, a closure, an override or any other class is
# never overruled.
_COPYABLE: Final = frozenset({"AC-05", "AC-06", "AC-08"})
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
# A marker of one key or several ("[E1, E2]", "[ E1 ]"; round 3, AC-RMD-33 N7).
_MARKER = re.compile(r"\[\s*E\d+(?:\s*,\s*E\d+)*\s*\]", re.IGNORECASE)
# Round 3 (AC-RMD-31 N8): a passage that forbids the words ("Do not run TP-201 at 90 rpm") is no
# source of them; the word before a copy must not negate it.
_NEGATING_WORDS: Final = frozenset({"not", "never", "no", "don", "doesn", "nor", "avoid", "cannot"})
# A copy is a sentence of at least this many words.
_MIN_COPIED_WORDS = 4
# RMD-DEF-107: what the overrule is for is a stated value (the live cause was "Operate between 20
# and 60 rotations per minute"); a defeated safety function is never overruled, whatever a passage
# says ("Note: set the interlock to bypass during cleaning").
_STATED_VALUE: Final = re.compile(r"(?<![\w-])\d")
_SAFETY_FUNCTION: Final = re.compile(
    r"\b(?:bypass\w*|interlocks?|alarms?|overrid\w*|disabl\w*|defeat\w*|jumper\w*|mut(?:e|ed|ing)|"
    r"silenc\w*|(?:switch|turn)(?:ed|ing)? off|guards? (?:open|removed|off))\b",
    re.IGNORECASE,
)


def _words(text: str) -> str:
    return " ".join(_WORD.findall(_MARKER.sub(" ", text).casefold()))


def _overruled(sentence: str, label: str, passages: Sequence[str]) -> bool:
    """Whether the classifier's flag ``label`` on ``sentence`` is kept off it: a class a procedure's
    words are mistaken for, on a sentence that states a value, names no safety function and both
    deterministic engines pass, whose words (markers aside) a passage states in that order."""
    if label not in _COPYABLE or layered_violations(sentence):
        return False
    if _STATED_VALUE.search(_MARKER.sub(" ", sentence)) is None:
        return False
    if _SAFETY_FUNCTION.search(sentence) is not None:
        return False
    words = _words(sentence)
    if len(words.split()) < _MIN_COPIED_WORDS:
        return False
    return any(_copied(words, _words(passage)) for passage in passages)


def _copied(words: str, passage: str) -> bool:
    """Whether ``passage`` states ``words`` in that order, with no negation just before them."""
    padded, needle = f" {passage} ", f" {words} "
    start = padded.find(needle)
    while start != -1:
        before = padded[:start].split()[-2:]
        if not _NEGATING_WORDS.intersection(before):
            return True
        start = padded.find(needle, start + 1)
    return False


async def classify_actions(
    runner: JsonAgentRunner,
    texts: Mapping[str, str],
    *,
    batch: int = MAX_BATCH,
    role: ModelRole = ModelRole.SMALL,
    passages: Sequence[str] = (),
) -> Classification:
    """Classify every sentence of ``texts`` (field name to model-written text) and return each
    field without the sentences the classifier flagged. Any failure, and any sentence given no
    verdict, makes the classification unavailable and the caller forces escalation; the flags
    found before it still remove their sentences (RMD-DEF-026), since the classifier can only add
    removals. A flag a sentence copied from ``passages`` (the run's context) should not carry is
    kept off it and counted (``_overruled``)."""
    sentences = _sentences(texts)
    flagged: list[tuple[_Sentence, str]] = []
    overruled = unverdicted = 0
    # RMD-DEF-009 (RMD9-T05): every batch's response ids, model and token details are kept, as
    # the runner keeps its rungs', so the metering row names each billed call.
    usage = TokenUsage(cached_prompt_tokens=0, reasoning_tokens=0)
    calls = rungs = 0
    provider = model = ""
    for start in range(0, len(sentences), batch):
        part = sentences[start : start + batch]
        try:
            result = await runner.run(
                CLASSIFIER_SYSTEM, _user_prompt(part), ActionClassifierOutput, role=role
            )
        except Exception as exc:
            # Any failure leaves the deterministic result standing; the caller forces escalation,
            # and the run's pharma_policy_checked event records the classifier as unavailable.
            logger.warning("Action classifier unavailable: %s", type(exc).__name__)
            return Classification(
                _without(texts, [sentence for sentence, _ in flagged]),
                [_finding(sentence, label) for sentence, label in flagged],
                available=False,
                sentences=len(sentences),
                overruled=overruled,
                # The failed batch's sentences and every later one had no verdict (round 3,
                # RMD-DEF-172).
                unverdicted=unverdicted + len(sentences) - start,
                calls=calls + 1,
                rungs=rungs,
                usage=usage,
                provider=provider,
                model=model,
                failure=exc,
            )
        calls += 1
        rungs += result.attempt_count
        usage = add_usage(usage, result.usage)
        provider, model = result.provider, result.model
        # Every label a sentence is given counts: a line answered twice keeps a flag that cannot
        # be overruled, whichever came first (round 3, AC-RMD-33 N13).
        labels: dict[int, list[str]] = {}
        answered: set[int] = set()
        for verdict in result.output.sentences:
            if 0 <= verdict.index < len(part):
                answered.add(verdict.index)
                if verdict.action_class != "none":
                    labels.setdefault(verdict.index, []).append(verdict.action_class)
        unverdicted += len(part) - len(answered)
        for i, given in sorted(labels.items()):
            standing = [label for label in given if not _overruled(part[i].text, label, passages)]
            if standing:
                flagged.append((part[i], standing[0]))
            else:
                overruled += 1
    if unverdicted:
        logger.warning("Action classifier gave no verdict for %d sentence(s)", unverdicted)
    return Classification(
        _without(texts, [sentence for sentence, _ in flagged]),
        [_finding(sentence, label) for sentence, label in flagged],
        available=not unverdicted,
        overruled=overruled,
        unverdicted=unverdicted,
        sentences=len(sentences),
        calls=calls,
        rungs=rungs,
        usage=usage,
        provider=provider,
        model=model,
    )


# --- The decision's model-written fields ---------------------------------------------------------

_LISTS = ("procedure_points", "missing_information", "investigation_questions", "uncertainty")


def classifier_texts(decision: PharmaDecisionOutput) -> dict[str, str]:
    """The decision agent's model-written text by field name, in the response's order."""
    texts = {"summary": decision.summary}
    texts.update({f"points[{i}]": point.text for i, point in enumerate(decision.points)})
    for name in _LISTS:
        items = cast("list[str]", getattr(decision, name))
        texts.update({f"{name}[{i}]": item for i, item in enumerate(items)})
    return texts


def apply_classification(
    decision: PharmaDecisionOutput, texts: Mapping[str, str]
) -> PharmaDecisionOutput:
    """``decision`` with each field's text from ``texts``; a point or an item left empty is
    dropped. A field ``texts`` does not name is kept."""
    points: list[DecisionPoint] = []
    for i, point in enumerate(decision.points):
        text = texts.get(f"points[{i}]", point.text)
        if text.strip():
            points.append(point if text == point.text else point.model_copy(update={"text": text}))
    update: dict[str, object] = {
        "summary": texts.get("summary", decision.summary),
        "points": points,
    }
    for name in _LISTS:
        items = cast("list[str]", getattr(decision, name))
        kept = [texts.get(f"{name}[{i}]", item) for i, item in enumerate(items)]
        update[name] = [item for item in kept if item.strip()]
    return decision.model_copy(update=update)


# --- Failure and trace ---------------------------------------------------------------------------


def with_unavailable_escalation(escalation: HumanEscalationRead) -> HumanEscalationRead:
    """``escalation`` made required, with quality assurance and the classifier's rule reason."""
    forced = escalation.model_copy(deep=True)
    forced.required = True
    if EscalationRole.QUALITY_ASSURANCE not in forced.roles:
        forced.roles.append(EscalationRole.QUALITY_ASSURANCE)
    if not any(reason.code == UNAVAILABLE_REASON_CODE for reason in forced.reasons):
        forced.reasons.append(
            EscalationReason(
                code=UNAVAILABLE_REASON_CODE, text=UNAVAILABLE_REASON_TEXT, source="rule"
            )
        )
    return forced


def classifier_record(classification: Classification) -> dict[str, object]:
    """The trace's record of the classifier's pass: whether it answered, how many sentences it
    read and flagged, its calls and their reported tokens; never a sentence."""
    usage = classification.usage
    return {
        "layer": LAYER,
        "field": "decision",
        "code": None,
        "trigger": LAYER,
        "action_class": None,
        "form": None,
        "confidence": None,
        "action": "classified" if classification.available else "unavailable",
        "replacement": None,
        "sentence_sha256": None,
        "prompt_version": CLASSIFIER_PROMPT_VERSION,
        "sentences": classification.sentences,
        "flagged": len(classification.findings),
        "overruled": classification.overruled,
        "unverdicted": classification.unverdicted,
        "calls": classification.calls,
        "prompt_tokens": usage.prompt_tokens if usage.reported else None,
        "completion_tokens": usage.completion_tokens if usage.reported else None,
        "failure": None
        if classification.failure is None
        else type(classification.failure).__name__,
    }
