"""The nine-section response: assembly, abstention and rendering (PDP7-T04; REQ-RSP-01 to 04).

``assemble_response`` builds the response from the decision agent's structured fields and the
deterministic results (anchored evidence, conflicts, sufficiency, escalation). When the evidence
is insufficient, or no decision was made, it returns ``abstention_response`` instead: every
section is written here from the deterministic results, so no model-written procedure text can
appear (REQ-SUF-02, PH-D-010). The workflow passes the result through the regulatory-boundary
check (``app.pharma.policy``) before anything is stored or shown.

``render_markdown`` writes the answer Markdown from a checked response only (REQ-RSP-02): the
human-review banner, then the nine headings in order, each present, with ``None identified.``
for an empty section. Every value is flattened onto its line and its Markdown control
characters are escaped, so model or document text cannot open a heading, a list, a link or an
HTML tag; the client's sanitizer remains the second layer.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.agents.prompts import flatten
from app.pharma.catalogue import (
    ESCALATION_RULES,
    HUMAN_REVIEW_BANNER,
    MODEL_ROLE_TEXTS,
    NONE_IDENTIFIED,
    POLICY_RULES,
    PROTOTYPE_LIMITATION,
    RESPONSE_SECTIONS,
)
from app.pharma.enums import (
    DataClassification,
    DispositionGate,
    EvidenceSufficiency,
    PharmaDisposition,
    PharmaQueryIntent,
)
from app.pharma.sufficiency import (
    REQUIREMENT_CONFLICT,
    REQUIREMENT_PREREQUISITE,
    RULE_VERSION_2,
    SUFFICIENT_COVERAGE,
    SUFFICIENT_COVERAGE_V2,
    SufficiencyAssessment,
)
from app.schemas.pharma import (
    Answerability,
    ConflictRead,
    ConflictsSection,
    EscalationReason,
    EvidenceItem,
    HumanEscalationRead,
    IncidentInput,
    PharmaDecisionOutput,
    PharmaResponse,
    RelevantProcedure,
    RiskFlag,
)

# A value on one rendered line; model text is bounded again here, after the schema's own clamp.
MAX_RENDERED_TEXT = 2_000
MAX_RESTATED_INPUT = 500

ABSTENTION_SUMMARY = (
    "No procedure answer is given: the controlled documents retrieved do not contain enough "
    "evidence for this question, so the analysis agents were not run."
)
ABSTENTION_NEXT_STEPS = (
    "Ask document control whether an Effective controlled document covers this question, and "
    "add it to the project if one does.",
    "Ask quality assurance and a subject-matter expert to answer from the site's own controlled "
    "records.",
)
ABSTENTION_LIMITATION = (
    "The assistant abstained because the evidence was classified insufficient; nothing in this "
    "response is a procedure answer."
)
MODEL_REPORTED_LABEL = "Reported by the analysis model:"
# The summary of an answer the regulatory-boundary check abstained from (PLV4-T03): a critical
# removal left no supported procedure statement.
POLICY_ABSTENTION_SUMMARY = (
    "No procedure answer is given: the regulatory-boundary check removed a statement that only "
    "qualified personnel may make, and no supported procedure statement remained."
)
# The grounding gate's wording (PLV7-T03, PLV-D-023): the abstention when no statement of the
# drafted answer is supported by the passages it cites, and the summary when only its own
# sentences were all withheld.
GROUNDING_ABSTENTION_SUMMARY = (
    "No procedure answer is given: no statement of the drafted answer is supported by the "
    "passages it cites, so the answer was withheld."
)
GROUNDING_ABSTENTION_LIMITATION = (
    "The assistant abstained because no statement of the drafted answer passed the support "
    "check; nothing in this response is a procedure answer."
)
GROUNDING_WITHHELD_SUMMARY = (
    "The drafted summary was withheld: none of its statements is supported by the passages it "
    "cites."
)
# The decision abstention (PLV9-T03, PLV-D-033, PLV-DEF-023): the decision agent read the passages
# and found that they do not state what the question asks.
DECISION_ABSTENTION_SUMMARY = (
    "No procedure answer is given: the controlled documents retrieved do not state what the "
    "question asks."
)
DECISION_ABSTENTION_LIMITATION = (
    "The assistant abstained because the analysis found that the retrieved passages do not "
    "answer this question; nothing in this response is a procedure answer."
)
# The refusal of a request to take a regulated decision (RMD3-T03): the regulatory-boundary
# detector read the question itself as asking the assistant to take one.
REQUEST_REFUSAL_SUMMARY = (
    "No procedure answer is given: the question asks the assistant to take a decision that only "
    "qualified personnel may take, such as a batch disposition, an approval, a closure or a "
    "classification."
)
REQUEST_REFUSAL_LIMITATION = (
    "The assistant refused because the question asks for a regulated decision, which belongs to "
    "the site's quality unit; nothing in this response is a procedure answer."
)
# The conflict answer (RMD3-T03, RE2E-DEF-032): the decision agent found Effective passages that
# state different values for what the question asks, each cited above.
CONFLICT_LIMITATION = (
    "The Effective documents state different values for what the question asks; document "
    "control and quality assurance must resolve which applies before either is used."
)
# The system's own words in a rendered answer: the banner, the abstention, the fixed limitation,
# the rule and policy reasons. The rendered-answer check (``policy.check_rendered``) reads every
# other line and masks these, which mention the acts they defer (PLV4-T03).
TRUSTED_TEXTS: tuple[str, ...] = (
    HUMAN_REVIEW_BANNER,
    PROTOTYPE_LIMITATION,
    NONE_IDENTIFIED,
    ABSTENTION_SUMMARY,
    *ABSTENTION_NEXT_STEPS,
    ABSTENTION_LIMITATION,
    MODEL_REPORTED_LABEL,
    POLICY_ABSTENTION_SUMMARY,
    GROUNDING_ABSTENTION_SUMMARY,
    GROUNDING_ABSTENTION_LIMITATION,
    GROUNDING_WITHHELD_SUMMARY,
    DECISION_ABSTENTION_SUMMARY,
    DECISION_ABSTENTION_LIMITATION,
    REQUEST_REFUSAL_SUMMARY,
    REQUEST_REFUSAL_LIMITATION,
    CONFLICT_LIMITATION,
    *(rule.text for rule in ESCALATION_RULES.values()),
    *MODEL_ROLE_TEXTS.values(),
    *(rule.notice for rule in POLICY_RULES.values()),
)

# A marker holds one key or a list of them (``[E5, E8]``, as the live model writes; RMD-DEF-025).
_KEY_IN_TEXT = re.compile(r"\[([A-Za-z0-9_-]{1,20}(?:\s*,\s*[A-Za-z0-9_-]{1,20})*)\]")
# CommonMark treats every ASCII punctuation character after a backslash as a literal; these are
# the ones that open structure inline (emphasis, code, links, images, HTML, entities, tables).
_INLINE_SPECIALS = re.compile(r"([\\`*_\[\]<>!|~&])")
# What would open a block at the start of a line: a heading, a list item or a rule marker, and
# an ordered-list number, whose punctuation (not its digits) takes the backslash.
_BLOCK_MARKER = re.compile(r"^([#+=-])")
_ORDERED_MARKER = re.compile(r"^(\d{1,9})([.)])")


@dataclass(frozen=True)
class ResponseInputs:
    """Everything the response is built from. Only ``decision`` holds model text; the rest is
    deterministic (anchored evidence, conflicts, risk flags already tied to evidence keys,
    the rules' escalation merged with model reasons)."""

    question: str
    intent: PharmaQueryIntent
    incident: IncidentInput | None
    assessment: SufficiencyAssessment
    escalation: HumanEscalationRead
    decision: PharmaDecisionOutput | None = None
    evidence: Sequence[EvidenceItem] = ()
    conflicts: Sequence[ConflictRead] = ()
    model_contradictions: Sequence[str] = ()
    risk_flags: Sequence[RiskFlag] = ()
    classifications: Sequence[DataClassification | None] = ()


# --- Assembly -----------------------------------------------------------------------------------


def point_texts(decision: PharmaDecisionOutput) -> list[str]:
    """The procedure points as text: version 2's structured points, each followed by the markers
    of its citation keys it does not already carry, or else version 1's ``procedure_points``."""
    if not decision.points:
        return list(decision.procedure_points)
    return [text for point in decision.points if (text := point.display_text())]


def assemble_response(inputs: ResponseInputs) -> PharmaResponse:
    """The nine sections from the decision and the deterministic results, or the abstention."""
    sufficiency = inputs.assessment.sufficiency
    if sufficiency is EvidenceSufficiency.INSUFFICIENT or inputs.decision is None:
        return abstention_response(inputs)
    decision = inputs.decision
    procedures, uncited = _procedures(inputs.evidence, point_texts(decision))
    missing = list(decision.missing_information)
    escalation = inputs.escalation.model_copy(deep=True)
    if sufficiency is EvidenceSufficiency.PARTIAL:
        # REQ-SUF-03: a partial answer always says what is missing and always escalates.
        missing = missing or missing_from(inputs.assessment)
        _require(escalation, "REQ-ESC-02")
    limitations = list(decision.uncertainty)
    if uncited:
        limitations.append(
            f"{uncited} procedure statement(s) without a citation to the evidence were left out."
        )
    if decision.cut_items:
        # RMD-DEF-023: a list cut at its limit is never shown as the whole answer.
        limitations.append(
            f"{decision.cut_items} further item(s) of the answer's lists were not shown; ask about "
            "one part of the question at a time to see them."
        )
    if decision.answerability is Answerability.CONFLICT:
        limitations.append(CONFLICT_LIMITATION)
    return PharmaResponse(
        incident_or_question_summary=decision.summary,
        relevant_procedures=procedures,
        evidence_and_citations=list(inputs.evidence),
        quality_and_operational_risks=list(inputs.risk_flags),
        conflicts_or_ambiguities=ConflictsSection(
            deterministic=list(inputs.conflicts),
            model_reported=list(inputs.model_contradictions),
        ),
        missing_information=missing,
        required_human_escalation=escalation,
        suggested_investigation_questions_or_next_steps=list(decision.investigation_questions),
        limitations_and_uncertainty=[*limitations, *_fixed_limitations(inputs)],
    )


def abstention_response(
    inputs: ResponseInputs, *, grounding: bool = False, decided: bool = False, request: bool = False
) -> PharmaResponse:
    """The response when the evidence is insufficient, (``grounding``) when no statement of the
    drafted answer is supported by its citations (PLV-D-023), or (``decided``) when the decision
    agent found that the passages do not state what the question asks (PLV-D-033), or
    (``request``) when the question asks the assistant to take a regulated decision (RMD3-T03): the
    mandatory wording, what is missing, the questions that would fill it, who must be involved and
    the next steps, all written here. No section holds model text or a document excerpt. A refused
    request is not restated: the boundary check would remove it, and its own abstention would
    replace the refusal."""
    escalation = inputs.escalation.model_copy(deep=True)
    _require(escalation, "REQ-ESC-02")
    restated = _restated(inputs)
    if request:
        summary, limitation = REQUEST_REFUSAL_SUMMARY, REQUEST_REFUSAL_LIMITATION
        restated = []
    elif decided:
        summary, limitation = DECISION_ABSTENTION_SUMMARY, DECISION_ABSTENTION_LIMITATION
    elif grounding:
        summary, limitation = GROUNDING_ABSTENTION_SUMMARY, GROUNDING_ABSTENTION_LIMITATION
    else:
        summary, limitation = ABSTENTION_SUMMARY, ABSTENTION_LIMITATION
    return PharmaResponse(
        incident_or_question_summary=" ".join([summary, *restated]),
        conflicts_or_ambiguities=ConflictsSection(deterministic=list(inputs.conflicts)),
        missing_information=missing_from(inputs.assessment),
        required_human_escalation=escalation,
        suggested_investigation_questions_or_next_steps=[
            *missing_information_questions(inputs.assessment),
            *ABSTENTION_NEXT_STEPS,
        ],
        limitations_and_uncertainty=[limitation, *_fixed_limitations(inputs)],
    )


def policy_abstention(response: PharmaResponse) -> PharmaResponse:
    """The checked response when the regulatory-boundary check removed a critical statement and
    left no supported procedure statement (PLV4-T03): the summary says no procedure answer is
    given, the procedures left without a statement are dropped, and every other checked section
    stays."""
    abstained = response.model_copy(deep=True)
    abstained.incident_or_question_summary = POLICY_ABSTENTION_SUMMARY
    abstained.relevant_procedures = [p for p in abstained.relevant_procedures if p.summary.strip()]
    return abstained


# --- Disposition (RMD3-T01, RE2E-DEF-034) --------------------------------------------------------

# The dispositions a result shows as an abstention: it gives no procedure answer.
ABSTAINING: frozenset[PharmaDisposition] = frozenset(
    {
        PharmaDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE,
        PharmaDisposition.REFUSE_PROHIBITED_DECISION,
        PharmaDisposition.OUT_OF_SCOPE,
    }
)


def disposition_of(
    sufficiency: EvidenceSufficiency,
    *,
    sufficiency_abstained: bool = False,
    decided: bool = False,
    grounding_abstained: bool = False,
    refused: bool = False,
    answerability: Answerability = Answerability.FULL,
    requested: bool = False,
) -> tuple[PharmaDisposition, DispositionGate]:
    """A result's disposition and the gate that decided it, from what each gate did: the policy
    abstention (``refused``) first, since it replaces whichever response the check read and is
    what the result shows, and the boundary's refusal of a request to take a regulated decision
    (``requested``, RMD3-T03); then the sufficiency rule's, the decision agent's and the grounding
    gate's abstentions; then the decision agent's conflict and partial readings (RMD3-T03); an
    answer is partial when the sufficiency rule classed the evidence partial, and otherwise the
    decision agent's."""
    if refused or requested:
        return PharmaDisposition.REFUSE_PROHIBITED_DECISION, DispositionGate.POLICY
    if sufficiency_abstained:
        return PharmaDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, DispositionGate.SUFFICIENCY
    if decided:
        return PharmaDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, DispositionGate.DECISION
    if grounding_abstained:
        return PharmaDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, DispositionGate.GROUNDING
    if answerability is Answerability.CONFLICT:
        return PharmaDisposition.CONFLICT_REQUIRES_REVIEW, DispositionGate.DECISION
    if answerability is Answerability.PARTIAL:
        return PharmaDisposition.PARTIAL_ANSWER, DispositionGate.DECISION
    if sufficiency is EvidenceSufficiency.PARTIAL:
        return PharmaDisposition.PARTIAL_ANSWER, DispositionGate.SUFFICIENCY
    return PharmaDisposition.ANSWER, DispositionGate.DECISION


def _restated(inputs: ResponseInputs) -> list[str]:
    restated = [f"Question asked: {flatten(inputs.question, MAX_RESTATED_INPUT)}"]
    if inputs.incident is not None:
        restated.append(
            f"Incident reported: {flatten(inputs.incident.summary, MAX_RESTATED_INPUT)}"
        )
    return restated


def _sufficient_coverage(assessment: SufficiencyAssessment) -> float:
    """The coverage the assessment's own rule version requires for Sufficient."""
    if assessment.rule_version == RULE_VERSION_2:
        return SUFFICIENT_COVERAGE_V2
    return SUFFICIENT_COVERAGE


def missing_from(assessment: SufficiencyAssessment) -> list[str]:
    """What the sufficiency rule found absent, in the rule's terms (requirements, section 15;
    rule version 2's requirements, PLV-D-024)."""
    missing = [
        f"Controlled evidence that mentions {anchor}"
        for anchor in assessment.anchors
        if anchor not in assessment.covered_anchors
    ]
    if assessment.eligible_count == 0:
        missing.append("Any eligible document: none remained after version resolution")
    elif assessment.effective_count == 0:
        missing.append("An Effective controlled procedure or reference document for this question")
    if REQUIREMENT_CONFLICT in assessment.capped_by:
        missing.append("One Effective revision of each document: two Effective revisions conflict")
    if REQUIREMENT_PREREQUISITE in assessment.capped_by:
        missing.append("A record that the procedure's prerequisites are complete")
    if assessment.best_coverage < _sufficient_coverage(assessment):
        missing.append(
            "A passage that addresses the question's terms "
            f"(best term coverage {assessment.best_coverage:.2f})"
        )
    return missing or [
        "Confirmation by a subject-matter expert that the cited passages cover the question"
    ]


def missing_information_questions(assessment: SufficiencyAssessment) -> list[str]:
    """The questions that would fill each gap the sufficiency assessment records, for a human to
    answer (PLV-D-023): one per uncovered anchor, then the document, conflict, prerequisite and
    coverage gaps; a question for the answering passage when the rule records no gap."""
    questions = [
        f"Which Effective controlled document covers {anchor}?"
        for anchor in assessment.anchors
        if anchor not in assessment.covered_anchors
    ]
    if assessment.eligible_count == 0:
        questions.append("Is there a controlled document for this question not yet in the project?")
    elif assessment.effective_count == 0:
        questions.append(
            "Is there an Effective controlled procedure or reference document for this question?"
        )
    if REQUIREMENT_CONFLICT in assessment.capped_by:
        questions.append("Which of the Effective revisions in the evidence is the one in use?")
    if REQUIREMENT_PREREQUISITE in assessment.capped_by:
        questions.append("Have the procedure's prerequisites been completed and recorded?")
    if assessment.best_coverage < _sufficient_coverage(assessment):
        questions.append("Which section of a controlled document addresses this question?")
    return questions or ["Which passage of a controlled document answers this question?"]


def with_rule(escalation: HumanEscalationRead, code: str) -> HumanEscalationRead:
    """A copy of ``escalation`` with ``code``'s rule part of it (the grounding gate's REQ-ESC-02
    when it withholds a claim, PLV-D-023)."""
    required = escalation.model_copy(deep=True)
    _require(required, code)
    return required


def _require(escalation: HumanEscalationRead, code: str) -> None:
    """Make ``code``'s rule part of the escalation if it is not already (never removes one)."""
    escalation.required = True
    if any(reason.code == code for reason in escalation.reasons):
        return
    rule = ESCALATION_RULES[code]
    escalation.roles.extend(role for role in rule.roles if role not in escalation.roles)
    escalation.reasons.append(EscalationReason(code=code, text=rule.text, source="rule"))


def cited_keys(text: str) -> list[str]:
    """Every key the markers of ``text`` name, in order, each key of a list included."""
    return [key.strip() for keys in _KEY_IN_TEXT.findall(text) for key in keys.split(",")]


def _procedures(
    evidence: Sequence[EvidenceItem], points: Sequence[str]
) -> tuple[list[RelevantProcedure], int]:
    """One entry per cited document version, in evidence order, carrying the procedure points
    that cite one of its keys. A point that cites no evidence key is left out and counted."""
    groups: dict[tuple[str, str], RelevantProcedure] = {}
    owner: dict[str, tuple[str, str]] = {}
    for item in evidence:
        anchor = item.anchor
        version = (anchor.document_number, anchor.revision)
        procedure = groups.setdefault(
            version,
            RelevantProcedure(
                document_number=anchor.document_number,
                revision=anchor.revision,
                title=anchor.title,
                lifecycle_status=anchor.lifecycle_status,
            ),
        )
        if anchor.section not in procedure.sections:
            procedure.sections.append(anchor.section)
        procedure.citation_keys.append(item.citation_key)
        owner[item.citation_key] = version
    summaries: dict[tuple[str, str], list[str]] = {version: [] for version in groups}
    uncited = 0
    for point in points:
        versions = [owner[key] for key in cited_keys(point) if key in owner]
        if not versions:
            uncited += 1
            continue
        summaries[versions[0]].append(point)
    for version, procedure in groups.items():
        procedure.summary = " ".join(_ended(point) for point in summaries[version])
    return list(groups.values()), uncited


_TRAILING_KEYS = re.compile(r"((?:\s*\[[A-Za-z0-9_-]{1,20}\])+)\s*$")


def _ended(point: str) -> str:
    """A procedure point that ends as a sentence, a full stop added before its trailing citation
    markers where it has no end punctuation, so joined points stay separate sentences for the
    policy check (PH-DEF-034)."""
    text = point.rstrip()
    keys = _TRAILING_KEYS.search(text)
    body, tail = (text[: keys.start()].rstrip(), keys.group(1)) if keys else (text, "")
    if body and body[-1] not in ".!?;":
        body = f"{body}."
    return f"{body}{tail}"


def _fixed_limitations(inputs: ResponseInputs) -> list[str]:
    """REQ-RSP-04: the prototype statement and the corpus classification, on every response."""
    assessment = inputs.assessment
    classes = sorted({c.value if c is not None else "unclassified" for c in inputs.classifications})
    corpus = (
        f"Corpus classification of the documents used: {', '.join(classes)}."
        if classes
        else "Corpus classification: no document was used."
    )
    return [
        f"Evidence sufficiency: {assessment.sufficiency.value} "
        f"(rule version {assessment.rule_version}).",
        PROTOTYPE_LIMITATION,
        corpus,
    ]


# --- Rendering ----------------------------------------------------------------------------------


def _escape(text: str) -> str:
    """One line of literal text: flattened, bounded, and every Markdown control character
    escaped, including a block marker at its start."""
    line = _INLINE_SPECIALS.sub(r"\\\1", flatten(text, MAX_RENDERED_TEXT).strip())
    line = _BLOCK_MARKER.sub(r"\\\1", line)
    return _ORDERED_MARKER.sub(r"\1\\\2", line)


def _label(value: str) -> str:
    return value.replace("_", " ")


def _bullets(items: Sequence[str]) -> list[str]:
    return [f"- {item}" for item in items] or [NONE_IDENTIFIED]


def _keys(keys: Sequence[str]) -> str:
    return f" ({', '.join(_escape(key) for key in keys)})" if keys else ""


def _summary(response: PharmaResponse) -> list[str]:
    text = _escape(response.incident_or_question_summary)
    return [text] if text else [NONE_IDENTIFIED]


def _relevant(response: PharmaResponse) -> list[str]:
    return _bullets(
        [
            f"**{_escape(p.document_number)}** revision {_escape(p.revision)} "
            f"({_label(p.lifecycle_status.value)}): {_escape(p.title)}. "
            f"Sections: {', '.join(_escape(s) for s in p.sections) or 'unknown'}."
            + (f" {_escape(p.summary)}" if p.summary.strip() else "")
            + _keys(p.citation_keys)
            for p in response.relevant_procedures
        ]
    )


def _evidence(response: PharmaResponse) -> list[str]:
    return _bullets(
        [
            f"{_escape('[' + item.citation_key + ']')} {_escape(item.anchor.document_number)} "
            f"revision {_escape(item.anchor.revision)}, section {_escape(item.anchor.section)}"
            + (f", page {item.anchor.page}" if item.anchor.page is not None else "")
            + f" ({_label(item.anchor.lifecycle_status.value)})"
            + (f": {_escape(item.excerpt)}" if item.excerpt.strip() else "")
            for item in response.evidence_and_citations
        ]
    )


def _risks(response: PharmaResponse) -> list[str]:
    return _bullets(
        [
            f"{flag.severity.value.capitalize()} {_label(flag.category.value)}: "
            f"{_escape(flag.description)}{_keys(flag.citation_keys)}"
            for flag in response.quality_and_operational_risks
        ]
    )


def _conflicts(response: PharmaResponse) -> list[str]:
    section = response.conflicts_or_ambiguities
    lines = [
        f"- {_label(conflict.kind.value).capitalize()}: {_escape(conflict.description)}"
        for conflict in section.deterministic
    ]
    reported = [_escape(text) for text in section.model_reported if text.strip()]
    if reported:
        # REQ-CON-02: labelled and never merged into the deterministic list.
        lines += ["", MODEL_REPORTED_LABEL, *(f"- {text}" for text in reported)]
    return lines or [NONE_IDENTIFIED]


def _escalation(response: PharmaResponse) -> list[str]:
    escalation = response.required_human_escalation
    roles = ", ".join(_label(role.value) for role in escalation.roles) or "none named"
    lines = [
        f"Required: {'yes' if escalation.required else 'no'}.",
        "",
        f"Roles: {roles}.",
    ]
    reasons = [
        f"{_escape(reason.code)} "
        f"({'rule' if reason.source == 'rule' else 'suggested by the analysis model'}): "
        f"{_escape(reason.text)}"
        for reason in escalation.reasons
    ]
    return [*lines, "", *_bullets(reasons)]


def _texts(items: Sequence[str]) -> list[str]:
    return _bullets([_escape(item) for item in items if item.strip()])


def render_markdown(response: PharmaResponse) -> str:
    """The answer Markdown, from the checked response only."""
    renderers: dict[str, Callable[[PharmaResponse], list[str]]] = {
        "incident_or_question_summary": _summary,
        "relevant_procedures": _relevant,
        "evidence_and_citations": _evidence,
        "quality_and_operational_risks": _risks,
        "conflicts_or_ambiguities": _conflicts,
        "missing_information": lambda r: _texts(r.missing_information),
        "required_human_escalation": _escalation,
        "suggested_investigation_questions_or_next_steps": (
            lambda r: _texts(r.suggested_investigation_questions_or_next_steps)
        ),
        "limitations_and_uncertainty": lambda r: _texts(r.limitations_and_uncertainty),
    }
    blocks = [f"> **Human review required.** {HUMAN_REVIEW_BANNER}"]
    for name, heading in RESPONSE_SECTIONS:
        blocks.append("\n".join([f"## {heading}", "", *renderers[name](response)]))
    return "\n\n".join(blocks) + "\n"
