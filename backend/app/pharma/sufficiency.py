"""Evidence sufficiency, pre-registered rule version 1 (REQ-SUF-01 to REQ-SUF-04).

``PHARMA_DOMAIN_REQUIREMENTS.md`` section 15, implemented clause by clause and never changed
silently: a change needs a recorded calibration on the development split.

1. Eligible evidence is the final context after version resolution (the caller passes it).
2. Anchors come from the question and the incident: equipment identifiers, document numbers,
   and the incident's equipment, area and process fields. An anchor is covered when an
   eligible chunk's text or its document's metadata contains it, ignoring case.
3. A chunk's term coverage is the share of the question's content terms that occur in it.
4. Sufficient: an eligible chunk from an Effective controlled or reference document, every
   anchor covered, and the best eligible chunk's term coverage at least 0.50.
5. Partial: not Sufficient, and an anchor covered or best coverage at least 0.25; or the only
   eligible evidence is Unknown-status.
6. Insufficient otherwise, including when no eligible evidence remains.
"""

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import ControlStatus, EvidenceSufficiency, LifecycleStatus
from app.pharma.ranking import PROCEDURE_TYPES
from app.schemas.pharma import IncidentInput

RULE_VERSION = "1"
SUFFICIENT_COVERAGE = 0.50
PARTIAL_COVERAGE = 0.25

_DOCUMENT_NUMBER = re.compile(r"\b[A-Z]{2,5}(?:-[A-Z0-9]{1,6}){1,5}\b")
_EQUIPMENT = re.compile(r"\b[A-Z]{1,5}-?[0-9]{1,5}[A-Z]?\b")
_TOKEN = re.compile(r"[a-z0-9]+")
# Fixed with the rule (version 1): function words and question words that carry no topic.
STOP_WORDS = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her",
        "was", "one", "our", "out", "has", "have", "his", "how", "its", "may", "who", "what",
        "when", "where", "which", "why", "with", "this", "that", "these", "those", "from",
        "into", "onto", "upon", "about", "after", "before", "does", "did", "should",
        "would", "could", "must", "shall", "will", "there", "their", "them", "they", "then",
        "than", "been", "being", "were", "also", "only", "such", "some", "each", "per", "use",
        "used", "using", "under", "over", "between", "during", "while", "within", "without",
        "your", "yours", "please", "tell", "give", "need", "needed",
    }
)  # fmt: skip
_CONTROLLED_OR_REFERENCE = frozenset({ControlStatus.CONTROLLED, ControlStatus.REFERENCE})


class EvidenceText(Protocol):
    @property
    def document_id(self) -> uuid.UUID: ...

    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class SufficiencyAssessment:
    """The classification and every input it was made from (REQ-SUF-04)."""

    sufficiency: EvidenceSufficiency
    rule_version: str
    anchors: tuple[str, ...]
    covered_anchors: tuple[str, ...]
    question_terms: tuple[str, ...]
    best_coverage: float
    eligible_count: int
    effective_count: int
    unknown_only: bool
    # Rule version 2 (PLV-D-022): the structured requirements the context does not meet, and the
    # anchors covered and best term coverage over all eligible evidence, authoritative or not (its
    # Insufficient test reads them; ``covered_anchors`` and ``best_coverage`` are authoritative).
    capped_by: tuple[str, ...] = ()
    covered_any: tuple[str, ...] = ()
    best_any_coverage: float = 0.0


def content_terms(text: str) -> set[str]:
    """Clause 3: lower-cased alphanumeric tokens of three or more characters, minus the stop
    words, with a trailing ``s`` removed."""
    terms: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        if len(token) < 3 or token in STOP_WORDS:
            continue
        terms.add(token.removesuffix("s"))
    return terms


def term_coverage(question_terms: set[str], text: str) -> float:
    """Clause 3: the share of the question's content terms that occur in ``text``."""
    if not question_terms:
        return 0.0
    return len(question_terms & content_terms(text)) / len(question_terms)


def extract_anchors(question: str, incident: IncidentInput | None) -> tuple[str, ...]:
    """Clause 2's anchors, in first-seen order, without duplicates (ignoring case)."""
    found: list[str] = []
    texts = [question]
    if incident is not None:
        texts += [incident.summary, incident.observations, incident.immediate_actions]
    for text in texts:
        numbers = _DOCUMENT_NUMBER.findall(text)
        found += numbers
        remainder = _DOCUMENT_NUMBER.sub(" ", text)
        found += _EQUIPMENT.findall(remainder)
    if incident is not None:
        found += [v.strip() for v in (incident.equipment_id, incident.area, incident.process_step)]
    unique: dict[str, str] = {}
    for anchor in found:
        if anchor and anchor.casefold() not in unique:
            unique[anchor.casefold()] = anchor
    return tuple(unique.values())


def _metadata_text(info: ControlledDocumentInfo | None) -> str:
    if info is None:
        return ""
    parts = [info.document_number, info.revision, info.title, info.area, info.process]
    return " ".join([*(p for p in parts if p), *info.equipment])


def _is_effective_controlled(info: ControlledDocumentInfo | None) -> bool:
    return (
        info is not None
        and info.lifecycle_status is LifecycleStatus.EFFECTIVE
        and info.control_status in _CONTROLLED_OR_REFERENCE
    )


def _is_unknown(info: ControlledDocumentInfo | None) -> bool:
    return info is None or info.lifecycle_status is LifecycleStatus.UNKNOWN


def classify_sufficiency(
    question: str,
    incident: IncidentInput | None,
    eligible: Sequence[EvidenceText],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
) -> SufficiencyAssessment:
    anchors = extract_anchors(question, incident)
    terms = content_terms(question)
    haystacks = [
        f"{chunk.text} {_metadata_text(infos.get(chunk.document_id))}".casefold()
        for chunk in eligible
    ]
    covered = tuple(a for a in anchors if any(a.casefold() in h for h in haystacks))
    best = max((term_coverage(terms, chunk.text) for chunk in eligible), default=0.0)
    effective = sum(1 for c in eligible if _is_effective_controlled(infos.get(c.document_id)))
    unknown_only = bool(eligible) and all(_is_unknown(infos.get(c.document_id)) for c in eligible)

    if effective > 0 and len(covered) == len(anchors) and best >= SUFFICIENT_COVERAGE:
        sufficiency = EvidenceSufficiency.SUFFICIENT
    elif eligible and (covered or best >= PARTIAL_COVERAGE or unknown_only):
        sufficiency = EvidenceSufficiency.PARTIAL
    else:
        sufficiency = EvidenceSufficiency.INSUFFICIENT
    return SufficiencyAssessment(
        sufficiency=sufficiency,
        rule_version=RULE_VERSION,
        anchors=anchors,
        covered_anchors=covered,
        question_terms=tuple(sorted(terms)),
        best_coverage=round(best, 4),
        eligible_count=len(eligible),
        effective_count=effective,
        unknown_only=unknown_only,
    )


# --- Rule version 2 (PLV7-T01; decision PLV-D-022), beside version 1 ---------------------------

RULE_VERSION_2: Final = "2"
# The thresholds the development calibration chose from the registered grid (PLV-D-022; PLV7-T01,
# accuracy 0.575 on development at 0.3 and 0.25, the highest of the 15 pairs, no tie): the
# sufficient threshold over 0.3 to 0.5, the partial threshold over 0.15 to 0.25.
SUFFICIENT_COVERAGE_V2: Final = 0.3
PARTIAL_COVERAGE_V2: Final = 0.25
REQUIREMENT_ANCHOR: Final = "R-ANCHOR"
REQUIREMENT_CONFLICT: Final = "R-CONFLICT"
REQUIREMENT_PREREQUISITE: Final = "R-PREREQ"
_CFR_SECTION = re.compile(r"(?<![\d.])(\d{3}\.\d{1,4})(?![\d.])")
_PERMISSION = re.compile(
    r"^\s*(?:can|may|could|is|are|should|shall)\b.*\b(?:start|restart|resume|continue|proceed"
    r"|release)",
    re.IGNORECASE | re.DOTALL,
)
# A permission question may follow a statement of what was done; each sentence is read.
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
_PREREQUISITE_HEADING = re.compile(
    r"^\s*#{1,6}\s+.*\b(?:prerequisite|precaution|warning|caution)", re.IGNORECASE
)


def _covered(
    anchors: Sequence[str],
    chunks: Sequence[EvidenceText],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
) -> tuple[str, ...]:
    haystacks = [
        f"{chunk.text} {_metadata_text(infos.get(chunk.document_id))}".casefold()
        for chunk in chunks
    ]
    return tuple(a for a in anchors if any(a.casefold() in h for h in haystacks))


def _conflicting_effective(
    chunks: Sequence[EvidenceText], infos: Mapping[uuid.UUID, ControlledDocumentInfo]
) -> bool:
    """Two Effective documents of one number and different revisions in the context, of which
    neither names the other as the revision it supersedes (R-CONFLICT)."""
    by_number: dict[str, dict[str, ControlledDocumentInfo]] = {}
    for chunk in chunks:
        info = infos.get(chunk.document_id)
        if info is not None and info.document_number and info.revision:
            by_number.setdefault(info.document_number, {})[info.revision] = info
    for number, revisions in by_number.items():
        held = list(revisions.values())
        for i, first in enumerate(held):
            for second in held[i + 1 :]:
                linked = (first.supersedes_number, first.supersedes_revision) == (
                    number,
                    second.revision,
                ) or (second.supersedes_number, second.supersedes_revision) == (
                    number,
                    first.revision,
                )
                if not linked:
                    return True
    return False


def _prerequisite_section(
    chunks: Sequence[EvidenceText], infos: Mapping[uuid.UUID, ControlledDocumentInfo]
) -> bool:
    for chunk in chunks:
        info = infos.get(chunk.document_id)
        first = chunk.text.lstrip().split("\n", 1)[0]
        if (
            info is not None
            and info.document_type in PROCEDURE_TYPES
            and _PREREQUISITE_HEADING.match(first)
        ):
            return True
    return False


def classify_sufficiency_v2(
    question: str,
    incident: IncidentInput | None,
    eligible: Sequence[EvidenceText],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    *,
    sufficient: float = SUFFICIENT_COVERAGE_V2,
    partial: float = PARTIAL_COVERAGE_V2,
) -> SufficiencyAssessment:
    """Rule version 2: version 1's inputs over authoritative evidence only, the CFR sections the
    question cites as anchors, and three structured requirements; ``capped_by`` names each
    requirement the context does not meet (PLV-D-022)."""
    anchors = list(extract_anchors(question, incident))
    for section in _CFR_SECTION.findall(question):
        if section.casefold() not in {a.casefold() for a in anchors}:
            anchors.append(section)
    terms = content_terms(question)
    authoritative = [c for c in eligible if _is_effective_controlled(infos.get(c.document_id))]
    covered = _covered(anchors, authoritative, infos)
    covered_any = _covered(anchors, eligible, infos)
    best = max((term_coverage(terms, chunk.text) for chunk in authoritative), default=0.0)
    best_any = max((term_coverage(terms, chunk.text) for chunk in eligible), default=0.0)
    unknown_only = bool(eligible) and all(_is_unknown(infos.get(c.document_id)) for c in eligible)
    capped: list[str] = []
    if len(covered) != len(anchors):
        capped.append(REQUIREMENT_ANCHOR)
    if _conflicting_effective(authoritative, infos):
        capped.append(REQUIREMENT_CONFLICT)
    asks_permission = any(_PERMISSION.search(s) for s in _SENTENCE_END.split(question))
    if asks_permission and _prerequisite_section(authoritative, infos):
        capped.append(REQUIREMENT_PREREQUISITE)

    if authoritative and not capped and best >= sufficient:
        sufficiency = EvidenceSufficiency.SUFFICIENT
    elif not eligible or (not covered_any and best_any < partial and not unknown_only):
        sufficiency = EvidenceSufficiency.INSUFFICIENT
    else:
        sufficiency = EvidenceSufficiency.PARTIAL
    return SufficiencyAssessment(
        sufficiency=sufficiency,
        rule_version=RULE_VERSION_2,
        anchors=tuple(anchors),
        covered_anchors=covered,
        question_terms=tuple(sorted(terms)),
        best_coverage=round(best, 4),
        eligible_count=len(eligible),
        effective_count=len(authoritative),
        unknown_only=unknown_only,
        capped_by=tuple(capped),
        covered_any=covered_any,
        best_any_coverage=round(best_any, 4),
    )
