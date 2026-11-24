"""Deterministic risk signals of a pharmaceutical inquiry (REQ-RSK, REQ-ESC-04, REQ-ESC-06).

Signals are read from the question and the incident fields (product quality, data integrity,
equipment, contamination, out of specification) and from the evidence and the incident fields
(suspected instruction text). Every text is normalised as the policy engine normalises it, so a
lookalike or zero-width spelling raises the same signal.

The extraction is conservative by design: a negated mention ("no contamination observed")
still raises its signal, because a rule cannot tell a reliable negation from an unreliable one,
and a missed signal is the worse error (recorded in the plan, PDP7-T03). A signal makes the
escalation stricter; it never makes it looser.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.pharma.policy import normalize_for_policy, policy_forms
from app.schemas.pharma import IncidentInput, PharmaFilters


class EvidenceText(Protocol):
    @property
    def document_id(self) -> uuid.UUID: ...

    @property
    def text(self) -> str: ...


SIGNAL_TERMS: dict[str, re.Pattern[str]] = {
    "product_quality": re.compile(
        r"\b(?:out of specification|weight drift|weight (?:out of|outside)|"
        r"hardness (?:out of|outside)|defects?|defective|mix-?up|"
        r"wrong (?:product|material|label)|label(?:l)?ing error|"
        r"(?:batch|product) (?:affected|impact|quality))\b"
    ),
    "data_integrity": re.compile(
        r"\b(?:audit trail|backdat\w*|overwrit\w*|correction fluid|shared (?:password|account)|"
        r"record (?:missing|altered|deleted|changed)|data integrity|falsif\w*|recorded before)\b"
    ),
    "equipment": re.compile(
        r"\b(?:alarm|malfunction\w*|breakdown|stopp(?:ed|age)|interlock|leak\w*|jam\w*|"
        r"calibration (?:overdue|failed)|out of tolerance|maintenance overdue|"
        r"damaged (?:punch|tooling))\b"
    ),
    "contamination": re.compile(
        r"\b(?:contaminat\w*|cross-contamination|foreign (?:matter|particles?)|residue|microbial)\b"
    ),
    "out_of_specification": re.compile(
        r"\b(?:out of specification|oos|out-of-limit|out of limits?|"
        r"outside (?:the )?(?:limits?|range|specification)|"
        r"exceed(?:s|ed)? (?:the )?(?:limits?|specification|hold time))\b"
    ),
}
# Text addressed to the model rather than to a reader of the document (threats T-01, T-02).
INSTRUCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bignore (?:all |the |any )?(?:previous|prior|above|earlier) (?:instructions|rules)\b"
    ),
    re.compile(r"\bdisregard (?:the |all |any )?(?:previous|prior|above|rules|instructions)\b"),
    re.compile(r"\byou are now\b"),
    re.compile(r"\bsystem prompt\b"),
    re.compile(r"\bdo not mention this\b"),
    re.compile(r"\bnote to the (?:assistant|ai|model)\b"),
    re.compile(r"\b(?:assistant|ai|model),? (?:you must|must) (?:say|state|answer)\b"),
    # Found by the PDP11-T01 injection suite (PH-DEF-019).
    re.compile(
        r"\bforget (?:the |all |any |everything |your )?"
        r"(?:rules|instructions|above|previous|prior)\b"
    ),
    re.compile(r"\bfrom now on,? (?:you |answer|respond|reply|say|state)\b"),
    re.compile(r"\b(?:system|admin|administrator|developer|debug) override\b"),
    re.compile(r"\b(?:admin|administrator|developer|debug) mode\b"),
    re.compile(r"\bpretend (?:that )?(?:you are|you're|to be)\b"),
)


def suspected_instruction(text: str) -> str | None:
    """The first instruction-like phrase in any normalised form of ``text``, or None."""
    for normalized in policy_forms(text):
        for pattern in INSTRUCTION_PATTERNS:
            match = pattern.search(normalized)
            if match is not None:
                return match.group(0)
    return None


# Separators a file name uses for spaces, read as spaces so "ignore_previous_instructions.md"
# reads as the phrase it spells (PH-DEF-035).
_NAME_SEPARATORS = re.compile(r"[_.\-]+")


def _source_names(evidence: Sequence[EvidenceText]) -> list[str]:
    """The file names of the passages, which the evidence block quotes to the model."""
    names: list[str] = []
    for chunk in evidence:
        name = getattr(chunk, "source_filename", None)
        if isinstance(name, str) and name:
            names.append(_NAME_SEPARATORS.sub(" ", name))
    return names


def _filter_texts(filters: PharmaFilters | None) -> list[str]:
    if filters is None:
        return []
    return [*filters.equipment_ids, *filters.areas, *filters.processes]


@dataclass(frozen=True)
class RiskSignals:
    product_quality: bool = False
    data_integrity: bool = False
    equipment: bool = False
    contamination: bool = False
    out_of_specification: bool = False
    instruction_text: bool = False
    matched: tuple[str, ...] = ()

    @property
    def quality_relevant(self) -> bool:
        """The signals REQ-ESC-04 escalates to quality assurance."""
        return (
            self.product_quality
            or self.data_integrity
            or self.contamination
            or self.out_of_specification
        )


def _incident_texts(incident: IncidentInput | None) -> list[str]:
    if incident is None:
        return []
    return [
        incident.summary,
        incident.area,
        incident.equipment_id,
        incident.process_step,
        incident.product_family,
        incident.observations,
        incident.immediate_actions,
    ]


def extract_signals(
    question: str,
    incident: IncidentInput | None,
    evidence: Sequence[EvidenceText],
    *,
    filters: PharmaFilters | None = None,
) -> RiskSignals:
    """The risk signals of an inquiry. Signal terms are read from the question, the incident fields
    and the filter values; instruction text also from every passage and its file name
    (PH-DEF-035)."""
    raw_inquiry = " \n ".join([question, *_incident_texts(incident), *_filter_texts(filters)])
    inquiry = normalize_for_policy(raw_inquiry)
    flags: dict[str, bool] = {}
    matched: list[str] = []
    for name, pattern in SIGNAL_TERMS.items():
        hit = pattern.search(inquiry)
        flags[name] = hit is not None
        if hit is not None:
            matched.append(f"{name}: {hit.group(0)}")
    instruction = next(
        (
            phrase
            for phrase in map(
                suspected_instruction,
                [raw_inquiry, *(c.text for c in evidence), *_source_names(evidence)],
            )
            if phrase is not None
        ),
        None,
    )
    if instruction is not None:
        matched.append(f"instruction_text: {instruction}")
    return RiskSignals(
        product_quality=flags["product_quality"],
        data_integrity=flags["data_integrity"],
        equipment=flags["equipment"],
        contamination=flags["contamination"],
        out_of_specification=flags["out_of_specification"],
        instruction_text=instruction is not None,
        matched=tuple(matched),
    )
