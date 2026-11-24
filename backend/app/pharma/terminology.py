"""Controlled terminology and query expansion for pharmaceutical retrieval (PLV6-T02).

A fixed table of manufacturing and quality terms, each a set of equivalent forms: an abbreviation
and its expansion, an equipment class and its common names, a document kind and its full name, a
process step and its spellings. The table is general vocabulary of the field (21 CFR 210 and 211
wording and the industry's standard abbreviations); no entry is taken from a scenario.

``expand_queries`` keeps every query as given, first and unchanged. A query that names one or
more terms gains one variant: the query followed by the other forms of each term it names, in the
table's order, at most ``MAX_ADDED_FORMS`` forms and never past ``MAX_EXPANDED_QUERY_CHARS``; a
query already at that length gains nothing. A form of three characters or fewer matches only as a
whole word in its own case ("QA", never "qa" or "aqua"); a longer form matches as whole words in
any case. The same queries always give the same expansion, and ``Expansion.log_fields`` holds
counts and term keys only, never query text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

TERMINOLOGY_VERSION: Final = "1"
MAX_ADDED_FORMS: Final = 8
# The workflow's bound on one retrieval query (``workflow.MAX_INCIDENT_QUERY_CHARS``); a test holds
# the two equal, since the workflow imports this module and not the other way round.
MAX_EXPANDED_QUERY_CHARS: Final = 200
SHORT_FORM: Final = 3

TermKind = Literal["quality", "document", "process", "equipment"]


@dataclass(frozen=True)
class TermEntry:
    key: str
    kind: TermKind
    forms: tuple[str, ...]


def _entry(key: str, kind: TermKind, *forms: str) -> TermEntry:
    return TermEntry(key, kind, forms)


TERMINOLOGY: Final[tuple[TermEntry, ...]] = (
    _entry("cgmp", "quality", "CGMP", "current good manufacturing practice", "GMP"),
    _entry("qa", "quality", "QA", "quality assurance"),
    _entry("qc", "quality", "QC", "quality control"),
    _entry("quality-unit", "quality", "quality unit", "quality control unit"),
    _entry("oos", "quality", "OOS", "out of specification", "out-of-specification"),
    _entry("oot", "quality", "OOT", "out of trend", "out-of-trend"),
    _entry("capa", "quality", "CAPA", "corrective and preventive action"),
    _entry("deviation", "quality", "deviation", "nonconformance", "non-conformance"),
    _entry("form-483", "quality", "Form 483", "FDA 483", "inspectional observations"),
    _entry("recall", "quality", "recall", "market withdrawal"),
    _entry("data-integrity", "quality", "data integrity", "ALCOA"),
    _entry("sop", "document", "SOP", "standard operating procedure", "written procedure"),
    _entry("wi", "document", "WI", "work instruction"),
    _entry("batch-record", "document", "batch record", "batch production record", "BMR"),
    _entry("logbook", "document", "logbook", "log book", "equipment log"),
    _entry("validation", "document", "process validation", "qualification"),
    _entry("line-clearance", "process", "line clearance", "line-clearance"),
    _entry("changeover", "process", "changeover", "change-over", "product changeover"),
    _entry("cleaning", "process", "cleaning", "sanitization", "sanitisation", "CIP"),
    _entry("sterilization", "process", "sterilization", "sterilisation", "SIP", "autoclave"),
    _entry("filtration", "process", "filtration", "filter integrity", "integrity test"),
    _entry(
        "environmental-monitoring",
        "process",
        "environmental monitoring",
        "settle plates",
        "air sampling",
    ),
    _entry("hold-time", "process", "hold time", "holding time"),
    _entry("set-up", "process", "set-up", "setup", "start-up", "startup"),
    _entry("filler", "equipment", "filler", "filling machine", "filling line"),
    _entry("press", "equipment", "tablet press", "compression machine"),
    _entry("coater", "equipment", "coater", "coating pan", "film coating"),
    _entry("blender", "equipment", "blender", "mixer", "V-blender"),
    _entry("blister", "equipment", "blister machine", "blister packer", "blister packaging"),
    _entry("isolator", "equipment", "isolator", "RABS", "restricted access barrier system"),
    _entry("air-handling", "equipment", "AHU", "air handling unit", "HVAC"),
    _entry("balance", "equipment", "balance", "scale", "weighing"),
)


def _pattern(form: str) -> re.Pattern[str]:
    flags = 0 if len(form) <= SHORT_FORM else re.IGNORECASE
    return re.compile(rf"(?<![\w-]){re.escape(form)}(?![\w-])", flags)


_PATTERNS: Final = tuple((entry, tuple(_pattern(f) for f in entry.forms)) for entry in TERMINOLOGY)


@dataclass(frozen=True)
class Expansion:
    """The queries to retrieve with, and what the expansion did."""

    queries: list[str]
    matched: list[str]
    added_forms: int
    queries_in: int

    def log_fields(self) -> dict[str, str | int | list[str]]:
        return {
            "terminology_version": TERMINOLOGY_VERSION,
            "queries_in": self.queries_in,
            "queries_out": len(self.queries),
            "terms": list(self.matched),
            "added_forms": self.added_forms,
        }


def _variant(query: str) -> tuple[str | None, list[str], int]:
    """The query's variant (``None`` when it names no term or has no room), its term keys and the
    forms added."""
    named: list[str] = []
    extra: list[str] = []
    for entry, patterns in _PATTERNS:
        present = [
            form
            for form, pattern in zip(entry.forms, patterns, strict=True)
            if pattern.search(query)
        ]
        if not present:
            continue
        named.append(entry.key)
        extra += [form for form in entry.forms if form not in present]
    added: list[str] = []
    text = query
    for form in extra:
        if len(added) == MAX_ADDED_FORMS:
            break
        candidate = f"{text} {form}"
        if len(candidate) > MAX_EXPANDED_QUERY_CHARS:
            continue
        text = candidate
        added.append(form)
    return (text if added else None), named, len(added)


def expand_queries(queries: Sequence[str]) -> Expansion:
    """``queries`` unchanged, then one variant for each query that names a term and has room."""
    out = list(queries)
    seen = {query.casefold() for query in out}
    matched: list[str] = []
    added = 0
    for query in queries:
        variant, named, count = _variant(query)
        matched += [key for key in named if key not in matched]
        if variant is not None and variant.casefold() not in seen:
            out.append(variant)
            seen.add(variant.casefold())
            added += count
    return Expansion(out, matched, added, len(queries))
