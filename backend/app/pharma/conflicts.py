"""Deterministic conflict detection (REQ-CON-01; REQ-CON-02 keeps the model's reports apart).

Two sources, merged into one deterministic list:

* version resolution's conflicts (superseded, retired and draft matches, unknown status,
  several Effective revisions, a superseded document still marked Effective), and
* the eligible evidence itself: a stated parameter (``Label: value unit`` on its own line) that
  differs between two Effective documents of the same scope, and two Effective site procedures
  of the same type that cover the same equipment and process and state the same parameter
  (overlapping scope, decision PH-D-025).

Same scope means the same process, or at least one shared equipment identifier. The same
parameter with the same value is not a conflict. Patterns are anchored to whole lines and
bounded, and the detector reads stored chunk text and stored metadata only.
"""

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import (
    ConflictKind,
    ControlledDocumentType,
    ControlStatus,
    LifecycleStatus,
    SourceAuthority,
)
from app.pharma.versions import VersionResolution

_NUMBER = r"[0-9]{1,6}(?:\.[0-9]{1,4})?"
_UNIT = r"[A-Za-z%°µ][A-Za-z0-9%°µ/.]{0,15}"
_AMOUNT = rf"{_NUMBER}(?:[ \t]?{_UNIT})?"
_PARAMETER = re.compile(
    rf"^(?P<label>[A-Z][A-Za-z0-9 ()/-]{{1,60}}?):[ \t]+"
    rf"(?P<value>{_AMOUNT}(?:[ \t]+to[ \t]+{_AMOUNT})?(?:[ \t]+{_UNIT}){{0,2}})[ \t]*$"
)
_PROCEDURES = frozenset({ControlledDocumentType.SOP, ControlledDocumentType.WORK_INSTRUCTION})
UNNUMBERED = "unknown"


class EvidenceText(Protocol):
    @property
    def document_id(self) -> uuid.UUID: ...

    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class StatedParameter:
    label: str
    value: str

    @property
    def key(self) -> str:
        return " ".join(self.label.casefold().split())

    @property
    def normalized_value(self) -> str:
        return " ".join(self.value.casefold().split())


@dataclass(frozen=True)
class DetectedConflict:
    kind: ConflictKind
    document_numbers: tuple[str, ...]
    description: str


def extract_parameters(text: str) -> list[StatedParameter]:
    """Every ``Label: value unit`` line of ``text``, in order."""
    found: list[StatedParameter] = []
    for line in text.splitlines():
        if len(line) > 200:
            continue
        match = _PARAMETER.match(line.strip())
        if match is not None:
            found.append(
                StatedParameter(match.group("label").strip(), match.group("value").strip())
            )
    return found


def _effective(info: ControlledDocumentInfo | None) -> bool:
    return info is not None and info.lifecycle_status is LifecycleStatus.EFFECTIVE


def same_scope(a: ControlledDocumentInfo, b: ControlledDocumentInfo) -> bool:
    if a.process and b.process and a.process.casefold() == b.process.casefold():
        return True
    shared = {e.upper() for e in a.equipment} & {e.upper() for e in b.equipment}
    return bool(shared)


def _site_procedure(info: ControlledDocumentInfo) -> bool:
    return (
        info.source_authority is SourceAuthority.SITE
        and info.control_status is ControlStatus.CONTROLLED
        and info.document_type in _PROCEDURES
    )


def detect_conflicts[C: EvidenceText](
    resolution: VersionResolution[C],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
) -> list[DetectedConflict]:
    found: dict[tuple[ConflictKind, tuple[str, ...]], DetectedConflict] = {}
    for conflict in resolution.conflicts:
        version_key = (conflict.kind, conflict.document_numbers)
        found.setdefault(version_key, DetectedConflict(*version_key, conflict.description))

    parameters: dict[uuid.UUID, dict[str, StatedParameter]] = {}
    for chunk in resolution.eligible:
        if not _effective(infos.get(chunk.document_id)):
            continue
        stated = parameters.setdefault(chunk.document_id, {})
        for parameter in extract_parameters(chunk.text):
            stated.setdefault(parameter.key, parameter)

    for left, right in combinations(sorted(parameters, key=str), 2):
        a, b = infos[left], infos[right]
        # Two copies of one number and revision (PH-DEF-029) are compared whatever their scope;
        # two revisions of one number are a version matter, resolved before this point.
        copies = (
            a.document_number is not None
            and a.document_number == b.document_number
            and a.revision == b.revision
        )
        if a.document_number is not None and a.document_number == b.document_number and not copies:
            continue
        if not copies and not same_scope(a, b):
            continue
        numbers = tuple(sorted({a.document_number or UNNUMBERED, b.document_number or UNNUMBERED}))
        shared = sorted(set(parameters[left]) & set(parameters[right]))
        for key in shared:
            pa, pb = parameters[left][key], parameters[right][key]
            if pa.normalized_value != pb.normalized_value:
                kind_key = (ConflictKind.PARAMETER_DISAGREEMENT, numbers)
                description = (
                    f"{pa.label}: two effective copies of {a.document_number} revision "
                    f"{a.revision} state {pa.value} and {pb.value}"
                    if copies
                    else f"{pa.label}: {a.document_number or UNNUMBERED} states {pa.value}; "
                    f"{b.document_number or UNNUMBERED} states {pb.value}"
                )
                found.setdefault(
                    kind_key,
                    DetectedConflict(ConflictKind.PARAMETER_DISAGREEMENT, numbers, description),
                )
        overlap = (
            not copies  # copies are a version conflict already (PH-DEF-029)
            and shared
            and _site_procedure(a)
            and _site_procedure(b)
            and a.document_type is b.document_type
            and a.process
            and b.process
            and a.process.casefold() == b.process.casefold()
            and {e.upper() for e in a.equipment} == {e.upper() for e in b.equipment}
        )
        if overlap:
            found.setdefault(
                (ConflictKind.OVERLAPPING_EFFECTIVE_SCOPE, numbers),
                DetectedConflict(
                    ConflictKind.OVERLAPPING_EFFECTIVE_SCOPE,
                    numbers,
                    f"{numbers[0]} and {numbers[1]} are both effective for the same equipment "
                    f"and process and state the same parameters",
                ),
            )
    return list(found.values())
