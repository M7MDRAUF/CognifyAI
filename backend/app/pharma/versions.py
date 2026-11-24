"""Version-aware partition of retrieval candidates (REQ-RET-02 to REQ-RET-07, REQ-CON-01).

Before reranking, a pharmaceutical run's candidates are split into the evidence the agents may
see and the evidence they may not, with exactly one recorded reason per exclusion:

1. ``filter_mismatch``: the document states a value for a filtered field and it is not one the
   inquiry named. A document stating no value is kept (REQ-RET-02).
2. ``superseded``, ``retired``, ``draft``: the document's own lifecycle status (REQ-RET-03).
3. ``superseded_by_effective``: an Effective document of the project names this one as the
   revision it supersedes, whatever this one's own status says (REQ-RET-05).
4. ``lower_revision``: another Effective document with the same number has a higher revision
   (REQ-RET-04).

A document with no controlled row, or whose header was missing or invalid, has lifecycle
``unknown``: it stays eligible and is reported, so the answer labels it and sufficiency never
counts it alone (REQ-RET-07). Pure functions: the eligible candidates keep their order, and the
same inputs always give the same partition and the same conflicts.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import ConflictKind, ExclusionReason, LifecycleStatus
from app.schemas.pharma import PharmaFilters

UNNUMBERED = "unknown"


class Candidate(Protocol):
    @property
    def document_id(self) -> uuid.UUID: ...


@dataclass(frozen=True)
class ExcludedCandidate[C: Candidate]:
    chunk: C
    reason: ExclusionReason


@dataclass(frozen=True)
class VersionConflict:
    kind: ConflictKind
    document_numbers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class VersionResolution[C: Candidate]:
    eligible: list[C]
    excluded: list[ExcludedCandidate[C]]
    conflicts: list[VersionConflict]


_STATUS_REASON = {
    LifecycleStatus.SUPERSEDED: (ExclusionReason.SUPERSEDED, ConflictKind.SUPERSEDED_MATCH),
    LifecycleStatus.RETIRED: (ExclusionReason.RETIRED, ConflictKind.RETIRED_MATCH),
    LifecycleStatus.DRAFT: (ExclusionReason.DRAFT, ConflictKind.DRAFT_MATCH),
}


def revision_key(revision: str | None) -> tuple[int, int, str]:
    """Orders revisions: numeric ones by value, then alphanumeric ones by text; none first."""
    if revision is None:
        return (-1, 0, "")
    if revision.isdigit():
        return (0, int(revision), revision)
    return (1, 0, revision)


def resolve_versions[C: Candidate](
    candidates: Sequence[C],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    filters: PharmaFilters | None = None,
) -> VersionResolution[C]:
    """Partition ``candidates``. ``infos`` holds every controlled document of the project (not
    only the candidates'), so a higher revision or a superseding document is seen even when it
    was not retrieved."""
    effective = [i for i in infos.values() if i.lifecycle_status is LifecycleStatus.EFFECTIVE]
    superseded_by: dict[tuple[str, str], ControlledDocumentInfo] = {}
    for info in effective:
        if info.supersedes_number and info.supersedes_revision:
            superseded_by[(info.supersedes_number, info.supersedes_revision)] = info
    highest: dict[str, ControlledDocumentInfo] = {}
    for info in effective:
        if info.document_number is None:
            continue
        current = highest.get(info.document_number)
        if current is None or revision_key(info.revision) > revision_key(current.revision):
            highest[info.document_number] = info

    eligible: list[C] = []
    excluded: list[ExcludedCandidate[C]] = []
    conflicts: dict[tuple[ConflictKind, tuple[str, ...]], VersionConflict] = {}

    def report(kind: ConflictKind, numbers: tuple[str, ...], description: str) -> None:
        conflicts.setdefault((kind, numbers), VersionConflict(kind, numbers, description))

    # PH-DEF-029: Effective documents that state the same number and revision. Neither can be
    # ranked above the other, so both stay eligible and the duplicate is a version conflict.
    copies: dict[tuple[str, str], set[uuid.UUID]] = {}
    for info in effective:
        if info.document_number and info.revision:
            copies.setdefault((info.document_number, info.revision), set()).add(info.document_id)

    for candidate in candidates:
        stored = infos.get(candidate.document_id)
        reason = _reason(stored, filters, superseded_by, highest, report)
        if reason is None:
            eligible.append(candidate)
            if stored is not None and stored.document_number and stored.revision:
                held = copies.get((stored.document_number, stored.revision), set())
                if len(held) > 1:
                    report(
                        ConflictKind.MULTIPLE_EFFECTIVE_REVISIONS,
                        (stored.document_number,),
                        f"{stored.document_number} revision {stored.revision} is stated by "
                        f"{len(held)} effective documents; each is used and they may differ",
                    )
        else:
            excluded.append(ExcludedCandidate(candidate, reason))
    return VersionResolution(eligible, excluded, list(conflicts.values()))


def _reason(
    info: ControlledDocumentInfo | None,
    filters: PharmaFilters | None,
    superseded_by: Mapping[tuple[str, str], ControlledDocumentInfo],
    highest: Mapping[str, ControlledDocumentInfo],
    report: "_Reporter",
) -> ExclusionReason | None:
    if info is None or info.lifecycle_status is LifecycleStatus.UNKNOWN:
        number = info.document_number if info and info.document_number else UNNUMBERED
        report(
            ConflictKind.UNKNOWN_STATUS,
            (number,),
            f"{number} has no valid controlled metadata; its lifecycle status is unknown",
        )
        return None
    if filters is not None and _mismatches(info, filters):
        return ExclusionReason.FILTER_MISMATCH
    number = info.document_number or UNNUMBERED
    label = f"{number} revision {info.revision or UNNUMBERED}"
    status_reason = _STATUS_REASON.get(info.lifecycle_status)
    if status_reason is not None:
        reason, kind = status_reason
        report(kind, (number,), f"{label} is {info.lifecycle_status.value} and was not used")
        return reason
    if info.document_number and info.revision:
        successor = superseded_by.get((info.document_number, info.revision))
        if successor is not None:
            report(
                ConflictKind.SUPERSEDED_MARKED_EFFECTIVE,
                (number,),
                f"{label} is marked effective but is superseded by revision "
                f"{successor.revision or UNNUMBERED}",
            )
            return ExclusionReason.SUPERSEDED_BY_EFFECTIVE
        top = highest.get(info.document_number)
        if top is not None and revision_key(top.revision) > revision_key(info.revision):
            report(
                ConflictKind.MULTIPLE_EFFECTIVE_REVISIONS,
                (number,),
                f"{number} has more than one effective revision; revision "
                f"{top.revision} is used and revision {info.revision} is not",
            )
            return ExclusionReason.LOWER_REVISION
    return None


class _Reporter(Protocol):
    def __call__(self, kind: ConflictKind, numbers: tuple[str, ...], description: str) -> None: ...


def _mismatches(info: ControlledDocumentInfo, filters: PharmaFilters) -> bool:
    if filters.equipment_ids and info.equipment:
        wanted = {value.upper() for value in filters.equipment_ids}
        if wanted.isdisjoint(value.upper() for value in info.equipment):
            return True
    areas = {value.casefold() for value in filters.areas}
    if areas and info.area is not None and info.area.casefold() not in areas:
        return True
    processes = {value.casefold() for value in filters.processes}
    if processes and info.process is not None and info.process.casefold() not in processes:
        return True
    return bool(
        filters.document_types
        and info.document_type is not None
        and info.document_type not in filters.document_types
    )
