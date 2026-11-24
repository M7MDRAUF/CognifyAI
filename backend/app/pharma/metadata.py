"""The controlled-document header contract and its parser (PDP3-T06).

A pharmaceutical project's Markdown or text document may open with a header block::

    ---
    document_number: SOP-SYN-MFG-010
    title: Tablet Press TP-201 Operation
    document_type: SOP
    revision: 04
    effective_date: 2026-03-01
    lifecycle_status: Effective
    supersedes: SOP-SYN-MFG-010 rev 03
    site: Synthetic Site A
    area: Compression suite
    equipment: TP-201, BAL-07
    process: Compression
    product_family: Synthetic product A
    source_authority: Site
    control_status: Controlled
    data_classification: Synthetic
    ---

The opening ``---`` must be the document's first line. Each value is checked against a strict
form (REQ-DOC-02). An invalid value is never stored: it is dropped, an issue names it, and the
lifecycle status becomes Unknown. The header text is returned apart from the body, so it is
neither chunked nor embedded (REQ-DOC-07). Author-controlled text is the attack surface here
(threat T-03): free-text values may not carry control characters, line separators, evidence
fence markers or chunk-id syntax. Parsing reads at most ``HEADER_MAX_LINES`` lines and never
scans the rest of the document for the header (threat T-11).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import cast

from app.pharma.enums import (
    ControlledDocumentType,
    ControlStatus,
    DataClassification,
    LifecycleStatus,
    MetadataStatus,
    SourceAuthority,
)
from app.schemas.pharma import EQUIPMENT_ID_PATTERN

SYNTHETIC_MARKING = "SYNTHETIC " + chr(0x2014) + " FOR RESEARCH AND EVALUATION ONLY"
HEADER_MAX_LINES = 40
HEADER_MAX_LINE_CHARS = 300
MAX_TITLE_LENGTH = 200
MAX_LABEL_LENGTH = 100
MAX_EQUIPMENT = 8

_BOM = chr(0xFEFF)
_DELIMITER = "---"
_KEY_LINE = re.compile(r"^([a-z_]{1,40}):(.*)$")
_DOCUMENT_NUMBER = re.compile(r"^[A-Z]{2,5}(?:-[A-Z0-9]{1,6}){1,5}$")
_REVISION = re.compile(r"^(?:[0-9]{1,4}|[A-Z]{1,2}[0-9]{0,2})$")
_EQUIPMENT_ID = re.compile(EQUIPMENT_ID_PATTERN)
_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_SUPERSEDES = re.compile(r"^(\S+) rev (\S+)$", re.IGNORECASE)
_HTTPS_URL = re.compile(r"^https://[^\s<>\"']{1,290}$")
# C0 and C1 controls (tab included: a header value is one line of text), and the two Unicode line
# breaks str.splitlines() honours outside them.
_FORBIDDEN_CHARACTERS = frozenset(
    [chr(c) for c in [*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)]] + [chr(0x2028), chr(0x2029)]
)
# Syntax the evidence formatter gives meaning to (`format_evidence`'s fences and provenance lines).
_FORBIDDEN_SUBSTRINGS = ("chunk_id", "<<<", ">>>", "```")

REQUIRED_KEYS = (
    "document_number",
    "title",
    "document_type",
    "revision",
    "lifecycle_status",
    "data_classification",
)


@dataclass(frozen=True)
class ControlledMetadataDraft:
    """The validated header values; ``None`` (or empty) where absent or refused."""

    document_number: str | None = None
    title: str | None = None
    document_type: ControlledDocumentType | None = None
    revision: str | None = None
    effective_date: date | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.UNKNOWN
    supersedes_number: str | None = None
    supersedes_revision: str | None = None
    site: str | None = None
    area: str | None = None
    equipment: tuple[str, ...] = ()
    process: str | None = None
    product_family: str | None = None
    source_authority: SourceAuthority | None = None
    control_status: ControlStatus | None = None
    data_classification: DataClassification | None = None
    source_url: str | None = None
    retrieved: date | None = None


@dataclass(frozen=True)
class ParsedHeader:
    metadata: ControlledMetadataDraft
    body: str
    status: MetadataStatus
    issues: tuple[str, ...] = field(default_factory=tuple)


def validate_document_number(value: str) -> bool:
    return _DOCUMENT_NUMBER.fullmatch(value) is not None


def validate_revision(value: str) -> bool:
    return _REVISION.fullmatch(value) is not None


def validate_equipment_id(value: str) -> bool:
    return _EQUIPMENT_ID.fullmatch(value) is not None


# RMD3-T03 (DV-RMD-22): the form a site takes on a label line, outside the fence: no quote, equals
# sign or line break, so a header cannot write another label field.
_SITE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,()'&/-]{0,59}$")


def validate_site(value: str) -> bool:
    return _SITE.fullmatch(value) is not None


class _RefusedValueError(ValueError):
    """A header value that failed its form; the message becomes the issue."""


def _clean_text(value: str, limit: int) -> str:
    if not value:
        raise _RefusedValueError("is empty")
    if any(character in _FORBIDDEN_CHARACTERS for character in value):
        raise _RefusedValueError("contains a control character or line break")
    lowered = value.lower()
    for marker in _FORBIDDEN_SUBSTRINGS:
        if marker in lowered:
            raise _RefusedValueError(f"contains reserved syntax {marker!r}")
    if len(value) > limit:
        raise _RefusedValueError(f"is longer than {limit} characters")
    return value


def _choice[
    EnumT: (
        ControlledDocumentType,
        LifecycleStatus,
        SourceAuthority,
        ControlStatus,
        DataClassification,
    )
](value: str, enum: type[EnumT]) -> EnumT:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    for member in enum:
        if member.value == normalized:
            return member
    allowed = ", ".join(member.value for member in enum)
    raise _RefusedValueError(f"must be one of {allowed}")


def _iso_date(value: str) -> date:
    if not _ISO_DATE.fullmatch(value):
        raise _RefusedValueError("must be a date written YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise _RefusedValueError("is not a calendar date") from error


def _document_number(value: str) -> str:
    if not validate_document_number(value):
        raise _RefusedValueError("must look like SOP-SYN-MFG-010")
    return value


def _revision(value: str) -> str:
    if not validate_revision(value):
        raise _RefusedValueError("must be 1 to 4 digits, or 1 to 2 capitals with up to 2 digits")
    return value


def _equipment(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(","))
    if len(items) > MAX_EQUIPMENT:
        raise _RefusedValueError(f"names more than {MAX_EQUIPMENT} items")
    for item in items:
        if not validate_equipment_id(item):
            raise _RefusedValueError(f"item {item!r} must look like TP-201")
    return items


def _supersedes(value: str) -> tuple[str, str]:
    match = _SUPERSEDES.fullmatch(value)
    if match is None:
        raise _RefusedValueError("must read '<document number> rev <revision>'")
    return _document_number(match.group(1)), _revision(match.group(2))


def _source_url(value: str) -> str:
    _clean_text(value, 300)
    if not _HTTPS_URL.fullmatch(value):
        raise _RefusedValueError("must be an https URL")
    return value


# key -> (draft field, converter). The converter raises _RefusedValueError with the reason.
_FIELDS: dict[str, Callable[[str], object]] = {
    "document_number": _document_number,
    "title": lambda v: _clean_text(v, MAX_TITLE_LENGTH),
    "document_type": lambda v: _choice(v, ControlledDocumentType),
    "revision": _revision,
    "effective_date": _iso_date,
    "lifecycle_status": lambda v: _choice(v, LifecycleStatus),
    "supersedes": _supersedes,
    "site": lambda v: _clean_text(v, MAX_LABEL_LENGTH),
    "area": lambda v: _clean_text(v, MAX_LABEL_LENGTH),
    "equipment": _equipment,
    "process": lambda v: _clean_text(v, MAX_LABEL_LENGTH),
    "product_family": lambda v: _clean_text(v, MAX_LABEL_LENGTH),
    "source_authority": lambda v: _choice(v, SourceAuthority),
    "control_status": lambda v: _choice(v, ControlStatus),
    "data_classification": lambda v: _choice(v, DataClassification),
    "source_url": _source_url,
    "retrieved": _iso_date,
}


def _missing(text: str) -> ParsedHeader:
    return ParsedHeader(ControlledMetadataDraft(), text, MetadataStatus.MISSING, ())


def _read_block(source: str) -> tuple[list[str], int] | None:
    """The lines between the opening and closing delimiters, and the offset after the closing one;
    ``None`` when no closing delimiter appears within ``HEADER_MAX_LINES`` lines."""
    first_end = source.find("\n")
    position = first_end + 1
    lines: list[str] = []
    for _ in range(HEADER_MAX_LINES + 1):
        newline = source.find("\n", position)
        end = newline if newline != -1 else len(source)
        line = source[position:end].rstrip("\r")
        if line == _DELIMITER:
            return lines, (newline + 1 if newline != -1 else len(source))
        lines.append(line)
        if newline == -1:
            return None
        position = newline + 1
    return None


def parse_controlled_header(text: str) -> ParsedHeader:
    """Split ``text`` into validated controlled metadata and the body that is indexed."""
    source = text.removeprefix(_BOM)
    first_end = source.find("\n")
    first_line = (source if first_end == -1 else source[:first_end]).rstrip("\r")
    if first_line != _DELIMITER:
        return _missing(text)
    block = _read_block(source)
    if block is None:
        issue = f"header block not closed within {HEADER_MAX_LINES} lines"
        return ParsedHeader(ControlledMetadataDraft(), text, MetadataStatus.INVALID, (issue,))
    lines, body_start = block
    body = source[body_start:]

    issues: list[str] = []
    values: dict[str, object] = {}
    seen: set[str] = set()
    for number, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        if len(line) > HEADER_MAX_LINE_CHARS:
            issues.append(f"line {number} is longer than {HEADER_MAX_LINE_CHARS} characters")
            continue
        # Checked on the whole raw line: str.strip() would silently drop a trailing U+0085, which
        # Python counts as whitespace, before any value check could see it.
        if any(character in _FORBIDDEN_CHARACTERS for character in line):
            issues.append(f"line {number} contains a control character or line break")
            continue
        match = _KEY_LINE.fullmatch(line)
        if match is None:
            issues.append(f"line {number} is not a 'key: value' line")
            continue
        key, raw = match.group(1), match.group(2).strip(" ")
        if key in seen:
            issues.append(f"duplicate key {key!r}")
            values.pop(key, None)
            continue
        seen.add(key)
        converter = _FIELDS.get(key)
        if converter is None:
            issues.append(f"unknown key {key!r}")
            continue
        try:
            values[key] = converter(raw)
        except _RefusedValueError as reason:
            issues.append(f"{key} {reason}")

    issues.extend(f"missing required key {key!r}" for key in REQUIRED_KEYS if key not in seen)
    draft = _draft(values)
    if draft.data_classification is DataClassification.PUBLIC and draft.source_url is None:
        issues.append("a public document needs its source_url")
    if draft.data_classification is DataClassification.SYNTHETIC and not any(
        line.strip() == SYNTHETIC_MARKING for line in body.splitlines()
    ):
        issues.append("a synthetic document must carry the synthetic marking line in its body")

    if issues:
        return ParsedHeader(
            replace(draft, lifecycle_status=LifecycleStatus.UNKNOWN),
            body,
            MetadataStatus.INVALID,
            tuple(issues),
        )
    return ParsedHeader(draft, body, MetadataStatus.VALID, ())


def _draft(values: dict[str, object]) -> ControlledMetadataDraft:
    # The converters fix these shapes: `_supersedes` returns (number, revision), `_equipment` a
    # tuple of identifiers.
    supersedes = values.get("supersedes")
    number: str | None = None
    revision: str | None = None
    if isinstance(supersedes, tuple):
        number, revision = cast("tuple[str, str]", supersedes)
    raw_equipment = values.get("equipment")
    equipment = cast("tuple[str, ...]", raw_equipment) if isinstance(raw_equipment, tuple) else ()
    return ControlledMetadataDraft(
        document_number=_str(values.get("document_number")),
        title=_str(values.get("title")),
        document_type=_of(values.get("document_type"), ControlledDocumentType),
        revision=_str(values.get("revision")),
        effective_date=_date(values.get("effective_date")),
        lifecycle_status=_of(values.get("lifecycle_status"), LifecycleStatus)
        or LifecycleStatus.UNKNOWN,
        supersedes_number=_str(number),
        supersedes_revision=_str(revision),
        site=_str(values.get("site")),
        area=_str(values.get("area")),
        equipment=equipment,
        process=_str(values.get("process")),
        product_family=_str(values.get("product_family")),
        source_authority=_of(values.get("source_authority"), SourceAuthority),
        control_status=_of(values.get("control_status"), ControlStatus),
        data_classification=_of(values.get("data_classification"), DataClassification),
        source_url=_str(values.get("source_url")),
        retrieved=_date(values.get("retrieved")),
    )


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _date(value: object) -> date | None:
    return value if isinstance(value, date) else None


def _of[EnumT](value: object, enum: type[EnumT]) -> EnumT | None:
    return value if isinstance(value, enum) else None
