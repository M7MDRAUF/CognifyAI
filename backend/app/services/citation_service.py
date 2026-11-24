"""Citation validation (FR-CITE-002).

A citation is kept only if it references a chunk that was actually retrieved
for this query. Snippets are repaired against the true chunk text so the
displayed quote always matches the source. Hallucinated citations are dropped.
No raw chunk identifier, whole (L-2) or shortened (FR-013), reaches the reader: see
`conceal_raw_identifiers`.
"""

import logging
import re
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.agents.prompts import evidence_key
from app.schemas.agents import CitationDraft, FinalDecisionOutput
from app.services.retrieval_types import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_SNIPPET = 300


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class CitationReport:
    """What citation validation kept, and what it removed and why (L-1, L-2).

    Counts rather than a bare list, because the *rate* is the measurement: a run that dropped
    eleven of eleven citations and a run that dropped none both previously produced a list of
    valid citations and no way to tell them apart.
    """

    kept: int = 0
    dropped_unretrieved: int = 0
    dropped_unanchored: int = 0
    dropped_duplicate: int = 0

    @property
    def dropped(self) -> int:
        return self.dropped_unretrieved + self.dropped_unanchored + self.dropped_duplicate

    @property
    def anchor_rate(self) -> float:
        """Fraction of citations that were actually referenced by the answer text."""
        total = self.kept + self.dropped
        return 1.0 if total == 0 else self.kept / total


def validate_citations(
    drafts: Sequence[CitationDraft],
    evidence: Sequence[RetrievedChunk],
    answer_markdown: str | None = None,
) -> tuple[list[CitationDraft], CitationReport]:
    """Keep a citation only when it points at retrieved evidence **and** the answer cites it.

    Two rules, and the second is new (L-1). Chunk membership was already enforced; anchoring was
    not, so on one live run **none of eleven** citation keys appeared anywhere in
    `answer_markdown`. The citations panel listed eleven sources the answer never referred to,
    which reads to a user as eleven supporting references.

    ``answer_markdown=None`` skips the anchoring rule rather than dropping everything -- a caller
    that has no answer yet is not making the claim that nothing is cited.

    L-2 needs its own mechanism, `conceal_raw_identifiers` below: anchoring keeps a citation the
    answer references, and a model that cites *by* chunk id references it by writing the id.
    """
    by_chunk = {item.chunk_id: item for item in evidence}
    valid: list[CitationDraft] = []
    seen: set[str] = set()
    unretrieved = unanchored = duplicate = 0
    haystack = _normalize(answer_markdown) if answer_markdown is not None else None

    for draft in drafts:
        chunk = by_chunk.get(draft.chunk_id)
        if chunk is None:
            logger.warning("Dropping citation to non-retrieved chunk %s", draft.chunk_id)
            unretrieved += 1
            continue
        if draft.citation_key in seen:
            duplicate += 1
            continue
        if haystack is not None and not _is_anchored(draft.citation_key, haystack):
            logger.warning(
                "Dropping citation %s: the answer never references it",
                draft.citation_key,
                extra={"event_type": "citation_not_anchored"},
            )
            unanchored += 1
            continue
        seen.add(draft.citation_key)
        snippet = draft.snippet.strip()
        if snippet == "" or _normalize(snippet) not in _normalize(chunk.text):
            # Replace an absent/altered quote with a real excerpt from the chunk.
            snippet = chunk.text[:_MAX_SNIPPET].strip()
        valid.append(draft.model_copy(update={"snippet": snippet}))

    report = CitationReport(
        kept=len(valid),
        dropped_unretrieved=unretrieved,
        dropped_unanchored=unanchored,
        dropped_duplicate=duplicate,
    )
    if report.dropped_unanchored:
        logger.warning(
            "Answer anchored %d of %d citations (%.0f%%)",
            report.kept,
            report.kept + report.dropped,
            report.anchor_rate * 100,
            extra={"event_type": "citation_anchor_rate"},
        )
    return valid, report


def _is_anchored(citation_key: str, normalized_answer: str) -> bool:
    """Whether the answer refers to this key.

    Matched against the normalized answer so whitespace and case differences do not decide it,
    and the bare key is accepted as well as the bracketed form: a model writing "see C1" has
    cited C1 as surely as one writing "[C1]", and dropping the first would silently discard a
    real reference.
    """
    key = _normalize(citation_key)
    if not key:
        return False
    return key in normalized_answer or f"[{key}]" in normalized_answer


# --- RMD8-T01: keys decide, and no unresolved marker is shown (RE2E-DEF-031) -------------------

# A bracket, with the space before it, that is not a Markdown link's text.
_MARKER_BRACKET = re.compile(r"(?P<space>[ \t]?)\[(?P<body>[^\[\]\n]+)\](?!\()")
# The members a citation marker is made of besides a drafted key: the placeholder concealment
# writes for an unnamed identifier, and the evidence keys of a keyed block.
_EVIDENCE_KEY = re.compile(r"C\d+", re.IGNORECASE)


def _is_marker(member: str, drafted: Collection[str]) -> bool:
    word = member.strip().lower()
    return word == "source" or _EVIDENCE_KEY.fullmatch(word) is not None or word in drafted


def remove_unresolved_markers(
    text: str, kept: Collection[str], drafted: Collection[str]
) -> tuple[str, int]:
    """``text`` without the citation markers that name no kept citation, and how many it removed.

    A marker is a member of a bracket, outside a Markdown link, that is ``source`` (what
    concealment writes for an unnamed identifier), an evidence key ``C<n>``, or a drafted citation
    key. It is unresolved when no kept citation has that key, compared without case. A bracket
    left with no member is removed with the space before it; one left with some keeps those,
    joined by ", ". Any other bracket -- prose, a checkbox, a key that is kept -- is untouched.
    Removing beats showing ``[source]``: that read as a reference and led nowhere (RE2E-DEF-031).
    """
    kept_keys = {key.strip().lower() for key in kept}
    drafted_keys = {key.strip().lower() for key in drafted}
    removed = 0

    def bracket(match: re.Match[str]) -> str:
        nonlocal removed
        members = re.split(r"[,;]", match.group("body"))
        unresolved = [
            member
            for member in members
            if _is_marker(member, drafted_keys) and member.strip().lower() not in kept_keys
        ]
        if not unresolved:
            return match.group(0)
        removed += len(unresolved)
        remaining = [member.strip() for member in members if member not in unresolved]
        if not any(remaining):
            return ""
        return f"{match.group('space')}[{', '.join(part for part in remaining if part)}]"

    return _MARKER_BRACKET.sub(bracket, text), removed


def without_unresolved_markers(
    decision: FinalDecisionOutput, kept: Collection[str]
) -> tuple[FinalDecisionOutput, int]:
    """The decision with every unresolved marker removed from each reader-facing field."""
    drafted = [citation.citation_key for citation in decision.citations]
    total = 0

    def clean(text: str) -> str:
        nonlocal total
        cleaned, removed = remove_unresolved_markers(text, kept, drafted)
        total += removed
        return cleaned

    cleaned = decision.model_copy(
        update={
            "answer_markdown": clean(decision.answer_markdown),
            "recommendation": clean(decision.recommendation),
            "uncertainty": clean(decision.uncertainty),
            "contradictory_evidence": [clean(item) for item in decision.contradictory_evidence],
            "next_steps": [clean(step) for step in decision.next_steps],
        }
    )
    return cleaned, total


def resolve_keyed_citations(
    decision: FinalDecisionOutput, evidence: Sequence[RetrievedChunk]
) -> FinalDecisionOutput:
    """Citations as the evidence keys say: the key decides which passage a citation names.

    ``evidence`` is the keyed block the decision read (``format_evidence(keyed=True)``), so
    ``C<n>`` is its n-th passage. A draft whose key is one of those takes that passage's chunk id,
    whatever chunk id the model wrote beside it; a ``[C<n>]`` in the answer with no citation entry
    becomes a citation of that passage, its snippet left for ``validate_citations`` to fill. A key
    that names no passage is left alone, to be dropped or removed downstream.
    """
    by_key = {evidence_key(position).lower(): item for position, item in enumerate(evidence)}
    citations: list[CitationDraft] = []
    seen: set[str] = set()
    for draft in decision.citations:
        passage = by_key.get(draft.citation_key.strip().lower())
        citations.append(
            draft if passage is None else draft.model_copy(update={"chunk_id": passage.chunk_id})
        )
        seen.add(draft.citation_key.strip().lower())
    for match in _MARKER_BRACKET.finditer(decision.answer_markdown):
        for member in re.split(r"[,;]", match.group("body")):
            key = member.strip()
            passage = by_key.get(key.lower())
            if passage is not None and key.lower() not in seen:
                citations.append(CitationDraft(citation_key=key, chunk_id=passage.chunk_id))
                seen.add(key.lower())
    if citations == decision.citations:
        return decision
    return decision.model_copy(update={"citations": citations})


# --- L-2: no raw chunk identifier reaches the reader ------------------------------------------

# A raw identifier, bare or in the brackets a model usually wraps a citation in, so "[<id>]"
# becomes "[C1]" rather than "[[C1]]". Whitespace is absorbed only *inside* brackets: absorbing it
# around a bare id would weld the label onto the word before it.
_ID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_RAW_IDENTIFIER = re.compile(rf"\[\s*(?P<bracketed>{_ID})\s*\]|(?P<bare>{_ID})")
# What an identifier becomes when no retained citation names it: the reader still sees that a
# source was referred to, and nothing that exposes the system's internal keys.
UNLABELLED_SOURCE = "[source]"


def _identifier(match: re.Match[str]) -> str:
    return (match.group("bracketed") or match.group("bare")).lower()


# FR-013: asked for chunk ids, a model also writes the first 8 to 12 hex characters of one --
# `[e307a19c]`, several in one bracket, a bare prefix in a table cell or in backticks -- and even
# keys a citation by one. None is a whole id, so the pass above left them for the reader, and none
# is a key, so anchoring (L-1) dropped the citations they referred to. A run is read against the
# identifiers in ``labels``: as a member of a bracket it is always a reference, and becomes the key
# of the one citation whose id it starts, the key it spells, or ``UNLABELLED_SOURCE``; bare, it is
# one only when it starts exactly one labelled id, and any other hex word is left as written.
_SHORT_ID = re.compile(r"[0-9a-fA-F]{8,12}")
_BARE_SHORT_ID = re.compile(r"(?<![0-9A-Za-z_\-])[0-9a-fA-F]{8,12}(?![0-9A-Za-z_\-])")
# A bracket, or a bare run outside one, so that each is rewritten once, in a single pass.
_SHORT_REFERENCE = re.compile(rf"\[(?P<group>[^\[\]\n]+)\]|{_BARE_SHORT_ID.pattern}")


def _short_label(run: str, labels: Mapping[str, str]) -> str | None:
    """The key of the one labelled identifier that ``run`` starts, or ``None``."""
    prefix = run.lower()
    keys = {
        key for identifier, key in labels.items() if identifier.replace("-", "").startswith(prefix)
    }
    return keys.pop() if len(keys) == 1 else None


def _conceal_short_identifiers(text: str, labels: Mapping[str, str]) -> str:
    spelled = {key.lower(): key for key in labels.values()}

    def bare(match: re.Match[str]) -> str:
        key = _short_label(match.group(0), labels)
        return f"[{key}]" if key else match.group(0)

    def member(run: str) -> str:
        key = _short_label(run, labels) or spelled.get(run.lower())
        return f"[{key}]" if key else UNLABELLED_SOURCE

    def reference(match: re.Match[str]) -> str:
        group = match.group("group")
        if group is None:
            return bare(match)
        members = [part.strip() for part in re.split(r"[,;]", group)]
        if not any(_SHORT_ID.fullmatch(part) for part in members):
            # A bracket of prose, or of keys: a run inside it is read as bare.
            return f"[{_BARE_SHORT_ID.sub(bare, group)}]"
        # Each member becomes its own bracket: the answer view links a bracket only when the
        # whole of it is one citation key.
        parts: list[str] = []
        for part in members:
            rendered = member(part) if _SHORT_ID.fullmatch(part) else f"[{part}]"
            if part and rendered not in parts:
                parts.append(rendered)
        return ", ".join(parts)

    return _SHORT_REFERENCE.sub(reference, text)


def conceal_identifiers(text: str, labels: Mapping[str, str]) -> str:
    """Replace every raw identifier in reader-facing text with the citation that names it (L-2).

    ``labels`` maps a lower-case identifier -- a chunk id, or a citation key that was one -- to
    the key a reader sees. An identifier no citation names becomes ``UNLABELLED_SOURCE``. Whole
    ids are replaced first, then shortened ones (FR-013, above).
    """

    def label(match: re.Match[str]) -> str:
        key = labels.get(_identifier(match))
        return f"[{key}]" if key else UNLABELLED_SOURCE

    return _conceal_short_identifiers(_RAW_IDENTIFIER.sub(label, text), labels)


def citation_labels(citations: Iterable[tuple[uuid.UUID, str]]) -> dict[str, str]:
    """``chunk id -> citation key`` for the citations a reader sees; the first citation wins.

    A key that is itself a raw identifier is never used as a label -- it would re-expose the very
    thing being concealed. Historical rows can hold one; new rows cannot, see below.
    """
    labels: dict[str, str] = {}
    for chunk_id, key in citations:
        if not _RAW_IDENTIFIER.fullmatch(key.strip()):
            labels.setdefault(str(chunk_id).lower(), key)
    return labels


def _key_identifier(key: str, drafted: Sequence[str]) -> str | None:
    """The identifier a citation key is, if it is one: a whole id (L-2), or the start of a drafted
    chunk id (FR-013). ``None`` for an ordinary key."""
    stripped = key.strip()
    raw = _RAW_IDENTIFIER.fullmatch(stripped)
    if raw is not None:
        return _identifier(raw)
    run = stripped.lower()
    if _SHORT_ID.fullmatch(run) and any(chunk.startswith(run) for chunk in drafted):
        return run
    return None


def conceal_raw_identifiers(decision: FinalDecisionOutput) -> FinalDecisionOutput:
    """The decision as the reader will see it: no raw chunk identifier anywhere in it (L-2).

    Anchoring (L-1) was relied on to cover this and cannot: a model that cites *by* chunk id --
    as the decision agent did on a live run, forty times in thirty-three report lines -- writes
    the id into the answer, so the citation is anchored and the id is rendered. Here a citation
    keyed by a raw id is given a positional key, and every raw id in the decision's prose becomes
    the key of the citation that names that chunk, or ``UNLABELLED_SOURCE``.

    A shortened id -- the first 8 to 12 hex characters of a drafted citation's chunk id -- is
    treated the same way (FR-013). A citation keyed by one is given a positional key, as two
    certified answers keyed eight citations each. In the prose, a shortened id becomes the key of
    the one citation whose chunk it starts, so anchoring then finds the reference and keeps it.

    Run before the answer is persisted and before citations are validated, so anchoring then
    checks the text the reader will actually get. A decision with nothing to conceal is returned
    as the same object.
    """
    drafted = [str(citation.chunk_id).replace("-", "") for citation in decision.citations]
    taken = {citation.citation_key for citation in decision.citations}
    renamed: dict[str, str] = {}
    counter = 0
    citations: list[CitationDraft] = []
    for citation in decision.citations:
        key = citation.citation_key
        identifier = _key_identifier(key, drafted)
        if identifier is not None:
            if identifier not in renamed:
                counter += 1
                while f"C{counter}" in taken:
                    counter += 1
                renamed[identifier] = f"C{counter}"
                taken.add(f"C{counter}")
            key = renamed[identifier]
        citations.append(citation.model_copy(update={"citation_key": key}))
    labels = {**citation_labels((c.chunk_id, c.citation_key) for c in citations), **renamed}
    answer = conceal_identifiers(decision.answer_markdown, labels)
    recommendation = conceal_identifiers(decision.recommendation, labels)
    contradictory = [conceal_identifiers(item, labels) for item in decision.contradictory_evidence]
    uncertainty = conceal_identifiers(decision.uncertainty, labels)
    next_steps = [conceal_identifiers(step, labels) for step in decision.next_steps]
    if not renamed and (answer, recommendation, contradictory, uncertainty, next_steps) == (
        decision.answer_markdown,
        decision.recommendation,
        decision.contradictory_evidence,
        decision.uncertainty,
        decision.next_steps,
    ):
        return decision
    return decision.model_copy(
        update={
            "citations": citations,
            "answer_markdown": answer,
            "recommendation": recommendation,
            "contradictory_evidence": contradictory,
            "uncertainty": uncertainty,
            "next_steps": next_steps,
        }
    )
