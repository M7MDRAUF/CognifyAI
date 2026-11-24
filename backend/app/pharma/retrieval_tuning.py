"""Retrieval tuning candidates for pharmaceutical runs (PLV6-T03; decision PLV-D-019).

A ``RetrievalTuning`` names options on top of the workflow's retrieval; each option is one small,
pure step, and ``CURRENT`` (every option off) reproduces the workflow exactly: the same queries,
the same version resolution and the same final context as ``ranking.authority_order``.

* ``include_question``: the question joins the incident's queries (the workflow searches the
  incident's equipment, step and summary, and the planner's queries, not the question itself);
* ``query_expansion``: the controlled-terminology variants of ``terminology.expand_queries``;
* ``reference_boost``: a chunk whose text or heading names a CFR section, or whose document is a
  document the question names by number, is raised within its tier;
* ``score_first``: the order is by score, the authority tier breaking ties only;
* ``demote_generic_sections``: Purpose and Scope sections follow the other sections of their
  tier;
* ``prerequisite_co_retrieval``: prerequisite, warning and precaution sections of a selected
  procedure, when among the candidates, take up to ``MAX_CO_RETRIEVED`` places;
* ``deduplicate``: a chunk whose case- and space-folded text repeats a higher one gives up its
  place;
* ``per_document_cap``: at most that many places per document while other documents have
  candidates left;
* ``keep_unsuperseded_revisions``: an effective revision excluded only as the lower of two
  effective revisions that do not supersede each other stays eligible; its conflict is still
  reported.

``ACTIVE`` is the configuration the workflow uses; PLV6-T04 sets it from the registered
selection rule, and until then it is ``CURRENT``.
"""

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Final

from app.pharma.anchors import ControlledDocumentInfo
from app.pharma.enums import ExclusionReason, PharmaQueryIntent
from app.pharma.ranking import PROCEDURE_TYPES, UNSCORED, authority_tier
from app.pharma.terminology import TERMINOLOGY_VERSION, Expansion, expand_queries
from app.pharma.versions import VersionResolution
from app.services.retrieval_types import RetrievedChunk

TUNING_VERSION: Final = "1"
# Hybrid scores lie in 0..1, so a bonus of 1 puts a chunk naming the reference ahead of every chunk
# of its tier that does not, and keeps the order among those that do.
REFERENCE_BONUS: Final = 1.0
MAX_CO_RETRIEVED: Final = 2
_GENERIC = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:purpose|scope)\b", re.IGNORECASE)
_PREREQUISITE = re.compile(r"\b(?:prerequisite|precaution|warning|caution|safety)", re.IGNORECASE)
_CFR_SECTION = re.compile(r"(?<![\d.])(\d{3}\.\d{1,4})(?![\d.])")
_DOCUMENT_NUMBER = re.compile(r"\b[A-Z]{2,5}(?:-[A-Z0-9]{1,6}){1,5}\b")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RetrievalTuning:
    name: str = "current"
    include_question: bool = False
    query_expansion: bool = False
    reference_boost: bool = False
    score_first: bool = False
    demote_generic_sections: bool = False
    prerequisite_co_retrieval: bool = False
    deduplicate: bool = False
    per_document_cap: int | None = None
    keep_unsuperseded_revisions: bool = False
    # DV-RMD-17 (RMD3-T03); in a configuration record only when on, so every earlier digest stands.
    procedure_co_retrieval: bool = False

    def options(self) -> list[str]:
        """The options that are on, in the order the fields declare them."""
        return [
            f.name
            for f in fields(self)
            if f.name != "name" and getattr(self, f.name) not in (False, None)
        ]


CURRENT: Final = RetrievalTuning()
# The greedy chain of PLV-D-019, from the development sweep of the single options: the options that
# raised nDCG@8 without lowering effective-version selection or adding misuse, in order of their
# gain (reference_boost, include_question, score_first, keep_unsuperseded_revisions), each added to
# the chain kept so far.
GREEDY: Final[tuple[RetrievalTuning, ...]] = (
    RetrievalTuning("greedy_1", reference_boost=True),
    RetrievalTuning("greedy_2", include_question=True, reference_boost=True),
    RetrievalTuning("greedy_3", include_question=True, reference_boost=True, score_first=True),
    RetrievalTuning(
        "greedy_4",
        include_question=True,
        reference_boost=True,
        score_first=True,
        keep_unsuperseded_revisions=True,
    ),
)
# DV-RMD-16 (RMD3-T03): greedy_4 with deduplicate. Every Scope section of the synthetic corpus
# reads the same, and under greedy_4 such repeats filled up to 6 of 8 places of a final context.
GREEDY_4_DEDUPLICATE: Final = replace(GREEDY[-1], name="greedy_4_deduplicate", deduplicate=True)
# DV-RMD-17 (RMD3-T03): with procedure_co_retrieval, a procedure present only by its Purpose or
# Scope brings its steps (SC-P08's changeover instruction).
GREEDY_4_DEDUPLICATE_PROCEDURES: Final = replace(
    GREEDY_4_DEDUPLICATE, name="greedy_4_deduplicate_procedures", procedure_co_retrieval=True
)
# Every option alone on ``CURRENT`` (PLV-D-019), then the greedy chain.
CANDIDATES: Final[tuple[RetrievalTuning, ...]] = (
    CURRENT,
    RetrievalTuning("include_question", include_question=True),
    RetrievalTuning("query_expansion", query_expansion=True),
    RetrievalTuning("reference_boost", reference_boost=True),
    RetrievalTuning("score_first", score_first=True),
    RetrievalTuning("demote_generic_sections", demote_generic_sections=True),
    RetrievalTuning("prerequisite_co_retrieval", prerequisite_co_retrieval=True),
    RetrievalTuning("deduplicate", deduplicate=True),
    RetrievalTuning("per_document_cap", per_document_cap=4),
    RetrievalTuning("keep_unsuperseded_revisions", keep_unsuperseded_revisions=True),
    *GREEDY,
    GREEDY_4_DEDUPLICATE,
    GREEDY_4_DEDUPLICATE_PROCEDURES,
)
# Frozen at PLV6-T04 by the registered selection rule (PLV-D-020; DV-PLV-20): the only finalist with
# effective-version selection 100% and misuse 0 on the test split. FROZEN_SHA256 is the SHA-256 of
# its configuration record, quoted in docs/pharma-live/evaluation/retrieval-selection.md.
FROZEN_SHA256: Final = "11c2277f4132e78fe910038a5fded8bf94fbf712aeb9d0db6cd210c8240ebe0b"
# DV-RMD-16 (RMD3-T03): the active configuration is greedy_4 with deduplicate. The selection rule
# above holds for it on the same test split (effective-version selection 100%, misuse 0 of 3,
# nDCG@8 equal to greedy_4's, at USD 0: docs/real-e2e-remediation/evidence/RMD3-T03/
# retrieval-selection/), and it gives repeated boilerplate's places to other evidence. DV-RMD-17
# adds procedure_co_retrieval, under the same rule and with the same figures.
ACTIVE: Final = GREEDY_4_DEDUPLICATE_PROCEDURES
ACTIVE_SHA256: Final = "e99e27137589b18877607efbe4c0205c2c9d76c5324796a497e904bdd53489ea"


def configuration_record(tuning: RetrievalTuning) -> dict[str, object]:
    """Everything that fixes the configuration's behaviour: its options and the versions and
    constants of the steps that read them. ``procedure_co_retrieval``, added after the earlier
    digests were taken, is recorded only when on (DV-RMD-17)."""
    options = asdict(tuning)
    if not tuning.procedure_co_retrieval:
        del options["procedure_co_retrieval"]
    return {
        "tuning_version": TUNING_VERSION,
        "terminology_version": TERMINOLOGY_VERSION,
        "reference_bonus": REFERENCE_BONUS,
        "max_co_retrieved": MAX_CO_RETRIEVED,
        **options,
    }


def configuration_sha256(tuning: RetrievalTuning) -> str:
    """The SHA-256 of the configuration record as canonical JSON (sorted keys, no spaces)."""
    text = json.dumps(configuration_record(tuning), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tuned_queries(
    queries: Sequence[str], question: str, tuning: RetrievalTuning
) -> tuple[list[str], Expansion | None]:
    """The queries to retrieve with, and the expansion's record when the expansion ran."""
    out = list(queries)
    known = {query.casefold() for query in out}
    if tuning.include_question and question.strip() and question.casefold() not in known:
        out.append(question)
    if not tuning.query_expansion:
        return out, None
    expansion = expand_queries(out)
    return expansion.queries, expansion


def tuned_resolution(
    candidates: Sequence[RetrievedChunk],
    resolution: VersionResolution[RetrievedChunk],
    tuning: RetrievalTuning,
) -> VersionResolution[RetrievedChunk]:
    """``resolution``, with the unsuperseded lower effective revisions eligible again when the
    tuning keeps them (in the candidates' order)."""
    if not tuning.keep_unsuperseded_revisions:
        return resolution
    readmitted = {
        item.chunk.chunk_id
        for item in resolution.excluded
        if item.reason is ExclusionReason.LOWER_REVISION
    }
    if not readmitted:
        return resolution
    kept = {chunk.chunk_id for chunk in resolution.eligible} | readmitted
    return VersionResolution(
        [chunk for chunk in candidates if chunk.chunk_id in kept],
        [item for item in resolution.excluded if item.chunk.chunk_id not in readmitted],
        resolution.conflicts,
    )


def _references(
    question: str, infos: Mapping[uuid.UUID, ControlledDocumentInfo]
) -> tuple[set[str], set[str]]:
    """The CFR sections the question cites, and the project's document numbers it names."""
    sections = set(_CFR_SECTION.findall(question))
    numbers = {info.document_number for info in infos.values() if info.document_number}
    named = {number for number in _DOCUMENT_NUMBER.findall(question) if number in numbers}
    return sections, named


def _names(
    chunk: RetrievedChunk,
    info: ControlledDocumentInfo | None,
    references: tuple[set[str], set[str]],
) -> bool:
    sections, numbers = references
    if info is not None and info.document_number in numbers:
        return True
    text = f"{chunk.section_title or ''} {chunk.text}"
    return any(re.search(rf"(?<![\d.]){re.escape(s)}(?!\d)", text) for s in sections)


def _fold(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _deduplicated(ordered: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    kept: list[RetrievedChunk] = []
    for chunk in ordered:
        folded = _fold(chunk.text)
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(chunk)
    return kept


def _capped(ordered: Sequence[RetrievedChunk], cap: int) -> list[RetrievedChunk]:
    """At most ``cap`` chunks per document first; the rest after, in their order, so they fill
    places only when no other document has candidates left."""
    counts: dict[uuid.UUID, int] = {}
    first: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    for chunk in ordered:
        count = counts.get(chunk.document_id, 0)
        (first if count < cap else overflow).append(chunk)
        counts[chunk.document_id] = count + 1
    return first + overflow


def _with_prerequisites(
    context: list[RetrievedChunk],
    rest: Sequence[RetrievedChunk],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    k: int,
) -> list[RetrievedChunk]:
    procedures = {
        chunk.document_id
        for chunk in context
        if (info := infos.get(chunk.document_id)) is not None
        and info.document_type in PROCEDURE_TYPES
    }
    extra = [
        chunk
        for chunk in rest
        if chunk.document_id in procedures and _PREREQUISITE.search(chunk.section_title or "")
    ][:MAX_CO_RETRIEVED]
    if not extra:
        return context
    room = max(0, k - len(extra))
    return context[:room] + extra


def _with_procedures(
    context: list[RetrievedChunk],
    rest: Sequence[RetrievedChunk],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
) -> list[RetrievedChunk]:
    """DV-RMD-17: an SOP or work instruction in the context only by its generic sections (Purpose,
    Scope), which name what it covers, brings its first other candidate section, in the order the
    documents first appear, up to ``MAX_CO_RETRIEVED`` sections. Each takes the place of the
    lowest-ranked chunk that is a generic section or comes from outside the procedure types; a
    procedure's other sections and the first place never give way (taking the last places, the
    first rule pushed out the steps it exists to bring, on SC-P08), and the context keeps at most
    ``k`` chunks. With no place to give, nothing is brought."""
    generic_only: list[uuid.UUID] = []
    for chunk in context:
        info = infos.get(chunk.document_id)
        if info is None or info.document_type not in PROCEDURE_TYPES:
            continue
        if chunk.document_id not in generic_only:
            generic_only.append(chunk.document_id)
    generic_only = [
        document
        for document in generic_only
        if all(
            _GENERIC.match(chunk.section_title or "")
            for chunk in context
            if chunk.document_id == document
        )
    ]
    extra: list[RetrievedChunk] = []
    for document in generic_only:
        section = next(
            (
                chunk
                for chunk in rest
                if chunk.document_id == document and not _GENERIC.match(chunk.section_title or "")
            ),
            None,
        )
        if section is not None:
            extra.append(section)
        if len(extra) == MAX_CO_RETRIEVED:
            break

    def yields(chunk: RetrievedChunk) -> bool:
        info = infos.get(chunk.document_id)
        if info is None or info.document_type not in PROCEDURE_TYPES:
            return True
        return bool(_GENERIC.match(chunk.section_title or ""))

    places = [index for index in range(len(context) - 1, 0, -1) if yields(context[index])]
    given = set(places[: len(extra)])
    kept = [chunk for index, chunk in enumerate(context) if index not in given]
    return kept + extra[: len(given)]


def refine_context(
    ranking: Sequence[RetrievedChunk],
    infos: Mapping[uuid.UUID, ControlledDocumentInfo],
    intent: PharmaQueryIntent,
    k: int,
    tuning: RetrievalTuning,
    *,
    question: str = "",
) -> list[RetrievedChunk]:
    """The final context of ``k`` chunks with dense ranks ``0..K-1``; under ``CURRENT`` it is
    ``authority_order(ranking, infos, intent, k)``."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    references = (
        _references(question, infos) if tuning.reference_boost else (set[str](), set[str]())
    )

    def key(
        chunk: RetrievedChunk,
    ) -> tuple[float, int, int, float, str] | tuple[float, int, float, int, str]:
        info = infos.get(chunk.document_id)
        score = chunk.rerank_score if chunk.rerank_score is not None else UNSCORED
        tier = authority_tier(info, intent)
        bonus = (
            REFERENCE_BONUS if tuning.reference_boost and _names(chunk, info, references) else 0.0
        )
        generic = int(
            tuning.demote_generic_sections and bool(_GENERIC.match(chunk.section_title or ""))
        )
        hybrid = -(chunk.hybrid_score + bonus)
        if tuning.score_first:
            return (-score, generic, hybrid, tier, str(chunk.chunk_id))
        return (-score, tier, generic, hybrid, str(chunk.chunk_id))

    ordered = sorted(ranking, key=key)
    if tuning.deduplicate:
        ordered = _deduplicated(ordered)
    if tuning.per_document_cap is not None:
        ordered = _capped(ordered, tuning.per_document_cap)
    context = ordered[:k]
    if tuning.prerequisite_co_retrieval:
        context = _with_prerequisites(context, ordered[k:], infos, k)
    if tuning.procedure_co_retrieval:
        context = _with_procedures(context, ordered[k:], infos)
    return [replace(chunk, rank=rank) for rank, chunk in enumerate(context)]
