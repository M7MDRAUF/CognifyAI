"""Markdown decision-report generation (PRD section 8.8).

Composes the persisted query result — answer, citations, evaluation, agent
trace, and the risk/contradiction detail from the agents' stored output — into
a single Markdown document with all ten required sections.
"""

import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ValidationError

from app.core.errors import (
    ConflictError,
    ProjectNotFoundError,
    QueryRunNotFoundError,
    ReportNotFoundError,
)
from app.domain.enums import AgentName, DomainProfile, QueryRunStatus
from app.domain.models import (
    AgentRun,
    Citation,
    EvaluationResult,
    PharmaRunRecord,
    QueryRun,
    Report,
)
from app.pharma.response import MODEL_REPORTED_LABEL
from app.repositories.project_repository import ProjectRepository
from app.repositories.query_repository import QueryRepository
from app.repositories.report_repository import ReportRepository
from app.schemas.agents import FinalDecisionOutput, RiskOutput
from app.schemas.pharma_api import PharmaResponse
from app.schemas.queries import EVALUATION_EXPLANATION_WITHHELD
from app.schemas.reports import ReportPage, ReportSummary
from app.services.answer_hygiene import CONTEXT_NOTICE, strip_context_lines, with_short_notice
from app.services.citation_service import citation_labels, conceal_identifiers
from app.services.interfaces import SupportsCommit
from app.services.pagination import decode_cursor, encode_cursor
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# T-23: what a pharmaceutical run's report says about itself, in every such report.
NOT_A_CONTROLLED_RECORD = (
    "This report is not a controlled record. It is decision support for review by qualified "
    "personnel: it does not approve a procedure, a deviation or a CAPA, release a batch, "
    "determine a root cause or establish compliance."
)


class PharmaRunRecords(Protocol):
    async def run_record(self, query_run_id: uuid.UUID) -> PharmaRunRecord | None: ...


class ReportService:
    def __init__(
        self,
        session: SupportsCommit,
        queries: QueryRepository,
        reports: ReportRepository,
        *,
        projects: ProjectRepository,
        pharma: PharmaRunRecords | None = None,
    ) -> None:
        self._session = session
        self._queries = queries
        self._reports = reports
        self._projects = projects
        # The pharmaceutical run records (PDP8-T04). None where none are read; a pharmaceutical
        # run's report then shows nothing from the model rather than its agent rows.
        self._pharma = pharma

    async def generate(self, query_run_id: uuid.UUID) -> Report:
        query_run = await self._queries.get(query_run_id)
        if query_run is None:
            raise QueryRunNotFoundError(f"Query run {query_run_id} does not exist.")
        if query_run.status is not QueryRunStatus.COMPLETED:
            raise ConflictError("A report can only be generated for a completed query.")
        citations = await self._queries.list_citations(query_run_id)
        agent_runs = await self._queries.list_agent_runs(query_run_id)
        evaluation = await self._queries.get_evaluation(query_run_id)
        checked = await self._pharma_sections(query_run)
        markdown = render_report_markdown(
            query_run, citations, agent_runs, evaluation, checked=checked
        )
        report = Report(
            id=new_uuid(),
            query_run_id=query_run_id,
            title=_title(query_run),
            markdown_content=markdown,
            created_at=utc_now(),
        )
        await self._reports.add(report)
        await self._session.commit()
        return report

    async def _pharma_sections(self, query_run: QueryRun) -> "Rendered[PharmaResponse] | None":
        """None for a General run; for a pharmaceutical one, its checked response or why there is
        none (PH-DEF-006). Chosen by the project's profile, not by whether a record exists, so a
        pharmaceutical run without one never falls through to its agent rows."""
        project = await self._projects.get(query_run.project_id)
        if project is None or project.domain_profile is not DomainProfile.PHARMA_MANUFACTURING:
            return None
        record = None if self._pharma is None else await self._pharma.run_record(query_run.id)
        return _pharma_sections_of(record)

    async def get(self, report_id: uuid.UUID) -> Report:
        report = await self._reports.get(report_id)
        if report is None:
            raise ReportNotFoundError(f"Report {report_id} does not exist.")
        return report

    async def list_for_project(
        self, project_id: uuid.UUID, *, limit: int, cursor: str | None = None
    ) -> ReportPage:
        """One page of the project's reports (CT-01): the project check, the page and the count.

        The page is fetched one row long to detect the next page; the next cursor names the last
        report returned.
        """
        if await self._projects.get(project_id) is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
        before = None if cursor is None else decode_cursor(cursor)
        rows = await self._reports.list_for_project(project_id, limit=limit, before=before)
        page = rows[:limit]
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
        total = await self._reports.count_for_project(project_id)
        return ReportPage(
            items=[ReportSummary.model_validate(row) for row in page],
            next_cursor=next_cursor,
            total=total,
        )


def _title(query_run: QueryRun) -> str:
    question = query_run.normalized_question or query_run.question
    return f"Decision Report: {question[:120]}"


def render_report_markdown(
    query_run: QueryRun,
    citations: Sequence[Citation],
    agent_runs: Sequence[AgentRun],
    evaluation: EvaluationResult | None,
    *,
    checked: "Rendered[PharmaResponse] | None" = None,
) -> str:
    """``checked`` is None for a General run. For a pharmaceutical run it is the policy-checked
    response, or why there is none, and the three model-text sections read it instead of the
    agent rows (PH-DEF-006); the report also states it is not a controlled record (T-23)."""
    # L-2: the answer is concealed before it is persisted, but the two sections below read the
    # decision agent's own output from its trace row, and a report generated for a run persisted
    # before concealment existed reads a raw answer too. Concealed here as well, where it renders.
    labels = citation_labels((citation.chunk_id, citation.citation_key) for citation in citations)
    answer = query_run.answer_markdown
    sections: list[str] = [
        f"# {_title(query_run)}",
        *([] if checked is None else [f"> {NOT_A_CONTROLLED_RECORD}"]),
        "## User Question\n\n" + query_run.question,
        "## Executive Summary\n\n" + _executive_summary(query_run, evaluation),
        "## Final Recommendation\n\n"
        + _demote_headings(
            _reader_answer(answer, labels) if answer else "_No answer was produced._"
        ),
        "## Evidence Table\n\n" + _evidence_table(citations),
        "## Contradictory Evidence\n\n" + _contradictions(agent_runs, labels, checked),
        "## Risk Matrix\n\n" + _risk_matrix(agent_runs, labels, checked),
        "## Uncertainty and Next Steps\n\n" + _uncertainty(agent_runs, labels, checked),
        "## Evaluation Metrics\n\n"
        + _evaluation_metrics(evaluation, labels, withheld=checked is not None),
        "## Agent Trace Summary\n\n" + _agent_trace(agent_runs),
        "## Appendix: Citations\n\n" + _citation_appendix(citations),
    ]
    return "\n\n".join(sections) + "\n"


def _reader_answer(answer: str, labels: Mapping[str, str]) -> str:
    """The stored answer as a reader sees it. One stored before RMD8-T02 may still repeat lines
    of its evidence context, and loses them here with the notice the filter writes (DV-RMD-57)."""
    stripped, lines = strip_context_lines(conceal_identifiers(answer, labels))
    if not lines:
        return stripped
    return f"{stripped}\n\n{CONTEXT_NOTICE}" if stripped else CONTEXT_NOTICE


def _reader_text(text: str, labels: Mapping[str, str]) -> str:
    """A General decision or risk text read from its trace row, as the detail reads it: concealed
    (L-2), and without the evidence context lines the model copied into it, with a short notice
    (RMD8-T02, DV-RMD-57)."""
    return with_short_notice(conceal_identifiers(text, labels))[0]


def _executive_summary(query_run: QueryRun, evaluation: EvaluationResult | None) -> str:
    parts = [f"Latency: {query_run.latency_ms} ms."]
    if evaluation is not None:
        parts.append(f"Faithfulness: {evaluation.faithfulness:.2f}.")
        parts.append(f"Citation accuracy: {evaluation.citation_accuracy:.2f}.")
    return " ".join(parts)


# --- containment -----------------------------------------------------------------------------
#
# Same invariant as _as_blockquote below, applied to the values that are not snippets: a report
# assembled out of document text must not let the documents it cites decide its structure. The
# interpolation site is the last point where that structure is still the code's decision, so the
# containment belongs there -- one helper per kind of place a value can land, and every one of
# them applied.

# Every pattern here is the CommonMark grammar the renderer actually implements (react-markdown
# with remark-gfm, frontend/src/components/ui/markdown.tsx). Where this scanner and that grammar
# disagree, the disagreement is the bug: reading a heading the renderer does not is a rewrite of
# the model's text, and missing one the renderer does emit is a forged report section.

# An ATX heading is one to six "#" at an indent of at most three spaces, followed by whitespace or
# nothing (CommonMark 4.2) -- so "#tag" is not one. Only levels 1 and 2 are demoted: those are the
# levels that open a sibling of the report's own "# " title and "## " sections, and level 3 and
# deeper are already subordinate to the section they sit in. Deeper levels are still *matched*,
# because a heading of any level closes the paragraph that a setext underline would attach to.
_ATX_HEADING = re.compile(r"^(?P<indent> {0,3})(?P<hashes>#{1,6})(?P<rest>[ \t].*|)$")

# A fence is a run of at least three backticks or tildes, indented at most three spaces, and the
# rest of the line is the info string (CommonMark 4.5). The rules that decide whether a given line
# opens or closes a block live in the two predicates below, because both directions were wrong and
# both cost something different: see _opens_fence and _closes_fence.
_CODE_FENCE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$")

# A setext underline is a run of "=" (level 1) or "-" (level 2) alone on its line (CommonMark 4.3).
# Unlike every other pattern here it cannot be judged from its own line: it is a heading only where
# it underlines a paragraph, and after anything else the same "---" is a thematic break or a table
# delimiter. That is what the block state in _demote_headings is for.
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")

# The lines that end a paragraph, or that open a block whose inner indentation this scanner does
# not track. A thematic break needs its own pattern because "***" and "- - -" are breaks that no
# other pattern here matches; the container pattern covers block quotes and list items. Raw HTML
# blocks are deliberately in neither: they have seven start conditions and their own end
# conditions (CommonMark 4.6), and lumping them in with the containers on a leading "<" is what
# let two forgeries through -- see _opens_html_block.
_THEMATIC_BREAK = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
_CONTAINER_START = re.compile(r"^ {0,3}(?:>|(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$))")
# Where a paragraph is already open, most of those do not start anything: a list interrupts a
# paragraph only if the item has content and, when ordered, numbers from 1 (CommonMark 5.2), so
# "2. item" under a line of prose is simply more of that prose -- and a "---" below it underlines
# the pair. Block quotes are here because they do interrupt one.
_INTERRUPTS_PARAGRAPH = re.compile(r"^ {0,3}(?:>|(?:[-+*]|1[.)])[ \t]+\S)")
# Four spaces or a tab is an indented code block wherever a paragraph is not already open. It
# needs no state of its own: every pattern above requires an indent of at most three spaces, so
# its lines fall through untouched, and it ends at the first line that does not have the indent.
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)")

# The markers a line inside a container carries, matched one at a time so that nesting in either
# order -- "> - " as readily as "- > " -- is followed through to the content behind it. Stripping
# them is how a heading written inside a block quote or a list item is recognised as the heading
# the renderer emits: the container nests it, but it is still a level-two heading, and the report
# is styled with Tailwind's "prose", which sizes an h2 the same wherever it sits.
#
# The gap after a list marker is capped at four, because five spaces is not a wider gap -- the
# item's content starts one space in and the rest is an indented code block (CommonMark 5.2), so
# "-     ## x" is code and demoting it would be a rewrite. A block quote marker takes at most one
# space of its own; any further indent belongs to the content and is left on it, where the
# at-most-three-spaces rule in _ATX_HEADING judges it.
_BLOCK_QUOTE_MARKER = re.compile(r"^ {0,3}>[ \t]?")
_LIST_MARKER = re.compile(
    r"^ {0,3}(?:[-+*]|(?P<number>\d{1,9})[.)])(?:(?P<gap>[ \t]{1,4})(?=[^ \t])|[ \t]*$)"
)

# CommonMark 4.6 type 6: the tag names whose block form ends only at a blank line.
_HTML_BLOCK_NAMES = frozenset(
    (
        "address", "article", "aside", "base", "basefont", "blockquote", "body", "caption",
        "center", "col", "colgroup", "dd", "details", "dialog", "dir", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "frame", "frameset", "h1", "h2",
        "h3", "h4", "h5", "h6", "head", "header", "hr", "html", "iframe", "legend", "li", "link",
        "main", "menu", "menuitem", "nav", "noframes", "ol", "optgroup", "option", "p", "param",
        "search", "section", "summary", "table", "tbody", "td", "tfoot", "th", "thead", "title",
        "tr", "track", "ul",
    )
)  # fmt: skip
# CommonMark 4.6 type 1: the tag names whose content is raw text up to the matching close tag.
_HTML_RAW_NAMES = ("script", "pre", "style", "textarea")
_HTML_RAW_CLOSERS = ("</script>", "</pre>", "</style>", "</textarea>")
_HTML_TAG_NAME = r"[A-Za-z][A-Za-z0-9-]*"
_HTML_ATTRIBUTE = (
    r"""(?:[ \t]+[a-zA-Z_:][a-zA-Z0-9_.:-]*"""
    r"""(?:[ \t]*=[ \t]*(?:[^ \t"'=<>`]+|'[^']*'|"[^"]*"))?)"""
)
# Type 7: one complete tag and nothing else on the line. It is the only start condition that
# cannot interrupt a paragraph, which is why the caller has to say whether one is open.
_HTML_COMPLETE_TAG = re.compile(
    rf"^ {{0,3}}(?:<{_HTML_TAG_NAME}{_HTML_ATTRIBUTE}*[ \t]*/?>|</{_HTML_TAG_NAME}[ \t]*>)[ \t]*$"
)
_HTML_DECLARATION = re.compile(r"^ {0,3}<![A-Za-z]")
_HTML_BLOCK_TAG = re.compile(rf"^ {{0,3}}</?(?P<name>{_HTML_TAG_NAME})(?=[ \t>]|/>|$)")


def _one_line(value: str) -> str:
    """Flatten an untrusted value so it cannot leave the line the report put it on.

    Every structural break in Markdown needs a line of its own: a heading, a thematic rule, the
    end of a table row, the end of a list. Collapsing whitespace runs to single spaces denies the
    value that line, which is the whole of the control for a value emitted inside running text.
    """
    return " ".join(value.split())


def _as_cell(value: str) -> str:
    r"""Contain an untrusted value inside one GFM table cell.

    Neither grammar break here is hypothetical. ``source_filename`` is the uploaded name exactly
    as the uploader typed it -- ``document_service.py`` stores the raw value and only the on-disk
    copy is sanitized -- and ``citation_key`` and ``mitigation`` are unconstrained model strings.

    A raw ``|`` re-cuts the row. GFM splits a row on unescaped pipes and discards the cells past
    the header's column count, so a filename carrying three pipes pushes the page number and
    relevance score the system computed off the end of the row and leaves the uploader's own
    numbers rendered in their place -- in the very table the report uses to evidence provenance.
    A raw newline is worse: it ends the row and the table, and everything after it is emitted as
    ordinary report body, which is how a ``## `` inside a cell opens a forged report section.

    Backslashes are doubled before pipes are escaped. Without that, ``a\|b`` would become
    ``a\\|b``, whose ``\\`` is a literal backslash and whose ``|`` still cuts the row.
    """
    return _no_inline_markup(_one_line(value)).replace("|", "\\|")


#: The inline constructs worth denying an untrusted value, and only those.
#:
#: Structural containment -- escaped pipes, flattened lines, demoted headings -- leaves INLINE
#: Markdown untouched, which browser verification demonstrated live: a filename of
#: ``[Invoice - paid in full](https://attacker.example/x).pdf`` renders as a clickable
#: attacker-controlled link inside a report a human forwards, and ``![x](https://...)`` makes the
#: reader's browser fetch the attacker's host on open. Neither is XSS -- the renderer escapes raw
#: HTML and neutralises ``javascript:`` -- but both are real in a document meant to be shared.
#:
#: Narrow on purpose. ``*`` and ``_`` are left alone: emphasis in a filename is cosmetic, and
#: escaping them would put backslashes through every ordinary ``my_report_v2.pdf`` in any reader
#: that is not GFM. ``(`` and ``)`` are left alone too -- a link needs its ``[...]`` half, so
#: escaping the brackets already breaks it, and parentheses are far too common in real
#: filenames (``Notes Final (v2).txt``) to disfigure for no gain. What is escaped is what
#: fabricates a LINK, an IMAGE, raw CODE or raw HTML.
_INLINE_MARKUP = ("\\", "[", "]", "!", "`", "<", ">")


def _no_inline_markup(value: str) -> str:
    """Escape the inline constructs that would turn an untrusted value into markup.

    The backslash goes first, so an escape this function adds cannot itself be escaped by a
    backslash the value already carried -- the same ordering the pipe escaping needs, and for
    the same reason.
    """
    for character in _INLINE_MARKUP:
        value = value.replace(character, "\\" + character)
    return value


def _opens_fence(line: str) -> str | None:
    """The fence marker a line opens a code block with, or None if it opens none.

    The two markers do not take the same info string. A tilde fence's may hold anything, but a
    backtick fence's may not hold a backtick (CommonMark 4.5), so ```` ```a`b ```` opens nothing
    at all -- the renderer reads it as an ordinary paragraph. Matching the backtick run alone and
    calling that a fence is the whole of what made the demotion skippable: the scanner believed
    it was inside code and stepped over the ``## `` beneath, while the renderer, agreeing with no
    one, emitted that line as a real second-level heading of the report.
    """
    match = _CODE_FENCE.match(line)
    if match is None:
        return None
    marker = match.group("marker")
    if marker[0] == "`" and "`" in match.group("info"):
        return None
    return marker


def _closes_fence(line: str, fence: str) -> bool:
    """Whether a line closes the open ``fence``.

    A closing fence repeats the opening character at least as many times and carries no info
    string (CommonMark 4.5). A second ``` ```python ``` inside a Python block is therefore code
    content and the block runs on. Reading it as a close was the mirror of the miss above and
    cost the other thing: it handed the renderer's code text back to the demotion, which rewrote
    a ``## `` the model had deliberately written *inside* a code block.
    """
    match = _CODE_FENCE.match(line)
    if match is None:
        return False
    marker = match.group("marker")
    return marker[0] == fence[0] and len(marker) >= len(fence) and not match.group("info").strip()


def _opens_html_block(line: str, *, in_paragraph: bool) -> int | None:
    """The kind of raw HTML block a line opens, or None if it opens none (CommonMark 4.6).

    This exists for two forged headings, not for tidiness. Treating every ``<`` line as one more
    container -- which is what ``_CONTAINER_START`` used to do -- got the grammar wrong in both
    directions, and both were live:

    * ``Proceed.`` over ``<foo bar=1>`` over ``Forged Section`` over ``===`` renders as a single
      level-ONE heading. A type-7 start is the one kind that cannot interrupt a paragraph, so the
      renderer keeps the paragraph open and lets the ``===`` underline all three lines. The scan
      believed the ``<`` had closed the paragraph and never looked at the underline.
    * ``<div>`` over ```` ``` ```` over ``</div>``, a blank line, then ``## Forged Section``
      renders that heading at level two. The three lines are one raw HTML block -- type 6, ended
      by the blank line -- so the ```` ``` ```` inside it is HTML text and not a fence. The scan
      opened a fence on it that never closed and skipped the rest of the document, which is the
      "raw HTML block containing a fence-looking line" residual turning out to be a forgery
      rather than the fidelity note it was recorded as.

    Knowing where such a block *ends* is what closes both, and the seven kinds do not end alike,
    so the kind is returned rather than a bool: 1 to 5 run to a closing string, 6 and 7 to the
    next blank line.
    """
    indent = len(line) - len(line.lstrip(" "))
    if indent > 3 or not line[indent:].startswith("<"):
        return None
    rest = line[indent:]
    lowered = rest.lower()
    for name in _HTML_RAW_NAMES:
        after = lowered[len(name) + 1 : len(name) + 2]
        if lowered.startswith(f"<{name}") and after in ("", " ", "\t", ">"):
            return 1
    if rest.startswith("<!--"):
        return 2
    if rest.startswith("<?"):
        return 3
    if rest.startswith("<![CDATA["):
        return 5
    if _HTML_DECLARATION.match(rest) is not None:
        return 4
    tag = _HTML_BLOCK_TAG.match(rest)
    if tag is not None and tag.group("name").lower() in _HTML_BLOCK_NAMES:
        return 6
    if not in_paragraph and _HTML_COMPLETE_TAG.match(rest) is not None:
        return 7
    return None


def _closes_html_block(line: str, kind: int) -> bool:
    """Whether a line meets the end condition of an open HTML block of ``kind`` 1 to 5.

    The condition may be met on the line that opened the block -- ``<!-- x -->`` is a whole
    comment and ``<!x>`` a whole declaration -- so the caller checks the opening line too. Kinds
    6 and 7 are not here: they end at a blank line, which the caller sees directly.
    """
    lowered = line.lower()
    if kind == 1:
        return any(closer in lowered for closer in _HTML_RAW_CLOSERS)
    if kind == 2:
        return "-->" in line
    if kind == 3:
        return "?>" in line
    if kind == 4:
        return ">" in line
    return "]]>" in line


def _table_header_cells(line: str) -> int | None:
    r"""The cells GFM reads in a line taken as a table header row, or None if it cannot be one.

    A transcription of the head-row half of ``micromark-extension-gfm-table``, the tokenizer
    behind the renderer this report is read in, because the cell count is the whole of what
    decides whether the line below is a delimiter row. A row must carry more than a lone pipe, an
    escaped ``\|`` stays inside its cell, and neither emphasis nor a code span protects a pipe
    from splitting the row.
    """
    content = line.lstrip(" \t")
    cells = 0
    tokens = 0
    expecting = False
    position = 0
    if content[:1] not in ("", "|"):
        expecting = True
        tokens += 1
    while position < len(content):
        if content[position] in " \t":
            position += 1
            continue
        tokens += 1
        if expecting:
            expecting = False
            cells += 1
        if content[position] == "|":
            expecting = True
            position += 1
            continue
        while position < len(content):
            if content[position] == "\\" and content[position + 1 : position + 2] in ("\\", "|"):
                position += 2
                continue
            if content[position] in "| \t":
                break
            position += 1
    return cells if tokens > 1 else None


def _table_delimiter_cells(line: str) -> int | None:
    """The cells GFM reads in a line taken as a table delimiter row, or None if it cannot be one.

    The other half of the same tokenizer. Every cell is a run of hyphens with an optional colon at
    either end, and the row as a whole must carry at least one ``:`` or ``|`` -- which is why a
    bare ``---`` under a paragraph is a setext underline and never a one-column table, and why
    ``:-`` carrying no pipe at all *is* one.
    """
    content = line.lstrip(" \t")
    if len(line) - len(content) > 4:
        return None
    cells = 0
    marked = False
    position = 0
    before_cell = True
    while True:
        if before_cell:
            if content[position : position + 1] == "|":
                marked = True
                position += 1
                while content[position : position + 1] in (" ", "\t"):
                    position += 1
            elif content[position : position + 1] not in ("-", ":"):
                return None
            head = content[position : position + 1]
            if head == ":":
                cells += 1
                marked = True
                position += 1
                if content[position : position + 1] != "-":
                    return None
            elif head == "-":
                cells += 1
            elif head == "":
                return cells if marked else None
            else:
                return None
            while content[position : position + 1] == "-":
                position += 1
            if content[position : position + 1] == ":":
                marked = True
                position += 1
            while content[position : position + 1] in (" ", "\t"):
                position += 1
            before_cell = False
        else:
            head = content[position : position + 1]
            if head == "|":
                before_cell = True
            elif head == "":
                return cells if marked else None
            else:
                return None


def _forms_a_table(header: str, delimiter: str) -> bool:
    """Whether two lines are a GFM table head, which is the one thing here that ends a paragraph.

    GFM forms a table only where the delimiter row has the same number of cells as the header row
    above it. The scan used to end the paragraph on any line carrying a pipe and a dash, which is
    where it and the grammar parted company: ``a`` over ``| --- | --- |`` is one cell over two,
    so the renderer forms no table, keeps the paragraph open, and lets a ``===`` two lines down
    underline the whole of it into a level-ONE heading -- a sibling of the report's own title,
    and worse than the level-two forgery the demotion was written to stop.
    """
    cells = _table_header_cells(header)
    return cells is not None and _table_delimiter_cells(delimiter) == cells


def _skip_definition_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\n":
        position += 1
    return position


def _label_end(text: str, position: int) -> int | None:
    """The index just past a link label, or None if one does not begin at ``position``."""
    if text[position : position + 1] != "[":
        return None
    position += 1
    size = 0
    seen = False
    while position < len(text) and size <= 999:
        character = text[position]
        if character == "[":
            return None
        if character == "]":
            return position + 1 if seen else None
        if character == "\n":
            position += 1
            continue
        if character == "\\" and text[position + 1 : position + 2] in ("[", "]", "\\"):
            position, size, seen = position + 2, size + 2, True
            continue
        seen = seen or character not in " \t"
        position, size = position + 1, size + 1
    return None


def _destination_end(text: str, position: int) -> int | None:
    """The index just past a link destination, or None if one does not begin at ``position``."""
    if text[position : position + 1] == "<":
        position += 1
        while position < len(text):
            character = text[position]
            if character == ">":
                return position + 1
            if character in "<\n":
                return None
            if character == "\\" and text[position + 1 : position + 2] in ("<", ">", "\\"):
                position += 2
                continue
            position += 1
        return None
    start = position
    balance = 0
    while position < len(text):
        character = text[position]
        if character == "\\" and text[position + 1 : position + 2] in ("(", ")", "\\"):
            position += 2
            continue
        if character == "(":
            balance += 1
        elif character == ")":
            if balance == 0:
                break
            balance -= 1
        elif character in " \t\n" or ord(character) < 32 or ord(character) == 127:
            break
        position += 1
    if balance or position == start:
        return None
    return position


def _title_end(text: str, position: int) -> int | None:
    """The index just past a link title and the blank rest of its line, or None if there is none.

    A title has to be preceded by whitespace and followed by nothing else on its line. Both are
    checked here rather than by the caller because a title that parses and is then followed by
    junk does not fall back to a definition without a title: the whole definition fails.
    """
    if text[position : position + 1] not in (" ", "\t", "\n"):
        return None
    position = _skip_definition_whitespace(text, position)
    opener = text[position : position + 1]
    if opener not in ('"', "'", "("):
        return None
    closer = ")" if opener == "(" else opener
    position += 1
    while position < len(text):
        character = text[position]
        if character == closer:
            position += 1
            break
        if character == "\\" and text[position + 1 : position + 2] in (closer, "\\"):
            position += 2
            continue
        position += 1
    else:
        return None
    while text[position : position + 1] in (" ", "\t"):
        position += 1
    return position if text[position : position + 1] in ("", "\n") else None


def _interrupts_a_paragraph(line: str) -> bool:
    """Whether a line opens a block that cuts an open paragraph off above it.

    Only asked of the lines *below* a link reference definition's label, and the answer decides
    how far that definition may reach. Block structure is settled before definitions are lifted
    out of the content, so ``[x]:`` over ``<pre>`` is not a definition whose destination is
    ``<pre>``: it is a one-line paragraph and a raw HTML block, and everything after it is HTML.

    A setext underline is on the list for the same reason and is the sharper case: ``[x]:`` over
    ``===`` looks like a definition whose destination is ``===`` and is nothing of the kind. The
    underline closes the content first, so what the renderer emits is a level-ONE heading reading
    ``[x]:`` -- and a label the model never chose is a heading the report never wrote.
    """
    return (
        _ATX_HEADING.match(line) is not None
        or _SETEXT_UNDERLINE.match(line) is not None
        or _THEMATIC_BREAK.match(line) is not None
        or _INTERRUPTS_PARAGRAPH.match(line) is not None
        or _opens_fence(line) is not None
        or _opens_html_block(line, in_paragraph=True) is not None
    )


def _block_ends(lines: Sequence[str]) -> list[int]:
    """For each index, the first line at or after it that ends an open paragraph.

    Computed once per scan in a single backward pass. ``_definition_span`` used to derive this
    per call by walking forward to the end of the block, which is the same answer every time
    because it depends only on line content -- so N consecutive definitions cost O(N^2)
    comparisons and N joins of O(N) text. Measured before this was hoisted: 20 KB of
    definitions took 3.75s and 100 KB took roughly 90s, on the event loop, reachable from an
    anonymous endpoint over a stored payload. Afterwards 100 KB takes milliseconds.

    The table is exact rather than a bound, so no decision the scanner makes changes.
    """
    ends = [0] * (len(lines) + 1)
    ends[len(lines)] = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip() or _interrupts_a_paragraph(lines[index]):
            ends[index] = index
        else:
            ends[index] = ends[index + 1]
    return ends


#: How much of a paragraph a link reference definition may be parsed from.
#:
#: CommonMark caps a link label at 999 characters (6.3); a destination and a title have no hard
#: limit, but a definition needing kilobytes of them is not one this scan can usefully recognise.
#: 8 KB leaves an order of magnitude of headroom over the label cap and bounds the work per call,
#: which is what stops consecutive definitions costing O(n^2).
#:
#: Erring small is the safe direction: a definition truncated by this budget is read as ordinary
#: paragraph text, which is the same way this function already fails when it is unsure. The
#: differential corpus is the check on whether it fails at all -- it scored 183 of 23,895 before
#: this change and must still score 183 after, or the budget has changed behaviour.
_DEFINITION_TEXT_BUDGET = 4096


def _definition_span(
    lines: Sequence[str], start: int, block_ends: Sequence[int] | None = None
) -> int:
    """How many lines a link reference definition at ``lines[start]`` runs over; 0 if it is none.

    A definition is consumed off the front of a paragraph rather than being one, and the whole of
    it -- label, destination and title -- may be spread over three lines. What the scan needs is
    therefore not "does this line look like a definition" but "how much of the text below is
    one", because a ``[x]:`` that is never given a destination is not a definition at all: the
    renderer reads it as ordinary paragraph text, reads on, and a ``===`` below underlines the
    lot into a level-one heading. Matching the label alone suppressed the paragraph and hid
    exactly that, while matching only a *complete* one-line definition would have hidden the
    ``[x]:`` whose destination is written on the line beneath it.

    The text a definition may be drawn from ends at the first blank line, and at the first line
    that opens a block of its own -- see _interrupts_a_paragraph, which is what keeps ``===``
    from being read as the destination of the label above it.
    """
    if block_ends is not None:
        end = block_ends[start + 1]
    else:
        end = start + 1
        while end < len(lines) and lines[end].strip() and not _interrupts_a_paragraph(lines[end]):
            end += 1
    # Take only as much of the block as a definition could possibly need. Joining all of it is
    # what made this quadratic: for consecutive definitions the block end IS the end of the
    # document, so every call rebuilt the whole remainder -- 655 million characters on a 100 KB
    # input, 59% of runtime inside str.join. Hoisting the forward scan in the previous change
    # removed a constant factor and left the complexity, and the stall was actually being held
    # off by the 64,000-character clamp on answer_markdown rather than by anything here.
    collected: list[str] = []
    budget = _DEFINITION_TEXT_BUDGET
    for line in lines[start:end]:
        collected.append(line)
        budget -= len(line) + 1
        if budget <= 0:
            break
    text = "\n".join(collected)
    indent = len(text) - len(text.lstrip(" "))
    if indent > 3:
        return 0
    position = _label_end(text, indent)
    if position is None or text[position : position + 1] != ":":
        return 0
    position = _skip_definition_whitespace(text, position + 1)
    after_destination = _destination_end(text, position)
    if after_destination is None:
        return 0
    position = _title_end(text, after_destination) or after_destination
    while text[position : position + 1] in (" ", "\t"):
        position += 1
    if text[position : position + 1] not in ("", "\n"):
        return 0
    return text[:position].count("\n") + 1


def _container_markers(line: str) -> tuple[str, ...]:
    """The container markers a line carries, as ``>`` for a block quote and ``-`` for a list item.

    Read off the line above so that the line below can be placed against it. Only the shape
    matters, not the bullet character or the number.
    """
    kinds: list[str] = []
    rest = line
    while True:
        quote = _BLOCK_QUOTE_MARKER.match(rest)
        if quote is not None:
            kinds.append(">")
            rest = rest[quote.end() :]
            continue
        item = _LIST_MARKER.match(rest)
        if item is None:
            return tuple(kinds)
        kinds.append("-")
        rest = rest[item.end() :]


def _behind_container_markers(
    line: str, *, after_content: bool, above: tuple[str, ...] = ()
) -> tuple[str, str]:
    """Split the container markers a line opens off the content written behind them.

    A block quote or a list item does not make a ``## `` stop being a level-two heading. The
    renderer nests it, but the report is styled with Tailwind's ``prose``, which sizes an h2 the
    same wherever it sits, so the reader sees a report section no agent wrote. The markers are
    therefore followed through to the content and the heading behind them is demoted in place.

    The one thing that can masquerade as a marker is a lazy continuation line. ``2. Forged`` under
    a line of prose is more of that prose, because an ordered list numbered from 2 cannot
    interrupt a paragraph -- so a list marker that cannot interrupt ends the walk and the line is
    left whole. ``after_content`` is what says whether there is anything to interrupt, and the
    renderer's answer is coarser than it looks: the flag stays set through every marker on the
    line. ``text`` over ``- 2. ## x`` is a list item holding a PARAGRAPH, not a nested list
    holding a heading, and ``[x]: /url`` over ``> 2. ## x`` likewise -- a line that turned out to
    hold nothing but a link reference definition still counts as content for this question, even
    though it leaves no paragraph for a setext underline to attach to. Block quote markers never
    end the walk: an ATX heading interrupts a paragraph wherever it is written.
    """
    prefix_length = 0
    rest = line
    depth = 0
    while True:
        quote = _BLOCK_QUOTE_MARKER.match(rest)
        if quote is not None:
            if depth < len(above) and above[depth] != ">":
                # The line above was in a different container here, and this marker has closed
                # it. Nothing of it is left open for the markers further in to interrupt.
                after_content = False
            prefix_length += quote.end()
            rest = rest[quote.end() :]
            depth += 1
            continue
        item = _LIST_MARKER.match(rest)
        if item is None:
            break
        if depth < len(above):
            # Any marker written where the line above had one of its own starts a new item and
            # closes what stood below it -- even the same bullet twice, because a list marker
            # never continues an item, it opens the next one.
            after_content = False
        elif after_content and (
            item.group("gap") is None or item.group("number") not in (None, "1")
        ):
            break
        prefix_length += item.end()
        rest = rest[item.end() :]
        depth += 1
    return line[:prefix_length], rest


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _continues_content(line: str) -> bool:
    """Whether a line is part of a content block, so that the line below it interrupts something.

    The coarser of the two questions the scan asks about the line above. It is what decides
    whether a list marker on the next line opens a container or is read as more of the text, and
    a link reference definition answers yes to it: block structure is settled before definitions
    are lifted off the front of the content, so the renderer is still in interrupt mode on the
    line below one.
    """
    inner = _behind_container_markers(line, after_content=False)[1]
    if not inner.strip():
        return False
    if _INDENTED_CODE.match(inner) is not None:
        # Four spaces in is never one of the patterns below: it is code, or more of a paragraph.
        return True
    inner = inner.strip()
    return not (
        _ATX_HEADING.match(inner)
        or _THEMATIC_BREAK.match(inner)
        or _opens_fence(inner) is not None
        or _opens_html_block(inner, in_paragraph=False) is not None
    )


def _leaves_a_paragraph_open(line: str) -> bool:
    """Whether a line could leave a paragraph open for the next line to continue.

    The finer of the two questions, and the one that decides when an open container ends. A line
    at column zero below a block quote or a list item is either a lazy continuation of a paragraph
    still open inside it -- in which case the container is still open and a ``---`` under it is a
    thematic break, not an underline -- or the start of a new top-level block, which closes the
    container. A paragraph is open only if the line above held paragraph text, so a blank line, a
    heading, a thematic break, a fence or a raw HTML block all answer no, whatever container
    markers they are written behind, and so does a finished link reference definition: it is
    content, but it is lifted off the front of that content and leaves no paragraph behind it.

    A run of ``=`` is deliberately not on that list. It closes a paragraph only where it is a
    setext underline, and the one place this question is asked -- under a container -- is the one
    place it cannot be: a lazy continuation line is not an underline, it is more of the paragraph.
    """
    inner = _behind_container_markers(line, after_content=False)[1]
    if not _continues_content(line):
        return False
    if _INDENTED_CODE.match(inner) is not None:
        return True
    return not _definition_span([inner.strip()], 0)


def _as_level_three(paragraph: list[str]) -> str:
    """Rewrite the paragraph a setext underline heads as the level-three ATX heading it becomes.

    The underline is dropped and the level moves onto the text, which is the only shape an ATX
    heading has. A setext heading may span several lines and an ATX heading may not, so the lines
    are joined by the single space a soft break inside a heading renders as: that keeps one
    heading holding the same words, where emitting only the first line would have left the rest
    of the model's own heading behind as a stray paragraph.

    One detail is not cosmetic: an ATX heading drops a trailing run of ``#`` as a closing sequence
    and a setext heading does not, so a heading whose text ends in ``#`` would lose it silently.
    Escaping the first ``#`` of that run keeps the renderer from reading the run as a closing
    sequence, and the backslash itself is not rendered.
    """
    head, *rest = (line.removesuffix("\r") for line in paragraph)
    indent = _indent_width(head)
    content = " ".join(part.strip() for part in [head[indent:], *rest])
    body = content.rstrip(" \t")
    hashes = len(body) - len(body.rstrip("#"))
    # Only a run preceded by whitespace is a closing sequence -- and after this rewrite a run that
    # opens the content is preceded by the space in "### ".
    if hashes and (hashes == len(body) or body[-hashes - 1] in " \t"):
        cut = len(body) - hashes
        content = f"{content[:cut]}\\{content[cut:]}"
    return f"{' ' * indent}### {content}"


def _demote_headings(markdown: str) -> str:
    """Keep the answer body's own headings inside the section that holds them.

    ``answer_markdown`` is the one field intended to be Markdown, so it must not be blockquoted
    -- the model's formatting is the point of it. But left alone its headings land at column zero
    of the generated report, where a ``## `` opens a sibling of the report's own sections rather
    than a subsection of Final Recommendation. Demoting levels 1 and 2 to level 3 preserves the
    formatting and fixes the nesting; deeper headings are already subordinate and are untouched.

    Both of Markdown's heading forms are demoted, because both reach the renderer as one. The ATX
    form (``## Title``) is decidable from its own line, wherever container markers put it. The
    setext form (``Title`` over ``---``) is not: the same ``---`` is a thematic break after a
    blank line, a delimiter row under a table header, and a plain rule after a list item. It is
    an underline only where it follows a paragraph, so the scan carries just enough block state
    to know that -- where the open paragraph began, and whether it is inside a container whose
    indentation this scanner does not track. Inside such a container the scan stands down rather
    than guess, so a setext heading written inside a list item or a block quote is left as it is:
    that residual is a heading the renderer nests inside that container, while guessing wrong the
    other way would corrupt a table or a thematic break, which is this function inventing content.

    What is *not* guessed at any more is the three block kinds whose grammar the scan used to
    approximate with a regex, each of which cost a forged heading: whether a pipe-and-dash line
    forms a GFM table (_forms_a_table), whether a ``[x]:`` is a link reference definition
    (_definition_span), and where a raw HTML block begins and ends (_opens_html_block). Fenced
    code and raw HTML are skipped entirely, so a ``# `` inside either stays what the model wrote.

    Still open, and left open deliberately:

    * A heading indented into a list item's content column -- ``    ## x`` under ``- item`` -- is
      four spaces from column zero but two from the item's content, so the renderer reads a
      heading where this scan reads indented code. Closing it needs the content column of every
      open list item, a container stack this function does not keep, and the same judgement
      leaves the setext case above alone. Pinned by a test so it cannot drift out of sight.
    * A raw HTML block opened at column zero inside a still-open list item is treated as
      top-level, because that is what the renderer usually does with it -- ``- item`` over
      ``<div>`` over ``## x`` is three lines of HTML and not a heading. Where the renderer
      instead keeps the block inside the item, the scan stands down over lines it would have
      demoted.

    Both residuals above fail in the same direction: a demotion forgone, never a line rewritten.

    That is true of *those two* and was for a while written here as though it were true of the
    whole function. It is not, and the differential corpus says so: of 23,895 cases, 17 rewrite
    an ordinary body -- one the renderer shows no level-1 or level-2 heading in, which this
    function therefore had no reason to touch at all -- and 21 change the word set.

    The cause is not a bug to be fixed in passing; it is inherent to converting setext to ATX.
    A setext heading is two lines, and an ATX heading is one, so demoting one *must* delete the
    underline and may join what sat above it. Where the scan reads a setext heading the renderer
    does not, both edits land on text that was never a heading::

        in   '    indented code\n<foo>\n---\n[x]: /url\n<foo bar=1>\n- \n==='
        out  '    indented code\n### <foo>\n[x]: /url\n### <foo bar=1>\n==='

    -- ``---`` and ``-`` are gone and ``<foo>`` has become a heading. In the worst measured case
    the join runs through a code payload, collapsing two fence lines into one.

    So the honest statement of the trade is narrower than the one that stood here. This function
    prefers forgoing a demotion to rewriting a line, and every *deliberate* stand-down above
    takes that side. But the setext path cannot take it: it edits two lines by construction, and
    when it is wrong about which two, it corrupts. 17 of 23,895 is the measured size of that,
    and it is recorded rather than described away -- the alternative is the reverse trade, where
    a missed forgery is a report section the model did not write.
    """
    source = markdown.split("\n")
    lines: list[str] = []
    fence: str | None = None
    # The kind of raw HTML block that is open (CommonMark 4.6) and whether it was opened inside a
    # container. A block of kind 1 to 5 runs to its closing string and a blank line does not end
    # it -- but any block ends with the container it was written in, so one opened inside a
    # container also ends at the first line back at column zero that carries no marker of its own.
    html: int | None = None
    html_nested = False
    html_blanked = False
    # Index in ``lines`` of the first line of the open paragraph, or None when none is open. The
    # underline applies to the whole paragraph, and only its first line can carry the "### ".
    paragraph: int | None = None
    # Set by a line that opens a block quote, a list item or a table: constructs whose content
    # indentation decides what their inner lines mean. It is cleared where the text demonstrably
    # leaves them -- see the dedent rule in the loop -- and not merely at a blank line, because a
    # list item stays open across one.
    opaque = False
    # Set where that container is in fact a GFM table, whose rows -- unlike a paragraph's
    # lines -- are not something the line below could be a lazy continuation of.
    table = False
    # The two questions the scan asks about the line above, which _continues_content and
    # _leaves_a_paragraph_open answer: whether it was content, which is what a marker on this
    # line would have to interrupt, and whether it left a paragraph open, which is what a setext
    # underline would attach to and what decides that an open container has ended. They differ
    # over a link reference definition, which is the first but not the second.
    after_content = False
    open_paragraph = False
    # The container markers the line above carried, so that this line's can be placed
    # against them. A marker written where that line had one of its own has closed what
    # stood below it, which is why ``- item`` over ``2. ## x`` IS the heading it looks
    # like while ``text`` over ``2. ## x`` is one more line of the same paragraph.
    above: tuple[str, ...] = ()

    # split("\n") rather than splitlines() so the join round-trips the text exactly. A CRLF line
    # ending leaves its carriage return on the line; it is a line ending, not content, so it is
    # ignored while classifying and put back when the line is emitted.
    stripped = [raw.removesuffix("\r") for raw in source]
    # Hoisted out of _definition_span, which used to derive it per call by walking to the end of
    # the block. The answer depends only on line content, so it is the same every time -- and
    # deriving it per call made N consecutive link reference definitions cost O(N^2).
    block_ends = _block_ends(stripped)
    index = 0
    while index < len(source):
        raw = source[index]
        line = stripped[index]
        index += 1
        tail = raw[len(line) :]

        if html is not None:
            prefix, inner = _behind_container_markers(line, after_content=False)
            blank = not (inner if html_nested else line).strip()
            # A block ends with the container it was written in, and a line back at column zero
            # carrying no marker of its own has left every container above it. For kinds 6 and 7
            # that is the whole rule; kinds 1 to 5 run on through a line like that and give way
            # only after a blank line, because it takes a blank line to end the paragraph a list
            # item is still holding open around them.
            left_containers = not blank and not prefix and not line.startswith((" ", "\t"))
            if html_nested and left_containers and (html >= 6 or html_blanked):
                # Read the line again from the top, as the block start it may well be.
                html, opaque, table, index = None, False, False, index - 1
                continue
            if blank:
                html_blanked = True
                if html >= 6:
                    html = None
                    paragraph, after_content, open_paragraph = None, False, False
                    lines.append(raw)
                    continue
            lines.append(raw)
            if html < 6 and _closes_html_block(line, html):
                html = None
            continue

        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
            lines.append(raw)
            continue

        opened = _opens_fence(line)
        if opened is not None:
            # Fenced code is never a lazy continuation, so a fence at column zero has no container
            # left open above it.
            if not line.startswith((" ", "\t")):
                opaque, table = False, False
            fence, paragraph, after_content, open_paragraph = opened, None, False, False
            above = ()
            lines.append(raw)
            continue

        if not line.strip():
            paragraph, table = None, False
            after_content, open_paragraph, above = False, False, ()
            lines.append(raw)
            continue

        # An open container ends at the first line back at column zero that is not a lazy
        # continuation of a paragraph still open inside it. Until then the scan stays inside it,
        # because there a "---" under a line of text is a thematic break and not an underline --
        # a setext underline cannot be a lazy continuation line (CommonMark 4.3).
        # ``startswith`` rather than an indent width: a leading tab is four columns of indent, and
        # a line inside a container reached by one has not come back to column zero at all.
        if opaque and not line.startswith((" ", "\t")) and not open_paragraph:
            opaque, table = False, False

        # A setext underline demotes the paragraph it underlines, and is consumed by it.
        if paragraph is not None and not opaque and _SETEXT_UNDERLINE.match(line) is not None:
            lines[paragraph:] = [_as_level_three(lines[paragraph:])]
            paragraph, after_content, open_paragraph, above = None, False, False, ()
            continue

        prefix, inner = _behind_container_markers(
            line,
            after_content=after_content,
            above=() if line.startswith((" ", "\t")) else above,
        )

        heading = _ATX_HEADING.match(inner)
        if heading is not None:
            demotable = len(heading.group("hashes")) <= 2
            lines.append(
                f"{prefix}{heading.group('indent')}###{heading.group('rest')}{tail}"
                if demotable
                else raw
            )
            paragraph, after_content, open_paragraph = None, False, False
            above = _container_markers(line)
            opaque = opaque or bool(prefix)
            continue

        kind = _opens_html_block(inner, in_paragraph=paragraph is not None or after_content)
        if kind is not None:
            # A block written behind a container marker ends with that container, so the loop
            # has to remember which kind it was. A block at column zero carrying no marker of its
            # own is NOT one of those, even where a list item is still open above it: the
            # renderer closes the item and puts the block at the top level, which is why
            # ``- item`` over ``<div>`` over ``## x`` is three lines of raw HTML and not a
            # heading. Standing on the marker rather than on the open container is deliberate --
            # see the residual noted in _demote_headings.
            html, html_nested, html_blanked = kind, bool(prefix), False
            if prefix:
                opaque = True
            elif not line.startswith((" ", "\t")):
                opaque = False
            paragraph, after_content, open_paragraph, above = None, False, False, ()
            lines.append(raw)
            if kind < 6 and _closes_html_block(line, kind):
                html = None
            continue

        if _THEMATIC_BREAK.match(line) is not None:
            paragraph, after_content, open_paragraph, above = None, False, False, ()
            lines.append(raw)
            continue

        container = _INTERRUPTS_PARAGRAPH if paragraph is not None else _CONTAINER_START
        if container.match(line) is not None:
            paragraph, opaque, table = None, True, False
            after_content = _continues_content(line)
            open_paragraph = _leaves_a_paragraph_open(line)
            above = _container_markers(line)
            lines.append(raw)
            continue

        if paragraph is not None:
            # Inside an open paragraph: either this line and the one above it are the header and
            # delimiter rows of a GFM table -- after which a bare "---" is a thematic break and
            # not an underline -- or it is another line of the paragraph.
            if _forms_a_table(lines[-1].removesuffix("\r"), line):
                # A table is its own block, so its rows are not content the line below could be
                # a lazy continuation of: a "> 2. " under one opens the list it looks like.
                paragraph, opaque, table = None, True, True
            after_content, open_paragraph, above = not table, True, ()
            lines.append(raw)
            continue

        # No paragraph is open. This line starts one unless it is not paragraph material at all:
        # an indented code block, which ends itself at the first line without the indent, or a
        # link reference definition, which is consumed off the front of a paragraph rather than
        # being one -- and which may run over the two lines below it.
        if not opaque and _INDENTED_CODE.match(line) is None:
            span = _definition_span(stripped, index - 1, block_ends)
            if span:
                lines.extend(source[index - 1 : index - 1 + span])
                index += span - 1
                after_content, open_paragraph, above = True, False, ()
                continue
            paragraph = len(lines)
        # A body row of an open table is not content either: the row belongs to the table, so a
        # "<foo bar=1>" under one opens the raw HTML block it would not open under a paragraph.
        after_content = _continues_content(line) and not table
        open_paragraph = _leaves_a_paragraph_open(line)
        above = _container_markers(line)
        lines.append(raw)

    return "\n".join(lines)


def _evidence_table(citations: Sequence[Citation]) -> str:
    if not citations:
        return "_No supporting evidence was cited._"
    rows = ["| Key | Source | Page | Relevance |", "| --- | --- | --- | --- |"]
    for citation in citations:
        page = str(citation.page_number) if citation.page_number is not None else "—"
        rows.append(
            f"| {_as_cell(citation.citation_key)} | {_as_cell(citation.source_filename)} | {page} "
            f"| {citation.relevance_score:.2f} |"
        )
    return "\n".join(rows)


def _contradictions(
    agent_runs: Sequence[AgentRun],
    labels: Mapping[str, str],
    checked: "Rendered[PharmaResponse] | None" = None,
) -> str:
    if checked is not None:
        return _checked_contradictions(checked, labels)
    rendered = _render_state(agent_runs, AgentName.FINAL_DECISION, FinalDecisionOutput)
    decision = rendered.value
    if decision is None or not decision.contradictory_evidence:
        return _absent_note(
            rendered.state if decision is None else RenderState.EMPTY,
            subject="final decision",
            empty="_No contradictory evidence was identified._",
        )
    return "\n".join(
        f"- {_one_line(_reader_text(item, labels))}" for item in decision.contradictory_evidence
    )


def _uncertainty(
    agent_runs: Sequence[AgentRun],
    labels: Mapping[str, str],
    checked: "Rendered[PharmaResponse] | None" = None,
) -> str:
    """Surface `uncertainty` and `next_steps` (C-15).

    Both fields exist on `FinalDecisionOutput`, both are explicitly requested by the prompt, and
    before this **no consumer read either**. The model was asked to state what it was unsure
    about, did so, and the answer was dropped on the floor -- which is worse than not asking,
    because the run pays for the tokens and the reader is left with an answer that looks more
    confident than the agent that wrote it.

    Decision D-7 chose to surface them rather than delete the fields.
    """
    if checked is not None:
        return _checked_uncertainty(checked, labels)
    rendered = _render_state(agent_runs, AgentName.FINAL_DECISION, FinalDecisionOutput)
    decision = rendered.value
    if decision is None:
        return _absent_note(
            rendered.state,
            subject="final decision",
            empty="_No uncertainty was recorded._",
        )

    blocks: list[str] = []
    uncertainty = _one_line(_reader_text(decision.uncertainty, labels)).strip()
    blocks.append(
        f"**Stated uncertainty.** {uncertainty}"
        if uncertainty
        else "_The final decision agent stated no uncertainty._"
    )
    if decision.next_steps:
        blocks.append("**Suggested next steps.**")
        blocks.extend(f"- {_one_line(_reader_text(step, labels))}" for step in decision.next_steps)
    else:
        blocks.append("_No next steps were suggested._")
    return "\n".join(blocks)


def _risk_matrix(
    agent_runs: Sequence[AgentRun],
    labels: Mapping[str, str],
    checked: "Rendered[PharmaResponse] | None" = None,
) -> str:
    if checked is not None:
        return _checked_risk_matrix(checked, labels)
    rendered = _render_state(agent_runs, AgentName.RISK, RiskOutput)
    risk_output = rendered.value
    if risk_output is None or not risk_output.risks:
        return _absent_note(
            rendered.state if risk_output is None else RenderState.EMPTY,
            subject="risk analysis",
            empty="_No risks were identified._",
        )
    rows = ["| Category | Severity | Likelihood | Mitigation |", "| --- | --- | --- | --- |"]
    for risk in risk_output.risks:
        mitigation = _as_cell(_reader_text(risk.mitigation, labels))
        rows.append(
            f"| {risk.category.value} | {risk.severity.value} "
            f"| {risk.likelihood.value} | {mitigation} |"
        )
    return "\n".join(rows)


def _evaluation_metrics(
    evaluation: EvaluationResult | None, labels: Mapping[str, str], *, withheld: bool = False
) -> str:
    """``withheld``: a pharmaceutical run's report shows the scores without the evaluator's
    explanation, which is model text the regulatory-boundary check does not cover (PH-D-030)."""
    if evaluation is None:
        return "_Evaluation was not available for this query._"
    explanation = (
        f"_{EVALUATION_EXPLANATION_WITHHELD}_"
        if withheld
        # Free model prose, contained the way a cited snippet already is -- and, like every
        # other section that renders a model's prose, with no raw identifier left in it (L-2).
        # The evaluator names chunks by id on live runs, where no scripted fixture did.
        else _as_blockquote(conceal_identifiers(evaluation.explanation, labels))
    )
    return "\n".join(
        [
            f"- Answer relevance: {evaluation.answer_relevance:.2f}",
            f"- Faithfulness: {evaluation.faithfulness:.2f}",
            f"- Citation accuracy: {evaluation.citation_accuracy:.2f}",
            f"- Context sufficiency: {evaluation.context_sufficiency:.2f}",
            f"- Hallucination risk: {evaluation.hallucination_risk:.2f}",
            "",
            explanation,
        ]
    )


def _agent_trace(agent_runs: Sequence[AgentRun]) -> str:
    if not agent_runs:
        return "_No agent activity was recorded._"
    rows = ["| Agent | Status | Latency (ms) |", "| --- | --- | --- |"]
    for run in agent_runs:
        latency = str(run.latency_ms) if run.latency_ms is not None else "—"
        rows.append(f"| {run.agent_name.value} | {run.status.value} | {latency} |")
    return "\n".join(rows)


def _as_blockquote(snippet: str) -> str:
    """Quote every line of a snippet, not just the first.

    A snippet is an arbitrary slice of a chunk, so it routinely contains blank lines and partial
    Markdown -- a table cut below its header row, a list cut mid-item. Prefixing only the first
    line meant the first blank line inside the snippet *ended* the blockquote, and everything
    after it leaked into the report as ordinary body text: that is why a cited table fragment
    reached the appendix as a run of bare pipes rather than as quoted evidence.

    Quoting every line also contains the snippet's own Markdown. An excerpt beginning "## " or
    "---" would otherwise restructure the generated document -- a report assembled out of
    document text must not let the documents it cites decide its structure. A blank line becomes
    a bare ">" so the quote survives it.
    """
    return "\n".join(f"> {line}" if line else ">" for line in snippet.splitlines())


def _citation_appendix(citations: Sequence[Citation]) -> str:
    if not citations:
        return "_No citations._"
    blocks: list[str] = []
    for citation in citations:
        quoted = _as_blockquote(citation.snippet)
        # Flattened rather than cell-escaped: this is a prose line, not a table row, so a pipe
        # here is harmless and escaping one would only leak a backslash into the appendix. The
        # newline is the break that matters, and flattening is what denies it.
        # Flattening is not enough here, and the reasoning above missed why. These two values
        # land inside a bracket pair THIS LINE SUPPLIES -- ``**[{key}]** {filename}`` -- so a
        # citation_key of ``x](https://attacker.example)`` closes the bracket the code opened
        # and completes a link the code never wrote. The evidence table was given
        # ``_no_inline_markup`` for exactly this; the appendix takes the SAME two values from
        # the SAME uploaded document. Pipe escaping still does not belong here, so not _as_cell.
        key = _no_inline_markup(_one_line(citation.citation_key))
        filename = _no_inline_markup(_one_line(citation.source_filename))
        blocks.append(f"**[{key}]** {filename}\n\n{quoted}")
    return "\n\n".join(blocks)


class RenderState(StrEnum):
    """Why a report section has nothing to show (N-011).

    `_validated` returned `None` for three genuinely different situations, and every caller
    rendered the same sentence for all of them. So `"_No risks were identified._"` -- a positive
    claim about the analysis -- was printed when the risk agent never ran, when its output could
    not be parsed, and when it genuinely found no risks. A reader could not tell a clean bill of
    health from a broken one.
    """

    NOT_RUN = "not_run"
    UNREADABLE = "unreadable"
    EMPTY = "empty"
    PRESENT = "present"


@dataclass(frozen=True)
class Rendered[ModelT: BaseModel]:
    """A validated agent output, or the reason there isn't one."""

    state: RenderState
    value: ModelT | None = None


def _render_state[ModelT: BaseModel](
    agent_runs: Sequence[AgentRun], agent_name: AgentName, model: type[ModelT]
) -> Rendered[ModelT]:
    """Which of the four states this section is in, and the output if there is one."""
    for run in agent_runs:
        if run.agent_name is not agent_name:
            continue
        if run.output_json is None:
            # The step ran and failed. `P2-T06` records why on the row itself; here it is
            # enough that the section says the step did not produce output rather than
            # claiming a result.
            return Rendered(RenderState.UNREADABLE)
        try:
            return Rendered(RenderState.PRESENT, model.model_validate(run.output_json))
        except ValidationError:
            logger.warning(
                "Report could not parse %s output; rendering it as unreadable rather than as "
                "an empty result",
                agent_name.value,
                extra={"event_type": "report_unreadable_output"},
            )
            return Rendered(RenderState.UNREADABLE)
    return Rendered(RenderState.NOT_RUN)


def _absent_note(state: RenderState, *, subject: str, empty: str) -> str:
    """The sentence for a section with nothing to show, distinguishing *why*."""
    if state is RenderState.NOT_RUN:
        return f"_The {subject} step did not run for this query._"
    if state is RenderState.UNREADABLE:
        return (
            f"_The {subject} step produced output that could not be read. This is not a "
            "statement that there were none._"
        )
    return empty


# --- the pharmaceutical branch (PDP8-T04) -----------------------------------------------------
#
# A pharmaceutical run's agent rows hold the model's text before the regulatory-boundary check,
# so its report reads the checked response in the run record instead (PH-DEF-006, threat T-16).
# Every value is contained exactly as the General sections contain theirs.


def _pharma_sections_of(record: PharmaRunRecord | None) -> Rendered[PharmaResponse]:
    """The record's checked response, or why there is none.

    NOT_RUN covers a run with no record and one whose recorder has not written its result; a
    stored result that does not validate is UNREADABLE. Neither falls back to the agent rows.
    """
    if record is None or record.response is None:
        return Rendered(RenderState.NOT_RUN)
    try:
        sections = PharmaResponse.model_validate(record.response.get("sections"))
    except ValidationError:
        logger.warning(
            "Report could not parse a pharmaceutical run's checked response; rendering it as "
            "unreadable",
            extra={"event_type": "report_unreadable_output"},
        )
        return Rendered(RenderState.UNREADABLE)
    return Rendered(RenderState.PRESENT, sections)


def _checked_contradictions(checked: Rendered[PharmaResponse], labels: Mapping[str, str]) -> str:
    response = checked.value
    if response is None:
        return _absent_note(checked.state, subject="policy check", empty="")
    section = response.conflicts_or_ambiguities
    blocks: list[str] = []
    if section.deterministic:
        blocks.append(
            "\n".join(
                f"- {conflict.kind.value.replace('_', ' ').capitalize()}: "
                f"{_one_line(conceal_identifiers(conflict.description, labels))}"
                for conflict in section.deterministic
            )
        )
    reported = [text for text in section.model_reported if text.strip()]
    if reported:
        # REQ-CON-02: labelled and never merged into the deterministic list.
        blocks.append(
            "\n".join(
                [
                    MODEL_REPORTED_LABEL,
                    *(f"- {_one_line(conceal_identifiers(text, labels))}" for text in reported),
                ]
            )
        )
    return "\n\n".join(blocks) or "_No contradictory evidence was identified._"


def _checked_risk_matrix(checked: Rendered[PharmaResponse], labels: Mapping[str, str]) -> str:
    response = checked.value
    if response is None:
        return _absent_note(checked.state, subject="policy check", empty="")
    if not response.quality_and_operational_risks:
        return "_No risks were identified._"
    rows = ["| Category | Severity | Description | Evidence |", "| --- | --- | --- | --- |"]
    for flag in response.quality_and_operational_risks:
        description = _as_cell(conceal_identifiers(flag.description, labels))
        evidence = _as_cell(", ".join(flag.citation_keys)) or "—"
        rows.append(
            f"| {flag.category.value} | {flag.severity.value} | {description} | {evidence} |"
        )
    return "\n".join(rows)


def _checked_uncertainty(checked: Rendered[PharmaResponse], labels: Mapping[str, str]) -> str:
    response = checked.value
    if response is None:
        return _absent_note(checked.state, subject="policy check", empty="")
    blocks: list[str] = []
    for heading, items, empty in (
        (
            "**Limitations and uncertainty.**",
            response.limitations_and_uncertainty,
            "_No limitations were recorded._",
        ),
        (
            "**Suggested investigation questions and next steps.**",
            response.suggested_investigation_questions_or_next_steps,
            "_No next steps were suggested._",
        ),
    ):
        lines = [
            f"- {_one_line(conceal_identifiers(item, labels))}" for item in items if item.strip()
        ]
        blocks.extend([heading, *lines] if lines else [empty])
    return "\n".join(blocks)
