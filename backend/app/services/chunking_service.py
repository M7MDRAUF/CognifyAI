"""Chunking service (FR-DOC-003).

Slides a window across the document's concatenated text: target 900 tokens
with 120 tokens of overlap (4 chars/token heuristic, see utils.text).
Window ends prefer whitespace so words are not cut mid-token. Each chunk
carries char offsets into the concatenated text plus the page number and
section title of the block where the window starts.
"""

from bisect import bisect_right
from dataclasses import dataclass

from app.core.config import upload_cap_mb
from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import ExtractedBlock
from app.utils.text import CHARS_PER_TOKEN, estimate_tokens

BLOCK_SEPARATOR = "\n\n"

# One accepted upload must not turn a linear byte budget into an unbounded number of
# chunks -- nor into an unbounded *bill*. Every 64 drafts that leave this function become one
# billed provider request in ``EmbeddingService.embed``, and the ingestion path is reachable
# anonymously and repeatably (``POST /documents/{id}/reindex``), while the only quota on it
# sums ``Document.file_size_bytes`` -- a column a reindex never moves. So this ceiling, not
# the byte quota, is what bounds one run's spend.
#
# It is therefore stated per megabyte of the *advertised upload cap*, not as a round number.
# A fixed 10,000 was calibrated against a 25 MB deployment and left only 16% headroom there, so
# an operator who raised MAX_UPLOAD_SIZE_MB past about 30 -- 29.8 MiB for prose, 28.7 for the
# densest ordinary layout below -- started having ordinary documents refused by a ceiling that
# had not moved with it, in response to changing a setting about bytes. Derived, the two cannot
# drift: raising the cap raises the bill the operator has agreed to, and lowering it tightens
# this in step.
#
# 440 is picked from measured strides, between the two edges it has to separate:
#
# * The densest *ordinary* layouts measured all sit at 336-349 chunks per MiB -- prose 336,
#   report paragraphs 337, base64 at 76-char lines 340, 600-character tokens 349 -- because a
#   window breaks at the first whitespace below the 3600-char target and advances ~3117 chars.
#   Text with no whitespace at all (CJK) is coarser still, at 292. So 440 leaves 26% headroom
#   over the worst ordinary shape rather than 16%.
# * The floor on the advance is 1800 chars, giving 582 chunks per MiB, and that layout is
#   refused with 24% to spare. The advance cannot be driven below it: a window that ends at
#   ``position + 1801`` needs whitespace there and none in the 1799 characters after it, which
#   is exactly what the *next* window would need for its own short end, so short windows cannot
#   repeat. Worth stating plainly, because this edge is not only adversarial -- a machine
#   generated file of 1799-character lines reaches the same 582 and is refused too.
#
# At the deployed 25 MB cap that is 11,000 chunks, or ~172 billed embedding requests for one
# document: about 7 per MB of whatever cap the operator sets.
_CHUNKS_PER_CAP_MB = 440


def default_max_chunks() -> int:
    """The production ceiling, derived from the cap the product advertises."""
    return upload_cap_mb() * _CHUNKS_PER_CAP_MB


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_number: int | None
    section_title: str | None
    token_estimate: int


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    page_number: int | None
    section_title: str | None


class ChunkingService:
    def __init__(
        self,
        target_tokens: int = 900,
        overlap_tokens: int = 120,
        max_chunks: int | None = None,
    ) -> None:
        if target_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
            msg = "target_tokens must be positive and overlap_tokens smaller than target."
            raise ValueError(msg)
        if max_chunks is not None and max_chunks <= 0:
            msg = "max_chunks must be positive when set."
            raise ValueError(msg)
        self._target_chars = target_tokens * CHARS_PER_TOKEN
        self._overlap_chars = overlap_tokens * CHARS_PER_TOKEN
        # None means "use the derived default", resolved per call rather than captured here
        # so that the derivation stays the single place the production ceiling is stated --
        # and so a cap change is picked up without rebuilding the service.
        self._max_chunks = max_chunks

    def build_chunks(self, blocks: list[ExtractedBlock]) -> list[ChunkDraft]:
        full_text, spans = self._assemble(blocks)
        if full_text.strip() == "":
            return []
        # _assemble emits spans in ascending start order, so their starts are searched
        # once per chunk instead of rescanned: a linear scan here cost O(blocks) per
        # chunk, i.e. O(chunks x blocks) for one upload, on the event-loop thread.
        span_starts = [span.start for span in spans]
        max_chunks = default_max_chunks() if self._max_chunks is None else self._max_chunks
        drafts: list[ChunkDraft] = []
        position = 0
        text_length = len(full_text)
        while position < text_length:
            # Refused here, inside the loop, rather than after the list is built: this is the
            # last decision point before any draft can reach a billed embedding request, and
            # stopping at the ceiling also bounds the peak memory a refused document holds.
            if len(drafts) >= max_chunks:
                msg = "Document produced too many chunks to index. Split it into smaller documents."
                raise ExtractionFailedError(msg)
            window_end = self._window_end(full_text, position, text_length)
            raw = full_text[position:window_end]
            stripped = raw.strip()
            if stripped != "":
                lead = len(raw) - len(raw.lstrip())
                char_start = position + lead
                span = self._span_at(spans, span_starts, char_start)
                drafts.append(
                    ChunkDraft(
                        chunk_index=len(drafts),
                        text=stripped,
                        char_start=char_start,
                        char_end=char_start + len(stripped),
                        page_number=span.page_number if span else None,
                        section_title=span.section_title if span else None,
                        token_estimate=estimate_tokens(stripped),
                    )
                )
            if window_end >= text_length:
                break
            position = self._next_start(full_text, window_end, position)
        return drafts

    def _next_start(self, full_text: str, window_end: int, previous_position: int) -> int:
        """Start the next window ~overlap before the end, aligned to a word boundary."""
        candidate = max(window_end - self._overlap_chars, previous_position + 1)
        while (
            0 < candidate < window_end
            and not full_text[candidate - 1].isspace()
            and not full_text[candidate].isspace()
        ):
            candidate += 1
        return candidate

    def _window_end(self, full_text: str, position: int, text_length: int) -> int:
        hard_end = min(position + self._target_chars, text_length)
        if hard_end >= text_length:
            return hard_end
        # Prefer breaking at whitespace, but never shrink below half a window.
        minimum = position + self._target_chars // 2
        for index in range(hard_end, minimum, -1):
            if full_text[index - 1].isspace():
                return index
        return hard_end

    @staticmethod
    def _assemble(blocks: list[ExtractedBlock]) -> tuple[str, list[_Span]]:
        parts: list[str] = []
        spans: list[_Span] = []
        offset = 0
        for block in blocks:
            text = block.text.strip()
            if text == "":
                continue
            if parts:
                offset += len(BLOCK_SEPARATOR)
            parts.append(text)
            spans.append(
                _Span(
                    start=offset,
                    end=offset + len(text),
                    page_number=block.page_number,
                    section_title=block.section_title,
                )
            )
            offset += len(text)
        return BLOCK_SEPARATOR.join(parts), spans

    @staticmethod
    def _span_at(spans: list[_Span], span_starts: list[int], position: int) -> _Span | None:
        """The span containing `position`, else the next one, else the last one.

        `span_starts` is the ascending list of `span.start` built by build_chunks.
        """
        if not spans:
            return None
        index = bisect_right(span_starts, position) - 1
        if index >= 0 and position < spans[index].end:
            return spans[index]
        # Position may fall on a separator; attribute it to the next block. Spans are
        # disjoint and ascending, so index + 1 is the first one starting after position.
        following = index + 1
        return spans[following] if following < len(spans) else spans[-1]
