"""Section-bounded chunking for controlled documents (REQ-CHK-01 to REQ-CHK-03, REQ-CHK-05).

A controlled procedure is cited by section, so a chunk never spans two sections: each extracted
block (one Markdown section, its heading line included) is chunked on its own, and the
section's heading becomes every one of its chunks' ``section_title``. A heading with no text of
its own (a document's title heading) yields no chunk. A short section is one chunk. A section
longer than the window is split between paragraphs, so a WARNING, CAUTION or NOTE paragraph
stays whole with the steps around it; only a single paragraph longer than the window is cut, by
the baseline ``ChunkingService`` windows.

The interface is ``ChunkingService.build_chunks``'s, and so is the chunk ceiling: one upload
must not become an unbounded number of billed embedding requests (see ``chunking_service``).
"""

import re

from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import ExtractedBlock
from app.services.chunking_service import (
    BLOCK_SEPARATOR,
    ChunkDraft,
    ChunkingService,
    default_max_chunks,
)
from app.utils.text import CHARS_PER_TOKEN, estimate_tokens

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_HEADING_ONLY = re.compile(r"#{1,6}[ \t][^\n]*")
TOO_MANY_CHUNKS = "Document produced too many chunks to index. Split it into smaller documents."


class SectionChunker:
    def __init__(
        self,
        target_tokens: int = 900,
        overlap_tokens: int = 120,
        max_chunks: int | None = None,
    ) -> None:
        self._windows = ChunkingService(target_tokens, overlap_tokens, max_chunks)
        self._target_chars = target_tokens * CHARS_PER_TOKEN
        self._max_chunks = max_chunks

    def build_chunks(self, blocks: list[ExtractedBlock]) -> list[ChunkDraft]:
        max_chunks = default_max_chunks() if self._max_chunks is None else self._max_chunks
        drafts: list[ChunkDraft] = []
        offset = 0
        first = True
        for block in blocks:
            text = block.text.strip()
            if text == "":
                continue
            if not first:
                offset += len(BLOCK_SEPARATOR)
            first = False
            if _HEADING_ONLY.fullmatch(text):
                # PH-DEF-013: a heading with no text of its own (a document's title followed
                # directly by its first section) holds only the title, which the stored metadata
                # keeps; as a chunk it outranked the sections that answer. Its place in the
                # joined text is kept so the next offsets stay true.
                offset += len(text)
                continue
            for piece, start in self._pieces(text):
                if len(drafts) >= max_chunks:
                    raise ExtractionFailedError(TOO_MANY_CHUNKS)
                drafts.append(
                    ChunkDraft(
                        chunk_index=len(drafts),
                        text=piece,
                        char_start=offset + start,
                        char_end=offset + start + len(piece),
                        page_number=block.page_number,
                        section_title=block.section_title,
                        token_estimate=estimate_tokens(piece),
                    )
                )
            offset += len(text)
        return drafts

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        """The section's chunk texts with their offsets inside ``text``."""
        if len(text) <= self._target_chars:
            return [(text, 0)]
        pieces: list[tuple[str, int]] = []
        current_start: int | None = None
        current_end = 0
        for start, end in _paragraphs(text):
            if end - start > self._target_chars:
                if current_start is not None:
                    pieces.append((text[current_start:current_end], current_start))
                    current_start = None
                pieces.extend(self._windowed(text[start:end], start))
                continue
            if current_start is not None and end - current_start > self._target_chars:
                pieces.append((text[current_start:current_end], current_start))
                current_start = None
            if current_start is None:
                current_start = start
            current_end = end
        if current_start is not None:
            pieces.append((text[current_start:current_end], current_start))
        return pieces

    def _windowed(self, paragraph: str, base: int) -> list[tuple[str, int]]:
        drafts = self._windows.build_chunks([ExtractedBlock(text=paragraph)])
        return [(draft.text, base + draft.char_start) for draft in drafts]


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each non-blank paragraph of ``text``, in order."""
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, len(text)))
    result: list[tuple[int, int]] = []
    for start, end in spans:
        segment = text[start:end]
        if segment.strip() == "":
            continue
        lead = len(segment) - len(segment.lstrip())
        trail = len(segment) - len(segment.rstrip())
        result.append((start + lead, end - trail))
    return result
