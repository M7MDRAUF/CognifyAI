"""TXT and Markdown extraction preserving heading structure when possible."""

import re

from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import ExtractedBlock

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionFailedError("Text file is not valid UTF-8.") from exc


def extract_plain_text(content: bytes) -> list[ExtractedBlock]:
    text = _decode(content)
    return [ExtractedBlock(text=paragraph) for paragraph in re.split(r"\n\s*\n", text)]


def extract_markdown(content: bytes) -> list[ExtractedBlock]:
    text = _decode(content)
    blocks: list[ExtractedBlock] = []
    current_section: str | None = None
    pending: list[str] = []

    def flush() -> None:
        nonlocal pending
        if pending:
            blocks.append(ExtractedBlock(text="\n".join(pending), section_title=current_section))
            pending = []

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is not None:
            flush()
            current_section = match.group(2).strip()
            pending.append(line)
        else:
            pending.append(line)
    flush()
    return blocks
