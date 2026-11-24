"""File-type-specific text extraction (FR-DOC-002).

Every parser returns a list of ``ExtractedBlock`` carrying page numbers and
section titles when the format provides them, so citation metadata survives
chunking. All parser failures surface as ``ExtractionFailedError``.
"""

from collections.abc import Callable

from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import ExtractedBlock
from app.integrations.parsers.docx import extract_docx
from app.integrations.parsers.pdf import extract_pdf
from app.integrations.parsers.tabular import extract_csv, extract_xlsx
from app.integrations.parsers.textual import extract_markdown, extract_plain_text

__all__ = ["ExtractedBlock", "extract_blocks"]

_EXTRACTORS: dict[str, Callable[[bytes], list[ExtractedBlock]]] = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "csv": extract_csv,
    "xlsx": extract_xlsx,
    "txt": extract_plain_text,
    "md": extract_markdown,
}


def extract_blocks(content: bytes, extension: str) -> list[ExtractedBlock]:
    """Dispatch to the parser for ``extension`` and drop empty blocks."""
    extractor = _EXTRACTORS.get(extension)
    if extractor is None:
        raise ExtractionFailedError(f"No parser is available for .{extension} files.")
    blocks = extractor(content)
    non_empty = [block for block in blocks if block.text.strip() != ""]
    if not non_empty:
        raise ExtractionFailedError(
            "No extractable text was found in the document. Scanned PDFs without a text "
            "layer are not supported in the MVP."
        )
    return non_empty
