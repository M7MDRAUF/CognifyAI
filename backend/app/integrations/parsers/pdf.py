"""PDF extraction preserving page numbers."""

import io

from pypdf import PdfReader

from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import PDF_UNREADABLE, ExtractedBlock


def extract_pdf(content: bytes) -> list[ExtractedBlock]:
    try:
        reader = PdfReader(io.BytesIO(content))
        blocks: list[ExtractedBlock] = []
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            blocks.append(ExtractedBlock(text=text, page_number=page_index + 1))
    except ExtractionFailedError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(PDF_UNREADABLE) from exc
    return blocks
