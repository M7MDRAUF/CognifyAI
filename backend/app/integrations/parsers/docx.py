"""DOCX extraction preserving paragraph structure and heading context."""

import io

from docx import Document as DocxDocument

from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import DOCX_UNREADABLE, ExtractedBlock


def extract_docx(content: bytes) -> list[ExtractedBlock]:
    try:
        document = DocxDocument(io.BytesIO(content))
        blocks: list[ExtractedBlock] = []
        current_section: str | None = None
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text == "":
                continue
            style = paragraph.style
            style_name = (style.name or "") if style is not None else ""
            if style_name.startswith("Heading"):
                current_section = text
            blocks.append(ExtractedBlock(text=text, section_title=current_section))
    except ExtractionFailedError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(DOCX_UNREADABLE) from exc
    return blocks
