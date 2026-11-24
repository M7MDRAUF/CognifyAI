"""Shared parser types and the fixed reasons a parser failure gives."""

from dataclasses import dataclass

# What a user reads next to Failed when a file of each format cannot be parsed (RMD6-T04,
# RE2E-DEF-022). Fixed text, never the library's exception text, which named its own internals
# ("File has not been decrypted") and gave no next step. Each parser raises these ``from`` the
# library's exception, so the ingestion log still records its class and message.
PDF_UNREADABLE = (
    "PDF text extraction failed: the file could not be read. It may be damaged, "
    "password-protected or not a PDF."
)
DOCX_UNREADABLE = (
    "DOCX text extraction failed: the file could not be read. It may be damaged or not a Word "
    "document."
)
CSV_UNREADABLE = (
    "CSV text extraction failed: the file could not be read. It must be comma-separated text in "
    "UTF-8."
)
XLSX_UNREADABLE = (
    "XLSX text extraction failed: the file could not be read. It may be damaged, "
    "password-protected or not an Excel workbook."
)


@dataclass(frozen=True)
class ExtractedBlock:
    text: str
    page_number: int | None = None
    section_title: str | None = None
