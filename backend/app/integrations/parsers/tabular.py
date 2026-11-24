"""CSV/XLSX extraction: rows become structured text blocks with column names."""

import csv
import io
from collections.abc import Iterable

from openpyxl import load_workbook

from app.core.config import upload_cap_mb
from app.core.errors import ExtractionFailedError
from app.integrations.parsers.base import CSV_UNREADABLE, XLSX_UNREADABLE, ExtractedBlock

ROWS_PER_BLOCK = 20

# Ceilings on the *expanded* document rather than on the bytes it arrived as. Upload validation
# bounds only the received representation; every expansion factor past it is chosen by the
# uploader, because a worksheet declares its own <dimension> and a single <row r="..."> makes
# openpyxl pad every absent row index up to it.
#
# Both halves of each ceiling are derived, and a document is charged against the smaller:
#
# 1. Per megabyte of the *advertised upload cap*. This is the memory bound, and it is the only
#    thing bounding memory: chunking refuses a document only after the whole grid is already
#    built. Measured with tracemalloc, the grid costs ~76 bytes per cell (21.7 MB for 300,000
#    cells, 60.2 MB for 1,000,000), plus each row's own list. The factors below therefore put
#    one document's peak grid at roughly 12x the cap -- about 300 MB on the deployed 25 MB
#    configuration -- which is the allocation an operator is agreeing to when they set the cap.
#
#    They are not set above the densest ordinary document, because no memory bound can be: a
#    three-column CSV of small integers carries 357,878 cells and 119,293 rows per MB of file,
#    so it reaches the cell ceiling at 45% of the cap. What matters is which ceiling bites
#    first, and for every shape measured it is the chunk ceiling -- that same CSV is stopped at
#    37% of the cap by ``chunking_service``, which refuses in the unit that is actually billed.
#    These two exist so that the refusal happens before the allocation, not instead of it.
#
# 2. Per byte of *this* file. A CSV cannot lie about its size: a row needs at least its line
#    terminator and a cell at least its separator, so one byte each is a proof, not an estimate.
#    A workbook can, because it arrives compressed -- which is why the ceiling on the cap alone
#    was never enough. A 4 KB worksheet declaring <row r="500000"> is refused by this half and by
#    nothing else.
#
#    The two workbook factors are measured against the *narrowest* shapes, not the average,
#    because those are the ones that fit the most rows into the fewest wire bytes. A
#    single-column sheet of one repeated short value -- as compressible as a real spreadsheet
#    gets -- costs 5.15 bytes a row, i.e. 0.194 rows per wire byte; 30 identical columns reach
#    0.467 cells per wire byte. Even a sheet of entirely empty rows, which carries no text at
#    all, only reaches 0.378. So 0.5 rows and 1.0 cells per wire byte clear every legitimate
#    shape measured (2.6x and 2.1x over the densest of them, and above the empty-row floor as
#    well) while still refusing the sparse-row bomb by more than two orders of magnitude.
#    A wider factor here is cheap and a narrow one is a silent false refusal, which is why these
#    are set from the narrow end rather than from the 7-column exports that are the common case.
#
# Neither of these is the spend ceiling. A document that survives both is still bounded by
# ``chunking_service.default_max_chunks()``, which is derived from the same cap and is what
# actually stops a large tabular document -- the ceilings here bound memory, that one bounds the
# vendor invoice.
ROWS_PER_CAP_MB = 80_000
CELLS_PER_CAP_MB = 160_000
_CSV_BYTES_PER_ROW = 1
_CSV_BYTES_PER_CELL = 1
_XLSX_BYTES_PER_ROW = 2
_XLSX_BYTES_PER_CELL = 1


class _ExpansionBudget:
    """Rows and cells one document may still materialize, charged as its grid is built.

    Charged per row *before* the row is copied, and every data row counts even when it is empty:
    openpyxl yields a zero-width padding row for each absent row index, so a budget that spent
    only on cells would never be spent at all by the cheapest amplification available.

    The header row of each table is charged for its cells but not as a row, so ``max_rows`` means
    exactly what the refusal says it means -- the data rows the document may carry. It used to be
    charged like any other, which refused a CSV of exactly ``max_rows`` records with a message
    naming ``max_rows`` as the limit.
    """

    def __init__(self, *, max_rows: int, max_cells: int) -> None:
        self._max_rows = max_rows
        self._max_cells = max_cells
        self._rows = max_rows
        self._cells = max_cells

    def charge(self, cells: int, *, header: bool = False) -> None:
        if not header:
            self._rows -= 1
        self._cells -= cells
        if self._rows < 0:
            raise ExtractionFailedError(
                f"Document exceeds the supported row count of {self._max_rows:,} data rows."
            )
        if self._cells < 0:
            raise ExtractionFailedError(
                f"Document exceeds the supported cell count of {self._max_cells:,} cells."
            )


def _budget_for(content: bytes, *, bytes_per_row: int, bytes_per_cell: int) -> _ExpansionBudget:
    """The smaller of what the upload cap affords and what this file could honestly hold.

    The ``+ 1`` on each per-file half is the row or cell that needs no separator: a final line
    without a terminator, a single-column row.
    """
    cap_mb = upload_cap_mb()
    return _ExpansionBudget(
        max_rows=min(cap_mb * ROWS_PER_CAP_MB, len(content) // bytes_per_row + 1),
        max_cells=min(cap_mb * CELLS_PER_CAP_MB, len(content) // bytes_per_cell + 1),
    )


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _rows_to_blocks(
    rows: Iterable[list[str]], *, section_title: str | None
) -> list[ExtractedBlock]:
    iterator = iter(rows)
    header = next(iterator, None)
    if header is None:
        return []
    columns = [name if name != "" else f"column_{i + 1}" for i, name in enumerate(header)]
    blocks: list[ExtractedBlock] = []
    pending: list[str] = []
    row_number = 1
    for row in iterator:
        row_number += 1
        cells = [
            f"{columns[i]}: {value}" for i, value in enumerate(row[: len(columns)]) if value != ""
        ]
        if cells:
            pending.append(f"Row {row_number} — " + "; ".join(cells))
        if len(pending) >= ROWS_PER_BLOCK:
            blocks.append(ExtractedBlock(text="\n".join(pending), section_title=section_title))
            pending = []
    if pending:
        blocks.append(ExtractedBlock(text="\n".join(pending), section_title=section_title))
    return blocks


def extract_csv(content: bytes) -> list[ExtractedBlock]:
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(decoded))
        budget = _budget_for(
            content, bytes_per_row=_CSV_BYTES_PER_ROW, bytes_per_cell=_CSV_BYTES_PER_CELL
        )
        rows: list[list[str]] = []
        for index, row in enumerate(reader):
            budget.charge(len(row), header=index == 0)
            rows.append([_stringify(cell) for cell in row])
        return _rows_to_blocks(rows, section_title=None)
    except ExtractionFailedError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(CSV_UNREADABLE) from exc


def extract_xlsx(content: bytes) -> list[ExtractedBlock]:
    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        blocks: list[ExtractedBlock] = []
        # One budget for the whole workbook, so extra sheets cannot each claim a fresh one. Each
        # sheet's first row is still exempt from the row half, which costs at most one row per
        # sheet and keeps the refusal's wording true for the single-sheet workbooks that are the
        # normal case.
        budget = _budget_for(
            content, bytes_per_row=_XLSX_BYTES_PER_ROW, bytes_per_cell=_XLSX_BYTES_PER_CELL
        )
        for sheet in workbook.worksheets:
            # A read-only worksheet takes its iteration bounds from the uploaded file's own
            # <dimension ref=...> element and pads every row out to the declared column count,
            # so a 1.6 KB workbook declaring A1:XFD1048576 hands us 16,384 values per row.
            # reset_dimensions() is openpyxl's own API for a worksheet whose declared dimensions
            # are wrong: it drops both bounds, after which each row is sized from the cells
            # actually stored. It does not bound how *many* rows arrive -- a sparse <row r="...">
            # still makes openpyxl emit one padding row per absent index -- which is the budget's
            # job.
            sheet.reset_dimensions()
            rows: list[list[str]] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                budget.charge(len(row), header=index == 0)
                rows.append([_stringify(cell) for cell in row])
            blocks.extend(_rows_to_blocks(rows, section_title=sheet.title))
        workbook.close()
    except ExtractionFailedError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(XLSX_UNREADABLE) from exc
    return blocks
