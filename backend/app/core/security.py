"""Upload validation and filename sanitization (SDD section 18).

Content signatures are checked with stdlib only: PDF by magic bytes,
DOCX/XLSX by ZIP structure, and text formats by UTF-8 decodability,
so a renamed binary cannot masquerade as a supported document.

The ZIP structure check is also where an archive's *expanded* size is bounded. The size limit
below applies to the bytes received, which for an archive is the compressed form; the member
count and the summed uncompressed size are the same limit applied to what those bytes become.
"""

import io
import re
import zipfile
from dataclasses import dataclass

from app.core.errors import FileTooLargeError, InvalidFileTypeError

ALLOWED_EXTENSIONS = frozenset({"pdf", "txt", "md", "docx", "csv", "xlsx"})

_ALLOWED_MIME_BY_EXTENSION: dict[str, frozenset[str]] = {
    "pdf": frozenset({"application/pdf"}),
    "txt": frozenset({"text/plain"}),
    "md": frozenset({"text/markdown", "text/plain", "text/x-markdown"}),
    "docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    "csv": frozenset({"text/csv", "application/csv", "application/vnd.ms-excel"}),
    "xlsx": frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
}

_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_BASENAME_LENGTH = 80

# The C0 and C1 control ranges, DEL, and the two Unicode line separators -- everything that
# ``str.splitlines()`` breaks on, plus everything that can move a terminal cursor. Unlike
# ``_UNSAFE_CHARS`` this keeps spaces, punctuation and non-Latin letters, because the name it
# cleans is read by a person and by a model rather than written to the filesystem.
_DISPLAY_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")

# Characters that occupy no width and change how the text around them is read: the bidi
# embedding, override and isolate controls (U+202A-U+202E, U+2066-U+2069), the zero-width
# characters with no orthographic role (U+200B, U+2060-U+2064, U+FEFF), and the Unicode tag
# block, which spells ASCII invisibly.
#
# Deliberately absent, and the reason this is a separate class from ``_DISPLAY_UNSAFE``:
#
# * U+200C and U+200D are orthography. Persian and the Indic scripts need U+200C between
#   morphemes, and an emoji sequence is joined with U+200D.
# * U+200E and U+200F are direction *marks*, not overrides, and appear in legitimately mixed
#   text.
# * Strong right-to-left letters are script, never a control. An Arabic or Hebrew name reads
#   right to left because of its own letters and must survive exactly as it was written.
#
# The trick this closes is the one that needs an explicit override: U+202E reverses everything
# after it, so ``invoice<RLO>fdp.exe`` is *displayed* as ``invoiceexe.pdf`` by any conforming
# renderer -- the frontend, the generated report, and a terminal reading a prompt alike.
_INVISIBLE = re.compile(
    "[\u200b\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]+"
)


def strip_invisible(text: str) -> str:
    """Remove bidi override/embedding controls and zero-width formatting characters.

    Removed rather than replaced with a space, unlike the control ranges above: these occupy no
    width, so a space would open a gap the author never wrote and would corrupt a name that is
    otherwise legitimate.
    """
    return _INVISIBLE.sub("", text)


_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_RECORD_LENGTH = 22
# The largest value the record's 16-bit comment-length field can hold, and therefore the window
# ``zipfile._EndRecData`` searches for the record (its ``ZIP_MAX_COMMENT``). Matched deliberately:
# a search narrower than CPython's would leave archives whose record the parser finds and this
# module does not -- unbounded for the same reason the entry count left them unbounded.
_EOCD_MAX_COMMENT_LENGTH = 65535

# The ZIP64 end-of-central-directory record and the locator that precedes the 32-bit record.
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_EOCD_LENGTH = 56
_ZIP64_EOCD_SIZE_OFFSET = 40
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_LOCATOR_LENGTH = 20

# The fixed part of one central-directory file header. Every record is this long plus its name,
# extra field and comment, so a directory of N bytes holds at most N // 46 records -- the bound
# ``_max_parsable_members`` rests on.
_CENTRAL_DIRECTORY_RECORD_LENGTH = 46

# An OOXML package is small: python-docx writes about a dozen parts, and even a figure-heavy
# report or a hundred-sheet workbook stays in the low hundreds. Within the per-file size limit a
# central directory can otherwise declare well over half a million entries, one ZipInfo each,
# allocated synchronously in the request path before any response is sent.
_MAX_ARCHIVE_MEMBERS = 2048
_ONE_MB = 1024 * 1024

# What an archive may declare it decompresses to. Two derived limits, both required, because
# neither one alone separates an office document from a bomb.
#
# The ratio is the discriminator. A previous fixed 128 MB was justified as "five times the 25 MB
# per-file limit", which is simply not what OOXML does: measured on workbooks openpyxl wrote
# itself, a realistic 120,000-row x 7-column sheet expands 8.7x (4.62 MB of wire bytes to
# 40.3 MB), and one whose cells repeat -- a category column, a status column, anything an
# ordinary export is full of -- reaches 23.5x. A four-paragraph .docx from python-docx is 22.5x
# on its boilerplate alone. So a full-size legitimate workbook expanded past that fixed ceiling
# and was refused at upload. A zip bomb is nothing like those: the 160 MB member the tests build
# arrives in ~160 KB, about 1000x. 64x sits above every legitimate ratio measured, with roughly
# 2.4x of headroom, and still refuses that bomb by more than an order of magnitude.
#
# The multiple of the per-file cap is the backstop, and it is needed because a ratio alone is
# unbounded in absolute terms: at 64x a full-size upload could declare 1.6 GB, and openpyxl
# materializes xl/sharedStrings.xml in full even in read-only mode, so "declared" would become
# "allocated". 12x admits the 8.7x realistic shape at full size with headroom and caps one
# archive at 300 MB on the deployed 25 MB configuration.
#
# Both are expressed against arguments rather than against a deployment: the cap arrives in
# ``validate_upload``'s own ``max_file_size_bytes``, so lowering MAX_UPLOAD_SIZE_MB tightens this
# without anything else being edited.
_MAX_ARCHIVE_EXPANSION_RATIO = 64
_MAX_ARCHIVE_EXPANSION_OF_CAP = 12


@dataclass(frozen=True)
class ValidatedUpload:
    extension: str
    sanitized_filename: str


def sanitize_filename(original_filename: str) -> str:
    """Return a safe basename: no path segments, no control or special chars."""
    # Take only the final path component regardless of separator style.
    basename = re.split(r"[\\/]", original_filename)[-1]
    basename = basename.strip().strip(".")
    stem, dot, extension = basename.rpartition(".")
    if dot == "":
        stem, extension = basename, ""
    stem = _UNSAFE_CHARS.sub("-", stem).strip("-_.")[:_MAX_BASENAME_LENGTH]
    extension = _UNSAFE_CHARS.sub("", extension).lower()[:10]
    if stem == "" or stem.lower() in _WINDOWS_RESERVED_NAMES:
        stem = "file"
    return f"{stem}.{extension}" if extension else stem


def sanitize_display_filename(original_filename: str, *, limit: int) -> str:
    """Return the user-visible name: readable, but unable to forge a line downstream.

    ``sanitize_filename`` above produces the on-disk basename and is deliberately far more
    aggressive than a displayed name should be. This is the other copy of the uploader's
    string -- ``Document.original_filename`` -- and it is the one that reaches the agent
    prompts, the Qdrant payload, the generated report and the API schema. PRD.md:542,
    SDD.md:1086 and the "Rules in force" section of docs/security.md all state that
    user-provided filenames are sanitized; until this existed that claim held only for the copy
    written to disk. That last reference is by section and not by line on purpose: the bullet it
    names has already moved once, and a line number into that file goes stale faster than the
    claim it points at does.

    Windows reserved names are deliberately left alone: this value never names a file.

    Backslash is kept as a path separator here even though this value never names a file and a
    Linux-legal ``my\\file.txt`` therefore loses its stem. The alternative is worse: a legacy
    ``Content-Disposition`` filename is a full Windows path, and preserving it would print the
    uploader's home directory into a report that gets shared. The cost is one rare filename
    rendering short; the benefit is not leaking a local directory tree.
    """
    basename = re.split(r"[\\/]", original_filename)[-1]
    cleaned = _DISPLAY_UNSAFE.sub(" ", strip_invisible(basename)).strip()
    return cleaned[:limit].strip() or "upload"


def _has_zip_signature(content: bytes) -> bool:
    return content.startswith(b"PK\x03\x04")


def _zip64_central_directory_size(content: bytes, eocd_index: int) -> int | None:
    """The central directory's size as the ZIP64 record states it, or None if there is none.

    Located the way ``zipfile._EndRecData64`` locates it: the locator sits immediately before the
    32-bit record, and the record itself is taken from the locator's own offset field when the
    signature is there, falling back to the position immediately before the locator.

    Read whenever the locator is present, not only when the 32-bit size carries the 0xFFFFFFFF
    sentinel, because CPython's override is unconditional as well -- ``_EndRecData64`` replaces
    the size and offset outright. An archive may therefore pair a truthful ZIP64 record with a
    32-bit field that understates it, and the parser will use the ZIP64 value; a guard reading
    the 32-bit field in that case would be reading a number nothing consults.

    Every slice below is taken at an offset the surrounding check has already proved is inside
    ``content``, so none of them can come back short -- which matters, because a short slice
    would read as a *smaller* integer and quietly weaken the bound built from it.
    """
    locator = eocd_index - _ZIP64_LOCATOR_LENGTH
    if locator < 0 or content[locator : locator + 4] != _ZIP64_LOCATOR_SIGNATURE:
        return None
    stated_offset = int.from_bytes(content[locator + 8 : locator + 16], "little")
    for candidate in (stated_offset, locator - _ZIP64_EOCD_LENGTH):
        if candidate < 0 or candidate + _ZIP64_EOCD_LENGTH > len(content):
            continue
        if content[candidate : candidate + 4] != _ZIP64_EOCD_SIGNATURE:
            continue
        start = candidate + _ZIP64_EOCD_SIZE_OFFSET
        return int.from_bytes(content[start : start + 8], "little")
    return None


def _max_parsable_members(content: bytes) -> int | None:
    """An upper bound on the ``ZipInfo`` objects ``zipfile`` can build from this archive.

    Read before ``zipfile.ZipFile`` is constructed: that constructor parses the whole central
    directory and allocates one ``ZipInfo`` per entry, so a count taken afterwards would not
    bound anything.

    The quantity read here is the directory's *byte size*, not its entry count, because the entry
    count is not what the parser obeys. ``ZipFile._RealGetContents`` walks ``while total < size_cd``
    and never looks at the end-of-central-directory record's total-entries field at all, so an
    archive that understates that field simply skips a guard built on it -- measured at roughly
    600x the intended cost, synchronously in the request path. Each record costs at least its
    46-byte fixed header, so ``size_cd // 46`` bounds the walk whatever else the archive claims,
    and the result is clamped by the uploaded length because ``fp.read(size_cd)`` cannot return
    more bytes than were uploaded.

    The bound overshoots: real records carry a name, so a package of N members usually reports
    room for something like 1.5N. That is the cost of a bound that cannot be lied about, and the
    ceiling has the headroom for it -- an OOXML package runs to a dozen parts, a figure-heavy
    report or a hundred-sheet workbook to the low hundreds.

    None means no readable record, and nothing needs bounding in that case: ``_EndRecData``
    returns None over the same window, so ``zipfile`` raises ``BadZipFile`` before it reads a
    single central-directory record.
    """
    if len(content) < _EOCD_RECORD_LENGTH:
        return None
    window = _EOCD_MAX_COMMENT_LENGTH + _EOCD_RECORD_LENGTH
    # The end bound keeps ``rfind`` off a truncated trailing signature that cannot be a record,
    # which CPython's fast path likewise steps over to reach the real one behind it.
    index = content.rfind(
        _EOCD_SIGNATURE,
        max(0, len(content) - window),
        len(content) - _EOCD_RECORD_LENGTH + len(_EOCD_SIGNATURE),
    )
    if index == -1:
        return None
    # The search above guarantees a whole record follows ``index``, so the 32-bit size field is
    # there to be read; it is used only when no ZIP64 record overrode it.
    size = _zip64_central_directory_size(content, index)
    if size is None:
        size = int.from_bytes(content[index + 12 : index + 16], "little")
    return min(size, len(content)) // _CENTRAL_DIRECTORY_RECORD_LENGTH


def _zip_member_prefixes(content: bytes, *, max_file_size_bytes: int) -> set[str]:
    """Top-level member names, rejecting an archive whose expanded form is out of bounds."""
    parsable_members = _max_parsable_members(content)
    if parsable_members is not None and parsable_members > _MAX_ARCHIVE_MEMBERS:
        raise InvalidFileTypeError(
            f"Archive's central directory has room for more than {_MAX_ARCHIVE_MEMBERS} members."
        )
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile:
        return set()
    # Unreachable while the bound above holds -- it is an upper bound on exactly this number --
    # and kept because it is exact and costs nothing: it is what notices if the directory walk
    # above is ever no longer the one CPython performs.
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise InvalidFileTypeError(f"Archive contains more than {_MAX_ARCHIVE_MEMBERS} members.")
    # file_size is the uncompressed length recorded in the central directory, and zipfile stops
    # handing out decompressed bytes once that many have been read, so no member can deliver more
    # than it declares here. Summing them bounds inflation before any parser opens the archive.
    expanded = sum(member.file_size for member in members)
    if expanded > len(content) * _MAX_ARCHIVE_EXPANSION_RATIO:
        raise InvalidFileTypeError(
            f"Archive decompresses to {expanded // max(len(content), 1)}x its uploaded size; "
            f"at most {_MAX_ARCHIVE_EXPANSION_RATIO}x is supported."
        )
    if expanded > max_file_size_bytes * _MAX_ARCHIVE_EXPANSION_OF_CAP:
        limit_mb = max_file_size_bytes * _MAX_ARCHIVE_EXPANSION_OF_CAP // _ONE_MB
        raise InvalidFileTypeError(f"Archive decompresses to more than {limit_mb} MB.")
    return {member.filename.split("/")[0] for member in members}


def _is_valid_text(content: bytes) -> bool:
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    return True


def _content_matches_extension(extension: str, content: bytes, *, cap_bytes: int) -> bool:
    if extension == "pdf":
        return content.startswith(b"%PDF-")
    if extension in {"docx", "xlsx"}:
        prefix = "word" if extension == "docx" else "xl"
        if not _has_zip_signature(content):
            return False
        return prefix in _zip_member_prefixes(content, max_file_size_bytes=cap_bytes)
    return _is_valid_text(content)


def validate_upload(
    original_filename: str,
    declared_mime_type: str,
    content: bytes,
    *,
    max_file_size_bytes: int,
) -> ValidatedUpload:
    """Validate extension, declared MIME type, size, and content signature."""
    sanitized = sanitize_filename(original_filename)
    _, dot, extension = sanitized.rpartition(".")
    if dot == "" or extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise InvalidFileTypeError(
            f"Unsupported file type for '{original_filename}'. Allowed types: {allowed}."
        )
    declared = declared_mime_type.split(";", maxsplit=1)[0].strip().lower()
    if declared not in _ALLOWED_MIME_BY_EXTENSION[extension]:
        raise InvalidFileTypeError(
            f"Declared content type '{declared}' does not match the .{extension} extension."
        )
    if len(content) == 0:
        raise InvalidFileTypeError("Uploaded file is empty.")
    if len(content) > max_file_size_bytes:
        limit_mb = max_file_size_bytes // (1024 * 1024)
        raise FileTooLargeError(f"File exceeds the {limit_mb} MB per-file limit.")
    if not _content_matches_extension(extension, content, cap_bytes=max_file_size_bytes):
        raise InvalidFileTypeError(f"File content does not look like a valid .{extension} file.")
    return ValidatedUpload(extension=extension, sanitized_filename=sanitized)
