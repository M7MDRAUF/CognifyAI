"""Opaque keyset cursors for the paged list endpoints (CT-06, CT-07).

A cursor names a position in a list ordered by ``(created_at DESC, id DESC)``: the last row a
client has seen. The next page is every row strictly after it in that order, which stays correct
when rows are added between requests and when two rows share a ``created_at`` -- the ``id`` breaks
the tie. Offsets do neither.

The token is unpadded URL-safe base64 of ``<created_at ISO-8601, microseconds, offset>|<uuid>``.
It carries no secret and grants no access: every list still filters by project. It is
caller-supplied, so decoding is strict and all-or-nothing. A token must be exactly what
``encode_cursor`` would produce for the position it decodes to; anything else is
``InvalidCursorError``, never a repaired position.
"""

import base64
import binascii
import re
import uuid
from datetime import datetime

from app.core.errors import InvalidCursorError

MAX_CURSOR_LENGTH = 256

_MESSAGE = "The page cursor is not valid. Reload the list."
_ALPHABET = re.compile(r"[A-Za-z0-9_-]+")


def encode_cursor(created_at: datetime, row_id: uuid.UUID) -> str:
    """The cursor for the row ``(created_at, row_id)``; ``created_at`` must be timezone-aware."""
    if created_at.utcoffset() is None:
        msg = "a cursor needs an aware datetime; a naive one names no instant"
        raise ValueError(msg)
    text = f"{created_at.isoformat(timespec='microseconds')}|{row_id}"
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> tuple[datetime, uuid.UUID]:
    """The position ``token`` names, or ``InvalidCursorError``."""
    if not token or len(token) > MAX_CURSOR_LENGTH or _ALPHABET.fullmatch(token) is None:
        raise InvalidCursorError(_MESSAGE)
    try:
        text = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise InvalidCursorError(_MESSAGE) from exc
    stamp, separator, ident = text.partition("|")
    if not separator or "|" in ident:
        raise InvalidCursorError(_MESSAGE)
    try:
        created_at = datetime.fromisoformat(stamp)
        row_id = uuid.UUID(ident)
    except ValueError as exc:
        raise InvalidCursorError(_MESSAGE) from exc
    if created_at.utcoffset() is None or encode_cursor(created_at, row_id) != token:
        raise InvalidCursorError(_MESSAGE)
    return created_at, row_id
