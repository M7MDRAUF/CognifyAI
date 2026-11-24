"""Timezone-aware time helpers. Never use naive datetimes in application code."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)
