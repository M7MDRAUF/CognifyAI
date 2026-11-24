"""Shared validation for free-text fields that reach the database.

Separate from ``core.security``'s sanitizers on purpose: those CLEAN a value on
its way to a display surface, while these REJECT at the API boundary. A control
character in a project name is not something to silently repair -- no legitimate
client sends one, and accepting it means storing a value no reader can see.
"""

import unicodedata

# C0 and C1 excluding tab, newline and carriage return, which are legitimate in a
# multi-line question. U+0000 is the one that matters: PostgreSQL cannot store it
# in a text column at all, so without this check it reaches asyncpg and surfaces
# as a 500 carrying the statement and its bound parameters.
_FORBIDDEN = frozenset(
    chr(code)
    for code in [*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)]
    if chr(code) not in "\t\n\r"
)


def reject_control_characters(value: str, *, field: str) -> str:
    """Raise if ``value`` carries a control character that cannot be stored or shown."""
    found = sorted({character for character in value if character in _FORBIDDEN})
    if found:
        names = ", ".join(f"U+{ord(character):04X}" for character in found)
        msg = f"{field} must not contain control characters ({names})."
        raise ValueError(msg)
    return value


def is_visually_blank(value: str) -> bool:
    """True when ``value`` renders as nothing.

    ``str.strip()`` alone is not enough: it keys off ``str.isspace()``, which is
    False for Cf format characters, so a value of U+200B alone survives it while
    rendering as nothing -- and on the query path that buys a full billed run.
    """
    return all(
        character.isspace() or unicodedata.category(character) in ("Cf", "Cc")
        for character in value
    )
