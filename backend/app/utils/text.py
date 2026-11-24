"""Text helpers shared by chunking and retrieval."""

import re

_WHITESPACE = re.compile(r"\s+")

# Documented heuristic: roughly 4 characters per token for English prose.
# Deterministic and dependency-free; the swap-in point if a real tokenizer
# is ever wanted is exactly this function.
CHARS_PER_TOKEN = 4


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if stripped == "":
        return 0
    return max(1, len(stripped) // CHARS_PER_TOKEN)
