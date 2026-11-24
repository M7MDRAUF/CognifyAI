"""Deterministic provider for tests and the dry-run mode.

Embeddings are seeded token-hash bag-of-words vectors: texts sharing words
land near each other under cosine distance, which is enough to exercise the
retrieval pipeline without a network call. Text/JSON responses are
programmable per test via ``script_*`` so the agent repair ladder and
failure paths (Phase 4) can be driven deterministically.
"""

import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Sequence

from app.core.errors import LLMProviderError
from app.integrations.llm.base import (
    JsonObject,
    JsonResult,
    TextRequest,
    TextResult,
    TokenUsage,
)
from app.integrations.llm.json_parsing import extract_json_object

_WORD = re.compile(r"[a-z0-9]+")


def _hash_to_index(word: str, dimension: int) -> int:
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimension


def deterministic_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        vector[_hash_to_index(word, dimension)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # Stable non-zero vector for empty/symbol-only text.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeProvider:
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self._text_script: deque[TextResult] = deque()
        self._json_script: deque[JsonResult | LLMProviderError] = deque()
        self.text_calls: list[TextRequest] = []
        self.json_calls: list[TextRequest] = []
        self.embed_calls: list[list[str]] = []

    def script_text(self, text: str) -> None:
        self._text_script.append(TextResult(text=text, usage=TokenUsage(1, 1)))

    def script_json(self, data: JsonObject) -> None:
        raw = json.dumps(data)
        self._json_script.append(JsonResult(raw_text=raw, data=data, usage=TokenUsage(1, 1)))

    def script_json_raw(self, raw_text: str) -> None:
        parsed = extract_json_object(raw_text) or {}
        self._json_script.append(JsonResult(raw_text=raw_text, data=parsed))

    def script_error(self, *, retryable: bool = False) -> None:
        self._json_script.append(LLMProviderError("scripted failure", retryable=retryable))

    async def generate_text(self, request: TextRequest) -> TextResult:
        self.text_calls.append(request)
        if self._text_script:
            return self._text_script.popleft()
        return TextResult(text="fake response", usage=TokenUsage(1, 1))

    async def generate_json(self, request: TextRequest) -> JsonResult:
        self.json_calls.append(request)
        if self._json_script:
            scripted = self._json_script.popleft()
            if isinstance(scripted, LLMProviderError):
                raise scripted
            return scripted
        return JsonResult(raw_text="{}", data={}, usage=TokenUsage(1, 1))

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [deterministic_embedding(text, self._dimension) for text in texts]
