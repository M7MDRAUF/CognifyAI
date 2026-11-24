"""Embedding generation (FR-DOC-004).

Batches texts to the provider, validates that every returned vector has the
configured dimension before anything reaches Qdrant, and surfaces provider
failures as ``EmbeddingFailedError``.

With a meter, each call to a provider that names its vendor gives one
``embedding_calls`` row, success or failure, which the job writes when it ends
(RMD9-T04, RE2E-DEF-029).
"""

import logging
import time
from collections.abc import Sequence

from app.core.errors import EmbeddingFailedError, LLMProviderError
from app.integrations.llm.base import EmbeddingProvider, MeteredEmbeddingProvider, TokenUsage
from app.repositories.metering_repository import EmbeddingMeter

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        dimension: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        meter: EmbeddingMeter | None = None,
    ) -> None:
        self._provider = provider
        self._dimension = dimension
        self._batch_size = batch_size
        # Metered only when a vendor answers: a FakeProvider names none, as no vendor answers a
        # fake-mode chat step and no `llm_calls` row is written for one.
        self._metered = provider if isinstance(provider, MeteredEmbeddingProvider) else None
        self._meter = meter if self._metered is not None else None

    @property
    def batch_size(self) -> int:
        """How many texts a caller should hand over at a time.

        ``embed`` batches the provider call but still returns every vector it was asked
        for, so it bounds the provider payload and nothing else. A caller that wants only
        one batch of vectors resident has to stream at this size.
        """
        return self._batch_size

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        try:
            vectors = await self._call(batch)
        except LLMProviderError as exc:
            raise EmbeddingFailedError(
                "Failed to generate embeddings from the LLM provider."
            ) from exc
        if len(vectors) != len(batch):
            raise EmbeddingFailedError(
                f"Embedding provider returned {len(vectors)} vectors for {len(batch)} inputs."
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingFailedError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {len(vector)}. Check LLM_EMBEDDING_DIMENSIONS and the model."
                )
        return vectors

    async def _call(self, batch: Sequence[str]) -> list[list[float]]:
        """One provider call, and its metering row when there is a meter."""
        meter, metered = self._meter, self._metered
        if meter is None or metered is None:
            return await self._provider.embed_texts(batch)
        started = time.perf_counter()
        try:
            vectors, usage = await metered.embed_texts_metered(batch)
        except LLMProviderError as exc:
            _record(meter, metered, batch, started, None, failure_kind=exc.kind.value)
            raise
        _record(meter, metered, batch, started, usage)
        return vectors


def _record(
    meter: EmbeddingMeter,
    metered: MeteredEmbeddingProvider,
    batch: Sequence[str],
    started: float,
    usage: TokenUsage | None,
    *,
    failure_kind: str | None = None,
) -> None:
    meter.record(
        provider=metered.embedding_provider,
        model=metered.embedding_model,
        input_count=len(batch),
        prompt_tokens=usage.prompt_tokens if usage is not None and usage.reported else None,
        latency_ms=int((time.perf_counter() - started) * 1000),
        outcome="success" if failure_kind is None else "failure",
        failure_kind=failure_kind,
    )
