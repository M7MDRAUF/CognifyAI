"""The embeddings client, split out from the chat provider so it stays pinned to a provider whose
embeddings API has actually been verified — independent of whatever ``LLM_PROVIDER`` (the *chat*
provider) happens to be, because ``LLM_EMBEDDING_PROVIDER`` can name a different one.

Only OpenAI's embeddings API has been verified against primary documentation. DeepSeek does not
document one at all; Z.ai's embedding-3 is reported to exist only on the China platform and to
reject requests on the global endpoint; NVIDIA and Moonshot host retrieval models whose ids and
vector dimensions we have not confirmed. Widening ``_EMBEDDING_CAPABLE`` requires verification,
not optimism.
"""

from collections.abc import Sequence
from typing import Final

from openai import AsyncOpenAI

from app.core.config import LLMProviderName, Settings
from app.core.errors import LLMFailureKind, LLMProviderError
from app.integrations.llm.base import TokenUsage
from app.integrations.llm.openai_provider import OpenAIProvider, request_timeout_for
from app.integrations.llm.providers import DIALECTS
from app.integrations.llm.registry import (
    ProviderCredentials,
    UnavailableReason,
    embedding_allow_list,
    embedding_availability_for,
    resolve_credentials,
    without_ambient_openai_env,
)

_EMBEDDING_CAPABLE: Final = frozenset({LLMProviderName.OPENAI})


class OpenAICompatibleEmbeddingProvider:
    """``EmbeddingProvider`` pinned to one verified, credentialed provider.

    Delegates the actual wire call to ``OpenAIProvider.embed_texts`` so embeddings get the same
    retry/backoff/error-classification behaviour as chat calls, and — for the default OpenAI
    configuration — send a byte-identical request (same model, same ``input``, same
    ``dimensions``) to what this codebase sent before the embedding path was split out of the
    chat provider.

    ``dimensions=`` is an OpenAI-specific request parameter, and ``OpenAIProvider.embed_texts``
    sends it unconditionally. That is only safe because this class is only ever constructed for a
    provider in ``_EMBEDDING_CAPABLE`` (enforced by ``build_embedding_provider`` below, the only
    place that constructs one) — membership in that set *is* the gate. If it ever widens to a
    provider that does not accept ``dimensions``, sending it unconditionally would break that
    provider, and gating will need to move here.
    """

    def __init__(self, settings: Settings, *, credentials: ProviderCredentials) -> None:
        dialect = DIALECTS[credentials.provider]
        # Same reasoning as `LLMRouter._build_chat_client`: the SDK reads `OPENAI_ORG_ID`,
        # `OPENAI_PROJECT_ID` and `OPENAI_CUSTOM_HEADERS` from the process environment and merges
        # them into every request's headers, and no constructor argument can fully contest that.
        # The embedding client ingests whole documents, so it is the last place an ambient
        # header should be able to redirect or re-authenticate a request.
        with without_ambient_openai_env():
            self._client = AsyncOpenAI(
                api_key=credentials.api_key.get_secret_value(),
                base_url=credentials.base_url,
                # L-9: an explicit `httpx.Timeout`, as the chat clients have had since P2-T10. A
                # bare float made the connect leg the whole budget, and ingestion has no run
                # deadline above this client.
                timeout=request_timeout_for(credentials.provider),
                max_retries=0,  # retries are handled inside OpenAIProvider
            )
        self._delegate = OpenAIProvider(
            settings, dialect=dialect, credentials=credentials, client=self._client
        )

    @property
    def embedding_provider(self) -> str:
        return self._delegate.embedding_provider

    @property
    def embedding_model(self) -> str:
        return self._delegate.embedding_model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._delegate.embed_texts(texts)

    async def embed_texts_metered(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], TokenUsage]:
        """The delegate's vectors and reported usage, for the metering row (RMD9-T04)."""
        return await self._delegate.embed_texts_metered(texts)

    async def aclose(self) -> None:
        await self._client.close()


def embedding_destination(settings: Settings) -> tuple[LLMProviderName | None, bool]:
    """Where document text is actually sent to be embedded: ``(provider, endpoint_is_default)``.

    Published by ``/llm/config`` so the UI can stop asserting a destination it cannot see. It had
    to: the frontend named "OpenAI" from a constant, reasoning that ``_EMBEDDING_CAPABLE`` pins
    the provider. It pins the *dialect*, not the *host* -- ``LLM_OPENAI_BASE_URL``, or the legacy
    ``LLM_BASE_URL`` self-host hatch that ``.env.example`` documents, redirects the same
    credential to an entirely different endpoint. A claim about where a user's documents go must
    be derived here, where the answer is known, not asserted in a component.

    ``(None, ...)`` means nothing is embedded remotely: fake/offline mode, or a configuration in
    which no embedding provider resolves at all. The second element is ``False`` whenever a base
    URL override is in play, which is the signal the UI needs to stop naming a vendor: the API
    dialect is still OpenAI's, but the operator, not OpenAI, chose the host.
    """
    if settings.llm_provider is LLMProviderName.FAKE:
        return None, False
    provider = settings.llm_embedding_provider
    availability = embedding_availability_for(settings, provider)
    if provider not in _EMBEDDING_CAPABLE or availability is None or not availability.available:
        return None, False
    credentials = resolve_credentials(settings, provider)
    if credentials is None:
        return None, False
    return provider, credentials.base_url == DIALECTS[provider].default_base_url


def build_embedding_provider(settings: Settings) -> OpenAICompatibleEmbeddingProvider:
    """Resolve ``settings.llm_embedding_provider`` to a client, or raise naming the fix.

    This is the startup guard: without it, setting ``LLM_PROVIDER=deepseek`` (or any provider
    outside ``_EMBEDDING_CAPABLE``) would boot happily and then fail every document ingestion and
    every query at runtime, inside a background task — the worst place to discover a missing key.
    Called both by ``LLMRouter.require_embeddings`` (wired into the lifespan, so this becomes a
    startup failure) and by ``LLMRouter.embeddings`` (the request-time accessor).

    Resolvable credentials are not enough: the provider must also be one this deployment allows
    to embed (``registry.embedding_availability_for``, the same computation ``/llm/config``'s
    embedding destination renders). Ingestion is the largest disclosure in the product — every
    chunk of every uploaded document is sent to this client — so an operator who excludes a
    vendor with ``LLM_ENABLED_EMBEDDING_PROVIDERS`` (or, when that is unset,
    ``LLM_ENABLED_PROVIDERS``) must not keep having documents embedded by it. Fake mode never
    reaches here at all: ``LLMRouter.embeddings`` resolves ``FAKE`` first.
    """
    provider = settings.llm_embedding_provider
    availability = embedding_availability_for(settings, provider)
    credentials: ProviderCredentials | None = None
    if provider in _EMBEDDING_CAPABLE and availability is not None and availability.available:
        credentials = resolve_credentials(settings, provider)
    if credentials is None:
        disabled = (
            availability is not None
            and availability.unavailable_reason is UnavailableReason.PROVIDER_DISABLED
        )
        list_name, list_value = embedding_allow_list(settings)
        allow_list = f" It is currently excluded by {list_name}={list_value!r}." if disabled else ""
        msg = (
            f"No usable embeddings provider is configured (LLM_EMBEDDING_PROVIDER="
            f"{provider.value!r}). Only {sorted(p.value for p in _EMBEDDING_CAPABLE)} "
            f"has a verified embeddings API today.{allow_list} Set LLM_EMBEDDING_PROVIDER=openai "
            "and LLM_OPENAI_API_KEY. Refusing to start rather than fail inside a background task."
        )
        raise LLMProviderError(msg, kind=LLMFailureKind.AUTH, provider=provider.value)
    return OpenAICompatibleEmbeddingProvider(settings, credentials=credentials)
