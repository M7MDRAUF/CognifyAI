"""Owns one lazily-constructed, cached provider client per configured LLM provider.

Before this module existed, ``app.integrations.llm.provider.build_llm_provider`` built exactly
one client for the single provider named by ``LLM_PROVIDER`` and refused everything else (see
that module's docstring). With per-provider credentials and a model registry in place
(``registry.py``), a deployment can have several providers configured at once and a request can
ask for any of them by ``option_id``. ``LLMRouter`` is what actually owns those clients.

**There is no role-binding wrapper here, and none should be added.** ``OpenAIProvider`` already
maps ``ModelRole.SMALL`` to ``credentials.model_fast`` and ``ModelRole.STRONG`` to
``credentials.model_powerful`` inside a single client, and a ``ModelEntry.option_id`` already
carries its provider (``"<provider>:<model id>"``). So ``chat_for(option_id)`` only has to resolve
the option to *which provider's client* to hand back; *which of that provider's two models*
actually runs a given request is still decided by the ``ModelRole`` on the request, exactly as it
is today. A wrapper that bound a client to one role would duplicate that routing for no benefit —
don't add one "for symmetry" with anything.

**Client construction is lazy and cached** in ``_clients``, keyed by provider. Building a client
for every configured provider eagerly would open an HTTP connection pool for providers nobody
ends up querying in a given process lifetime, and ``AsyncOpenAI`` raises when handed a blank key —
so eager construction would turn "I configured two of five providers" into a boot failure for the
other three. A provider whose credentials do not resolve never gets a client at all; asking for
one raises a typed, provider-naming ``LLMProviderError`` instead.

**Every construction goes through the availability gate** (``_require_available``), which asks
``registry.availability_for`` — the same computation ``/llm/config`` and ``/health`` publish.
Resolvable credentials are necessary but not sufficient: a provider excluded by
``LLM_ENABLED_PROVIDERS``, or any real provider while the deployment runs offline, is refused
here even though its key resolves perfectly well. Without that gate the allow-list was advisory,
applied only where availability was *rendered* and never where a vendor was actually called.

**Closing:** the router builds its own ``AsyncOpenAI`` instance for every real client (injecting
it into ``OpenAIProvider`` via that class's existing ``client=`` parameter) instead of letting
``OpenAIProvider`` build one internally, purely so it has a handle to close on shutdown —
``OpenAIProvider`` has no public way to hand that back out, and this module has no license to add
one (see ``openai_provider.py``'s docstring: it is frozen except for one unrelated cleanup for
this task). The per-provider timeout table and the blank-key guard both live in
``openai_provider.py`` as the single source of truth — ``timeout_seconds_for`` and
``ensure_api_key_present`` below are imports, not copies, so this module's client construction
(the path production actually uses, since every real client is built here with ``client=``
injected into ``OpenAIProvider``) can never silently drift from that module's own.
"""

import logging

from openai import AsyncOpenAI

from app.core.config import LLMProviderName, Settings
from app.core.errors import LLMFailureKind, LLMProviderError
from app.integrations.llm import registry
from app.integrations.llm.base import ChatProvider, EmbeddingProvider, LLMProvider
from app.integrations.llm.embeddings import (
    OpenAICompatibleEmbeddingProvider,
    build_embedding_provider,
)
from app.integrations.llm.fake_provider import FakeProvider
from app.integrations.llm.openai_provider import (
    OpenAIProvider,
    ensure_api_key_present,
    request_timeout_for,
)
from app.integrations.llm.providers import DIALECTS
from app.integrations.llm.registry import ModelEntry, ProviderAvailability, ProviderCredentials

logger = logging.getLogger(__name__)


class LLMRouter:
    """Resolves model options and provider names to cached, lazily-built provider clients."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._override: LLMProvider | None = None
        self._clients: dict[LLMProviderName, LLMProvider] = {}
        self._raw_clients: dict[LLMProviderName, AsyncOpenAI] = {}
        self._embedding_client: OpenAICompatibleEmbeddingProvider | None = None
        self._closed = False

    @classmethod
    def from_single_provider(cls, settings: Settings, provider: LLMProvider) -> "LLMRouter":
        """Every accessor returns this one provider. For tests and dry runs."""
        router = cls(settings)
        router._override = provider
        return router

    def available_models(self) -> tuple[ModelEntry, ...]:
        return registry.available_models(self._settings)

    def provider_availability(self) -> tuple[ProviderAvailability, ...]:
        return registry.provider_availability(self._settings)

    def default_option_id(self) -> str | None:
        return registry.default_option_id(self._settings)

    def is_live(self) -> bool:
        """True when at least one real provider will actually be called -- see
        ``registry.is_live`` for why this decision cannot live on ``Settings`` itself."""
        return registry.is_live(self._settings)

    def find_model(self, option_id: str) -> ModelEntry | None:
        return registry.find_model(self._settings, option_id)

    def chat_for(self, option_id: str | None = None) -> ChatProvider:
        """Resolve ``option_id`` to its provider's client, or the deployment default's when None.

        ``None`` resolves through ``_default_client``, which is the same answer
        ``registry.default_option_id`` advertises through ``/llm/config``.


        An unknown or unavailable ``option_id`` raises ``ValueError`` — callers are expected to
        validate against ``available_models``/``find_model`` before reaching here.
        """
        if self._override is not None:
            return self._override
        if option_id is None:
            return self._default_client()
        entry = registry.find_model(self._settings, option_id)
        if entry is None:
            msg = f"Unknown or unavailable model option {option_id!r}."
            raise ValueError(msg)
        return self._client_for(entry.provider)

    def chat_for_provider(self, provider: str | None) -> ChatProvider:
        """Client for a provider named by its string value. ``None`` -> the deployment default.

        Built for ``run_query_job``, which resolves a chat client from a ``QueryRun``'s
        persisted ``resolved_provider`` — a plain ``str | None`` column, not an
        ``LLMProviderName``, since rows written before this feature (or by a fake-mode run)
        leave it ``NULL``. ``None`` and any string that is not a recognised provider both fall
        back to ``_default_client`` rather than raising: a background job has no caller to
        report a validation error to, and the *shape* of the value was already validated, if it
        needed to be, back when the query was created.

        What creation-time validation cannot speak for is *availability*, which is deployment
        configuration and can change between the row being written and the job running it — an
        operator who adds ``LLM_ENABLED_PROVIDERS`` and restarts has already-queued rows naming
        the provider they just switched off. ``_client_for``'s gate therefore still applies on
        this path, and a run whose ``resolved_provider`` is no longer available fails with a
        typed ``LLMProviderError`` rather than quietly billing that vendor.
        """
        if self._override is not None:
            return self._override
        if provider is None:
            return self._default_client()
        try:
            provider_name = LLMProviderName(provider)
        except ValueError:
            logger.warning(
                "Unrecognised resolved_provider %r; falling back to the default provider.",
                provider,
            )
            return self._default_client()
        return self._client_for(provider_name)

    def embeddings(self) -> EmbeddingProvider:
        """The embeddings client, pinned to ``LLM_EMBEDDING_PROVIDER`` (see ``embeddings.py``)."""
        if self._override is not None:
            return self._override
        if self._settings.llm_provider is LLMProviderName.FAKE:
            return self._client_for(LLMProviderName.FAKE)
        # Checked here as well as in `_client_for`: this accessor builds and caches its own
        # client, so without it a late ingestion job would open an HTTP pool `aclose` has
        # already finished walking and will never revisit.
        self._ensure_open(self._settings.llm_embedding_provider)
        if self._embedding_client is None:
            self._embedding_client = build_embedding_provider(self._settings)
        return self._embedding_client

    def require_embeddings(self) -> EmbeddingProvider:
        """Same resolution as ``embeddings()``, called once at startup to fail fast.

        Raises the same typed, ``LLM_OPENAI_API_KEY``-naming error ``embeddings()`` would raise
        lazily on the first ingestion job — see ``embeddings.py``'s ``build_embedding_provider``.
        Wired into the lifespan so a misconfigured embedding provider is a boot failure, not a
        background-task failure discovered only once a document is ingested.
        """
        return self.embeddings()

    def default_provider(self) -> ChatProvider:
        """The **chat** client for the provider ``LLM_PROVIDER`` names.

        Typed as ``ChatProvider``, not ``LLMProvider``, even though the object behind it is an
        ``OpenAIProvider`` that also implements ``embed_texts``. That incidental capability is
        how ``scripts/compare_baseline.py`` came to hand the *chat* client to
        ``EmbeddingService`` and send embedding requests to the chat vendor, bypassing
        ``LLM_EMBEDDING_PROVIDER`` and the ``_EMBEDDING_CAPABLE`` pin. Embeddings come from
        ``embeddings()``; this accessor must not be a second, unguarded door to them.
        """
        if self._override is not None:
            return self._override
        return self._client_for(self._settings.llm_provider)

    async def aclose(self) -> None:
        """Close every cached client's underlying ``AsyncOpenAI`` (and the embedding client's).

        Shutdown is *total*: every client that exists is closed exactly once, whatever any other
        client's ``close()`` does, and whatever else is running on the loop. Three properties,
        each of which was previously absent:

        - **Snapshot first.** The caches are copied and cleared *before* the first ``await``.
          Iterating ``self._raw_clients.values()`` across an ``await`` meant a background query
          job calling ``chat_for_provider``/``embeddings`` mid-shutdown (``workers/jobs.py``
          does exactly that, on an inner task) raised ``RuntimeError: dictionary changed size
          during iteration`` — after which *nothing* was closed and every pool stayed open.
        - **Per-item isolation.** One flaky vendor's ``close()`` must not abandon the others or
          the embedding client, for the same reason ``main.py``'s lifespan cleanup spells out:
          an earlier failure that skips later cleanups leaks every pool they hold. Failures are
          collected and re-raised together as an ``ExceptionGroup``, mirroring that ``finally``.
        - **Closed stays closed.** ``_client_for`` refuses afterwards, so a late arrival cannot
          build a client that nothing will ever close.
        """
        self._closed = True
        raw_clients = list(self._raw_clients.values())
        embedding_client = self._embedding_client
        self._raw_clients.clear()
        self._clients.clear()
        self._embedding_client = None

        failures: list[Exception] = []
        for raw_client in raw_clients:
            try:
                await raw_client.close()
            # Broad on purpose: every remaining client must still get its turn.
            except Exception as exc:
                failures.append(exc)
        if embedding_client is not None:
            try:
                await embedding_client.aclose()
            # Broad for the same reason as the loop above.
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("LLM client shutdown failed", failures)

    def _default_client(self) -> LLMProvider:
        """The one resolution of "no option was given": ``LLM_DEFAULT_MODEL``'s provider when it
        names an available entry, otherwise ``LLM_PROVIDER``'s.

        ``/llm/config`` advertises ``default_model_option_id`` as the id a query with no
        ``model_option_id`` will use, and ``registry.default_option_id`` answers
        ``LLM_DEFAULT_MODEL`` first. This method is the half that makes that branch true: without
        it the API named the operator's chosen option while this path quietly answered — and
        billed — from ``LLM_PROVIDER``'s vendor instead, on a run whose ``resolved_provider`` is
        ``NULL``, so nothing afterwards recorded which vendor had actually served it.

        **This is the only inference permitted here, and it is not really an inference.**
        ``LLM_DEFAULT_MODEL`` is one exact ``"<provider>:<model>"`` id the operator typed: a
        *more* specific instruction than ``LLM_PROVIDER``, not a competing one. Everything else
        the registry may advertise stays out: it must never become "use whichever vendor happens
        to hold a key", which would silently bill a vendor nobody selected — the same defect
        pointed the other way, and the one
        ``test_llm_router.py::test_chat_for_default_with_no_credentials_raises_naming_provider``
        and ``test_provider_factory_guard.py::test_provider_with_no_credentials_is_refused``
        exist to forbid. ``registry.find_model`` returns only entries the catalogue actually
        offers, so a stale or mistyped value redirects nothing, and the availability gate in
        ``_client_for`` still applies to whichever provider is resolved.

        ``default_provider()`` deliberately does *not* route through here: it means "the client
        for the provider ``LLM_PROVIDER`` names", which is a different question with its own
        callers (``build_llm_provider``, the operator scripts).
        """
        configured_default = self._settings.llm_default_model
        if configured_default is not None:
            entry = registry.find_model(self._settings, configured_default)
            if entry is not None:
                return self._client_for(entry.provider)
        return self._client_for(self._settings.llm_provider)

    def default_available(self) -> bool:
        """Whether the option-less path, ``_default_client``, would find its provider available.

        The same resolution, asked without building a client: ``LLM_DEFAULT_MODEL``'s entry when it
        names one the catalogue offers (``find_model`` returns only available entries), otherwise
        ``LLM_PROVIDER``'s availability. ``FAKE`` is always available, as ``_client_for`` treats it.
        """
        configured_default = self._settings.llm_default_model
        if configured_default is not None and registry.find_model(
            self._settings, configured_default
        ):
            return True
        provider = self._settings.llm_provider
        if provider is LLMProviderName.FAKE:
            return True
        availability = registry.availability_for(self._settings, provider)
        return availability is not None and availability.available

    def _client_for(self, provider: LLMProviderName) -> LLMProvider:
        cached = self._clients.get(provider)
        if cached is not None:
            return cached
        if provider is LLMProviderName.FAKE:
            # Resolved before the availability gate below: `FakeProvider` constructs no client,
            # places no call and spends nothing, so "is this provider available" does not apply
            # to it -- and `provider_availability` does not publish a row for it outside fake
            # mode anyway.
            fake = FakeProvider(dimension=self._settings.llm_embedding_dimensions)
            self._clients[provider] = fake
            return fake
        self._ensure_open(provider)
        self._require_available(provider)
        credentials = registry.resolve_credentials(self._settings, provider)
        if credentials is None:
            dialect = DIALECTS[provider]
            msg = (
                f"No credentials resolve for provider {provider.value!r}; set "
                f"{dialect.api_key_env}. Refusing to build a client with no key."
            )
            raise LLMProviderError(msg, kind=LLMFailureKind.AUTH, provider=provider.value)
        client, raw_client = self._build_chat_client(provider, credentials)
        self._clients[provider] = client
        self._raw_clients[provider] = raw_client
        return client

    def _ensure_open(self, provider: LLMProviderName) -> None:
        """Refuse to open a *new* connection pool once ``aclose`` has run.

        ``aclose`` closes the clients that exist when it starts; anything built afterwards would
        be closed by nobody. ``CONNECTION`` rather than ``AUTH`` because this is a transport
        condition, not a configuration one — and non-retryable, since the process is on its way
        down and retrying cannot bring it back.
        """
        if not self._closed:
            return
        msg = (
            f"This router has been closed; refusing to build a new client for "
            f"{provider.value!r} during shutdown."
        )
        raise LLMProviderError(msg, kind=LLMFailureKind.CONNECTION, provider=provider.value)

    def _require_available(self, provider: LLMProviderName) -> None:
        """Refuse a provider the catalogue reports unavailable, before any client is built.

        ``LLM_ENABLED_PROVIDERS`` and offline mode used to be applied only in
        ``registry.provider_availability`` — i.e. only on the path that *renders* the answer to
        "is an external LLM in use". This method is the enforcement half: with the allow-list
        excluding the default provider, ``/llm/config`` reported ``configured: false`` and the UI
        told the user every answer was local, while this method's absence let ``chat_for_provider``
        and ``embeddings()`` hand out live clients for that very provider and bill it. Same
        computation, now consulted by both halves.

        ``LLMFailureKind.AUTH`` for every reason, matching the no-credentials refusal just below
        it: all three are configuration refusals to build a client at all, none is retryable, and
        a second kind would only split one condition across two branches of every caller.
        """
        availability = registry.availability_for(self._settings, provider)
        if availability is not None and availability.available:
            return
        reason = availability.unavailable_reason if availability is not None else None
        if reason is registry.UnavailableReason.PROVIDER_DISABLED:
            detail = (
                f"it is excluded by LLM_ENABLED_PROVIDERS={self._settings.llm_enabled_providers!r}"
            )
        elif reason is registry.UnavailableReason.OFFLINE_MODE:
            detail = "this deployment runs offline (LLM_PROVIDER=fake)"
        elif reason is registry.UnavailableReason.MISSING_API_KEY:
            # Named explicitly rather than left to the generic branch below. Adding this gate in
            # front of `_client_for`'s no-credentials refusal made that refusal -- the one message
            # in this module that told an operator *which variable to set* -- unreachable, because
            # a provider with no key is reported unavailable here first. A missing key is also the
            # commonest reason by a wide margin, so the regression replaced the most useful
            # message with the least useful one on the most travelled path.
            detail = f"no API key is configured; set {DIALECTS[provider].api_key_env}"
        else:
            detail = "it is not currently available"
        msg = (
            f"Provider {provider.value!r} is not available for use: {detail}. Refusing to build "
            "a client for a provider this deployment reports as unavailable."
        )
        raise LLMProviderError(msg, kind=LLMFailureKind.AUTH, provider=provider.value)

    def _build_chat_client(
        self, provider: LLMProviderName, credentials: ProviderCredentials
    ) -> tuple[OpenAIProvider, AsyncOpenAI]:
        dialect = DIALECTS[provider]
        api_key = credentials.api_key.get_secret_value()
        # `registry.resolve_credentials` already filters a blank key to `None` before a
        # `ProviderCredentials` is ever built, so this should be unreachable in practice — but
        # this is the path production actually uses (every real client is built here, always
        # with `client=` injected into `OpenAIProvider`, which is exactly what makes
        # `OpenAIProvider`'s own guard dead code for production). Defend here too rather than
        # relying solely on the caller having done it.
        ensure_api_key_present(api_key, provider)
        # `without_ambient_openai_env` is what keeps the resolved credential the *only* thing on
        # the wire: the SDK otherwise reads `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and
        # `OPENAI_CUSTOM_HEADERS` from the process environment and merges them into the headers
        # of every request, to every vendor. See its docstring for why no argument here can
        # close that.
        with registry.without_ambient_openai_env():
            raw_client = AsyncOpenAI(
                api_key=api_key,
                base_url=credentials.base_url,
                timeout=request_timeout_for(provider),
                max_retries=0,  # retries are handled inside OpenAIProvider, same as every client
            )
        wrapped = OpenAIProvider(
            self._settings, dialect=dialect, credentials=credentials, client=raw_client
        )
        return wrapped, raw_client
