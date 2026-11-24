"""Turns settings plus provider dialects into the catalogue of selectable models.

``config.py`` stays dumb on purpose (see its module docstring and the layering rule in the
Phase 2 design notes): it knows only flat fields and one generic accessor,
``Settings.provider_setting``. This module is where resolution actually happens — it imports the
dialects and combines them with settings to answer "what can this deployment actually call, and
with what credentials and model ids."

Import direction, never violated: ``registry.py`` -> ``providers/`` -> ``dialects.py`` ->
``config.py``.

**The escape hatch:** ``available_models`` builds its catalogue from *resolved* credentials
rather than a frozen list of known model ids. A model named in ``LLM_<PROVIDER>_MODEL_FAST`` (or
``_POWERFUL``) that this codebase has never heard of still produces a valid ``ModelEntry``,
carrying that provider's default capabilities. Vendors ship new model ids constantly; picking one
up is a one-line ``.env`` change, never a code change.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import SecretStr

from app.core.config import LLMProviderName, Settings
from app.domain.enums import AnswerModel
from app.integrations.llm.dialects import ProviderDialect
from app.integrations.llm.providers import DIALECTS

_FAST_DESCRIPTION = "Quicker and cheaper synthesis"
_POWERFUL_DESCRIPTION = "Highest quality synthesis (recommended)"

# The values `backend/.env.example` ships as keys (lines 11 and 68-74). Each is syntactically a
# key, so copied verbatim it made its provider available: `/llm/config` offered its options and
# every query to them failed at the vendor (RE2E-DEF-003). Exact values only: a local
# OpenAI-compatible server accepts any key, and the live profiles key theirs `local`.
PLACEHOLDER_API_KEYS: Final = frozenset({"sk-replace-me", "nvapi-replace-me", "replace-me"})

# Context windows with a recorded source, by provider and model id. DeepSeek's Models & Pricing
# page documents 1M tokens for `deepseek-flash` (DeepSeek-V4.1-Flash-1017), as
# DEEPSEEK_V41_FLASH_1017_PROVIDER_PREFLIGHT.md records it, read the way the DeepSeek dialect
# reads "384K" of output as 384,000. Any other model is unknown and published as null: a guess
# would be shown to the user as a fact (DV-RMD-48).
_DOCUMENTED_CONTEXT_WINDOW_TOKENS: Final = {
    (LLMProviderName.DEEPSEEK, "deepseek-flash"): 1_000_000,
}

# Every environment variable `openai.AsyncOpenAI.__init__` consults when the corresponding
# argument is unset (`openai==2.41.1`, `_client.py:173-224`). Enumerated rather than matched by
# an `OPENAI_*` prefix so adding one is a deliberate edit against the pinned SDK, not a guess.
_AMBIENT_OPENAI_ENV_VARS: Final = (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_WEBHOOK_SECRET",
)


@contextmanager
def without_ambient_openai_env() -> Generator[None]:
    """Hide the SDK's ambient configuration variables for the duration of a client construction.

    This is the third leg of the same guarantee ``resolve_credentials`` makes for the key and the
    base URL, which is why it lives here rather than next to either construction site: *nothing
    but the credentials this module resolved may reach the wire.* The pinned SDK treats the
    process environment as a configuration source for seven values, and two of them are
    cross-vendor hazards on every provider, not just OpenAI: ``OPENAI_ORG_ID``/
    ``OPENAI_PROJECT_ID`` stamp the operator's OpenAI account identifiers onto a DeepSeek or Kimi
    request, and ``OPENAI_CUSTOM_HEADERS`` injects *arbitrary* headers — including a replacement
    ``Authorization``, which silently substitutes the bearer token the app resolved.

    **Argument-level defences cannot close this.** ``default_headers={}`` does not work:
    ``_client.py:224`` is ``{**parsed, **default_headers}``, so an empty mapping merges to the
    env-parsed headers unchanged. ``organization=Omit()``/``project=Omit()`` plus an explicit
    ``Authorization`` entry closes those three names, but any *other* injected header still
    rides along, because only keys you name can be overridden. The environment is the only
    place the whole channel can be shut.

    Scrubbing is scoped to the constructor call rather than done once at process start because
    this package has no startup hook of its own, and a process-wide mutation would be invisible
    to anyone reading the construction site. ``AsyncOpenAI.__init__`` is synchronous and contains
    no ``await``, so no other coroutine on the event loop can observe the scrubbed environment;
    it is restored in a ``finally`` regardless of how construction ends.
    """
    saved = {name: os.environ[name] for name in _AMBIENT_OPENAI_ENV_VARS if name in os.environ}
    for name in saved:
        del os.environ[name]
    try:
        yield
    finally:
        os.environ.update(saved)


@dataclass(frozen=True)
class ProviderCredentials:
    provider: LLMProviderName
    api_key: SecretStr
    base_url: str | None
    model_fast: str
    model_powerful: str


@dataclass(frozen=True)
class ModelEntry:
    option_id: str  # "<provider>:<model id>", e.g. "deepseek:deepseek-v4-pro"
    provider: LLMProviderName
    model: str  # the id sent on the wire
    tier: AnswerModel
    display_name: str
    description: str


class UnavailableReason(StrEnum):
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_DISABLED = "provider_disabled"
    OFFLINE_MODE = "offline_mode"


@dataclass(frozen=True)
class ProviderAvailability:
    provider: LLMProviderName
    label: str
    available: bool
    unavailable_reason: UnavailableReason | None
    is_default: bool


def resolve_credentials(
    settings: Settings, provider: LLMProviderName
) -> ProviderCredentials | None:
    """Resolve the API key, base URL and model ids to use for ``provider``.

    Precedence for each value: the per-provider ``LLM_<PROVIDER>_*`` setting; then, only when
    ``provider`` is the one named by ``LLM_PROVIDER``, the matching legacy field; then the
    dialect default (no legacy fallback exists for the API key, since there is nothing safe to
    default a credential to). Returns ``None`` when no API key resolves at all — that is what
    "unavailable" means.

    **The key and the base URL are resolved as one unit, not by two independent ladders.** The
    legacy ``LLM_BASE_URL`` is only consulted when the key *itself* came from the legacy
    ``LLM_API_KEY``. Gating both on "is this the provider ``LLM_PROVIDER`` names" looks
    symmetric but is not: the *sources* can differ even when that gate agrees. A deployment that
    uses the documented self-host escape hatch (``LLM_PROVIDER=openai`` plus
    ``LLM_BASE_URL=http://ollama.local:11434/v1``, ``.env.example:23``) and *later* adds a real
    ``LLM_OPENAI_API_KEY`` would otherwise pair that real key with the leftover URL and send it,
    as a bearer token, to a host that is not OpenAI. An explicitly-configured per-provider key
    therefore gets either its own ``LLM_<PROVIDER>_BASE_URL`` or the dialect default, never the
    legacy one.

    The mirror direction — a legacy key with a per-provider ``LLM_<PROVIDER>_BASE_URL`` — is
    deliberately left open: that URL is configured *for the very provider* whose key resolved, so
    no vendor mixing can occur, and it is how an operator points the legacy configuration at a
    proxy for that same vendor.

    A ``.env.example`` placeholder (``PLACEHOLDER_API_KEYS``) is read as no key wherever a key is
    read, exactly like a blank one, so it never makes a provider available (RE2E-DEF-003).
    """
    dialect = DIALECTS[provider]
    is_legacy_provider = provider is settings.llm_provider

    api_key = _explicit_key(settings, provider)
    used_legacy_key = False
    if api_key is None and is_legacy_provider:
        api_key = _usable_key(settings.llm_api_key.get_secret_value())
        used_legacy_key = api_key is not None
    if api_key is None:
        return None

    base_url = settings.provider_setting(provider, "base_url")
    if base_url is None and used_legacy_key:
        # Blank (e.g. an uncommented `LLM_BASE_URL=` in .env.example) means unset, matching
        # `provider_setting`'s own blank handling — not "use an empty base URL".
        base_url = (settings.llm_base_url or "").strip() or None
    if base_url is None:
        base_url = dialect.default_base_url

    model_fast = settings.provider_setting(provider, "model_fast")
    if model_fast is None and is_legacy_provider:
        # Blank (e.g. an uncommented `LLM_TEXT_MODEL_SMALL=`) means unset, matching
        # `provider_setting`'s own blank handling -- not "send an empty model id on the wire".
        model_fast = settings.llm_text_model_small.strip() or None
    if model_fast is None:
        model_fast = dialect.default_model_fast

    model_powerful = settings.provider_setting(provider, "model_powerful")
    if model_powerful is None and is_legacy_provider:
        # Same blank handling as model_fast above.
        model_powerful = settings.llm_text_model_strong.strip() or None
    if model_powerful is None:
        model_powerful = dialect.default_model_powerful

    return ProviderCredentials(
        provider=provider,
        api_key=SecretStr(api_key),
        base_url=base_url,
        model_fast=model_fast,
        model_powerful=model_powerful,
    )


def _usable_key(raw: str | None) -> str | None:
    """``raw`` stripped, or ``None`` when it is blank or a ``.env.example`` placeholder."""
    key = (raw or "").strip()
    return key if key and key not in PLACEHOLDER_API_KEYS else None


def _explicit_key(settings: Settings, provider: LLMProviderName) -> str | None:
    """The usable key set in ``provider``'s own ``LLM_<PROVIDER>_API_KEY``, if any."""
    return _usable_key(settings.provider_setting(provider, "api_key"))


def _allow_list(raw: str | None) -> frozenset[str] | None:
    """None means "no allow-list configured"; otherwise the lower-cased names it permits.

    Blank or whitespace-only (``LLM_ENABLED_PROVIDERS=``) is treated the same as unset, consistent
    with ``Settings.provider_setting``'s own blank handling — otherwise an operator clearing that
    line (``.env.example`` ships it commented out) gets an allow-list that permits nothing, and
    every provider becomes unavailable.
    """
    if raw is None or not raw.strip():
        return None
    return frozenset(name.strip().lower() for name in raw.split(",") if name.strip())


def _enabled_provider_names(settings: Settings) -> frozenset[str] | None:
    """The chat allow-list, ``LLM_ENABLED_PROVIDERS`` (see ``_allow_list``)."""
    return _allow_list(settings.llm_enabled_providers)


def embedding_allow_list(settings: Settings) -> tuple[str, str | None]:
    """The setting that decides which providers may embed document text, and its raw value.

    ``LLM_ENABLED_EMBEDDING_PROVIDERS``, or the chat list ``LLM_ENABLED_PROVIDERS`` when it is
    unset or blank, so a configuration without it behaves as it did before it existed.
    """
    own = settings.llm_enabled_embedding_providers
    if own is not None and own.strip():
        return "LLM_ENABLED_EMBEDDING_PROVIDERS", own
    return "LLM_ENABLED_PROVIDERS", settings.llm_enabled_providers


def _configured_availability(
    settings: Settings,
    provider: LLMProviderName,
    enabled: frozenset[str] | None,
    *,
    is_default: bool,
) -> ProviderAvailability:
    """One real provider's row outside fake mode: the allow-list ``enabled``, then its key."""
    label = DIALECTS[provider].label
    if enabled is not None and provider.value.lower() not in enabled:
        return ProviderAvailability(
            provider=provider,
            label=label,
            available=False,
            unavailable_reason=UnavailableReason.PROVIDER_DISABLED,
            is_default=is_default,
        )
    resolved = resolve_credentials(settings, provider) is not None
    return ProviderAvailability(
        provider=provider,
        label=label,
        available=resolved,
        unavailable_reason=None if resolved else UnavailableReason.MISSING_API_KEY,
        is_default=is_default,
    )


def provider_availability(settings: Settings) -> tuple[ProviderAvailability, ...]:
    """Report, for every known, *selectable* provider, whether it is usable and why not when
    it isn't.

    Availability is derived from configuration only — never from a network probe. A working
    ``GET /v1/models`` does not prove chat works (some NVIDIA accounts get exactly that
    combination without the chat entitlement), and the endpoint this feeds must stay fast and
    always-200.

    ``FAKE`` is excluded entirely unless it is the active provider (``settings.llm_provider``):
    it has no API key of its own to be "missing", so outside of an actual fake-mode deployment
    it would only ever show up as permanently, uninterestingly unavailable. Excluding it here --
    rather than reporting it as unavailable -- is what makes it genuinely *unknown* to every
    caller of this function: ``/llm/config`` (``app/api/routes/llm.py``) never publishes it as a
    provider to pick, and ``QueryService`` (``app/services/query_service.py``) must therefore
    reject a ``fake:`` option id as ``invalid_model_option`` (400, unrecognised) rather than
    ``model_option_unavailable`` (409, known-but-down) in that case. This is the single
    statement of "fake is not selectable unless it is the active provider" -- both callers read
    it from here instead of each encoding their own copy of the rule.

    When ``FAKE`` *is* the active provider, every entry here must agree with
    ``available_models``, which returns exactly the two synthetic ``FAKE_DIALECT`` entries in
    that mode and nothing else -- no real provider's credentials, however genuinely configured
    (the test suite's own default environment force-sets a per-provider sentinel key for all
    five real providers; see ``tests/conftest.py``), are ever turned into a selectable option
    while running fake/offline. So ``FAKE`` is reported available unconditionally (it needs no
    credential to be "usable" here), every real provider is reported unavailable regardless of
    its own resolved credentials or the ``LLM_ENABLED_PROVIDERS`` allow-list (neither one changes
    what ``available_models`` offers in fake mode either), and the allow-list/missing-key
    machinery below is skipped entirely for this branch -- applying it would risk a real
    provider's credentials making it look available again, which is exactly the contradiction
    this guards against.
    """
    if settings.llm_provider is LLMProviderName.FAKE:
        return tuple(
            ProviderAvailability(
                provider=provider,
                label=DIALECTS[provider].label,
                available=provider is LLMProviderName.FAKE,
                unavailable_reason=(
                    None if provider is LLMProviderName.FAKE else UnavailableReason.OFFLINE_MODE
                ),
                is_default=provider is LLMProviderName.FAKE,
            )
            for provider in LLMProviderName
        )

    enabled = _enabled_provider_names(settings)
    return tuple(
        _configured_availability(
            settings, provider, enabled, is_default=provider is settings.llm_provider
        )
        for provider in LLMProviderName
        if provider is not LLMProviderName.FAKE
    )


def availability_for(settings: Settings, provider: LLMProviderName) -> ProviderAvailability | None:
    """The row ``provider_availability`` publishes for ``provider``, or ``None`` when it
    publishes none.

    This is the single enforcement point the execution path uses. Before it existed, availability
    was computed here and consumed *for display only* — ``/llm/config`` and ``/health`` read it,
    while ``LLMRouter._client_for`` and ``build_embedding_provider`` went straight to
    ``resolve_credentials`` and happily built a live client for a provider the very same API had
    just reported unavailable. Both now ask this function first, so "what we told the user" and
    "what we will actually call" are derived from one computation rather than two.

    ``None`` means the provider is not published at all: ``provider_availability`` omits ``FAKE``
    outside fake mode (see its docstring). Callers must treat that as "not usable" too — except
    for ``FAKE`` itself, which builds no client and places no call, and which
    ``LLMRouter._client_for`` therefore resolves before consulting this.
    """
    for availability in provider_availability(settings):
        if availability.provider is provider:
            return availability
    return None


def embedding_availability_for(
    settings: Settings, provider: LLMProviderName
) -> ProviderAvailability | None:
    """Whether ``provider`` may embed document text: ``availability_for``'s rule, with the
    embedding allow-list (``embedding_allow_list``) in place of the chat one (RE2E-DEF-002).

    The chat list used to decide both, so limiting chat to DeepSeek disabled an OpenAI-dialect
    embedding server too and the backend refused to start; leaving it unset was the only working
    configuration, and it offered that embedding server as an OpenAI chat model. Fake mode and
    ``FAKE`` itself keep ``availability_for``'s answer.
    """
    if settings.llm_provider is LLMProviderName.FAKE or provider is LLMProviderName.FAKE:
        return availability_for(settings, provider)
    _, raw = embedding_allow_list(settings)
    return _configured_availability(
        settings,
        provider,
        _allow_list(raw),
        is_default=provider is settings.llm_embedding_provider,
    )


def context_window_tokens(provider: LLMProviderName, model: str) -> int | None:
    """``model``'s documented context window on ``provider``, or ``None`` when none is recorded."""
    return _DOCUMENTED_CONTEXT_WINDOW_TOKENS.get((provider, model))


def _model_entry(
    *, provider: LLMProviderName, model: str, tier: AnswerModel, dialect: ProviderDialect
) -> ModelEntry:
    description = _FAST_DESCRIPTION if tier is AnswerModel.FAST else _POWERFUL_DESCRIPTION
    return ModelEntry(
        option_id=f"{provider.value}:{model}",
        provider=provider,
        model=model,
        tier=tier,
        display_name=f"{dialect.label} ({model})",
        description=description,
    )


def available_models(settings: Settings) -> tuple[ModelEntry, ...]:
    """The catalogue of selectable models: two entries (fast, powerful) per available provider.

    When ``settings.llm_provider`` is ``FAKE`` this returns exactly the two synthetic
    ``FAKE_DIALECT`` entries (``fake:fake-small``, ``fake:fake-strong``) and nothing else, so a
    dry-run deployment still renders a capability list without contacting any real provider.
    """
    if settings.llm_provider is LLMProviderName.FAKE:
        fake_dialect = DIALECTS[LLMProviderName.FAKE]
        return (
            _model_entry(
                provider=LLMProviderName.FAKE,
                model=fake_dialect.default_model_fast,
                tier=AnswerModel.FAST,
                dialect=fake_dialect,
            ),
            _model_entry(
                provider=LLMProviderName.FAKE,
                model=fake_dialect.default_model_powerful,
                tier=AnswerModel.POWERFUL,
                dialect=fake_dialect,
            ),
        )

    entries: list[ModelEntry] = []
    for availability in provider_availability(settings):
        if not availability.available:
            continue
        credentials = resolve_credentials(settings, availability.provider)
        if credentials is None:
            continue
        dialect = DIALECTS[availability.provider]
        if credentials.model_fast == credentials.model_powerful:
            # Fast and powerful resolve to the same model id (a self-hosted NIM or a locked-down
            # Azure deployment, say) — there is genuinely only one model here. Emitting two
            # entries would give them identical `option_id`s ("<provider>:<model>" carries no
            # tier), and `find_model`/`default_option_id` would then silently resolve to the FAST
            # one even when a later phase's routing meant to select POWERFUL. One entry, tagged
            # with the default tier, avoids the collision entirely.
            entries.append(
                _model_entry(
                    provider=availability.provider,
                    model=credentials.model_powerful,
                    tier=AnswerModel.POWERFUL,
                    dialect=dialect,
                )
            )
            continue
        entries.append(
            _model_entry(
                provider=availability.provider,
                model=credentials.model_fast,
                tier=AnswerModel.FAST,
                dialect=dialect,
            )
        )
        entries.append(
            _model_entry(
                provider=availability.provider,
                model=credentials.model_powerful,
                tier=AnswerModel.POWERFUL,
                dialect=dialect,
            )
        )
    return tuple(entries)


def is_live(settings: Settings) -> bool:
    """True when at least one real (non-fake) provider will actually be used for a chat call --
    i.e. this deployment is not effectively running offline.

    Moved out of ``Settings.llm_is_live`` (see ``config.py``'s module docstring and the layering
    rule there): expressing this needs ``available_models``, which imports the provider dialects
    -- an import ``app.core.config`` must never make, since the dialects import
    ``LLMProviderName`` *from* config. ``/llm/config`` and ``/health`` both call this (via
    ``LLMRouter.is_live``) instead of each keeping their own copy of "is a provider actually
    configured". That was exactly the bug this replaces: the old ``Settings.llm_is_live`` looked
    only at the legacy ``LLM_API_KEY``, so a deployment with that field blank and a real
    ``LLM_OPENAI_API_KEY`` (or any other per-provider key) reported ``configured: false`` while
    still resolving a provider and billing it on every query.

    ``FAKE`` is handled explicitly rather than by relying on ``available_models`` being empty:
    when ``settings.llm_provider`` is ``FAKE``, ``available_models`` returns the two synthetic
    fake entries (see its own docstring) -- never empty, and neither one is a real provider call.
    """
    if settings.llm_provider is LLMProviderName.FAKE:
        return False
    return bool(available_models(settings))


def default_option_id(settings: Settings) -> str | None:
    """Pick the default model option id -- the one the option-less path will actually use.

    ``LLM_DEFAULT_MODEL`` when it names an available entry; otherwise, when the default
    provider's own credentials resolved only through the legacy ``LLM_API_KEY`` fallback, an
    explicitly-configured alternative provider's powerful entry (see below); otherwise the
    default provider's powerful entry; otherwise ``None``.

    **Never the first available entry.** ``/llm/config`` publishes this id as "the id
    ``POST /queries`` will use when ``model_option_id`` is omitted" (``docs/api.md``), and the
    path a query with no ``model_option_id`` takes resolves ``LLM_PROVIDER``
    (``LLMRouter.chat_for(None)``) -- which must keep doing so, since routing to whichever vendor
    happens to hold a key would bill one the operator never selected (pinned by
    ``test_llm_router.py::test_chat_for_default_with_no_credentials_raises_naming_provider`` and
    ``test_provider_factory_guard.py::test_provider_with_no_credentials_is_refused``). A
    terminal ``models[0]`` fallback made that claim false: with ``LLM_PROVIDER=openai``
    unconfigured and a working DeepSeek key, this advertised ``deepseek:deepseek-flash``,
    ``/llm/config`` reported ``configured: true``, the query was accepted with 202, and the
    worker then hard-failed on ``Provider 'openai' is not available for use``. ``models[0]`` is
    enum declaration order -- no operator ever signalled it, and it is not even tier-correct
    (it is the FAST entry, while ``default_tier`` is POWERFUL). ``None`` is the honest answer:
    the catalogue still offers the other providers' entries to a caller that selects one
    explicitly, and ``providers[].is_default`` still names the unavailable default, which is
    what makes the misconfiguration visible instead of papering over it.

    Of the two branches that can still name another provider, ``LLM_DEFAULT_MODEL`` is an exact
    option id the operator typed, and ``LLMRouter``'s option-less path follows it for that
    reason. The placeholder branch below is a *heuristic*, so execution deliberately does not
    follow it; it only fires when the default provider is itself available, so the option-less
    path still answers -- from the default provider rather than from the entry preselected here.
    Closing that last gap means resolving this id once, at request time, and persisting it on the
    run (``QueryService.create``), not widening what execution infers.

    The second branch exists because a legacy key nobody set on purpose cannot be told apart
    from a real one. ``.env.example``'s own placeholders now resolve as no key at all
    (``PLACEHOLDER_API_KEYS``, RE2E-DEF-003), but a key left over from an older setup is still
    syntactically a key. Without this branch it would silently outrank a provider the operator
    genuinely configured via its own ``LLM_<PROVIDER>_API_KEY`` -- selecting it by default,
    401ing on every query, while the operator's real key sits unused on another provider. It
    only fires when the default provider is *not* itself explicitly configured and *is*
    available (resolved via the legacy fallback) -- when the default provider has no entry at
    all, the existing "first available" fallback below already picks the right thing, and this
    must not override an operator who did explicitly configure the default provider.
    """
    models = available_models(settings)
    if not models:
        return None

    if settings.llm_default_model is not None:
        for entry in models:
            if entry.option_id == settings.llm_default_model:
                return entry.option_id

    default_provider_available = any(entry.provider is settings.llm_provider for entry in models)
    default_provider_is_explicit = _explicit_key(settings, settings.llm_provider) is not None
    if default_provider_available and not default_provider_is_explicit:
        for entry in models:
            if (
                entry.provider is not settings.llm_provider
                and entry.tier is AnswerModel.POWERFUL
                and _explicit_key(settings, entry.provider) is not None
            ):
                return entry.option_id

    for entry in models:
        if entry.provider is settings.llm_provider and entry.tier is AnswerModel.POWERFUL:
            return entry.option_id

    return None


def find_model(settings: Settings, option_id: str) -> ModelEntry | None:
    for entry in available_models(settings):
        if entry.option_id == option_id:
            return entry
    return None
