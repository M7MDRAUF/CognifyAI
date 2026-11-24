"""Classifies vendor SDK exceptions into the provider-agnostic ``LLMFailureKind`` taxonomy.

``app.core.errors`` holds the taxonomy itself (``LLMFailureKind``, ``RETRYABLE_KINDS``,
``LLMProviderError``) because ``core`` must not import from ``app.integrations``. This module is
where the OpenAI SDK's exception types actually get inspected: pure functions, no I/O, no
logging, so they are trivial to unit test without a network or a client.
"""

import secrets
from collections.abc import Mapping
from typing import Final, TypeGuard

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.config import LLMProviderName
from app.core.errors import LLMFailureKind

# The workflow this sits inside is a background task behind an SSE stream that pings every 15
# seconds. Honouring a hostile or careless `Retry-After: 3600` (or a jittered backoff computed
# from one) would wedge a run for an hour, so both `retry_after_seconds` and `backoff_delay`
# below cap at this ceiling.
_MAX_RETRY_AFTER_SECONDS = 20.0

_BASE_BACKOFF_SECONDS = 0.5

_KIMI_ERROR_KIND_BY_TYPE: dict[str, LLMFailureKind] = {
    "exceeded_current_quota_error": LLMFailureKind.BILLING,
    "engine_overloaded_error": LLMFailureKind.OVERLOADED,
    "rate_limit_reached_error": LLMFailureKind.RATE_LIMIT,
    "content_filter": LLMFailureKind.CONTENT_FILTER,
    "invalid_authentication_error": LLMFailureKind.AUTH,
    "incorrect_api_key_error": LLMFailureKind.AUTH,
    "permission_denied_error": LLMFailureKind.AUTH,
    "resource_not_found_error": LLMFailureKind.NOT_FOUND,
}

# Z.ai's error-code table, from `docs.z.ai/api-reference/api-code` -- a first-class
# API-reference page linked from `docs.z.ai/llms.txt`. (`providers/glm.py`'s docstring and
# `docs/providers.md` still assert that Z.ai publishes no error schema; neither file is this
# module's to edit, but the premise they state is false and this table is the evidence.)
#
# Only codes whose kind *differs* from what `_generic_status_kind` would return are listed:
# this map exists to correct the status mapping, not to restate it. The collision it corrects is
# the whole defect: Z.ai returns 429 where OpenAI and DeepSeek return 402/403, so an empty
# balance, an expired package and a plan that excludes the model were all being retried as
# transient throttling.
#
# `1302` -- "rate limit reached for requests", the one 429 in Z.ai's table that genuinely *is*
# throttling -- is deliberately **absent** for the same reason, though it reads as an exception
# at first glance. An entry for it would map 429 to RATE_LIMIT, which is exactly what the status
# mapping already returns, so it changed no behaviour: deleting it left the suite green because
# there was nothing to observe. Listing it would restate the rule this map exists to correct, and
# the map's value is precisely that every line in it is a correction.
#
# Codes the audit's remediation suggested but whose vendor rows it did not quote (1308, 1313,
# 1221, 1222) are deliberately absent: they fall through to the status mapping, which is exactly
# the behaviour they have today, rather than being guessed at.
_GLM_ERROR_KIND_BY_CODE: dict[str, LLMFailureKind] = {
    "1113": LLMFailureKind.BILLING,  # 429 insufficient balance or no resource package
    "1211": LLMFailureKind.NOT_FOUND,  # 400 unknown model, please check the model code
    "1301": LLMFailureKind.CONTENT_FILTER,  # 400 potentially unsafe or sensitive content
    "1305": LLMFailureKind.OVERLOADED,  # 429 service may be temporarily overloaded
    "1309": LLMFailureKind.BILLING,  # 429 GLM Coding Plan package has expired
    "1310": LLMFailureKind.BILLING,  # 429 weekly/monthly limit exhausted
    "1311": LLMFailureKind.BILLING,  # 429 subscription plan does not include this model
    "1314": LLMFailureKind.BILLING,  # 429 enterprise package has expired
    "1315": LLMFailureKind.BILLING,  # 429 key is limited to the enterprise coding package
}

# OpenAI multiplexes transient throttling and permanent quota exhaustion onto HTTP 429 and tells
# them apart only in the body. Its error-codes guide marks every code below "Retry advised: No",
# while `slow_down` -- same status -- is "Retry advised: Yes". Without this set, the single most
# common way an OpenAI integration dies in production was reported as a rate limit and retried.
_OPENAI_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)


# The keys that mark an object as an error payload in its own right rather than a bare envelope
# around one. `_error_payload` uses this to decide whether descending into a nested `error` is
# unwrapping or over-unwrapping; see its docstring for why the difference is not academic.
_ERROR_FIELDS = frozenset({"message", "type", "code", "param"})


def _generic_status_kind(status_code: int) -> LLMFailureKind:
    """The fallback mapping every provider gets when it has no override for this status.

    404 and 410 both map to ``NOT_FOUND``: 410 Gone is a permanent, more specific variant of "this
    doesn't exist" (a vendor that knows the resource is retired, e.g. an end-of-life model id,
    rather than merely unrecognised). Without this, a 410 fell through to the catch-all
    ``BAD_REQUEST`` below and lost that information entirely -- see ``user_safe_message``'s
    ``NOT_FOUND`` case for the operator-facing text that recovers it.

    408 and 409 are the two 4xx codes that are not the caller's mistake, and they are named here
    for the same reason: the catch-all ``BAD_REQUEST`` made them non-retryable, so a transient
    gateway timeout in front of any provider failed a whole agent step on the first attempt. The
    pinned SDK's own ``_should_retry`` (``openai/_base_client.py:807-815``) retries both by name
    -- "Retry on request timeouts" and "Retry on lock timeouts" -- and since every client in this
    application is constructed with ``max_retries=0`` and the retry loop is reimplemented on top
    of this taxonomy, not matching it made the reimplementation strictly worse than the policy it
    replaced. 408 is a server-side request timeout, so it joins the client-side ``APITimeoutError``
    under ``TIMEOUT``. 409 has no kind of its own in the taxonomy (which lives in ``app.core`` and
    is not this module's to extend); ``SERVER`` is the closest honest label -- vendor-side and
    transient -- and it carries the retry the SDK considered worth making. Both are cheap to be
    wrong about: every wire call this application makes is a stateless chat-completion or
    embedding read, so a retry costs at most two extra attempts.
    """
    if status_code == 429:
        return LLMFailureKind.RATE_LIMIT
    if status_code in (401, 403):
        return LLMFailureKind.AUTH
    if status_code in (404, 410):
        return LLMFailureKind.NOT_FOUND
    if status_code == 408:
        return LLMFailureKind.TIMEOUT
    if status_code == 409:
        return LLMFailureKind.SERVER
    if status_code >= 500:
        return LLMFailureKind.SERVER
    return LLMFailureKind.BAD_REQUEST


def _is_str_keyed_dict(value: object) -> TypeGuard[dict[str, object]]:
    """A plain ``isinstance`` check, wrapped as a ``TypeGuard``.

    A bare ``isinstance(value, dict)`` narrows an ``object``-typed value to ``dict[Unknown,
    Unknown]`` under pyright strict, and every subsequent ``.get()`` on it then reports as
    partially unknown. The ``TypeGuard`` return annotation is what fixes that: the narrowing
    contract for callers is the declared ``dict[str, object]``, not whatever pyright would have
    inferred from the ``isinstance`` check alone. Still an ``isinstance`` check underneath, never
    a ``cast`` -- a vendor JSON error body always has string keys, so this reflects that fact
    rather than asserting away a real unknown.
    """
    return isinstance(value, dict)


def _error_payload(body: object) -> dict[str, object] | None:
    """The inner error object, from either shape an ``APIStatusError`` body can arrive in.

    This is the fix for the defect that made every body-driven table in this module dead code.
    The SDK **unwraps** the ``error`` envelope before it constructs the exception --
    ``openai/_client.py:1088``, ``data = body.get("error", body) if is_mapping(body) else body``
    -- so what ``exc.body`` actually holds in production is the *inner* object,
    ``{"message", "type", "code"}``, with no ``error`` key left on it. The Kimi classifier below
    was written against the wire shape and required ``body["error"]`` to be a dict, a condition
    that can never hold for a real Moonshot response: the whole ``_KIMI_ERROR_KIND_BY_TYPE``
    table was unreachable, and an exhausted Kimi balance was retried as a rate limit. The unit
    tests passed only because they hand-built the envelope the SDK had already stripped.

    Both shapes are accepted, rather than swapping one guess for another: the unwrapped object
    the SDK delivers today, and the raw envelope -- which is what a caller hand-building an
    exception passes, what a future SDK that stopped unwrapping would deliver, and what a
    gateway that re-wraps an upstream error can produce.

    Returns ``None`` only when the body is not a JSON object at all (HTML from an edge proxy, a
    bare string, a list), so the caller can fall back to the status mapping. A body that *is* an
    object but carries nothing the caller recognises is the caller's decision, not this
    function's.

    Unwrapping happens **at most once**, and only when the outer object is a bare envelope --
    that is, when it carries no error fields of its own. Descending into a nested ``error`` on an
    object that already looks like an error payload would invert the classification it exists to
    get right: a Moonshot body of ``{"type": "exceeded_current_quota_error", ..., "error":
    {"type": "rate_limit_reached_error"}}`` would be read as the inner rate limit and retried,
    turning a permanently-failing exhausted balance back into three billed attempts. That is
    precisely the bug this function was written to fix, reintroduced through the other door. No
    vendor is documented to nest that way, which is the point: this module has already been wrong
    once about what shape a body arrives in, and the cost of not assuming again is one condition.
    """
    if not _is_str_keyed_dict(body):
        return None
    nested = body.get("error")
    if _is_str_keyed_dict(nested) and not _ERROR_FIELDS & body.keys():
        return nested
    return body


def _classify_kimi_body(body: object) -> LLMFailureKind | None:
    """Kimi's taxonomy is entirely body-driven (see ``providers/kimi.py``'s docstring).

    The body is untyped under mypy strict, so it is narrowed with ``isinstance`` checks rather
    than ``cast``. Returns ``None`` whenever the body yields no error ``type`` this module has an
    entry for, so the caller falls back to the generic status mapping.

    That covers two cases which used to be treated differently and should not have been. A body
    that does not parse at all -- Moonshot's edge returning HTML for a 502, or an empty body from
    a proxy -- obviously carries no information. But a body that *does* parse and names a ``type``
    this map has no entry for used to return ``UNKNOWN``, which is not retryable, so a Kimi 5xx
    carrying a documented error name was handled *worse* than the same 5xx carrying nothing
    (measured: one attempt versus three). Moonshot's own ``server_error``, ``unexpected_output``
    and ``server_unavailable`` all took that path. "We do not recognise this name" is not the same
    claim as "we do not know whether this is transient", and the HTTP status answers the second
    question by itself -- so the status is never discarded now, and one new vendor error string
    can no longer disable every Kimi 5xx retry at once.
    """
    payload = _error_payload(body)
    if payload is None:
        return None
    error_type = payload.get("type")
    if not isinstance(error_type, str):
        return None
    return _KIMI_ERROR_KIND_BY_TYPE.get(error_type)


def _classify_glm_body(body: object) -> LLMFailureKind | None:
    """GLM's taxonomy is driven by the documented ``error.code``.

    Returns ``None`` when the body yields no code ``_GLM_ERROR_KIND_BY_CODE`` lists, so the
    caller falls back to the generic status mapping -- the same residual path Kimi takes, and
    the behaviour GLM had for everything when it had no override at all.

    The code is accepted as either a JSON string (the form Z.ai's documented example quotes) or
    a JSON number, because tolerating that one variation costs nothing and cannot misclassify,
    whereas guessing wrong about it would put an un-retryable billing failure straight back into
    the retry ladder. That is the same mistake, in miniature, that ``_error_payload`` exists to
    undo.
    """
    payload = _error_payload(body)
    if payload is None:
        return None
    code = payload.get("code")
    if isinstance(code, int):
        code = str(code)
    if not isinstance(code, str):
        return None
    return _GLM_ERROR_KIND_BY_CODE.get(code)


def _names_openai_quota_exhaustion(body: object) -> bool:
    """Whether an OpenAI 429 body names one of the permanent quota/spend codes.

    ``insufficient_quota`` is published as the error ``type`` on older responses and as the
    ``code`` on current ones, so both fields are checked against the same set. Each value is
    ``isinstance``-checked before the membership test rather than passed to ``in`` directly: a
    body whose ``code`` is itself an object would raise ``TypeError`` on an unhashable value,
    and a classifier is the last place that should be able to throw.
    """
    payload = _error_payload(body)
    if payload is None:
        return False
    return any(
        isinstance(value, str) and value in _OPENAI_QUOTA_CODES
        for value in (payload.get("code"), payload.get("type"))
    )


def _nvidia_404_is_entitlement_gap(body: object) -> bool:
    """Whether a NVIDIA 404's body actually names the entitlement trap, rather than being a
    plain unknown model id.

    Live evidence for the two shapes:

    - the trap: ``{"title":"Not Found","detail":"Function '<uuid>' not found"}`` -- NVIDIA's
      internal function-routing layer never had this account's key wired to any function at all.
    - a plain unknown model (e.g. ``gpt-4o-mini`` sent to NVIDIA, which is not one of its model
      ids): also a 404, but with no ``"Function ... not found"`` wording. Reporting *that* as
      "missing the Public API Endpoints entitlement" is simply wrong and sends an operator after
      the wrong problem.

    Deliberately conservative -- requires both "function" and "not found" in the detail text --
    so an unrecognised NVIDIA 404 shape falls through to the generic mapping (``NOT_FOUND``)
    rather than being guessed as the entitlement problem.
    """
    if not _is_str_keyed_dict(body):
        return False
    detail = body.get("detail")
    if not isinstance(detail, str):
        return False
    lowered = detail.lower()
    return "function" in lowered and "not found" in lowered


def _classify_status_error(exc: APIStatusError, provider: LLMProviderName) -> LLMFailureKind:
    status = exc.status_code

    if provider is LLMProviderName.DEEPSEEK and status == 402:
        # DeepSeek: out-of-balance. The generic 4xx bucket would call this BAD_REQUEST, and
        # retrying it would be pointless -- the account needs topping up, not a retry.
        return LLMFailureKind.BILLING

    if (
        provider is LLMProviderName.OPENAI
        and status == 429
        and _names_openai_quota_exhaustion(exc.body)
    ):
        # OpenAI: quota or spend exhausted. It shares HTTP 429 with ordinary throttling and is
        # distinguishable only by the body, so without this the most common way an OpenAI
        # account dies was retried three times and reported as a rate limit. Any other 429 --
        # `slow_down`, or no body at all -- falls through to RATE_LIMIT below and keeps its
        # retries. Same reasoning as DeepSeek's 402 above, different status.
        return LLMFailureKind.BILLING

    if provider is LLMProviderName.KIMI:
        kimi_kind = _classify_kimi_body(exc.body)
        if kimi_kind is not None:
            return kimi_kind
        # The body named no error type this module knows -- either it didn't parse as JSON at
        # all, or it named a type with no entry in the table. Either way the HTTP status is the
        # only signal left, and it is never worth discarding: fall through to the generic
        # mapping below rather than losing retries on a 502/429/etc.

    if provider is LLMProviderName.GLM:
        glm_kind = _classify_glm_body(exc.body)
        if glm_kind is not None:
            return glm_kind
        # Same residual path as Kimi's: a body with no code Z.ai's table lists (an HTML page
        # from a proxy, or a code added after this map was written) falls through to the status
        # mapping, which is what GLM did for everything before it had an override.

    if provider is LLMProviderName.NVIDIA:
        if status == 401:
            # A 401 carries no body worth inspecting the way a 404 does here, and it is always a
            # genuine credential-shaped failure -- possibly still the entitlement trap (see the
            # module docstring: 401 and 404 are its two indistinguishable symptoms), so it stays
            # unconditionally AUTH.
            return LLMFailureKind.AUTH
        if status == 404 and _nvidia_404_is_entitlement_gap(exc.body):
            return LLMFailureKind.AUTH
        # A 404 whose body does not name the entitlement trap -- e.g. a plain unknown/misspelled
        # model id -- falls through to the generic mapping below (NOT_FOUND), instead of being
        # blanket-classified as the entitlement problem it may well not be.

    return _generic_status_kind(status)


def classify(exc: Exception, provider: LLMProviderName) -> LLMFailureKind:
    """Turn a vendor SDK exception into a provider-agnostic ``LLMFailureKind``.

    Handles the SDK exception types providers already import -- ``APITimeoutError`` and
    ``APIConnectionError`` have no HTTP status to inspect, so they are classified directly.
    ``RateLimitError`` and every other ``APIStatusError`` subclass (``AuthenticationError``,
    ``NotFoundError``, ...) carry a ``status_code`` and go through the generic mapping, with
    per-provider overrides layered on top for the cases where that generic mapping is wrong.
    """
    if isinstance(exc, APITimeoutError):
        return LLMFailureKind.TIMEOUT
    if isinstance(exc, APIConnectionError):
        return LLMFailureKind.CONNECTION
    if isinstance(exc, APIStatusError):
        return _classify_status_error(exc, provider)
    return LLMFailureKind.UNKNOWN


# What each failure kind means for the person reading it, and what they or the owner can do
# (RMD7-T04, RE2E-DEF-042). The page used to show only "LLM request failed (deepseek, auth).":
# internal values, which could not tell a misconfiguration only the owner can fix from an outage
# that passes. NOT_FOUND and NVIDIA's AUTH are worded in ``user_safe_message`` itself.
_TRY_AGAIN = "Try again in a moment."
_OWNER_LOG = "This is a configuration or software problem: the owner should check the server log."
_MEANING: Final[Mapping[LLMFailureKind, tuple[str, str]]] = {
    LLMFailureKind.AUTH: (
        "The AI model provider rejected the application's API key.",
        "The owner must check the provider key in the server configuration.",
    ),
    LLMFailureKind.BILLING: (
        "The AI model provider refused the request for a billing reason: no credit left, or a "
        "spending limit reached.",
        "The owner must check the provider account.",
    ),
    LLMFailureKind.RATE_LIMIT: (
        "The AI model provider is limiting how many requests it accepts right now.",
        "Wait a minute, then try again.",
    ),
    LLMFailureKind.OVERLOADED: (
        "The AI model provider is overloaded.",
        "Try again in a few minutes.",
    ),
    LLMFailureKind.TIMEOUT: ("The AI model provider did not answer in time.", _TRY_AGAIN),
    LLMFailureKind.CONNECTION: (
        "The AI model provider could not be reached.",
        "Check the server's network connection, then try again.",
    ),
    LLMFailureKind.SERVER: (
        "The AI model provider had an internal error.",
        "Try again in a moment; if it keeps happening, the provider may be having an outage.",
    ),
    LLMFailureKind.CONTENT_FILTER: (
        "The AI model provider declined the request under its content policy.",
        "Rephrase the question, then try again.",
    ),
    LLMFailureKind.BAD_REQUEST: (
        "The AI model provider rejected the request as invalid.",
        _OWNER_LOG,
    ),
    LLMFailureKind.UNKNOWN: (
        "The AI model provider request failed for a reason the application could not identify.",
        "Try again; if it keeps happening, the owner should check the server log.",
    ),
}


def _meaning(provider: LLMProviderName, kind: LLMFailureKind) -> tuple[str, str]:
    if provider is LLMProviderName.NVIDIA and kind is LLMFailureKind.AUTH:
        # See providers/nvidia.py: a 401 and a genuinely entitlement-shaped 404 from NVIDIA NIM's
        # chat endpoint are both folded into AUTH here because they are indistinguishable
        # symptoms of a missing "Public API Endpoints" entitlement. Naming that explicitly saves
        # an operator from hunting for a wrong model name (what a plain NOT_FOUND label would
        # suggest instead). A 404 that does *not* name the entitlement trap is classified
        # NOT_FOUND instead (see `_nvidia_404_is_entitlement_gap`) and falls through to the case
        # below.
        return (
            "The AI model provider rejected the request; this usually means the account is "
            "missing the 'Public API Endpoints' entitlement, not that the key is wrong.",
            "The owner must enable that entitlement for the provider account.",
        )
    if kind is LLMFailureKind.NOT_FOUND:
        # Covers both a plain unknown/misspelled model id (404) and a retired one (410 Gone,
        # which a vendor's response can name an explicit end-of-life date for -- see
        # `_generic_status_kind`). `classify` only returns the kind, not the original status
        # code or body, so the two are not distinguished here; naming both possibilities and
        # pointing at the fix is still strictly more useful than the previous behaviour, where a
        # 410 fell through to a bare, unexplained "bad_request".
        env_prefix = f"LLM_{provider.value.upper()}"
        return (
            "The configured model was not found at the AI model provider. Its id may be "
            "misspelled, or the provider may have retired it (a 410 response names an explicit "
            "end-of-life date).",
            f"The owner must set {env_prefix}_MODEL_FAST and/or {env_prefix}_MODEL_POWERFUL to a "
            "model id the provider currently supports.",
        )
    return _MEANING[kind]


def user_safe_message(
    provider: LLMProviderName, kind: LLMFailureKind, *, detail: str | None = None
) -> str:
    """A message safe to show a user: what happened, what to do, then the operator reference
    ``(provider: kind)`` -- the application's own values, so an operator can say what failed.

    ``detail`` is a sentence the provider adapter observed itself (an empty or cut-off reply),
    placed between the meaning and the next step. Never the API key, the base URL, or the raw
    response body -- vendor SDK exceptions can carry request context including the base URL, and
    Kimi/GLM error bodies are not guaranteed to be free of account-identifying detail.
    """
    meaning, next_step = _meaning(provider, kind)
    parts = [meaning]
    if detail:
        sentence = detail.strip()
        parts.append(sentence[:1].upper() + sentence[1:])
    parts.append(next_step)
    return f"{' '.join(parts)} ({provider.value}: {kind.value})"


def retry_after_seconds(exc: Exception) -> float | None:
    """Read a ``Retry-After`` header off ``exc``, in the integer-seconds form only.

    None of the five providers documents the HTTP-date form, so parsing it would be
    speculative -- unparseable or negative values return ``None``. The result is capped at
    ``_MAX_RETRY_AFTER_SECONDS``.
    """
    if not isinstance(exc, APIStatusError):
        return None
    raw = exc.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = int(raw.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(float(seconds), _MAX_RETRY_AFTER_SECONDS)


def backoff_delay(attempt: int, *, retry_after: float | None = None) -> float:
    """Full-jitter exponential backoff, capped so a retry storm never wedges the SSE stream.

    ``secrets.SystemRandom`` is used instead of the ``random`` module: ruff's ``S311`` flags
    ``random`` as a non-cryptographic generator, and while this jitter has nothing to do with
    security, ``SystemRandom`` is the approved source in this codebase. Jitter matters here
    beyond politeness -- with several providers configured, a synchronised retry storm across
    agents would hit every vendor in lockstep.

    When ``retry_after`` is present it takes precedence over the computed exponential delay, but
    is still jittered rather than honoured verbatim: a bare ``Retry-After`` would have every
    agent that hit the same 429 retry in exact lockstep, which is the synchronised storm jitter
    exists to prevent in the first place. The jitter is drawn from ``[retry_after, retry_after *
    1.5]`` -- never *below* what the server asked for, but spread out above it -- with both ends
    still capped at ``_MAX_RETRY_AFTER_SECONDS``.
    """
    if retry_after is not None:
        lower = min(retry_after, _MAX_RETRY_AFTER_SECONDS)
        upper = min(retry_after * 1.5, _MAX_RETRY_AFTER_SECONDS)
        if upper <= lower:
            return lower
        return secrets.SystemRandom().uniform(lower, upper)
    upper = min(_MAX_RETRY_AFTER_SECONDS, _BASE_BACKOFF_SECONDS * 2 ** (attempt - 1))
    return secrets.SystemRandom().uniform(0, upper)
