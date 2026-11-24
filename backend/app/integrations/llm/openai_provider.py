"""OpenAI-compatible provider (SDD section 10.2).

Works against OpenAI or any OpenAI-wire-compatible endpoint via ``base_url`` and a
``ProviderDialect`` (DeepSeek, NVIDIA NIM, GLM, Kimi, ...). Retries timeouts/rate-limits/5xx with
bounded, jittered exponential backoff and wraps every SDK failure in ``LLMProviderError`` so the
rest of the app never imports the SDK's exception types.

Not every failure is an SDK exception. A 2xx response can still carry no usable answer -- a reply
truncated at the token cap, a reasoning model that spent its whole budget thinking, an endpoint
that answers "still working" with no choices at all. ``_read_content`` classifies those into the
same ``LLMProviderError`` taxonomy rather than returning an empty string, which is what they used
to become.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Final

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from app.core.config import LLMProviderName, Settings
from app.core.errors import RETRYABLE_KINDS, LLMFailureKind, LLMProviderError
from app.integrations.llm.base import (
    JsonResult,
    JsonValue,
    ModelRole,
    TextRequest,
    TextResult,
    ThinkingMode,
    TokenUsage,
)
from app.integrations.llm.dialects import ProviderDialect, build_chat_payload
from app.integrations.llm.errors import (
    backoff_delay,
    classify,
    retry_after_seconds,
    user_safe_message,
)
from app.integrations.llm.json_parsing import extract_json_object
from app.integrations.llm.providers.openai import OPENAI_DIALECT
from app.integrations.llm.registry import ProviderCredentials, without_ambient_openai_env

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3

# The ``finish_reason`` every one of the five vendors documents for "stopped at the token cap".
_LENGTH_FINISH_REASON = "length"

# Per-provider request timeout. This belongs on `ProviderDialect` once a later phase needs it
# there; `dialects.py` is frozen and owned by another task right now, so it lives here as a
# module-level mapping instead. DeepSeek's queueing is concurrency-based and a queued request can
# sit; NVIDIA and GLM get a modest bump over the OpenAI default; Kimi's k3 model always reasons
# and Moonshot's own ceiling is 900s, but a task held that long behind a 15s-ping SSE stream is
# not acceptable, hence 180s rather than the vendor max.
#
# This is the single source of truth for the table: every path that actually builds an
# `AsyncOpenAI` client -- this module's own constructor below, and `LLMRouter._build_chat_client`
# (the path production actually uses, since it always injects `client=`) -- calls
# `timeout_seconds_for` rather than keeping its own copy. A hand-copied duplicate is how a raised
# Kimi timeout here previously changed nothing at runtime: the router's copy, not this one, was
# what production read.
DEFAULT_TIMEOUT_SECONDS = 60.0
PROVIDER_TIMEOUTS_SECONDS: Mapping[LLMProviderName, float] = {
    LLMProviderName.OPENAI: 60.0,
    LLMProviderName.DEEPSEEK: 120.0,
    LLMProviderName.NVIDIA: 90.0,
    LLMProviderName.GLM: 90.0,
    LLMProviderName.KIMI: 180.0,
    LLMProviderName.FAKE: 60.0,
}


def timeout_seconds_for(provider: LLMProviderName) -> float:
    """The request timeout for ``provider``, from the single per-vendor table above."""
    return PROVIDER_TIMEOUTS_SECONDS.get(provider, DEFAULT_TIMEOUT_SECONDS)


# How long a connection may take to establish. Separate from the total, and much shorter: a host
# that will not answer a TCP handshake in 10s is not going to answer a completion in 180.
_CONNECT_TIMEOUT_SECONDS: Final = 10.0


def request_timeout_for(provider: LLMProviderName) -> httpx.Timeout:
    """An **explicit, total** request budget -- level 8 of the deadline hierarchy (SDD 13.0.1).

    A bare ``float`` handed to ``AsyncOpenAI`` becomes an ``httpx`` timeout whose ``read`` leg is
    **per-read**: every byte received resets it. A provider that drips one token every 30 seconds
    therefore runs indefinitely under a "120-second timeout", and a run's only real bound was the
    600-second ``asyncio.timeout`` around the whole workflow -- which a single stalled call can
    consume entirely (L-9).

    ``httpx.Timeout`` has no single "whole request" leg, so the total budget is expressed as the
    read/write/pool ceiling with a short connect leg in front of it. That bounds a stalled
    *transfer*, which is the observed failure. The remaining case -- a response that keeps
    trickling bytes forever, each within the read window -- is bounded by the runner-side
    per-call deadline above this layer, which is why SDD 13.0.1 specifies both and not either.
    """
    total = timeout_seconds_for(provider)
    return httpx.Timeout(total, connect=min(_CONNECT_TIMEOUT_SECONDS, total))


def ensure_api_key_present(api_key: str, provider: LLMProviderName) -> None:
    """Refuse a blank API key rather than let ``AsyncOpenAI`` silently fall back to the ambient
    ``OPENAI_API_KEY`` environment variable -- with several non-OpenAI providers configurable,
    that is a genuine credential-confusion route (a blank DeepSeek key could silently pick up an
    OpenAI key from the environment). Shared by every path that actually builds an
    ``AsyncOpenAI`` client -- this module's own constructor below, and
    ``LLMRouter._build_chat_client`` -- so the guard applies wherever a client is actually built,
    not only on the branch production never takes.
    """
    if not api_key.strip():
        msg = (
            f"No API key configured for provider {provider.value!r}; refusing to construct a "
            "client with a blank key."
        )
        raise LLMProviderError(msg, kind=LLMFailureKind.AUTH, provider=provider.value)


class OpenAIProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        dialect: ProviderDialect = OPENAI_DIALECT,
        credentials: ProviderCredentials | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._provider_name = dialect.name
        self._capabilities = dialect.default_capabilities

        if credentials is not None:
            api_key = credentials.api_key.get_secret_value()
            base_url = credentials.base_url
            model_fast = credentials.model_fast
            model_powerful = credentials.model_powerful
        else:
            # No credentials supplied: behave exactly as before per-provider credentials
            # existed, so every existing test and the legacy single-provider configuration keep
            # working unchanged.
            api_key = settings.llm_api_key.get_secret_value()
            base_url = settings.llm_base_url
            model_fast = settings.llm_text_model_small
            model_powerful = settings.llm_text_model_strong

        if client is not None:
            self._client = client
        else:
            ensure_api_key_present(api_key, dialect.name)
            # Scrubbed for the same reason `router.py` scrubs its own construction: with the
            # ambient `OPENAI_*` variables visible, `OPENAI_CUSTOM_HEADERS` replaces the
            # `Authorization` header on every request this client makes -- to *any* vendor -- and
            # `OPENAI_ORG_ID`/`OPENAI_PROJECT_ID` stamp the operator's OpenAI account onto it.
            # This branch is unreachable from application code today (the router always injects
            # `client=`), which is exactly why it is worth closing: an unreachable path that is
            # one call site away from being reachable, and differs in security behaviour from the
            # path beside it, is how the hole gets reopened. It also covers the legacy branch
            # above passing `base_url=None`, since the scrub hides `OPENAI_BASE_URL` too.
            with without_ambient_openai_env():
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=request_timeout_for(dialect.name),
                    max_retries=0,  # retries handled here for uniform error wrapping
                )

        self._model_by_role = {
            ModelRole.SMALL: model_fast,
            ModelRole.STRONG: model_powerful,
        }
        self._embedding_model = settings.llm_embedding_model
        self._embedding_dimensions = settings.llm_embedding_dimensions

    async def generate_text(self, request: TextRequest) -> TextResult:
        model = self._model_by_role[request.role]
        content, usage = await self._chat_and_read(request, json_mode=False)
        return TextResult(
            text=content,
            usage=usage,
            provider=self._provider_name.value,
            model=model,
        )

    async def generate_json(self, request: TextRequest) -> JsonResult:
        model = self._model_by_role[request.role]
        raw, usage = await self._chat_and_read(request, json_mode=True)
        data = extract_json_object(raw) or {}
        return JsonResult(
            raw_text=raw,
            data=data,
            usage=usage,
            provider=self._provider_name.value,
            model=model,
        )

    def _mode_for(self, request: TextRequest) -> ThinkingMode | None:
        """The thinking mode this request is sent with: its own, else the dialect's default; None
        for a dialect without DeepSeek's switch, whose requests carry no thinking field
        (PLV-D-026)."""
        default = self._capabilities.default_mode
        if default is None:
            return None
        return request.mode or default

    async def _chat_and_read(
        self, request: TextRequest, *, json_mode: bool
    ) -> tuple[str, TokenUsage]:
        """Send the request and read its content **inside** the retry boundary (N-007).

        Content validation used to sit outside ``_with_retry``, so a response classified
        ``SERVER``/``retryable=True`` -- NVIDIA's 202-with-no-choices, a genuinely transient
        condition the adapter already knew how to name -- could not re-enter the bounded path.
        The ``retryable`` flag on those errors was therefore dead in the one place it was
        deliberately set to ``True``.

        Usage is still read before validation, for N-002: ``_read_content`` raises for a
        billed-but-unusable response, and evaluating usage afterwards loses every charged
        failure's spend.
        """
        model = self._model_by_role[request.role]
        messages: list[ChatCompletionMessageParam] = [
            _to_message_param(message.role, message.content) for message in request.messages
        ]
        mode = self._mode_for(request)
        payload = build_chat_payload(request, self._capabilities, json_mode=json_mode, mode=mode)
        if mode is not None:
            # SRC-REF-002: the switch and the effort go in the request body, through
            # ``extra_body``, beside anything the dialect already put there.
            switch: dict[str, JsonValue] = {
                "thinking": {"type": "enabled" if mode.enabled else "disabled"}
            }
            if mode.effort is not None:
                switch["reasoning_effort"] = mode.effort
            payload["extra_body"] = {**payload.get("extra_body", {}), **switch}

        async def call() -> tuple[str, TokenUsage]:
            completion = await self._client.chat.completions.create(
                model=model, messages=messages, stream=False, **payload
            )
            # The usage names the call it counts (PLV-D-027): every metering path receives it.
            response_id = _named(completion, "id")
            usage = replace(
                _usage(completion.usage),
                response_ids=(response_id,) if response_id else (),
                response_model=_named(completion, "model"),
                system_fingerprint=_named(completion, "system_fingerprint"),
                mode=mode,
            )
            content = self._read_content(completion, model=model, usage=completion.usage)
            return content, usage

        return await self._with_retry(call, model=model)

    def _read_content(
        self,
        completion: ChatCompletion,
        *,
        model: str,
        usage: CompletionUsage | None = None,
    ) -> str:
        """The assistant text of the first choice, or a classified failure explaining its absence.

        This replaces ``completion.choices[0].message.content or ""``, which turned three
        distinguishable outcomes into either an empty string or an unclassified ``TypeError``:

        * **No choices at all.** NVIDIA NIM documents a ``202`` "result is pending, poll using the
          requestId" response for ``POST /v1/chat/completions``. 202 is a 2xx, so the SDK does not
          raise and leniently constructs a ``ChatCompletion`` whose ``choices`` is ``None``;
          indexing it raised ``TypeError``, which ``_with_retry`` does not catch and the agent
          layer does not catch either, so it surfaced as an unclassified HTTP 500. It is a
          transient, genuinely retryable condition, and is classified as one here.
        * **``finish_reason == "length"``.** The reply was cut off by the output-token budget.
          OpenAI's own guidance is to check for this *before* parsing, because the JSON will be
          partial -- and a partial body that happens to parse is worse than one that does not.
        * **Empty ``content``.** A reasoning model that spends its whole budget on the chain of
          thought returns ``content: null`` with the trace in a separate ``reasoning_content``
          field this application does not consume. That used to read as an empty answer with no
          error anywhere.

        The last two are raised **non-retryable** on purpose. Re-issuing the identical request
        under the identical budget cannot succeed, and the agent repair ladder would do exactly
        that three times over: because ``LLMProviderError`` is not a ``ValidationError``, raising
        here means the ladder is skipped entirely and one billed call reports the problem instead
        of three producing a misleading "malformed output" message.
        """
        choices = completion.choices
        if not choices:
            raise self._failure(
                model=model,
                kind=LLMFailureKind.SERVER,
                retryable=True,
                usage=usage,
                detail=(
                    "the provider accepted the request but returned no completion. Some "
                    "endpoints answer a queued or cold-starting model this way; retrying is "
                    "usually enough."
                ),
            )

        choice = choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == _LENGTH_FINISH_REASON:
            raise self._failure(
                model=model,
                kind=LLMFailureKind.BAD_REQUEST,
                retryable=False,
                usage=usage,
                detail=(
                    "the reply was cut off by the output-token budget, so it is incomplete and "
                    "must not be parsed. If this model reasons before answering, the reasoning "
                    "is charged against the same budget -- raise this provider's "
                    "default_max_output_tokens, or ask for a shorter answer."
                ),
            )
        if not content.strip():
            raise self._failure(
                model=model,
                kind=LLMFailureKind.BAD_REQUEST,
                retryable=False,
                usage=usage,
                detail=(
                    "the model returned its internal reasoning and no answer. The reasoning is "
                    "charged against the same output-token budget as the answer -- raise this "
                    "provider's default_max_output_tokens."
                    if _has_reasoning_trace(choice)
                    else "the model returned an empty response."
                ),
            )
        return content

    def _failure(
        self,
        *,
        model: str,
        kind: LLMFailureKind,
        retryable: bool,
        detail: str,
        usage: CompletionUsage | None = None,
    ) -> LLMProviderError:
        """An ``LLMProviderError`` whose message is ``user_safe_message``'s, with ``detail`` in it
        (RMD7-T04: the meaning and next step first, then the reference).

        ``detail`` must be safe to show a user: never a response body, a base URL or a key.

        ``usage`` is the charged-attempt metadata from N-002. It is carried on the error rather
        than logged and dropped, because the component that has to *account* for the spend is the
        metering decorator several frames up, not this adapter.
        """
        return LLMProviderError(
            user_safe_message(self._provider_name, kind, detail=detail),
            retryable=retryable,
            kind=kind,
            provider=self._provider_name.value,
            model=model,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
        )

    @property
    def embedding_provider(self) -> str:
        return self._provider_name.value

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors, _ = await self.embed_texts_metered(texts)
        return vectors

    async def embed_texts_metered(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], TokenUsage]:
        """The vectors and the usage the response reported (RMD9-T04, RE2E-DEF-029).

        ``usage`` used to be dropped here, so no embedding call could be metered. A vendor that
        sends no usage block -- or one without ``prompt_tokens`` -- reports nothing: the usage is
        unreported, never zero tokens (F-003).
        """
        if not texts:
            return [], TokenUsage(reported=False)
        inputs = list(texts)

        async def call() -> CreateEmbeddingResponse:
            return await self._client.embeddings.create(
                model=self._embedding_model,
                input=inputs,
                dimensions=self._embedding_dimensions,
            )

        response = await self._with_retry(call, model=self._embedding_model)
        vectors = [list(item.embedding) for item in response.data]
        response_model = getattr(response, "model", None) or ""
        prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        if not isinstance(prompt_tokens, int):
            return vectors, TokenUsage(reported=False, response_model=response_model)
        return vectors, TokenUsage(prompt_tokens, 0, response_model=response_model)

    @staticmethod
    def is_retryable(kind: LLMFailureKind) -> bool:
        """**The** retry decision (SDD 13.0, ADR-008).

        Public because it *is* the contract: ADR-008 names it as the single place the decision is
        made, and a contract test asserting "exactly these five categories retry" has to be able
        to reach it.

        One function, consulted by one caller, deciding from the category alone. The previous
        arrangement expressed the decision twice -- a ``retryable`` boolean on every error, and
        a ``kind not in RETRYABLE_KINDS`` test that never read it -- which is two sources of
        truth for one question, and the one place the flag was deliberately set to ``True`` was
        outside the loop that could have acted on it.
        """
        return kind in RETRYABLE_KINDS

    async def _with_retry[T](self, call: Callable[[], Awaitable[T]], *, model: str) -> T:
        last_error: LLMProviderError | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            retry_after: float | None = None
            try:
                return await call()
            except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as exc:
                kind = classify(exc, self._provider_name)
                retry_after = retry_after_seconds(exc)
                last_error = LLMProviderError(
                    user_safe_message(self._provider_name, kind),
                    retryable=self.is_retryable(kind),
                    kind=kind,
                    provider=self._provider_name.value,
                    model=model,
                )
            except LLMProviderError as exc:
                # Raised by `_read_content` for a 2xx carrying no usable answer. It reaches this
                # handler only because validation now runs inside the boundary (N-007); before,
                # it was raised a frame above and could never be retried.
                kind = exc.kind
                last_error = exc
            # Derived from the decision, never from the delay: a backoff can legitimately be
            # zero (a `Retry-After: 0`, or a test clock), and reading `delay > 0` as "retried"
            # reports "not retrying" on a line immediately followed by another attempt.
            will_retry = self.is_retryable(kind) and attempt < _MAX_ATTEMPTS
            delay = backoff_delay(attempt, retry_after=retry_after) if will_retry else 0.0
            self._log_attempt(
                attempt=attempt, kind=kind, model=model, delay=delay, retried=will_retry
            )
            if not self.is_retryable(kind):
                raise last_error
            if will_retry:
                await asyncio.sleep(delay)
        raise last_error or LLMProviderError(
            user_safe_message(self._provider_name, LLMFailureKind.UNKNOWN),
            provider=self._provider_name.value,
            model=model,
        )

    def _log_attempt(
        self, *, attempt: int, kind: LLMFailureKind, model: str, delay: float, retried: bool
    ) -> None:
        """One structured line per failed attempt (L-10).

        This module previously contained a ``logger`` and **zero calls to it**: retries, rate
        limits and timeouts were entirely invisible, and the only evidence a call had been
        retried three times was the wall-clock gap.

        Deliberately no response body, no base URL and no key -- provider and model identity are
        first-class fields, and ``JsonLogFormatter`` redacts the message besides.
        """
        logger.warning(
            "LLM attempt %d/%d failed (%s) for provider=%s model=%s; %s",
            attempt,
            _MAX_ATTEMPTS,
            kind.value,
            self._provider_name.value,
            model,
            f"retrying in {delay:.2f}s" if retried else "not retrying",
            extra={"event_type": "llm_attempt_failed"},
        )


def _has_reasoning_trace(choice: Choice) -> bool:
    """Whether the message carries a non-empty ``reasoning_content``.

    Not a declared field of the SDK's ``ChatCompletionMessage`` -- it is a vendor extension that
    DeepSeek, GLM, Kimi and NVIDIA all return alongside ``content`` -- so it arrives in
    ``model_extra``. Read only to tell an operator *why* the answer is missing; the trace itself
    is never returned, logged or persisted.
    """
    extra = choice.message.model_extra or {}
    reasoning = extra.get("reasoning_content")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def _named(completion: ChatCompletion, name: str) -> str:
    """A detail the response names (its id, model or system fingerprint), "" when it names none.
    Read leniently: an endpoint may omit a field the SDK declares, and the SDK then builds the
    completion without it."""
    value = getattr(completion, name, None)
    return value if isinstance(value, str) else ""


def _to_message_param(role: str, content: str) -> ChatCompletionMessageParam:
    if role == "system":
        return {"role": "system", "content": content}
    if role == "assistant":
        return {"role": "assistant", "content": content}
    return {"role": "user", "content": content}


def _usage(usage: CompletionUsage | None) -> TokenUsage:
    """The usage block, with the cache and reasoning details when the response reports them
    (SRC-REF-003, PLV-DEF-003): the cached prompt tokens from ``prompt_tokens_details``, else
    DeepSeek's ``prompt_cache_hit_tokens``; the reasoning tokens from
    ``completion_tokens_details``. A detail the response omits is None, unknown, never zero."""
    if usage is None:
        # F-003: a response without a usage block is unknown usage, not zero usage.
        return TokenUsage(reported=False)
    prompt_details = usage.prompt_tokens_details
    cached = None if prompt_details is None else prompt_details.cached_tokens
    if cached is None:
        hit = (usage.model_extra or {}).get("prompt_cache_hit_tokens")
        cached = hit if isinstance(hit, int) else None
    completion_details = usage.completion_tokens_details
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_prompt_tokens=cached,
        reasoning_tokens=None
        if completion_details is None
        else completion_details.reasoning_tokens,
    )
