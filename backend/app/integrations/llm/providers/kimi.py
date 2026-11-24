"""Kimi (Moonshot AI) dialect constants.

This is the dialect that proves the shared payload builder was worth having: a naive
OpenAI-shaped call would 400 against Kimi outright. Two documented differences from the
OpenAI defaults, both expressed purely as ``ModelCapabilities`` data rather than as a
conditional at any call site:

1. ``max_tokens`` is deprecated on Kimi's chat-completions endpoint; ``max_completion_tokens``
   is the supported replacement, hence ``MaxTokensParam.MAX_COMPLETION_TOKENS``.
2. Kimi's chat reference lists ``temperature`` (and ``top_p``) under *unsupported* parameters,
   hence ``supports_temperature=False``. The payload builder omits the key entirely in that
   case — it never sends ``temperature: null`` — because present-and-null is still sending the
   parameter, and that is exactly what this dialect must not do.

Deliberately ``StructuredOutputMode.JSON_OBJECT``, not ``JSON_SCHEMA``, even though Kimi *does*
have a schema mode: ``response_format: {"type": "json_schema", ...}``. That mode only accepts
Kimi's own "Moonshot Flavored JSON Schema" (MFJS) dialect, not plain JSON Schema, and this
application generates plain (Pydantic-derived) JSON Schema. Whether a Pydantic-generated schema
happens to validate as MFJS has not been verified against a live endpoint. Claiming
``JSON_SCHEMA`` support here would be asserting an unverified capability, so this dialect
degrades to ``json_object`` and leans on the repair ladder underneath instead. Do not "fix" this
to ``JSON_SCHEMA`` without first probing a live Kimi endpoint with a real generated schema and
confirming it is accepted — that is the whole reason this comment exists.

Error taxonomy (for the retry phase that consumes it later, not implemented here): Kimi returns
``{"error": {"message", "type", "code"}}``. ``exceeded_current_quota_error`` is a billing failure
and must never be retried; ``engine_overloaded_error`` should honour ``Retry-After``;
``rate_limit_reached_error`` covers RPM/TPM/TPD/concurrency limits collectively. Numeric RPM/TPM
figures are UNVERIFIED — Moonshot documents the error types but not the numbers.

A request with no response for 900 seconds gets back a 504, and Moonshot's docs recommend
``stream: true`` for long generations. This application does not stream, so a long Kimi
generation is bounded by this application's own per-provider timeout rather than by that 504.
"""

from app.core.config import LLMProviderName
from app.integrations.llm.capabilities import (
    REASONING_OUTPUT_TOKENS,
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)
from app.integrations.llm.dialects import ProviderDialect

KIMI_DIALECT = ProviderDialect(
    name=LLMProviderName.KIMI,
    label="Kimi (Moonshot)",
    default_base_url="https://api.moonshot.ai/v1",
    api_key_env="LLM_KIMI_API_KEY",
    default_capabilities=ModelCapabilities(
        structured_output=StructuredOutputMode.JSON_OBJECT,
        supports_temperature=False,
        max_tokens_param=MaxTokensParam.MAX_COMPLETION_TOKENS,
        # 131,072 is Moonshot's documented *default* for ``kimi-k3``, not its maximum (that is
        # 1,048,576). It is kept as the clamp deliberately: this one number covers both models in
        # this dialect, and Moonshot publishes no per-model ceiling for ``kimi-k2.6``, whose 256K
        # context implies a smaller one. A clamp set to K3's maximum would be a ceiling neither
        # model is known to accept.
        max_output_tokens=131_072,
        # ``kimi-k3`` always has thinking mode enabled and cannot be asked not to; the trace is
        # returned in ``reasoning_content`` and is charged against this budget.
        default_max_output_tokens=REASONING_OUTPUT_TOKENS,
    ),
    default_model_fast="kimi-k2.6",
    default_model_powerful="kimi-k3",
)
