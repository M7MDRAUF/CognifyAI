"""DeepSeek dialect constants.

DeepSeek's chat-completion API is OpenAI-wire-compatible, with a few provider-specific facts
worth recording here:

1. ``frequency_penalty`` and ``presence_penalty`` are deprecated/unsupported by DeepSeek. This
   application has never sent them, so the restriction is already satisfied — which is exactly
   why there is no ``supports_penalties`` capability field. The lane test asserts their absence
   so the guarantee is locked rather than assumed.
2. Under load DeepSeek emits SSE keep-alive comments (``: keep-alive``) and, for non-streaming
   requests, repeated empty lines. The OpenAI SDK's own parser handles both. Nobody should ever
   be tempted to hand-roll SSE parsing for this provider.
3. Rate limiting is concurrency-based (2500 concurrent for ``deepseek-flash``, 500 for
   ``deepseek-v4-pro``), not RPM/TPM, and a request that has not begun inference within 10
   minutes has its connection closed. HTTP 402 means out-of-balance and must never be retried —
   the error taxonomy that acts on this arrives in a later phase; this is just the fact record.
4. Thinking mode is **on by default**, at effort ``high``, and DeepSeek documents that the
   generation cap covers "both the visible output tokens and the reasoning tokens" — the chain of
   thought comes back in a separate ``reasoning_content`` field this application does not read.
   That is why ``default_max_output_tokens`` below is the reasoning budget and not the shared
   direct-answer one: with the old flat 1,024, a high-effort chain of thought consumed the entire
   budget and ``content`` came back empty.
5. Every request now states its mode (PLV8-T01, PLV-D-026, PLV-DEF-004). The live-validation
   protocol runs every call site with thinking disabled, where ``temperature`` 0.0 takes effect
   (SRC-REF-002), so that is the default below for every call site; a request that names its own
   ``mode`` (an experiment that turns thinking on at one call site) is sent as named.
"""

from typing import Final

from app.core.config import LLMProviderName
from app.integrations.llm.base import ThinkingMode
from app.integrations.llm.capabilities import (
    REASONING_OUTPUT_TOKENS,
    THINKING_OFF_OUTPUT_TOKENS,
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)
from app.integrations.llm.dialects import ProviderDialect

# Point 5: the protocol's mode at every call site (live-validation protocol, section 1).
CALL_SITE_DEFAULT_MODE: Final = ThinkingMode(enabled=False)

DEEPSEEK_DIALECT = ProviderDialect(
    name=LLMProviderName.DEEPSEEK,
    label="DeepSeek",
    default_base_url="https://api.deepseek.com",
    api_key_env="LLM_DEEPSEEK_API_KEY",
    default_capabilities=ModelCapabilities(
        structured_output=StructuredOutputMode.JSON_OBJECT,
        supports_temperature=True,
        max_tokens_param=MaxTokensParam.MAX_TOKENS,
        max_output_tokens=384_000,
        # Point 4 of the docstring: ``deepseek-v4-pro`` thinks by default, and the generation cap
        # covers the chain of thought as well as the visible answer.
        default_max_output_tokens=REASONING_OUTPUT_TOKENS,
        default_mode=CALL_SITE_DEFAULT_MODE,
        # With thinking switched off the model answers directly (PLV-D-033, PLV-DEF-022).
        direct_max_output_tokens=THINKING_OFF_OUTPUT_TOKENS,
    ),
    default_model_fast="deepseek-flash",
    default_model_powerful="deepseek-v4-pro",
)
