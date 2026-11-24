"""NVIDIA NIM dialect constants.

NVIDIA's structured-generation guide recommends ``guided_json`` over
``response_format={"type": "json_object"}``: the latter lets the model emit any valid JSON
including ``{}``, while guided decoding constrains the output to match a schema. The wire shape
is ``extra_body={"nvext": {"guided_json": <schema>}}`` with no ``response_format`` key at all —
see ``StructuredOutputMode.NVEXT_GUIDED_JSON`` in ``dialects.py`` for the payload construction.

Two traps specific to this provider:

1. The entitlement trap. An account without the "Public API Endpoints" entitlement gets a
   working ``GET /v1/models`` while ``POST /v1/chat/completions`` returns 401/404
   "Function not found". A green model listing is therefore not proof that chat works, and
   provider availability must never be inferred from discovery succeeding — only from a key
   being configured.
2. Self-hosted NIM containers do not validate the API key header at all, so a self-hosted
   deployment pointed at by ``LLM_NVIDIA_BASE_URL`` will accept any non-empty key. Do not treat
   a successful call against a self-hosted endpoint as evidence that a key is valid.

``default_model_fast``/``default_model_powerful`` below were verified callable against the live
API on 2026-09-17 -- an actual successful chat completion, on an account with the "Public API
Endpoints" entitlement, not just a vendor documentation sample. That is stronger provenance than
this file previously had: the prior defaults included ``meta/llama-3.3-70b-instruct``, sourced
from NVIDIA's own current documentation sample, and it (along with the prior fast default) still
returned HTTP 410 Gone, end-of-life 2026-08-26, when probed live the same day. A vendor sample can
go stale; a live call that day cannot lie about that day.

NVIDIA model availability is per account, though: many catalogue entries returned by
``GET /v1/models`` 404 on ``POST /v1/chat/completions`` for accounts without that specific
model's entitlement (see the entitlement trap above). These two ids are a working default for the
account they were verified against, not a cross-account guarantee -- confirm against
``GET /v1/models`` (or a live chat call) for any other account before relying on them unmodified.
"""

from app.core.config import LLMProviderName
from app.integrations.llm.capabilities import (
    REASONING_OUTPUT_TOKENS,
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)
from app.integrations.llm.dialects import ProviderDialect

NVIDIA_DIALECT = ProviderDialect(
    name=LLMProviderName.NVIDIA,
    label="NVIDIA NIM",
    default_base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="LLM_NVIDIA_API_KEY",
    default_capabilities=ModelCapabilities(
        structured_output=StructuredOutputMode.NVEXT_GUIDED_JSON,
        supports_temperature=True,
        max_tokens_param=MaxTokensParam.MAX_TOKENS,
        # What the hosted endpoint accepts *is* documented: ``max_tokens`` has minimum 1, maximum
        # 131,072, default 2,048. What remains unpublished is the per-model cap, so the old
        # ``# UNVERIFIED`` marker overstated the ignorance. The conservative choice that marker
        # was really describing now lives in ``default_max_output_tokens``, where it belongs --
        # this field is the endpoint's ceiling, not our budget.
        max_output_tokens=131_072,
        # ``deepseek-ai/deepseek-v4-flash-0731`` is a reasoning model and the endpoint's own
        # ``reasoning_effort`` default is ``high``. NVIDIA's parameter reference is explicit that
        # "reasoning shares this budget, so leave headroom on reasoning-heavy prompts" -- which is
        # exactly what a flat 1,024 did not do.
        default_max_output_tokens=REASONING_OUTPUT_TOKENS,
    ),
    # Verified callable 2026-09-17 (see module docstring). Both prior defaults were confirmed
    # dead (HTTP 410 Gone) on the same date.
    default_model_fast="meta/llama-3.2-11b-vision-instruct",
    default_model_powerful="deepseek-ai/deepseek-v4-flash-0731",
)
