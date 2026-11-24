"""GLM (Z.ai) dialect constants.

Two platforms, two key namespaces. The international platform is ``api.z.ai``; the legacy
China platform is ``open.bigmodel.cn/api/paas/v4``. They are separate services with separate
credentials, so a key issued on one will not authenticate against the other. Operators pointing
at the China platform set ``LLM_GLM_BASE_URL``.

The trailing slash is intentional. Z.ai documents the OpenAI-SDK base URL as
``https://api.z.ai/api/paas/v4/``. The OpenAI Python client normalises a missing trailing slash
itself, so either form works, but we store the documented one verbatim.

Rate limits are undocumented: Z.ai's HTTP guide offers only general advice ("set reasonable
timeouts", "use exponential backoff") and publishes no numeric limits. The error body shape,
however, **is** documented -- ``docs.z.ai/api-reference/api-code`` publishes
``{"error": {"code", "message"}}`` and a code table -- so the statement that once stood here,
that classifying GLM failures by HTTP status alone was the only honest option, was made on a
false premise. How failures are actually classified is ``errors.py``'s business, not this
constant's; the correction is recorded here because this docstring is where the false premise
was asserted.

GLM also accepts several non-OpenAI parameters this application does not send: ``thinking``
(with ``type`` / ``clear_thinking``), ``do_sample``, ``reasoning_effort``, ``request_id`` and
``tool_stream``, and its ``tools`` array supports server-side ``web_search`` and ``retrieval``
types beyond OpenAI's ``function``. None are used; they are the extension surface, not
capabilities this dialect exercises.

Not sending ``thinking`` is **not** the same as thinking being off, and an earlier version of
this docstring implied it was. Z.ai documents ``thinking.type`` as defaulting to ``enabled``,
``reasoning_effort`` as defaulting to ``max``, and ``glm-4.7`` specifically as *forced* thinking
that cannot be disabled at all. The reasoning trace is returned in ``reasoning_content``,
alongside ``content``, and this application reads only ``content`` — hence
``default_max_output_tokens`` below.
"""

from app.core.config import LLMProviderName
from app.integrations.llm.capabilities import (
    REASONING_OUTPUT_TOKENS,
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)
from app.integrations.llm.dialects import ProviderDialect

GLM_DIALECT = ProviderDialect(
    name=LLMProviderName.GLM,
    label="GLM (Z.ai)",
    default_base_url="https://api.z.ai/api/paas/v4/",
    api_key_env="LLM_GLM_API_KEY",
    default_capabilities=ModelCapabilities(
        structured_output=StructuredOutputMode.JSON_OBJECT,
        supports_temperature=True,
        max_tokens_param=MaxTokensParam.MAX_TOKENS,
        # Z.ai *does* publish this, contrary to the "UNVERIFIED — no per-model output cap" note
        # that stood here: the chat-completion reference gives ``max_tokens`` a range of 1-131,072
        # and states that the GLM-4.7 series supports 128K maximum output. The old 8,192 was a
        # 16x under-statement, and it would have clamped the reasoning budget below.
        max_output_tokens=128_000,
        default_max_output_tokens=REASONING_OUTPUT_TOKENS,
    ),
    default_model_fast="glm-4.7-flash",
    default_model_powerful="glm-4.7",
)
