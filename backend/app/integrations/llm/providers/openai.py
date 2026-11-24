"""OpenAI dialect constants.

``default_base_url`` is deliberately explicit rather than ``None``. The pinned OpenAI SDK
(``openai/_client.py``) falls back to the ``OPENAI_BASE_URL`` environment variable whenever it is
constructed with ``base_url=None`` -- so leaving this ``None`` would let any ambient
``OPENAI_BASE_URL`` (a LiteLLM/vLLM/Ollama shim, a corporate gateway, another tool's leftover
shell export, a stray line in ``backend/.env``, which ``docker-compose.yml`` passes wholesale into
the container) silently retarget every chat and embedding call, sending the real API key to that
host as a bearer token. This is the same credential-confusion vector ``ensure_api_key_present``
already guards on the *key* side; pinning the default here closes it on the *url* side.
"""

from app.core.config import LLMProviderName
from app.integrations.llm.capabilities import (
    THINKING_OFF_OUTPUT_TOKENS,
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)
from app.integrations.llm.dialects import ProviderDialect

OPENAI_DIALECT = ProviderDialect(
    name=LLMProviderName.OPENAI,
    label="OpenAI",
    default_base_url="https://api.openai.com/v1",
    api_key_env="LLM_OPENAI_API_KEY",
    default_capabilities=ModelCapabilities(
        structured_output=StructuredOutputMode.JSON_OBJECT,
        supports_temperature=True,
        max_tokens_param=MaxTokensParam.MAX_TOKENS,
        # 16,384 is ``gpt-4o``'s documented *output* ceiling. The 128,000 that stood here is its
        # context window, which is a different quantity -- and since this field is the clamp
        # ``build_chat_payload`` applies, the effect of the mix-up was that the clamp could never
        # bind. Neither OpenAI default model reasons before answering, so the budget requested is
        # the one a direct, thinking-off decision gets: the shared 1,024 was below a measured
        # pharmaceutical decision reply (1,202 tokens), refused as truncated (RMD-DEF-022).
        max_output_tokens=16_384,
        default_max_output_tokens=THINKING_OFF_OUTPUT_TOKENS,
    ),
    default_model_fast="gpt-4o-mini",
    default_model_powerful="gpt-4o",
)
