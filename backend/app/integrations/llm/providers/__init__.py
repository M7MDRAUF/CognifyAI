"""Per-provider dialect constants and the mapping that ties them to the provider enum.

Each module here exports exactly one ``ProviderDialect`` constant describing how a vendor
differs from the OpenAI wire format it is otherwise compatible with. All the variation lives in
data: a renamed parameter, an omitted parameter, a different structured-output mechanism. No
module here contains behaviour.

**Adding a sixth provider** is therefore three mechanical steps and no new logic:

1. add a member to ``LLMProviderName`` in ``app.core.config``;
2. add ``providers/<name>.py`` exporting one ``ProviderDialect`` constant, modelled on any
   existing one, with every value traced to that vendor's own documentation;
3. add the constant to ``DIALECTS`` below and add a table-driven test asserting the payload
   ``build_chat_payload`` produces for it.

Anything a vendor supports that this application does not use — tool calling, token streaming,
reasoning-trace capture — is deliberately absent from ``ModelCapabilities`` and documented in
``docs/providers.md`` instead, so that no unread configuration accumulates here.
"""

from collections.abc import Mapping

from app.core.config import LLMProviderName
from app.integrations.llm.dialects import ProviderDialect
from app.integrations.llm.providers.deepseek import DEEPSEEK_DIALECT
from app.integrations.llm.providers.fake import FAKE_DIALECT
from app.integrations.llm.providers.glm import GLM_DIALECT
from app.integrations.llm.providers.kimi import KIMI_DIALECT
from app.integrations.llm.providers.nvidia import NVIDIA_DIALECT
from app.integrations.llm.providers.openai import OPENAI_DIALECT

DIALECTS: Mapping[LLMProviderName, ProviderDialect] = {
    LLMProviderName.OPENAI: OPENAI_DIALECT,
    LLMProviderName.DEEPSEEK: DEEPSEEK_DIALECT,
    LLMProviderName.NVIDIA: NVIDIA_DIALECT,
    LLMProviderName.GLM: GLM_DIALECT,
    LLMProviderName.KIMI: KIMI_DIALECT,
    LLMProviderName.FAKE: FAKE_DIALECT,
}

__all__ = [
    "DEEPSEEK_DIALECT",
    "DIALECTS",
    "FAKE_DIALECT",
    "GLM_DIALECT",
    "KIMI_DIALECT",
    "NVIDIA_DIALECT",
    "OPENAI_DIALECT",
]
