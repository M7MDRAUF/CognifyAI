"""Fake/offline provider dialect constants.

``default_base_url`` is safe to leave ``None`` here, unlike ``OPENAI_DIALECT`` (see that module's
docstring for the ``OPENAI_BASE_URL`` redirection risk): ``FakeProvider`` never constructs an
``AsyncOpenAI`` client or makes a network call at all (see ``fake_provider.py``), so this value is
never handed to the SDK constructor where a ``None`` would matter.

``api_key_env`` is set here for the same reason: ``ProviderDialect`` is one shared shape across
all six dialects, and the field is a real, production-read value for the five real ones (see
``router.py``'s "no credentials resolve" error message). There is no ``Settings`` field named
after this value, and ``router.py`` never actually reads ``FAKE_DIALECT.api_key_env`` -- its
``_client_for`` returns a ``FakeProvider`` unconditionally for ``LLMProviderName.FAKE`` before
that error path could ever be reached. Keeping the field populated here (rather than special
casing ``ProviderDialect`` to make it optional just for this one degenerate entry) is a
deliberate, low-cost consequence of sharing the shape, not a bug.
"""

from app.core.config import LLMProviderName
from app.integrations.llm.capabilities import ModelCapabilities
from app.integrations.llm.dialects import ProviderDialect

FAKE_DIALECT = ProviderDialect(
    name=LLMProviderName.FAKE,
    label="Offline (fake provider)",
    default_base_url=None,
    api_key_env="LLM_FAKE_API_KEY",
    default_capabilities=ModelCapabilities(),
    default_model_fast="fake-small",
    default_model_powerful="fake-strong",
)
