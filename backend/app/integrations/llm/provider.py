"""Provider factory: selects a concrete implementation from configuration.

Thin wrapper around ``LLMRouter.default_provider`` — kept as its own function, with its original
name, for callers that only ever want the one provider ``LLM_PROVIDER`` names and never the full
router. Building a router just to immediately call ``default_provider()`` on it costs nothing
(construction does no I/O and builds no clients — see ``router.py``), so there is no reason for
those callers to depend on the router type at all. The two operator scripts no longer go through
here: both need the *pinned* embedding client alongside a chat client, and one router owning both
is what keeps them from sharing one object for two roles (see below).

Every ``LLMProviderName`` is routable now: ``LLMRouter`` resolves credentials per provider through
the registry and refuses (with a typed, provider-naming error) only when a provider's credentials
do not actually resolve — never because the provider is simply unimplemented. The old
``_WIRED_PROVIDERS`` allow-list this function used to enforce is gone along with the vendor
fallthrough bug it guarded against: today, building a client always goes through
``resolve_credentials`` for the *requested* provider, so there is no path left that can send one
vendor's key to another vendor's endpoint.

**This returns a ``ChatProvider``, and deliberately not an ``LLMProvider``.** The object behind it
is an ``OpenAIProvider``, which happens to implement ``embed_texts`` as well as the chat methods —
so while this was typed as ``LLMProvider`` it satisfied ``EmbeddingService``'s parameter, and both
operator scripts passed it there. That bypassed ``LLM_EMBEDDING_PROVIDER`` and the
``_EMBEDDING_CAPABLE`` pin entirely: with ``LLM_PROVIDER=deepseek``, ``scripts.seed_demo`` sent
``POST https://api.deepseek.com/embeddings`` carrying OpenAI's ``text-embedding-3-small`` id — a
404 at best, and at worst (a vendor that does host an OpenAI-shaped embeddings endpoint at the
same width) vectors from a foreign embedding space written into the collection the API later
queries with OpenAI vectors, corrupting retrieval with nothing to catch it. Embeddings come from
``LLMRouter.embeddings()``; the narrower return type is what makes reaching for this one instead
a type error rather than a silent wrong answer.
"""

from app.core.config import Settings
from app.integrations.llm.base import ChatProvider
from app.integrations.llm.router import LLMRouter


def build_llm_provider(settings: Settings) -> ChatProvider:
    return LLMRouter(settings).default_provider()
