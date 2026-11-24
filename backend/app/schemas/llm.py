"""LLM capability schemas backing the chat model selector (FR-CHAT-001)."""

from pydantic import BaseModel

from app.core.config import LLMProviderName
from app.domain.enums import AnswerModel


class LLMModelOption(BaseModel):
    """One selectable answer model, resolved from configured settings."""

    tier: AnswerModel
    model_name: str
    description: str


class LLMProviderInfo(BaseModel):
    """Whether one provider is currently usable, for rendering a provider picker."""

    id: str
    label: str
    available: bool
    unavailable_reason: str | None = None
    is_default: bool = False


class LLMModelOptionEntry(BaseModel):
    """One concrete, selectable provider+model pair. ``id`` is ``"<provider>:<model>"``."""

    id: str
    provider: str
    tier: AnswerModel
    model_name: str
    label: str
    description: str
    context_window_tokens: int | None = None
    available: bool
    unavailable_reason: str | None = None


class LLMConfigResponse(BaseModel):
    """Non-secret view of the LLM configuration.

    Never carries the API key: only the provider name, whether a real external
    provider will be called, and the display names of the configured models.

    ``contract_version`` 2 adds ``providers``/``model_options``/``default_model_option_id``
    on top of the unchanged v1 fields above, so a caller that only understands v1 keeps
    working (it simply ignores the new keys) while a v2-aware caller can offer a full
    provider+model picker instead of just a tier toggle.
    """

    provider: LLMProviderName
    configured: bool
    default_tier: AnswerModel
    models: list[LLMModelOption]
    contract_version: int = 2
    default_model_option_id: str | None = None
    providers: list[LLMProviderInfo] = []
    model_options: list[LLMModelOptionEntry] = []
    # Where document text goes to be embedded. Separate from `provider` because embeddings are a
    # separate axis: a user selecting a local or offline chat model does not stop ingestion
    # sending every chunk of every uploaded document to the embedding endpoint. `None` means
    # nothing is embedded remotely. `embedding_endpoint_is_default` is False when a base URL
    # override is configured -- the provider id then fixes only the API dialect, not the host, so
    # a client must not name that vendor as the destination.
    embedding_provider: LLMProviderName | None = None
    # No default: the one construction site always computes it, so a default here would be
    # unreachable, and an unreachable fail-safe is worse than none -- it reads as a
    # guarantee while a future second construction site could silently take it.
    embedding_endpoint_is_default: bool
