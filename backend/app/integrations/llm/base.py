"""LLM provider interface and transfer types (SDD section 10.1).

JSON values are modelled with an explicit recursive alias rather than
``Any`` so the whole provider layer stays type-checked. Agents validate
the returned ``dict`` against their own Pydantic schemas.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class ModelRole(StrEnum):
    """Routes a request to the small/fast or the strong model (SDD 10.3)."""

    SMALL = "small"
    STRONG = "strong"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


THINKING_EFFORTS: frozenset[str] = frozenset({"low", "high", "max"})


@dataclass(frozen=True)
class ThinkingMode:
    """DeepSeek's thinking switch and, with thinking, its effort (SRC-REF-002; PLV-D-026)."""

    enabled: bool
    effort: str | None = None

    def __post_init__(self) -> None:
        if self.effort is not None and (not self.enabled or self.effort not in THINKING_EFFORTS):
            msg = f"a thinking effort is one of {sorted(THINKING_EFFORTS)}, with thinking enabled"
            raise ValueError(msg)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # False when the vendor's response carried no usage block at all (F-003). The counts are then
    # zero only as a placeholder: a metering row records NULL -- unknown -- never a zero that
    # the rate table would price as free (ADR-004).
    reported: bool = True
    # The prompt tokens served from the vendor's cache (priced apart) and the completion tokens
    # spent on hidden reasoning (PLV8-T01, PLV-DEF-003). None when the response does not report
    # the detail: unknown, never zero.
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    # The call or calls these tokens were spent on (PLV8-T02, PLV-D-027): the id each response
    # names, in order, and the last response's model, system fingerprint and the thinking mode it
    # was sent with. The usage is what every metering path receives, so it carries them.
    response_ids: tuple[str, ...] = ()
    response_model: str = ""
    system_fingerprint: str = ""
    mode: ThinkingMode | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class _NamedByUsage:
    """What the response names, read from the result's usage (one source; PLV-D-027)."""

    usage: TokenUsage

    @property
    def response_id(self) -> str:
        return self.usage.response_ids[-1] if self.usage.response_ids else ""

    @property
    def response_model(self) -> str:
        return self.usage.response_model

    @property
    def system_fingerprint(self) -> str:
        return self.usage.system_fingerprint

    @property
    def mode(self) -> ThinkingMode | None:
        return self.usage.mode


@dataclass(frozen=True)
class TextRequest:
    messages: Sequence[ChatMessage]
    role: ModelRole = ModelRole.SMALL
    temperature: float = 0.0
    # ``None`` means "use the budget this model needs", which ``build_chat_payload`` reads from
    # ``ModelCapabilities.default_max_output_tokens``. It is deliberately not a number: this
    # field used to default to a flat 1024, and because nothing in ``app/`` ever overrode it,
    # every provider -- including four whose default "powerful" models reason before answering --
    # got a budget sized for a direct-answering OpenAI chat model. Only a caller that genuinely
    # wants a specific budget (the operator probe in ``scripts/check_providers.py``) sets an int.
    max_tokens: int | None = None
    # JSON Schema of the expected output, supplied so a provider that supports
    # schema-constrained decoding can use it; providers that do not simply ignore it.
    json_schema: JsonObject | None = None
    # The thinking mode for this request; None means the dialect's default mode, and a dialect
    # without a thinking switch ignores it (PLV-D-026).
    mode: ThinkingMode | None = None


@dataclass(frozen=True)
class TextResult(_NamedByUsage):
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    # Which provider/model actually answered (SDD 5.3). Left as "" by
    # FakeProvider and the dry-run path: no external model answered, and ""
    # must persist as SQL NULL rather than as an empty string pretending to
    # be a provider name.
    provider: str = ""
    model: str = ""
    # What the response itself names (PLV8-T01) -- its id, model and system fingerprint, "" when
    # it names none, and the thinking mode the request was sent with -- is read from ``usage``.


@dataclass(frozen=True)
class JsonResult(_NamedByUsage):
    raw_text: str
    data: JsonObject
    usage: TokenUsage = field(default_factory=TokenUsage)
    # See TextResult.provider/model and the response details above.
    provider: str = ""
    model: str = ""


@runtime_checkable
class ChatProvider(Protocol):
    async def generate_text(self, request: TextRequest) -> TextResult: ...

    async def generate_json(self, request: TextRequest) -> JsonResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class MeteredEmbeddingProvider(EmbeddingProvider, Protocol):
    """An embedding client that names its vendor and returns the usage each call reported, so
    the call can be metered (RMD9-T04, RE2E-DEF-029). ``FakeProvider`` is not one: no vendor
    answers it, as no vendor answers a fake-mode chat step."""

    @property
    def embedding_provider(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    async def embed_texts_metered(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], TokenUsage]: ...


@runtime_checkable
class LLMProvider(ChatProvider, EmbeddingProvider, Protocol):
    """Both capabilities in one object. Kept for FakeProvider and the operator scripts."""
