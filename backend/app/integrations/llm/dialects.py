"""Pure per-provider request shaping.

Holds only the dialect dataclass, the payload ``TypedDict`` and the pure builder that turns a
``TextRequest`` plus ``ModelCapabilities`` into the kwargs a provider client sends over the wire.
Must import nothing from ``app.integrations.llm.providers`` — every provider module imports
``ProviderDialect`` from here, so importing back would be circular.
"""

from dataclasses import dataclass
from typing import TypedDict, cast

from openai.types.shared_params import (
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
)

from app.core.config import LLMProviderName
from app.integrations.llm.base import JsonValue, TextRequest, ThinkingMode
from app.integrations.llm.capabilities import (
    MaxTokensParam,
    ModelCapabilities,
    StructuredOutputMode,
)


@dataclass(frozen=True)
class ProviderDialect:
    """Everything that differs between OpenAI-wire-compatible vendors.

    Adding a provider is one of these constants plus a registry entry — no new behaviour.
    """

    name: LLMProviderName
    label: str
    default_base_url: str | None  # None means "use the SDK default" -- only fake leaves this
    # unset today; every dialect that can build a real ``AsyncOpenAI`` client must set an explicit
    # value, or that SDK falls back to reading ``OPENAI_BASE_URL`` from the environment (see
    # ``providers/openai.py``'s docstring for the credential-redirection risk that creates).
    api_key_env: str  # named in the "no credentials resolve" error message (see router.py)
    default_capabilities: ModelCapabilities
    default_model_fast: str
    default_model_powerful: str


class ChatPayload(TypedDict, total=False):
    temperature: float
    max_tokens: int
    max_completion_tokens: int
    response_format: ResponseFormatJSONObject | ResponseFormatText | ResponseFormatJSONSchema
    extra_body: dict[str, JsonValue]


def build_chat_payload(
    request: TextRequest,
    capabilities: ModelCapabilities,
    *,
    json_mode: bool,
    mode: ThinkingMode | None = None,
) -> ChatPayload:
    """Turn a provider-agnostic request into the provider's chat-completion kwargs.

    Pure: no I/O, no logging, no client. Takes data, returns data. ``mode`` is the thinking mode
    the request is sent with, for a dialect that takes DeepSeek's switch.
    """
    payload: ChatPayload = {}

    if capabilities.supports_temperature:
        payload["temperature"] = request.temperature

    # An unset ``max_tokens`` means "whatever this model needs" -- see ``ModelCapabilities`` for
    # why the budget we request and the ceiling we clamp to are two separate numbers. With
    # thinking switched off, a reasoning dialect's model answers directly, so it asks for its
    # direct budget where it has one (PLV-D-033).
    default = capabilities.default_max_output_tokens
    if mode is not None and not mode.enabled and capabilities.direct_max_output_tokens is not None:
        default = capabilities.direct_max_output_tokens
    requested = default if request.max_tokens is None else request.max_tokens
    max_tokens = min(requested, capabilities.max_output_tokens)
    if capabilities.max_tokens_param is MaxTokensParam.MAX_COMPLETION_TOKENS:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    response_format: ResponseFormatJSONObject | ResponseFormatText | ResponseFormatJSONSchema
    match capabilities.structured_output:
        case StructuredOutputMode.JSON_OBJECT:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            payload["response_format"] = response_format
        case StructuredOutputMode.JSON_SCHEMA:
            if not json_mode:
                response_format = {"type": "text"}
            elif request.json_schema is not None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_output",
                        "strict": False,
                        "schema": cast("dict[str, object]", request.json_schema),
                    },
                }
            else:
                # Never send a schema-mode request with no schema: fall back safely.
                response_format = {"type": "json_object"}
            payload["response_format"] = response_format
        case StructuredOutputMode.NVEXT_GUIDED_JSON:
            if json_mode and request.json_schema is not None:
                extra_body: dict[str, JsonValue] = {"nvext": {"guided_json": request.json_schema}}
                payload["extra_body"] = extra_body
        case StructuredOutputMode.NONE:
            pass

    return payload
