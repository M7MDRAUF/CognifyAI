"""LLM capability endpoint powering the chat model selector (FR-CHAT-001).

Separate from ``/health`` on purpose. Health answers "are dependencies ready"
and returns 503 when degraded; this endpoint always returns 200 so the UI can
render a disabled-with-warning control instead of failing.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.domain.enums import AnswerModel
from app.integrations.llm.embeddings import embedding_destination
from app.integrations.llm.registry import ProviderAvailability, context_window_tokens
from app.integrations.llm.router import LLMRouter
from app.schemas.llm import LLMConfigResponse, LLMModelOption, LLMModelOptionEntry, LLMProviderInfo

router = APIRouter(tags=["llm"])

_FAST_DESCRIPTION = "Quicker and cheaper synthesis"
_POWERFUL_DESCRIPTION = "Highest quality synthesis (recommended)"


def _reason(availability: ProviderAvailability) -> str | None:
    reason = availability.unavailable_reason
    return reason.value if reason is not None else None


@router.get("/llm/config", response_model=LLMConfigResponse)
async def get_llm_config(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> LLMConfigResponse:
    """Report which answer models are selectable, and whether selection is live.

    ``LLMRouter`` is instantiated locally rather than taken from application resources:
    every accessor used below (``available_models``, ``provider_availability``,
    ``default_option_id``) is a pure read over ``settings`` with no I/O and no cached
    client, so building one here is free and keeps this endpoint reflecting exactly the
    ``settings`` dependency it was given — the same one the v1 fields below already use.
    """
    llm_router = LLMRouter(settings)
    availability_by_provider = {row.provider: row for row in llm_router.provider_availability()}
    providers = [
        LLMProviderInfo(
            id=availability.provider.value,
            label=availability.label,
            available=availability.available,
            unavailable_reason=_reason(availability),
            is_default=availability.is_default,
        )
        # `provider_availability()` already excludes the offline/fake provider unless it is
        # genuinely the one this deployment is running (see its docstring in registry.py) --
        # no filter is needed here, and none should be re-added: that would just create a
        # second copy of the same rule for the route and the validator to drift apart on.
        for availability in availability_by_provider.values()
    ]
    model_options = [
        LLMModelOptionEntry(
            id=entry.option_id,
            provider=entry.provider.value,
            tier=entry.tier,
            model_name=entry.model,
            label=entry.display_name,
            description=entry.description,
            context_window_tokens=context_window_tokens(entry.provider, entry.model),
            # Read from the option's provider row rather than stamped (RE2E-DEF-003). In practice
            # true and null: `available_models()` lists only available providers' entries.
            available=availability_by_provider[entry.provider].available,
            unavailable_reason=_reason(availability_by_provider[entry.provider]),
        )
        for entry in llm_router.available_models()
    ]
    embedding_provider, embedding_endpoint_is_default = embedding_destination(settings)
    return LLMConfigResponse(
        provider=settings.llm_provider,
        configured=llm_router.is_live(),
        default_tier=AnswerModel.POWERFUL,
        models=[
            LLMModelOption(
                tier=AnswerModel.FAST,
                model_name=settings.llm_text_model_small,
                description=_FAST_DESCRIPTION,
            ),
            LLMModelOption(
                tier=AnswerModel.POWERFUL,
                model_name=settings.llm_text_model_strong,
                description=_POWERFUL_DESCRIPTION,
            ),
        ],
        contract_version=2,
        default_model_option_id=llm_router.default_option_id(),
        providers=providers,
        model_options=model_options,
        embedding_provider=embedding_provider,
        embedding_endpoint_is_default=embedding_endpoint_is_default,
    )
