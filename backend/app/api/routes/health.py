"""Health endpoint reporting dependency readiness (SDD section 7.1)."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import HealthProbe, get_app_settings, get_postgres_probe, get_qdrant_probe
from app.core.config import LLMProviderName, Settings
from app.integrations.llm.embeddings import embedding_destination
from app.integrations.llm.registry import ProviderCredentials, resolve_credentials
from app.integrations.llm.router import LLMRouter
from app.schemas.health import DependencyStatus, EmbeddingStatus, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# RE2E-DEF-005: health checked PostgreSQL and Qdrant only, so an unreachable embedding endpoint
# showed nowhere until an upload failed on it inside its background task. The check is a GET of
# the endpoint's model list: free on OpenAI and on an OpenAI-compatible local server, and it
# carries no document text (an embedding request would be billed). One deadline bounds the whole
# request, and the answer, either way, is kept for the window below (DV-RMD-49).
EMBEDDING_PROBE_TIMEOUT_SECONDS: Final = 2.0
EMBEDDING_PROBE_CACHE_SECONDS: Final = 30.0

EmbeddingCheck = Callable[[Settings], Awaitable[EmbeddingStatus]]


class EmbeddingProbe:
    """Whether the configured embedding endpoint answers, asked at most once per window."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._answers: dict[tuple[LLMProviderName, str], tuple[float, bool]] = {}

    async def __call__(self, settings: Settings) -> EmbeddingStatus:
        if settings.llm_provider is LLMProviderName.FAKE:
            return "offline"
        provider, _ = embedding_destination(settings)
        credentials = resolve_credentials(settings, provider) if provider is not None else None
        # No base URL only happens for a dialect that embeds nothing; there is nothing to reach.
        if credentials is None or credentials.base_url is None:
            return "error"
        key = (credentials.provider, credentials.base_url)
        now = self._clock()
        kept = self._answers.get(key)
        if kept is not None and now - kept[0] < EMBEDDING_PROBE_CACHE_SECONDS:
            return "ok" if kept[1] else "error"
        reachable = await self._reach(credentials, credentials.base_url)
        self._answers[key] = (now, reachable)
        return "ok" if reachable else "error"

    async def _reach(self, credentials: ProviderCredentials, base_url: str) -> bool:
        # Only the failure's class or status is logged: a self-hosted base URL may carry
        # credentials (`.env.example` documents it), so the URL never reaches a log line.
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {credentials.api_key.get_secret_value()}"}
        try:
            async with (
                asyncio.timeout(EMBEDDING_PROBE_TIMEOUT_SECONDS),
                httpx.AsyncClient(transport=self._transport) as client,
            ):
                response = await client.get(url, headers=headers)
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.warning(
                "The embedding endpoint did not answer its check (%s)", type(exc).__name__
            )
            return False
        if response.is_success:
            return True
        logger.warning(
            "The embedding endpoint answered its check with HTTP %s", response.status_code
        )
        return False


_EMBEDDING_PROBE = EmbeddingProbe()


def get_embedding_check() -> EmbeddingCheck:
    return _EMBEDDING_PROBE


@router.get("/health", response_model=HealthResponse)
async def get_health(
    settings: Annotated[Settings, Depends(get_app_settings)],
    postgres_probe: Annotated[HealthProbe, Depends(get_postgres_probe)],
    qdrant_probe: Annotated[HealthProbe, Depends(get_qdrant_probe)],
    embedding_check: Annotated[EmbeddingCheck, Depends(get_embedding_check)],
) -> JSONResponse:
    postgres: DependencyStatus = "ok" if await postgres_probe() else "error"
    qdrant: DependencyStatus = "ok" if await qdrant_probe() else "error"
    embeddings = await embedding_check(settings)
    # `LLMRouter(settings).is_live()` is a pure read over `settings` (no I/O, no cached client --
    # see `routes/llm.py`'s docstring), so building one here is free. Previously this checked only
    # the legacy `LLM_API_KEY` directly, which reported "unconfigured" while a per-provider key
    # still resolved and billed a vendor (see `registry.is_live`'s docstring).
    configured = LLMRouter(settings).is_live()
    healthy = postgres == "ok" and qdrant == "ok" and embeddings != "error"
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        postgres=postgres,
        qdrant=qdrant,
        llm_provider="configured" if configured else "unconfigured",
        embeddings=embeddings,
    )
    status_code = 200 if body.status == "ok" else 503
    return JSONResponse(status_code=status_code, content=body.model_dump())
