"""Application factory and lifespan startup validation."""

import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.deps import AppResources
from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.errors import (
    build_unexpected_error_handler,
    describe_validation_envelope,
    register_exception_handlers,
)
from app.core.logging import configure_logging, correlation_id_var, redact_url_credentials
from app.integrations.llm.router import LLMRouter
from app.integrations.postgres.database import build_engine, build_session_factory, check_postgres
from app.integrations.qdrant.client import QdrantService, build_qdrant_client
from app.services.event_bus import EventBus
from app.services.run_reconciler import RunReconciler
from app.services.run_registry import RunRegistry
from app.utils.ids import new_uuid

logger = logging.getLogger(__name__)

# A DSN embeds its credentials as `scheme://user:password@host:port/db` -- strips exactly that
# userinfo segment, leaving the scheme, host, port and path (database name) intact. Applied before
# either startup-failure message below is ever built: raised out of the lifespan, uvicorn logs the
# full traceback -- including this message -- into whatever aggregates CI job logs or an
# observability backend, which are broadly readable and long-retained. Host and database name are
# useful for diagnosing the failure; the user and password are not, and must never reach that log.
#
# The obvious pattern for this -- ``//[^@/]+@`` -- is wrong in a way that only shows up with
# certain passwords, which is the worst way for a redactor to be wrong. ``[^@/]`` stops at the
# first ``/``, so a password containing one (routine: ``openssl rand -base64`` emits ``/``
# regularly) never matches and the whole DSN is logged verbatim; a password containing ``@``
# leaks its tail. SQLAlchemy parses both of those DSNs fine, so the connection succeeds and
# nothing signals that redaction silently did nothing. ``redact_url_credentials`` backtracks to
# the *last* ``@`` in the run, matching SQLAlchemy's own parse.
redact_dsn = redact_url_credentials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate every dependency at startup and fail fast on misconfiguration."""
    settings = get_settings()
    engine = build_engine(settings.database_url)
    if not await check_postgres(engine):
        await engine.dispose()
        msg = (
            "PostgreSQL is not reachable at startup "
            f"(DATABASE_URL={redact_dsn(settings.database_url)!r})."
        )
        raise RuntimeError(msg)
    qdrant = QdrantService(
        build_qdrant_client(settings.qdrant_url), settings.qdrant_collection_name
    )
    router = LLMRouter(settings)
    try:
        if not await qdrant.check_ready():
            msg = (
                "Qdrant is not reachable at startup "
                f"(QDRANT_URL={redact_dsn(settings.qdrant_url)!r})."
            )
            raise RuntimeError(msg)
        await qdrant.ensure_collection(settings.llm_embedding_dimensions)
        # Fails fast on a misconfigured embedding provider (wrong LLM_EMBEDDING_PROVIDER, or a
        # missing key) here, at startup, instead of inside the first ingestion or query
        # background task — see LLMRouter.require_embeddings.
        router.require_embeddings()
    except BaseException:
        await qdrant.close()
        await engine.dispose()
        raise
    resources = AppResources(
        engine=engine,
        session_factory=build_session_factory(engine),
        qdrant=qdrant,
        llm=router,
        event_bus=EventBus(),
        run_registry=RunRegistry(),
    )
    app.state.resources = resources
    # RMD5-T01: closes the runs no process owns -- those a dead process left `RUNNING`, among
    # them any this process's predecessor held -- once at startup and then on an interval.
    reconciler = RunReconciler(
        resources.session_factory, resources.event_bus, resources.run_registry, settings
    )
    await reconciler.start()
    logger.info("CognifyAI backend ready", extra={"event_type": "startup_completed"})
    try:
        yield
    finally:
        # Each cleanup runs independently: an earlier one raising (e.g. `qdrant.close()`
        # hitting a network error) must never skip the others, or every HTTP pool `router` and
        # `engine` hold open would leak. Collected and re-raised together rather than swallowed.
        # The reconciler stops first, while the engine it uses is still open.
        cleanup_errors: list[Exception] = []
        try:
            await reconciler.stop()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            await qdrant.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            await router.aclose()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            await engine.dispose()
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            raise ExceptionGroup("lifespan cleanup failed", cleanup_errors)


# A correlation id is caller-supplied, bound into every log record of the request, and reflected
# in the response header. Unvalidated, that is three separate problems at once:
#
#   - **Log shaping.** The formatter writes `"correlation_id": <value>` into a JSON line. A value
#     carrying a newline splits one record into two, and the second is attacker-authored -- it can
#     forge a whole log entry, including a different level and a different run id. Every endpoint
#     here is anonymous, so this needs no account to reach.
#   - **Header reflection.** The value is echoed back verbatim; a control character or a CR/LF in a
#     response header is a response-splitting primitive against anything in front of this service.
#   - **Unbounded size.** The value is copied into every record of the request. A megabyte header
#     becomes a megabyte per log line.
#
# The rule is deliberately narrow: a canonical UUID, or a bounded token of the characters an id
# is actually made of. Anything else is **replaced**, never sanitised -- a repaired value is still
# attacker-chosen, and a caller whose id was rejected must not be able to tell by reading it back.
MAX_CORRELATION_ID_LENGTH = 128
_CORRELATION_ID_PATTERN = re.compile(
    rf"\A[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_CORRELATION_ID_LENGTH - 1}}}\Z"
)


def normalize_correlation_id(incoming: str | None) -> str:
    """The caller's id when it is safe to use, otherwise a fresh server-owned one.

    Never raises and never returns the input modified: the answer is either exactly what the
    caller sent or a value this process generated.
    """
    if incoming and _CORRELATION_ID_PATTERN.match(incoming):
        return incoming
    return str(new_uuid())


class CorrelationMiddleware:
    """Binds the request's correlation id for the whole request (RMD9-T01; RE2E-DEF-013,
    RE2E-DEF-024).

    A pure ASGI middleware, so the id is bound from before the application runs until its call
    returns: through the response start, every body chunk and the background tasks Starlette runs
    after the last one. The `BaseHTTPMiddleware` step it replaces reset the id when `call_next`
    returned, before the response was sent, and uvicorn writes its access line from inside
    `send(http.response.start)`, so almost every access line had no id. Worse, uvicorn resumes
    reading a keep-alive connection from inside a request's `receive`, so the connection's next
    request began in a context that request had bound and, the id reset again before its own
    response, logged the earlier request's id. Bound here for the whole call, a request's own id
    shadows whatever its task started with, in its access line and in every line it logs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        correlation_id = normalize_correlation_id(Headers(scope=scope).get("X-Correlation-ID"))
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)["X-Correlation-ID"] = correlation_id
            await send(message)

        token = correlation_id_var.set(correlation_id)
        try:
            try:
                await self.app(scope, receive, send_with_id)
            except Exception as exc:
                # F-007: an unhandled exception used to escape to Starlette's
                # ServerErrorMiddleware, outside this step, so a 500 carried no header, a `null`
                # envelope `correlation_id` and an uncorrelated log record. Answered here, while
                # the id is bound, it carries all three. Once the response has started, a second
                # start would be a protocol error, so the exception goes on as before.
                if started:
                    raise
                response = await build_unexpected_error_handler()(Request(scope), exc)
                await response(scope, receive, send_with_id)
        finally:
            correlation_id_var.reset(token)


class UnexpectedErrorMiddleware:
    """Answers an unhandled exception from *inside* `CORSMiddleware` (SE-002).

    `CorrelationMiddleware` is outermost, so the 500 it builds (F-007) never passes back through
    CORS: the browser reported a network failure and could not read the error or its reference.
    Answered here instead, the 500 is decorated by CORS on its way out, and the correlation layer
    still binds the id first and sets the header on the way out. `CorrelationMiddleware` keeps
    its own catch as the backstop for anything raised outside this layer.

    It answers only while nothing has been sent. Once `http.response.start` has gone out (a
    streaming response such as SSE failing mid-body), a second start would be a protocol error, so
    the exception is re-raised exactly as before. Lifespan and websocket scopes pass through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def send_noting_start(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_noting_start)
        except Exception as exc:
            if started:
                raise
            response = await build_unexpected_error_handler()(Request(scope), exc)
            await response(scope, receive, send)


class CognifyApp(FastAPI):
    """The application, with its OpenAPI document describing the 422 body it actually serves
    (RMD7-T03, ``errors.describe_validation_envelope``)."""

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is None:
            describe_validation_envelope(super().openapi())
        return super().openapi()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.app_log_level)
    app = CognifyApp(title="CognifyAI", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved
    # Starlette makes the last-added middleware outermost, so the order is: `CorrelationMiddleware`,
    # then CORS, then the error layer, innermost. See `UnexpectedErrorMiddleware`.
    app.add_middleware(UnexpectedErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_allowed_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["X-Correlation-ID"],
    )
    app.add_middleware(CorrelationMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    return app
