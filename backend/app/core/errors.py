"""Domain error hierarchy and HTTP translation (SDD section 13).

Route handlers never catch domain errors themselves; the handlers
registered here translate them centrally so the error envelope
``{code, message, correlation_id}`` is identical everywhere.
"""

import logging
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, cast

import sqlalchemy.exc
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import correlation_id_var

logger = logging.getLogger(__name__)


class LLMFailureKind(StrEnum):
    """Why an LLM provider call failed, independent of any vendor SDK's exception types.

    Lives here rather than in ``app.integrations.llm`` because ``core`` must not import from
    ``app.integrations`` (see the layering rule in the Phase 2 design notes), and
    ``LLMProviderError`` below needs to carry one of these.
    """

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER = "server"
    AUTH = "auth"
    BILLING = "billing"
    CONTENT_FILTER = "content_filter"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


RETRYABLE_KINDS = frozenset(
    {
        LLMFailureKind.TIMEOUT,
        LLMFailureKind.CONNECTION,
        LLMFailureKind.RATE_LIMIT,
        LLMFailureKind.OVERLOADED,
        LLMFailureKind.SERVER,
    }
)


class CognifyError(Exception):
    """Base class for all domain errors. Message must be safe to show users."""

    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProjectNotFoundError(CognifyError):
    code = "project_not_found"
    status_code = 404


class DocumentNotFoundError(CognifyError):
    code = "document_not_found"
    status_code = 404


class QueryRunNotFoundError(CognifyError):
    code = "query_run_not_found"
    status_code = 404


class ReportNotFoundError(CognifyError):
    code = "report_not_found"
    status_code = 404


class InvalidFileTypeError(CognifyError):
    code = "invalid_file_type"
    status_code = 400


class InvalidCursorError(CognifyError):
    """A paged list's ``cursor`` is not one this server issued (``app.services.pagination``).

    400, not 404: the request is malformed, and repeating it unchanged can never succeed. The
    cursor is refused as a whole and never repaired into a nearby position, so a client that
    tampered with it, or kept one across a format change, is told to start over from the top.
    """

    code = "invalid_cursor"
    status_code = 400


class FileTooLargeError(CognifyError):
    code = "file_too_large"
    status_code = 413


class ConflictError(CognifyError):
    code = "conflicting_state"
    status_code = 409


class InvalidModelOptionError(CognifyError):
    """Raised when a query names a ``model_option_id`` the registry has never heard of."""

    code = "invalid_model_option"
    status_code = 400


class ModelOptionUnavailableError(CognifyError):
    """Raised when a query names a real ``model_option_id`` whose provider is not currently
    available (missing credentials or disabled), so it cannot be honoured right now."""

    code = "model_option_unavailable"
    status_code = 409


class InvalidRoutingProfileError(CognifyError):
    """A request named a routing profile this deployment does not define (SDD 10.3.3).

    400 and not 409: the name is wrong, not temporarily unavailable, and retrying it unchanged
    can never succeed. Refused rather than coerced to the default -- silently substituting a
    profile bills a vendor the caller did not choose.
    """

    code = "invalid_routing_profile"
    status_code = 400


class DomainProfileDisabledError(CognifyError):
    """A pharmaceutical project was requested while ``PHARMA_PROFILE_ENABLED`` is off
    (REQ-PRF-04). 400: the deployment refuses the profile; a General project is unaffected."""

    code = "domain_profile_disabled"
    status_code = 400


class PharmaInputNotAllowedError(CognifyError):
    """A ``pharma`` object was sent with a query on a General project (REQ-API-03). 400: the
    request asks for a workflow the project does not run; nothing is created."""

    code = "pharma_input_not_allowed"
    status_code = 400


class RoutingProfileUnavailableError(CognifyError):
    """A request named a known routing profile whose provider is unavailable (SDD 10.3.3 (4)).

    409, like ``model_option_unavailable``: the name is right and the condition is the
    deployment's, so it can clear. Refused before anything is created, never routed elsewhere.
    """

    code = "routing_profile_unavailable"
    status_code = 409


class QueryRunNotCancellableError(CognifyError):
    """Raised when a cancel request targets a query run that cannot be cancelled: it is
    already in a terminal state, or it is not registered as running in this process (never
    started here, already finished, or running on a different worker process -- see
    ``RunRegistry``'s docstring)."""

    code = "query_run_not_cancellable"
    status_code = 409


# The error code of a run the reconciler closed because no process owned it any more: its lease
# lapsed, or, for a row with no lease, its deadline passed (RMD5-T01; RE2E-DEF-038, RE2E-DEF-041).
# Never raised through HTTP: it is the code the run's terminal event carries
# (``app.services.run_reconciler``), so a client can tell this end from an ordinary failure.
RUN_ABANDONED = "run_abandoned"


class ExtractionFailedError(CognifyError):
    code = "extraction_failed"
    status_code = 422


class EmbeddingFailedError(CognifyError):
    code = "embedding_failed"
    status_code = 503


class RetrievalFailedError(CognifyError):
    code = "retrieval_failed"
    status_code = 503


@dataclass(frozen=True)
class FieldError:
    """One failing field of an agent's output: where it is (``loc``, the schema's field names and
    list indexes) and what kind of failure (``type``, Pydantic's error type or ``no_schema_field``).
    Never the value the model wrote (RE2E-DEF-010)."""

    loc: tuple[str | int, ...]
    type: str

    def as_dict(self) -> dict[str, object]:
        return {"loc": list(self.loc), "type": self.type}

    def render(self) -> str:
        path = ".".join(str(part) for part in self.loc) or "(root)"
        return f"{path}: {self.type}"


class AgentOutputValidationError(CognifyError):
    """Raised when an agent's repair ladder is exhausted without a usable output.

    Every rung was a billed call, so the error carries what they were billed for (M-7, L-12).
    Without it the costliest failure in the ladder -- three paid rungs and no answer -- was the
    one failure that recorded no spend at all, because only ``LLMProviderError`` carried usage.

    It also carries which fields failed and every rung's response details (RE2E-DEF-010). With only
    the message, the decision step's failures on the real model could not be told apart, and their
    charged-failure rows kept no response id to find the calls by.
    """

    code = "agent_output_invalid"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        rungs: int = 0,
        validation_errors: tuple[FieldError, ...] = (),
        response_ids: tuple[str, ...] = (),
        cached_prompt_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        # `None` when no rung reported usage -- "unknown", not "free", as on `LLMProviderError`.
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.rungs = rungs
        self.validation_errors = validation_errors
        # Each rung's response id in order, and the cache and reasoning tokens summed, `None` when
        # a rung did not report them (the call details of a success row, PLV8-T02).
        self.response_ids = response_ids
        self.cached_prompt_tokens = cached_prompt_tokens
        self.reasoning_tokens = reasoning_tokens

    @property
    def charged_tokens(self) -> int | None:
        """Total tokens the exhausted ladder was charged for, or ``None`` when none reported."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class LLMProviderError(CognifyError):
    """Raised for any LLM provider call failure. ``message`` is returned to the user verbatim
    in the ``{code, message, correlation_id}`` envelope by the handler below — never interpolate
    a raw provider response body or exception string into it. Vendor SDK exceptions can carry
    request context including the base URL, so provider and model identity belong in the
    attributes below and in log records, not in ``message``.
    """

    code = "llm_provider_error"
    status_code = 503

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        kind: LLMFailureKind = LLMFailureKind.UNKNOWN,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind
        self.provider = provider
        self.model = model
        # Charged-attempt metadata (N-002). A 2xx the vendor billed can still carry no usable
        # answer -- a reply cut off at the token cap, a reasoning model that spent its whole
        # budget thinking, a 202 with no choices. Those are failures to the caller and invoices
        # to the vendor, and the usage has to survive the raise or the spend record understates
        # what was actually bought.
        #
        # `None` means the response reported no usage at all, and is deliberately distinct from
        # `0`, which means the vendor reported zero. A metering row that cannot tell "unknown"
        # from "free" cannot be reconciled against a bill.
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def charged_tokens(self) -> int | None:
        """Total tokens this attempt was charged for, or ``None`` when the vendor reported none."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class QdrantOperationError(CognifyError):
    code = "qdrant_operation_failed"
    status_code = 503


class ServiceUnavailableError(CognifyError):
    """A dependency the request needs -- the database -- cannot be reached (RMD7-T01)."""

    code = "service_unavailable"
    status_code = 503


INTERNAL_ERROR_MESSAGE = "An unexpected internal error occurred."
SERVICE_UNAVAILABLE_MESSAGE = (
    "A service this application depends on is unavailable right now. Try again in a moment."
)

# What a dependency outage reaches a route as (RMD7-T01, RE2E-DEF-004): a refused or dropped
# connection, a pool that cannot hand one out, a server shutting down or not yet accepting. The
# connection subset of ``OSError`` only: a missing file is an ``OSError`` and nothing is down.
_OUTAGE_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    sqlalchemy.exc.OperationalError,
    sqlalchemy.exc.InterfaceError,
    sqlalchemy.exc.TimeoutError,
)

# PostgreSQL's own codes for a connection that failed or was refused (class 08) and for a server
# shutting down or not yet accepting connections (57P01 to 57P03). The driver puts them on its
# exceptions and SQLAlchemy's asyncpg adapter copies them onto its own, so a protocol code decides,
# not a driver's class names.
_CONNECTION_SQLSTATE_CLASS = "08"
_UNAVAILABLE_SQLSTATES = frozenset({"57P01", "57P02", "57P03"})


def _has_outage_sqlstate(exc: BaseException) -> bool:
    sqlstate = getattr(exc, "sqlstate", None)
    return isinstance(sqlstate, str) and (
        sqlstate.startswith(_CONNECTION_SQLSTATE_CLASS) or sqlstate in _UNAVAILABLE_SQLSTATES
    )


# Enough links for a driver error wrapped by the ORM and re-raised by the application, bounded so
# a pathological chain cannot keep the handler busy.
_MAX_CHAIN_LINKS = 16


def is_dependency_outage(exc: BaseException) -> bool:
    """Whether ``exc``, or anything in its cause, context or driver-original chain, is a
    dependency outage. The type decides, never the message's words."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < _MAX_CHAIN_LINKS:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, _OUTAGE_TYPES) or _has_outage_sqlstate(current):
            return True
        if isinstance(current, sqlalchemy.exc.DBAPIError):
            if current.connection_invalidated:
                return True
            if isinstance(current.orig, BaseException):
                pending.append(current.orig)
        pending.extend(
            link for link in (current.__cause__, current.__context__) if link is not None
        )
    return False


def _error_body(code: str, message: str) -> dict[str, str | None]:
    return {"code": code, "message": message, "correlation_id": correlation_id_var.get()}


async def handle_domain_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, CognifyError):  # pragma: no cover - registration guarantees type
        raise exc
    return JSONResponse(status_code=exc.status_code, content=_error_body(exc.code, exc.message))


def build_unexpected_error_handler() -> Callable[
    [Request, Exception], Coroutine[None, None, JSONResponse]
]:
    """The answer to an exception no domain handler took, the same in every environment.

    RE2E-DEF-004 (RMD7-T01): in ``APP_ENV=development`` -- the shipped default -- the 500 used to
    end with ``Detail: {exc}``, whatever raised it, so any caller could read a socket error, a DSN
    or a path. The detail goes to the log under the correlation ID, redacted by the formatter, and
    the body is fixed text. A dependency outage is answered as what it is, 503
    ``service_unavailable``, not as an internal error.
    """

    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        if is_dependency_outage(exc):
            logger.error("Request failed: a dependency is unavailable", exc_info=exc)
            return JSONResponse(
                status_code=ServiceUnavailableError.status_code,
                content=_error_body(ServiceUnavailableError.code, SERVICE_UNAVAILABLE_MESSAGE),
            )
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500, content=_error_body("internal_error", INTERNAL_ERROR_MESSAGE)
        )

    return handle_unexpected_error


VALIDATION_ERROR_CODE = "validation_error"

# The bounds pydantic reports in an error's ``ctx``, kept because the page words the limit ("at most
# 4,000 characters"); every other ``ctx`` entry -- a ``value_error``'s exception object, a pattern,
# an expected literal -- is dropped (RMD7-T03, DV-RMD-45).
_BOUND_KEYS = frozenset(
    {
        "max_length",
        "min_length",
        "ge",
        "gt",
        "le",
        "lt",
        "multiple_of",
        "max_digits",
        "decimal_places",
    }
)


def _validation_item(error: Mapping[str, Any]) -> dict[str, object]:
    """One rejected field: where, pydantic's own sentence, and its type -- never ``input``."""
    item: dict[str, object] = {
        "loc": list(error.get("loc", ())),
        "msg": str(error.get("msg", "")),
        "type": str(error.get("type", "")),
    }
    ctx = error.get("ctx")
    if isinstance(ctx, Mapping):
        bounds = {
            str(key): value
            for key, value in cast(Mapping[object, object], ctx).items()
            if key in _BOUND_KEYS and isinstance(value, int | float) and not isinstance(value, bool)
        }
        if bounds:
            item["ctx"] = bounds
    return item


async def handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """A rejected request in the envelope every other error carries (RMD7-T03, RE2E-DEF-030).

    FastAPI's default handler answered ``{"detail": [... "input": ...]}``: no code, no correlation
    ID for the page to show, and the submitted value repeated back -- an incident object, or
    thousands of characters of question. The envelope's message is the first error's own sentence;
    ``detail`` keeps the location, message and type the frontend parses, and ``ctx`` only for the
    numeric bounds it words.
    """
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - registration guarantees
        raise exc
    detail = [_validation_item(error) for error in exc.errors()]
    message = str(detail[0]["msg"]) if detail else "The request was not valid."
    body: dict[str, object] = {**_error_body(VALIDATION_ERROR_CODE, message), "detail": detail}
    return JSONResponse(status_code=422, content=body)


def describe_validation_envelope(document: dict[str, Any]) -> None:
    """Rewrite a generated OpenAPI document's 422 components to the body
    ``handle_validation_error`` serves, in place: the envelope with ``detail``, and a ``detail``
    item without ``input``. FastAPI's own module-level definitions are left alone."""
    schemas = cast(dict[str, Any], document.get("components", {}).get("schemas", {}))
    if "ValidationError" in schemas:
        schemas["ValidationError"] = {
            "title": "ValidationError",
            "type": "object",
            "properties": {
                "loc": {
                    "title": "Location",
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                },
                "msg": {"title": "Message", "type": "string"},
                "type": {"title": "Error Type", "type": "string"},
                "ctx": {
                    "title": "Context",
                    "type": "object",
                    "description": "The rule's numeric bounds only, such as max_length.",
                },
            },
            "required": ["loc", "msg", "type"],
        }
    if "HTTPValidationError" in schemas:
        schemas["HTTPValidationError"] = {
            "title": "HTTPValidationError",
            "type": "object",
            "properties": {
                "code": {"title": "Code", "type": "string", "const": VALIDATION_ERROR_CODE},
                "message": {"title": "Message", "type": "string"},
                "correlation_id": {
                    "title": "Correlation Id",
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
                "detail": {
                    "title": "Detail",
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/ValidationError"},
                },
            },
            "required": ["code", "message", "correlation_id", "detail"],
        }


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CognifyError, handle_domain_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, build_unexpected_error_handler())
