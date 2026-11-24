"""Structured JSON logging with correlation-ID support.

Log records never include API keys, full document contents, or full
prompts (SDD section 14). Every record this handler formats has both its
message and its exception chain passed through ``redact_secrets``, and the
vendor SDK loggers that would otherwise emit request URLs and whole chat
payloads are capped below the application's own level -- see
``configure_logging``. Context fields are injected from contextvars
so services do not need to thread IDs through every call.
"""

import json
import logging
import re
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import cache
from typing import Final, cast

from app.domain import enums as domain_enums
from app.domain.enums import EventType
from app.pharma import enums as pharma_enums
from app.pharma.enums import (
    ConflictKind,
    EscalationRole,
    EvidenceSufficiency,
    ExclusionReason,
    PolicyCode,
)

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
project_id_var: ContextVar[str | None] = ContextVar("project_id", default=None)
query_run_id_var: ContextVar[str | None] = ContextVar("query_run_id", default=None)
agent_name_var: ContextVar[str | None] = ContextVar("agent_name", default=None)

# RE2E-DEF-026 (RMD9-T02): the ingestion success line's two counts, which the formatter dropped.
_EXTRA_FIELDS = ("event_type", "latency_ms", "chunks_embedded", "chunks_reused")

# FR-015: `event_type` names what happened -- the field a log search or an alert filters on -- and
# every value is a constant written in this codebase, never user data. F-006 redacted it like any
# other text, so `_OPAQUE_TOKEN`'s 20-character rule below turned most of the vocabulary
# (`citation_not_anchored`, `qdrant_operation_failed`, `rerank_non_finite_score`, ...) into
# `[redacted]`. An exact member of this set is written as is; any other value, including one that
# merely starts with a member, is still redacted, which keeps F-006's guarantee for whatever a
# future caller passes there. `test_every_emitted_event_type_is_known` holds this set equal to
# the `extra` literals under `app/` plus the `EventType` values, so it cannot drift either way.
KNOWN_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        *(event.value for event in EventType),
        "agent_output_invalid",
        "answer_hygiene_applied",
        "citation_anchor_rate",
        "citation_key_relabelled",
        "citation_markers_removed",
        "citation_not_anchored",
        "document_indexed",
        "embedding_metering_failed",
        "hallucinated_reference",
        "hydration_drop",
        "llm_attempt_failed",
        "pharma_abstained",
        "pharma_conflicts_detected",
        "pharma_escalation_decided",
        "pharma_policy_checked",
        "pharma_signals_extracted",
        "pharma_sufficiency_classified",
        "pharma_unknown_references_removed",
        "pharma_versions_resolved",
        "planner_premise_guarded",
        "qdrant_operation_failed",
        "report_unreadable_output",
        "rerank_non_finite_score",
        "reranker_fallback",
        "run_finished",
        "run_started",
        "startup_completed",
    }
)

# PDP13-T03: a pharmaceutical step event's `pharma` field (`app/pharma/workflow.py`,
# `step_fields`) holds counts, booleans and codes. Its codes are enumeration values written in this
# codebase, as `event_type` is, and six of them have 20 or more word characters
# (`superseded_by_effective`), which `_OPAQUE_TOKEN` below would turn into `[redacted]`. An exact
# member is written as is; any other string is redacted like the message (F-006).
_STEP_CODE_ENUMS: Final[tuple[type[StrEnum], ...]] = (
    ConflictKind,
    EscalationRole,
    EvidenceSufficiency,
    ExclusionReason,
    PolicyCode,
)
KNOWN_STEP_CODES: Final[frozenset[str]] = frozenset(
    member.value for enumeration in _STEP_CODE_ENUMS for member in enumeration
)

# `formatException` below walks `__cause__`, and `openai_provider.py` deliberately chains
# `raise last_error from exc`, where `exc` is an `APIStatusError` whose `__str__` is
# `f"Error code: {status} - {body}"` -- the vendor's body verbatim. A self-hosted NIM, a proxy, or
# GLM answering a 401 with `{"error": "invalid api key: sk-abc123..."}` would otherwise put that
# key straight into this module's own JSON output, contradicting this module's docstring above.
#
# Mirrors `scripts/check_providers.py`'s own `_redact_secrets`, but widened for two gaps a
# reviewer found in that pattern:
#   - it matched only a single long opaque run, missing a dotted vendor key format such as GLM's
#     `<id>.<secret>`, where each half can be shorter than 20 characters on its own;
#   - it never touched URLs at all, which can carry a credential in a query string or simply
#     reveal a redirected endpoint that should not be logged.
# The dotted pattern requires 8+ characters on each side of the dot so it does not fire on
# ordinary dotted identifiers that show up constantly in tracebacks -- module paths
# (`app.core.logging`), filenames (`openai_provider.py`), qualified exception names
# (`openai.APIStatusError`) -- all of which have at least one side shorter than that.
_DOTTED_SECRET = r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"  # noqa: S105 -- a regex, not a credential  # nosec B105
_OPAQUE_TOKEN = r"[A-Za-z0-9_-]{20,}"  # noqa: S105 -- a regex, not a credential  # nosec B105
# A canonical UUID is matched first and passed through untouched. Both the dotted and the opaque
# rule would otherwise swallow one, and the only production log lines that interpolate a value at
# all interpolate an id (`jobs.py` "Query run %s vanished", `citation_service.py` "Dropping
# citation to non-retrieved chunk %s"). Ids are not credentials here -- `query_run_id` and
# `correlation_id` are already emitted unredacted as first-class payload fields below -- and none
# of the configured vendors issues a UUID-shaped key.
#
# The exemption applies only where the UUID *ends*, which is what the trailing lookahead is for:
# it must exempt an id, not merely the leading 36 characters of something longer. Without that
# boundary a `<uuid>-<tail>` token had its UUID consumed here and scanning resumed at the tail,
# which then survived in clear whenever it fell under `_OPAQUE_TOKEN`'s 20-character floor --
# coverage `_OPAQUE_TOKEN` did have before this branch was added, and a narrowing that reaches
# past the five LLM vendors the exemption was reasoned about, since `redact_secrets` runs on every
# record from every library. No matching lookbehind is needed: a token whose UUID is *preceded* by
# a token character is already claimed whole by `_OPAQUE_TOKEN`, which begins matching at that
# earlier character and never reaches this branch.
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_SECRET_LIKE_TOKEN = re.compile(
    f"(?P<keep>{_UUID})(?![A-Za-z0-9_-])|{_DOTTED_SECRET}|{_OPAQUE_TOKEN}"
)
_URL = re.compile(r"https?://\S+")
# `_URL` covers http(s) only, so a DSN (`postgresql+asyncpg://user:pw@host/db`) kept its password.
# The greedy `\S*` backtracks to the *last* `@` in the run rather than stopping at the first, so
# a password containing `/` or `@` -- routine for `openssl rand -base64` output, and what
# SQLAlchemy itself accepts -- is covered too. Host, port and database name survive on purpose:
# the line has to stay diagnosable. The cost of that greed is that a URL with no userinfo but an
# `@` in its path redacts the path instead; over-redacting a rare path is the right way round for
# a redactor, and `_URL` already collapses http(s) URLs wholesale before this rule sees them.
#
# The scheme is length-bounded rather than `*`. Unbounded, this pattern is quadratic in the
# longest run of scheme-legal characters: the engine retries the run from every start position,
# each time scanning to its end looking for a `://` that is not there. Measured before the bound:
# 5k characters 0.02s, 40k 1.38s, 80k 5.64s. That only became reachable when redaction was
# extended from the exception field to *every* record's message -- and this application logs
# SQLAlchemy errors whose `[parameters: ...]` carries uploaded document text, so the long run is
# attacker-supplied. 31 is comfortably above the longest scheme in use here (`postgresql+asyncpg`,
# 18), and turns each attempt into constant work.
_URI_USERINFO = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]{0,31}://)\S*@")


@cache
def _application_names() -> frozenset[str]:
    """The exact names this application spells, which the opaque-token rule spares (RMD7-T02,
    RE2E-DEF-001): every setting's field name and the environment name it is read from
    (``config.py`` reads ``FIELD.upper()``), the event types and step codes above, and the values
    of the application's own enumerations.

    A startup failure tells the operator which setting to fix, and the 20-character rule turned
    ``LLM_EMBEDDING_PROVIDER`` into ``[redacted]`` while the rejected values stayed readable. Only
    exact strings are spared, never a pattern, and nothing read from the environment is in here:
    a value is a value whatever name it was set under. Built on first use, because ``config``
    imports this module.
    """
    # Imported here, not at the top: `config` imports this module at its own top, so either order
    # of the two top-level imports fails.
    from app.core.config import Settings  # noqa: PLC0415 -- an import cycle with config

    names: set[str] = {*KNOWN_EVENT_TYPES, *KNOWN_STEP_CODES}
    for field_name in Settings.model_fields:
        names.update((field_name, field_name.upper()))
    for module in (domain_enums, pharma_enums):
        for member in vars(module).values():
            if (
                isinstance(member, type)
                and issubclass(member, Enum)
                and member.__module__ == module.__name__
            ):
                names.update(str(item.value) for item in member)
    return frozenset(names)


def _mask_secret_like(match: re.Match[str]) -> str:
    kept = match.group("keep")
    if kept is not None:
        return kept
    token = match.group(0)
    return token if token in _application_names() else "[redacted]"


def redact_url_credentials(url: str) -> str:
    """Strip the ``user:password@`` userinfo segment from a URL or DSN, keeping the scheme, host,
    port and path so the endpoint stays identifiable.

    Use this rather than ``redact_secrets`` wherever the endpoint itself is the thing being
    reported: ``redact_secrets`` collapses any http(s) URL to ``[redacted-url]``, which would
    erase the fact such a line exists to carry.
    """
    return _URI_USERINFO.sub(r"\g<scheme>[redacted]@", url)


def redact_secrets(text: str) -> str:
    """Best-effort redaction of anything credential- or URL-shaped in free text before it is
    logged or printed.

    Defense in depth, not the primary guarantee -- callers should not be interpolating secrets
    into messages in the first place -- but a formatter or logger must never be the thing that
    leaks one, even if some future error path forgets to redact.
    """
    without_urls = _URL.sub("[redacted-url]", text)
    without_userinfo = redact_url_credentials(without_urls)
    return _SECRET_LIKE_TOKEN.sub(_mask_secret_like, without_userinfo)


# Ceiling for any free-text excerpt that leaves this process -- a trace record, a persisted
# `agent_runs.error_message`, a report field. It exists because the two places that most want to
# quote a model's offending output are also the two least able to bound it: a provider body and an
# agent payload are both attacker-influenced and both arbitrarily long. 2,000 characters is enough
# to diagnose a schema mismatch and far short of a document.
EXCERPT_CHAR_LIMIT: Final = 2000

_TRUNCATION_MARKER: Final = "... [truncated]"


def bounded_excerpt(text: str, limit: int = EXCERPT_CHAR_LIMIT) -> str:
    """A redacted, length-bounded excerpt of free text, safe to persist or to put in a trace.

    **Redaction runs before truncation, not after.** The other order can cut a credential in half
    and leave the first half in clear: ``_OPAQUE_TOKEN`` needs 20 characters to fire, so a
    40-character key truncated to its first 19 stops matching and survives. Redacting first means
    the key is already ``[redacted]`` by the time any cut is made.

    Truncation is marked. An excerpt that was silently cut reads as a complete short value, and a
    reader diagnosing malformed output would conclude the model returned exactly that.
    """
    redacted = redact_secrets(text)
    if len(redacted) <= limit:
        return redacted
    return redacted[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


@contextmanager
def bound_context(
    *,
    correlation_id: object = None,
    project_id: object = None,
    query_run_id: object = None,
    agent_name: object = None,
) -> Generator[None]:
    """Bind log context for the duration of the block and reset it on **every** exit path.

    SDD 14 names six fields every backend record must carry. Four are the contextvars above; two
    are per-record extras. Until this existed, exactly one of the four -- ``correlation_id``,
    bound by ``main.py``'s middleware -- was ever written. The other three were declared, read by
    the formatter below, and set by nothing.

    The four parameters are named rather than ``**kwargs`` so a misspelled field is a type error
    at the call site instead of a runtime surprise, and so this signature is the one place that
    states which fields exist. They are ``object`` rather than ``str`` because every caller has a
    ``UUID`` or an enum in hand; each is stringified here, once.

    ``try/finally`` rather than a plain set, because the paths that most need the reset are the
    ones least likely to reach a tidy end: an agent raising, a run hitting its deadline, a worker
    task cancelled mid-step. ``CancelledError`` is a ``BaseException`` and would skip an
    ``except``-based cleanup entirely; ``finally`` still runs.

    **Reset, not set-to-None.** A contextvar token restores whatever was bound *before* this
    block, so nesting works: binding the agent name inside a block that already bound the query
    run leaves the query run intact on the way out. Setting ``None`` would erase the outer binding
    and make the next sibling step log without a run id.

    A value of ``None`` is skipped rather than bound, so a caller can pass an optional id
    unconditionally without blanking an outer binding with it.

    Cross-run isolation comes free from ``contextvars``: each asyncio task copies the context at
    creation, so two runs on one event loop cannot observe each other's values.
    """
    pairs: tuple[tuple[ContextVar[str | None], object], ...] = (
        (correlation_id_var, correlation_id),
        (project_id_var, project_id),
        (query_run_id_var, query_run_id),
        (agent_name_var, agent_name),
    )
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    try:
        for var, value in pairs:
            if value is None:
                continue
            tokens.append((var, var.set(str(value))))
        yield
    finally:
        # Reversed so nested binds unwind in the order they were made.
        for var, token in reversed(tokens):
            var.reset(token)


def _step_value(value: object) -> object:
    """One value of a step event's `pharma` field, as it is written into the line."""
    if isinstance(value, bool | int):
        return value
    if isinstance(value, list | tuple):
        return [_step_value(item) for item in cast("Sequence[object]", value)]
    text = str(value)
    return text if text in KNOWN_STEP_CODES else redact_secrets(text)


# RE2E-DEF-010: an agent's failing field is named by the schema's own field names (`base.py` writes
# no other text there) and Pydantic's error type, all lowercase identifiers. A name of 20 or more
# characters (`investigation_questions`, `contradictory_evidence`) is what `_OPAQUE_TOKEN` would
# turn into `[redacted]`. A lowercase identifier is written as is; any other string, which the
# runner never writes, is redacted like the message.
_FIELD_ERROR_NAME = re.compile(r"[a-z_][a-z0-9_]{0,63}|\*")


def _field_error_part(part: object) -> object:
    if isinstance(part, int) and not isinstance(part, bool):
        return part
    text = str(part)
    return text if _FIELD_ERROR_NAME.fullmatch(text) else redact_secrets(text)


def _field_error_value(error: object) -> object:
    """One entry of a log line's `validation_errors`: `{"loc": [...], "type": ...}`."""
    if not isinstance(error, dict):
        return redact_secrets(str(error))
    fields = cast("Mapping[str, object]", error)
    loc = fields.get("loc")
    parts = cast("Sequence[object]", loc) if isinstance(loc, list | tuple) else ()
    return {
        "loc": [_field_error_part(part) for part in parts],
        "type": _field_error_part(fields.get("type", "")),
    }


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            # The message, not only the exception chain: `configure_logging` makes root the sink
            # for every library logger, so a third-party record's message reaches this formatter
            # too, and `httpx`'s carries `str(request.url)` -- which, unlike `repr`, does not
            # mask userinfo.
            "message": redact_secrets(record.getMessage()),
        }
        context: dict[str, str | None] = {
            "correlation_id": correlation_id_var.get(),
            "project_id": project_id_var.get(),
            "query_run_id": query_run_id_var.get(),
            "agent_name": agent_name_var.get(),
        }
        for key, value in context.items():
            if value is not None:
                payload[key] = value
        for key in _EXTRA_FIELDS:
            if key in record.__dict__:
                value = record.__dict__[key]
                # F-006: copied into the line, so redacted like the message. A number is kept as a
                # number; anything else is text by the time it is serialised, and is redacted as
                # the text it will become.
                numeric = isinstance(value, int | float) and not isinstance(value, bool)
                if numeric or value is None:
                    payload[key] = value
                elif key == "event_type" and isinstance(value, str) and value in KNOWN_EVENT_TYPES:
                    # FR-015: a known constant, kept readable. `str()` so an `EventType` member is
                    # written as its value.
                    payload[key] = str(value)
                else:
                    payload[key] = redact_secrets(str(value))
        errors = record.__dict__.get("validation_errors")
        if isinstance(errors, list):
            # RE2E-DEF-010: an exhausted ladder's failing fields, written as objects.
            payload["validation_errors"] = [
                _field_error_value(error) for error in cast("list[object]", errors)
            ]
        step = record.__dict__.get("pharma")
        if isinstance(step, dict):
            # A pharmaceutical step event's counts and codes, written as an object (PDP13-T03).
            fields = cast("Mapping[object, object]", step)
            payload["pharma"] = {
                redact_secrets(str(name)): _step_value(value) for name, value in fields.items()
            }
        if record.exc_info is not None:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


# Vendor SDK loggers are held at WARNING or quieter whatever APP_LOG_LEVEL says. Root is the sink
# for every library logger, so without this ceiling:
#   - `httpx` logs `HTTP Request: %s %s "..."` at INFO -- the shipped level -- on every LLM,
#     embedding and Qdrant call, using `str(request.url)`, which does not mask userinfo;
#   - at DEBUG, `openai._base_client` logs `Request options: %s` including `json_data`, i.e. the
#     whole chat payload: the user's question and every retrieved document chunk, ~10 times per
#     query run.
# Raising APP_LOG_LEVEL to DEBUG to diagnose a provider problem is the most likely reason anyone
# raises it, and it must not silently start copying the corpus into the log aggregator. This
# ceiling is therefore a deliberate decision, not an omission: to inspect wire traffic, raise
# these loggers by name in a throwaway process, never through APP_LOG_LEVEL.
_VENDOR_LOGGERS: Final = ("openai", "httpx", "httpcore")

# Uvicorn is the one library that is *not* covered by "root is the sink for every library logger"
# above. It installs its own handlers on these three and sets `propagate = False`, so their
# records never reach root, never meet `JsonLogFormatter`, and never pass through
# `redact_secrets`. `uvicorn.access` formats the request target including its query string, so a
# credential a client puts in a URL is written to stdout verbatim. Observed, not theorised:
#
#     INFO:  127.0.0.1:48625 - "GET /api/v1/health?token=sk-CANARY... HTTP/1.1" 200 OK
#
# The application never builds such a URL itself, so this is defence in depth rather than a live
# leak -- but a logger must never be the thing that leaks a credential, which is exactly the
# standard `redact_secrets` is held to. It also left the stream half plain-text and half JSON,
# so anything parsing these logs structurally silently dropped every request line.
#
# Taking their handlers away and turning propagation back on routes them through the same single
# root handler as everything else. Levels are deliberately *not* capped the way `_VENDOR_LOGGERS`
# are: uvicorn's records are its own one-line summaries, never request bodies or prompts, and
# `uvicorn.error` carries startup and lifespan failures an operator must not lose.
_SERVER_LOGGERS: Final = ("uvicorn", "uvicorn.error", "uvicorn.access")


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # `logging.config.fileConfig` and `dictConfig` default to `disable_existing_loggers=True`,
    # which sets `disabled = True` on every logger that already exists and is not named in the
    # config they are given -- a flag nothing else here resets, and which silences a logger even
    # with a level and a reachable handler. `alembic/env.py` calls `fileConfig` on import, so any
    # process that runs a migration before serving (an entrypoint doing `alembic upgrade head`
    # then starting the app in-process, or a test session that migrates then exercises the app)
    # loses every application log line afterwards, with nothing to indicate why.
    #
    # `configure_logging` is the single place that establishes this process's logging, so it
    # asserts the state it intends rather than assuming nothing else has touched it.
    root.disabled = False
    # `root.level` is the numeric level `setLevel` just resolved; a higher number is quieter, so
    # `max` never makes a vendor logger noisier than the application itself.
    vendor_level = max(root.level, logging.WARNING)
    for name in _VENDOR_LOGGERS:
        logging.getLogger(name).setLevel(vendor_level)
    # Must run after root's handler is installed: these loggers have no handler of their own
    # afterwards, so until root has one their records would have nowhere to go.
    for name in _SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
        server_logger.disabled = False
