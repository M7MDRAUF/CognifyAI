"""Base agent: typed prompt → validated JSON with a repair ladder (SDD 8.2).

When the model returns invalid JSON the runner: (1) tries a deterministic
parse+validate, (2) asks the model to repair given the validation errors,
(3) retries once with a stricter instruction, then (4) raises
``AgentOutputValidationError``. Every rung is covered by tests using the
scriptable FakeProvider.

"Invalid" means more than "Pydantic raised". Three of the six output models
give every field a default, so an empty or entirely unrelated payload validates
as a complete, empty answer; see ``_addresses_the_schema`` for how a payload
that never engaged with the schema is told apart from one that genuinely found
nothing.
"""

import logging
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, ValidationError

from app.core.errors import AgentOutputValidationError, FieldError
from app.integrations.llm.base import (
    ChatMessage,
    ChatProvider,
    JsonObject,
    ModelRole,
    TextRequest,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# How much of a failing response the repair rung may quote back. Generous enough to contain
# any genuine agent payload and far short of the sizes that made the repair call the most
# expensive one in the ladder.
MAX_REPAIR_EXCERPT_CHARS = 8_000

_STRICT_INSTRUCTION = (
    "Your previous response was not valid JSON for the required schema. Respond again with "
    "ONLY a single valid JSON object and no surrounding text or code fences."
)


class AgentResult[OutputT: BaseModel]:
    def __init__(
        self,
        output: OutputT,
        usage: TokenUsage,
        *,
        provider: str = "",
        model: str = "",
        attempt_count: int = 1,
    ) -> None:
        self.output = output
        self.usage = usage
        # Provider/model of the rung that produced this *validated* output --
        # not necessarily the first attempt. If the first call returns
        # malformed JSON and a repair rung succeeds, this is the repair
        # rung's provider/model.
        self.provider = provider
        self.model = model
        # Which rung produced it: 1 first try, 2 after a repair, 3 after the strict retry.
        # `usage` already accumulates across rungs, so the tokens were never lost -- but a
        # three-rung step and a one-rung step with a verbose model looked identical, and only
        # this tells them apart. `agent_runs.attempt_count` is written from it.
        self.attempt_count = attempt_count


class JsonAgentRunner:
    """Drives the validate→repair→retry ladder for one agent invocation."""

    def __init__(self, provider: ChatProvider, agent_label: str) -> None:
        self._provider = provider
        self._agent_label = agent_label

    async def run[OutputT: BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputT],
        *,
        role: ModelRole,
    ) -> AgentResult[OutputT]:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        # Every detail starts known and zero, so a rung that does not report one makes the step's
        # unknown (PLV8-T02, PLV-D-027).
        total = TokenUsage(cached_prompt_tokens=0, reasoning_tokens=0)
        schema = output_model.model_json_schema()
        names = _schema_field_names(schema)
        errors = _FieldErrors(names)

        result = await self._provider.generate_json(
            TextRequest(messages=messages, role=role, json_schema=schema)
        )
        total = add_usage(total, result.usage)
        parsed = _try_validate(result.data, output_model, errors)
        if parsed is not None:
            return AgentResult(
                parsed, total, provider=result.provider, model=result.model, attempt_count=1
            )

        repair_messages = [
            *messages,
            # Bounded for the same reason as the repair instruction: this turn re-sends the
            # identical payload, so leaving it unbounded would make that bound decorative.
            ChatMessage(role="assistant", content=result.raw_text[:MAX_REPAIR_EXCERPT_CHARS]),
            ChatMessage(
                role="user",
                content=self._repair_prompt(result.raw_text, output_model, tuple(errors.found)),
            ),
        ]
        repair = await self._provider.generate_json(
            TextRequest(messages=repair_messages, role=role, json_schema=schema)
        )
        total = add_usage(total, repair.usage)
        parsed = _try_validate(repair.data, output_model, errors)
        if parsed is not None:
            return AgentResult(
                parsed, total, provider=repair.provider, model=repair.model, attempt_count=2
            )

        strict_messages = [
            ChatMessage(role="system", content=f"{system_prompt}\n\n{_STRICT_INSTRUCTION}"),
            ChatMessage(role="user", content=user_prompt),
        ]
        strict = await self._provider.generate_json(
            TextRequest(messages=strict_messages, role=role, temperature=0.0, json_schema=schema)
        )
        total = add_usage(total, strict.usage)
        parsed = _try_validate(strict.data, output_model, errors)
        if parsed is not None:
            return AgentResult(
                parsed, total, provider=strict.provider, model=strict.model, attempt_count=3
            )

        # RE2E-DEF-010: the line says which fields failed and how, never what the model wrote.
        logger.error(
            "Agent %s produced invalid output after repair",
            self._agent_label,
            extra={
                "event_type": "agent_output_invalid",
                "validation_errors": [error.as_dict() for error in errors.found],
            },
        )
        # All three rungs were billed; the error carries the accumulated usage so the failure is
        # metered like any other charged attempt (M-7, L-12), with every rung's response details.
        raise AgentOutputValidationError(
            f"The {self._agent_label} step returned malformed output and could not be repaired.",
            provider=strict.provider or None,
            model=strict.model or None,
            prompt_tokens=total.prompt_tokens,
            completion_tokens=total.completion_tokens,
            rungs=3,
            validation_errors=tuple(errors.found),
            response_ids=total.response_ids,
            cached_prompt_tokens=total.cached_prompt_tokens,
            reasoning_tokens=total.reasoning_tokens,
        )

    @staticmethod
    def _repair_prompt(
        raw_text: str, output_model: type[BaseModel], failed: Sequence[FieldError] = ()
    ) -> str:
        """The repair rung's prompt, with the offending payload **bounded** (L-8).

        The failing response was embedded verbatim, so an agent whose output was too large
        was asked to fix it in a call *larger than the one that overran* -- and if that
        failed, the strict rung paid again. The bound is what stops an over-long output
        from costing three escalating calls instead of one.

        Marked when truncated: a model handed a silently-cut fragment sees valid JSON that
        simply ends, and "fix this" is the wrong instruction for a payload we cut ourselves.

        It names the fields that failed and how (RMD2-T03, RE2E-DEF-009): told only that the JSON
        did not match, the model repeated the same shape on the repair rung.
        """
        schema_keys = ", ".join(output_model.model_fields)
        excerpt = raw_text
        if len(excerpt) > MAX_REPAIR_EXCERPT_CHARS:
            excerpt = (
                excerpt[:MAX_REPAIR_EXCERPT_CHARS]
                + "\n... [truncated; your response exceeded the size limit]"
            )
        listed = "; ".join(
            f"{'.'.join(str(part) for part in error.loc) or '(root)'} ({error.type})"
            for error in failed
        )
        fields = f" These fields failed: {listed}." if listed else ""
        return (
            "The JSON you returned did not match the required schema. Required top-level "
            f"fields: {schema_keys}.{fields} Return ONLY a corrected JSON object.\n\n"
            f"Your previous response was:\n{excerpt}"
        )


def _addresses_the_schema(data: JsonObject, output_model: type[BaseModel]) -> bool:
    """Whether ``data`` is a payload that engaged with ``output_model`` at all.

    Pydantic cannot answer this on its own here. ``ResearchOutput``, ``CriticOutput`` and
    ``RiskOutput`` give *every* field a default and inherit the default ``extra='ignore'``, so
    ``model_validate`` accepts ``{}``, ``{"bad": 1}`` and ``{"error": "I cannot answer that"}``
    alike and returns a fully-defaulted instance. Combined with ``generate_json`` coercing an
    unparseable body to ``{}``, that meant a refusal, a content-filter block or a 200-with-prose
    completed the workflow with empty findings, no risks and ``hallucination_risk = 0.0`` -- a
    confidently wrong answer -- and the repair ladder never ran, because its only trigger is a
    ``ValidationError`` these three models cannot raise.

    The test is "at least one top-level key belongs to the schema", which is the weakest thing
    that distinguishes the two cases that matter:

    * a model that genuinely found nothing answers the question it was asked, in the shape it was
      asked for: ``{"risks": []}``, ``{"key_findings": [], "themes": [], "contradictions": []}``.
      Those carry a schema key and are accepted, empty lists and all.
    * a model that did not answer returns something that is *about* not answering -- ``{}``, a
      refusal envelope, an unrelated object, or the ``{}`` synthesised from unparsable prose.
      None of those names a field of the schema.

    A partial answer (some fields present, the rest left to their defaults) still passes, which is
    deliberate: those fields have defaults because omitting them is legitimate. Making the fields
    required instead would reject that case *and* change the persisted contract for rows already
    written, which is why the fix lives here and not on the models.
    """
    return any(key in output_model.model_fields for key in data)


def _try_validate[OutputT: BaseModel](
    data: JsonObject, output_model: type[OutputT], errors: "_FieldErrors"
) -> OutputT | None:
    if not _addresses_the_schema(data, output_model):
        errors.add((), NO_SCHEMA_FIELD)
        return None
    try:
        return output_model.model_validate(data)
    except ValidationError as exc:
        # Location and type only: Pydantic's `input` and `ctx` hold the model's values.
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            errors.add(tuple(error["loc"]), error["type"])
        return None


# The type recorded for a payload that names no field of the schema (`_addresses_the_schema`).
NO_SCHEMA_FIELD = "no_schema_field"
# How many failing fields one exhausted ladder reports. A list field of 25 bad items is 25
# errors; the log line and the excerpt must stay bounded however many the model produced.
MAX_FIELD_ERRORS = 20
# A location part that is not one of the schema's own field names (a union member's name, say)
# is written as this, so no text from outside the schema can reach a log line or an excerpt.
OTHER_LOCATION = "*"


class _FieldErrors:
    """The failing fields of every rung, each once, in the order first seen, and at most
    ``MAX_FIELD_ERRORS`` of them."""

    def __init__(self, names: frozenset[str]) -> None:
        self._names = names
        self.found: list[FieldError] = []

    def add(self, loc: tuple[int | str, ...], error_type: str) -> None:
        kept = tuple(
            part if isinstance(part, int) or part in self._names else OTHER_LOCATION for part in loc
        )
        error = FieldError(kept, error_type)
        if error not in self.found and len(self.found) < MAX_FIELD_ERRORS:
            self.found.append(error)


def _schema_field_names(schema: JsonObject) -> frozenset[str]:
    """Every property name the output model's JSON schema declares, its nested models included."""
    models = [schema, *cast("dict[str, JsonObject]", schema.get("$defs", {})).values()]
    return frozenset(
        name for model in models for name in cast("dict[str, object]", model.get("properties", {}))
    )


def add_usage(total: TokenUsage, usage: TokenUsage) -> TokenUsage:
    """Two calls' usage as one, every call detail kept. Public since RMD9-T05: the action
    classifier sums its batches with it (RMD-DEF-009); start from known zeros, as ``run`` does."""
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
        completion_tokens=total.completion_tokens + usage.completion_tokens,
        # F-003: one rung whose usage went unreported makes the step's sum a lower bound, and the
        # step's metering row records it as unknown rather than as that partial count.
        reported=total.reported and usage.reported,
        # PLV8-T02 (PLV-D-027): the call details of every rung -- the cache and reasoning
        # tokens summed, unknown as soon as one rung did not report them; each response's id in
        # order; the latest response's model, fingerprint and mode.
        cached_prompt_tokens=_known_sum(total.cached_prompt_tokens, usage.cached_prompt_tokens),
        reasoning_tokens=_known_sum(total.reasoning_tokens, usage.reasoning_tokens),
        response_ids=total.response_ids + usage.response_ids,
        response_model=usage.response_model or total.response_model,
        system_fingerprint=usage.system_fingerprint or total.system_fingerprint,
        mode=usage.mode or total.mode,
    )


def _known_sum(total: int | None, count: int | None) -> int | None:
    return None if total is None or count is None else total + count
