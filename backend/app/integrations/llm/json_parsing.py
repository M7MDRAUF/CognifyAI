"""Best-effort extraction of a JSON object from an LLM response.

Models sometimes wrap JSON in prose or code fences; this recovers the
first balanced object before validation. It never raises — callers
decide what to do with ``None`` (the agent repair ladder in Phase 4).
"""

import json

from app.integrations.llm.base import JsonObject, JsonValue


def extract_json_object(text: str) -> JsonObject | None:
    candidate = _strip_code_fence(text).strip()
    decoded = _try_load(candidate)
    if decoded is not None:
        return decoded
    span = _first_balanced_object(candidate)
    if span is None:
        return None
    return _try_load(span)


def _try_load(text: str) -> JsonObject | None:
    try:
        parsed: JsonValue = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    without_open = stripped.split("\n", 1)
    if len(without_open) != 2:
        return text
    body = without_open[1]
    fence_end = body.rfind("```")
    return body[:fence_end] if fence_end != -1 else body


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
