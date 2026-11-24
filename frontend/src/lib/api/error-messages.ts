/**
 * User-facing wording for every failure the API client can raise.
 *
 * The backend's envelope `{code, message, correlation_id}` is the contract (plan section 10); the
 * wording shown to people is owned here, keyed by `code`. Three kinds of raw text never reach the
 * page: the browser's "Failed to fetch", pydantic's validation sentences, and the `Detail:` suffix an
 * unexpected 500 carried in development before RMD7-T01 (a backend that predates it may still send
 * it). A domain message the backend wrote for users and that tells
 * them what to change (a refused file, a full admission queue) is kept verbatim, because replacing
 * it would hide the one fact they need.
 */

export type ApiErrorKind = "http" | "network";

/** The first item of a FastAPI 422 `detail` array, as far as the wording needs it. */
export interface ValidationFacts {
  type: string;
  loc: readonly (string | number)[];
  ctx: Readonly<Record<string, unknown>> | null;
}

export interface ErrorFacts {
  /** HTTP status; `0` when no response could be read at all. */
  status: number;
  code: string;
  kind: ApiErrorKind;
  /** The backend's (or the browser's) own text, kept for diagnosis, never shown raw by default. */
  serverMessage: string | null;
  correlationId: string | null;
  validation: ValidationFacts | null;
}

type Copy = string | ((facts: ErrorFacts) => string);

/** Keep the backend's sentence: `CognifyError` messages are written to be shown to users. */
const keepServerMessage =
  (fallback: string): Copy =>
  (facts) =>
    facts.serverMessage !== null && facts.serverMessage.trim() !== ""
      ? facts.serverMessage
      : fallback;

const COPY: Readonly<Record<string, Copy>> = {
  project_not_found: "This project no longer exists.",
  document_not_found: "This document no longer exists.",
  query_run_not_found: "This analysis could not be found.",
  report_not_found: "This report could not be found.",
  invalid_file_type: keepServerMessage("This file type is not supported."),
  invalid_cursor:
    "This list changed while it was loading. Reload the page to start from the newest entries.",
  file_too_large: keepServerMessage("This file is larger than the server accepts."),
  conflicting_state: keepServerMessage(
    "That action conflicts with the current state. Refresh and try again.",
  ),
  invalid_model_option:
    "The selected model is not offered by this server. Reload the page to refresh the model list.",
  model_option_unavailable:
    "The selected model's provider is not available right now. Choose another model or try again later.",
  invalid_routing_profile: "The routing profile is not recognised by this server.",
  routing_profile_unavailable:
    "The routing profile's provider is not available right now. Try again later or choose another model.",
  query_run_not_cancellable: "This analysis has already finished and can no longer be stopped.",
  extraction_failed: "The document's text could not be extracted.",
  embedding_failed: "The document could not be indexed right now.",
  retrieval_failed: "Evidence retrieval failed.",
  agent_output_invalid: "An agent returned output that could not be used.",
  llm_provider_error: keepServerMessage(
    "The language model provider could not complete the request.",
  ),
  qdrant_operation_failed:
    "The search index could not be updated. The change may still have been saved; refresh to check.",
  // RMD7-T01 (RE2E-DEF-004): the database could not be reached. Nothing the person did caused it,
  // and it is usually over in moments.
  service_unavailable:
    "The service is temporarily unavailable: a system it depends on could not be reached. Try again in a moment.",
  internal_error: "The server hit an unexpected error.",
  validation_error: validationMessage,
  // The pharmaceutical profile (REQ-API-06). The server's sentences name the request field and the
  // setting; these say what the person can do instead.
  domain_profile_disabled:
    "This server does not offer the Pharmaceutical manufacturing profile. Create the project with the General profile, or ask the operator to enable it.",
  pharma_input_not_allowed:
    "Pharmaceutical inquiry details can only be sent on a Pharmaceutical manufacturing project.",
};

/** Every backend error code this module words (the `CognifyError` codes plus FastAPI's 422). */
export const BACKEND_ERROR_CODES: readonly string[] = Object.keys(COPY);

const NETWORK_COPY =
  "The server could not be reached, or it failed without a readable response. Check that the " +
  "service is running and try again.";

const INVALID_RESPONSE_COPY =
  "The server sent a response this page could not read. Reload the page; if it keeps happening, " +
  "the frontend and backend versions may not match.";

const FIELD_LABELS: Readonly<Record<string, string>> = {
  question: "The question",
  name: "The project name",
  description: "The description",
  domain: "The domain",
  objective: "The objective",
  file: "The file",
  retrieval_mode: "The retrieval mode",
  response_style: "The response style",
  answer_model: "The model tier",
  model_option_id: "The model",
  routing_profile: "The routing profile",
};

const PATH_LABELS: Readonly<Record<string, string>> = {
  project_id: "project",
  document_id: "document",
  query_run_id: "analysis",
  report_id: "report",
};

function validationMessage(facts: ErrorFacts): string {
  const validation = facts.validation;
  if (validation === null) {
    return "The request was not valid.";
  }
  const where = validation.loc[0];
  const field = String(validation.loc[validation.loc.length - 1] ?? "");
  if (where === "path") {
    const thing = PATH_LABELS[field] ?? "item";
    return `This address does not point to a valid ${thing}. Check the link and try again.`;
  }
  const label = FIELD_LABELS[field] ?? "A field";
  const lowerLabel = label.charAt(0).toLowerCase() + label.slice(1);
  switch (validation.type) {
    case "string_too_long": {
      const max = validation.ctx?.max_length;
      return typeof max === "number"
        ? `${label} is too long (at most ${max.toLocaleString("en-US")} characters).`
        : `${label} is too long.`;
    }
    case "string_too_short":
    case "missing":
      return `${label} is required.`;
    case "value_error": {
      // The backend's own validators word these; pydantic only adds the "Value error, " prefix.
      const own = facts.serverMessage?.replace(/^Value error,\s*/, "") ?? "";
      return own !== "" ? own : `${label} is not valid.`;
    }
    default:
      return `Check ${lowerLabel}: it is not valid.`;
  }
}

function withReference(message: string, facts: ErrorFacts): string {
  if (facts.status >= 500 && facts.correlationId !== null && facts.correlationId !== "") {
    return `${message} Reference: ${facts.correlationId}`;
  }
  return message;
}

/** The sentence a person sees for this failure. Never returns the browser's or pydantic's text. */
export function userFacingMessage(facts: ErrorFacts): string {
  if (facts.kind === "network") {
    return NETWORK_COPY;
  }
  if (facts.code === "invalid_response") {
    return INVALID_RESPONSE_COPY;
  }
  const copy = COPY[facts.code];
  if (copy === undefined) {
    return withReference(
      `The request failed (HTTP ${String(facts.status)}). Try again in a moment.`,
      facts,
    );
  }
  const base = typeof copy === "string" ? copy : copy(facts);
  return withReference(base, facts);
}
