import type { z } from "zod";
import { z as zod } from "@/lib/zod";

import {
  type ApiErrorKind,
  type ErrorFacts,
  type ValidationFacts,
  userFacingMessage,
} from "@/lib/api/error-messages";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

interface ApiErrorDetails {
  kind?: ApiErrorKind;
  correlationId?: string | null;
  serverMessage?: string | null;
  validation?: ValidationFacts | null;
}

/**
 * Every failure the API client raises. `message` is the user-facing sentence
 * (`error-messages.ts`); `code`, `status` and `correlationId` stay machine-readable for the few
 * views that branch on them, and `serverMessage` keeps the backend's own text for diagnosis.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  /**
   * `network`: no readable response (offline, the server down, or a response a browser refuses).
   */
  readonly kind: ApiErrorKind;
  readonly correlationId: string | null;
  readonly serverMessage: string | null;
  readonly validation: ValidationFacts | null;

  constructor(status: number, code: string, message: string, details: ApiErrorDetails = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.kind = details.kind ?? "http";
    this.correlationId = details.correlationId ?? null;
    this.serverMessage = details.serverMessage ?? null;
    this.validation = details.validation ?? null;
  }

  static fromFacts(facts: ErrorFacts): ApiError {
    return new ApiError(facts.status, facts.code, userFacingMessage(facts), facts);
  }
}

const errorEnvelopeSchema = zod.object({
  code: zod.string(),
  message: zod.string(),
  correlation_id: zod.string().nullish(),
});

const validationErrorSchema = zod.object({
  detail: zod.array(
    zod.object({
      msg: zod.string(),
      type: zod.string().optional(),
      loc: zod.array(zod.union([zod.string(), zod.number()])).optional(),
      ctx: zod.record(zod.string(), zod.unknown()).nullish(),
    }),
  ),
});

async function toApiError(response: Response): Promise<ApiError> {
  const facts: ErrorFacts = {
    status: response.status,
    code: "unknown_error",
    kind: "http",
    serverMessage: null,
    correlationId: response.headers.get("X-Correlation-ID"),
    validation: null,
  };
  try {
    const body: unknown = await response.json();
    const envelope = errorEnvelopeSchema.safeParse(body);
    // A 422 carries `detail[]` beside the envelope (RMD7-T03); a backend that predates it sends
    // `detail[]` alone. Either way the field, its rule and its limit come from the first item.
    const validation = validationErrorSchema.safeParse(body);
    const first = validation.success ? validation.data.detail[0] : undefined;
    if (envelope.success) {
      facts.code = envelope.data.code;
      facts.serverMessage = envelope.data.message;
      facts.correlationId = envelope.data.correlation_id ?? facts.correlationId;
    } else if (first !== undefined) {
      facts.code = "validation_error";
      facts.serverMessage = first.msg;
    }
    if (first !== undefined && facts.code === "validation_error") {
      facts.validation = {
        type: first.type ?? "unknown",
        loc: first.loc ?? [],
        ctx: first.ctx ?? null,
      };
    }
  } catch {
    // Non-JSON error body; the generic wording for the status applies.
  }
  return ApiError.fromFacts(facts);
}

/**
 * `fetch` rejects with a `TypeError` when no response is readable: the server is down, the
 * network is gone, or the browser refuses a cross-origin response. The backend's unexpected 500s
 * are readable since SE-002, so they arrive here as responses. An abort is the caller's own
 * decision and is rethrown untouched.
 */
async function send(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_BASE_URL}${path}`, init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw cause;
    }
    throw ApiError.fromFacts({
      status: 0,
      code: "network_error",
      kind: "network",
      serverMessage: cause instanceof Error ? cause.message : null,
      correlationId: null,
      validation: null,
    });
  }
}

export async function requestJson<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  return readJson(await send(path, init), schema);
}

/** A response read against `schema`: the error envelope on failure, `invalid_response` on drift. */
async function readJson<T>(response: Response, schema: z.ZodType<T>): Promise<T> {
  if (!response.ok) {
    throw await toApiError(response);
  }
  const body: unknown = await response.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    throw ApiError.fromFacts({
      status: response.status,
      code: "invalid_response",
      kind: "http",
      serverMessage: parsed.error.message,
      correlationId: response.headers.get("X-Correlation-ID"),
      validation: null,
    });
  }
  return parsed.data;
}

export async function requestVoid(path: string, init?: RequestInit): Promise<void> {
  const response = await send(path, init);
  if (!response.ok) {
    throw await toApiError(response);
  }
}

export interface UploadOptions {
  /** The fraction of the request body sent so far, 0 to 1. */
  onProgress?: (fraction: number) => void;
  /** Aborts the request; the promise then rejects with the DOM `AbortError`. */
  signal?: AbortSignal;
}

/**
 * A response header, read only if CORS exposes it. The backend exposes `X-Correlation-ID` since
 * SE-002, but `getResponseHeader` still logs a console error for any header a response does not
 * expose (another origin, an older backend); the full header list holds only the exposed ones and
 * never logs.
 */
function exposedHeader(xhr: XMLHttpRequest, name: string): string | null {
  const prefix = `${name.toLowerCase()}:`;
  const line = xhr
    .getAllResponseHeaders()
    .split("\r\n")
    .find((entry) => entry.toLowerCase().startsWith(prefix));
  return line === undefined ? null : line.slice(prefix.length).trim();
}

function abortError(): DOMException {
  return new DOMException("The upload was cancelled.", "AbortError");
}

/**
 * A multipart POST that reports upload progress, which `fetch` cannot (FE-049). It fails exactly
 * as `requestJson` does: an abort rejects with the `AbortError`, no readable response is a
 * `network_error`, an error status is read as the backend's envelope, and a body that does not
 * match `schema` is `invalid_response`.
 */
export function postFormWithProgress<T>(
  path: string,
  schema: z.ZodType<T>,
  body: FormData,
  { onProgress, signal }: UploadOptions = {},
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(abortError());
      return;
    }
    const xhr = new XMLHttpRequest();
    const onAbort = () => {
      xhr.abort();
    };
    const settle = () => {
      signal?.removeEventListener("abort", onAbort);
    };
    xhr.open("POST", `${API_BASE_URL}${path}`);
    xhr.upload.onprogress = (event: ProgressEvent) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress?.(event.loaded / event.total);
      }
    };
    xhr.onabort = () => {
      settle();
      reject(abortError());
    };
    xhr.onerror = () => {
      settle();
      reject(
        ApiError.fromFacts({
          status: 0,
          code: "network_error",
          kind: "network",
          serverMessage: null,
          correlationId: null,
          validation: null,
        }),
      );
    };
    xhr.onload = () => {
      settle();
      const correlationId = exposedHeader(xhr, "X-Correlation-ID");
      const response = new Response(xhr.responseText, {
        status: xhr.status,
        headers: correlationId === null ? {} : { "X-Correlation-ID": correlationId },
      });
      readJson(response, schema).then(resolve, reject);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    xhr.send(body);
  });
}

export function jsonInit(method: "POST" | "PUT" | "PATCH", payload: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}
