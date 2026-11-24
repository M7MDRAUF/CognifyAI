export const ALLOWED_EXTENSIONS = ["pdf", "txt", "md", "docx", "csv", "xlsx"] as const;
type AllowedExtension = (typeof ALLOWED_EXTENSIONS)[number];

/**
 * The server's limits (FE-025): `MAX_UPLOAD_SIZE_MB` and `MAX_PROJECT_SIZE_MB` in
 * `backend/.env.example`. The backend does not publish them, so a contract test holds these
 * constants to that file, and the copy and the checks both read them.
 */
export const MAX_FILE_SIZE_MB = 25;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
export const MAX_PROJECT_SIZE_MB = 250;
export const MAX_PROJECT_SIZE_BYTES = MAX_PROJECT_SIZE_MB * 1024 * 1024;

/**
 * The declared types the server accepts per extension (`backend/app/core/security.py`,
 * `_ALLOWED_MIME_BY_EXTENSION`; a contract test keeps the two equal). The first entry is the one
 * sent when the browser reports none, or one the server would refuse.
 */
export const ALLOWED_MIME_BY_EXTENSION: Record<AllowedExtension, readonly string[]> = {
  pdf: ["application/pdf"],
  txt: ["text/plain"],
  md: ["text/markdown", "text/plain", "text/x-markdown"],
  docx: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
  csv: ["text/csv", "application/csv", "application/vnd.ms-excel"],
  xlsx: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
};

function extensionOf(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function isAllowedExtension(extension: string): extension is AllowedExtension {
  return (ALLOWED_EXTENSIONS as readonly string[]).includes(extension);
}

/**
 * The file as the server will accept its declared type: browsers often report no type for `.md`
 * (sent as `application/octet-stream`, which the server refuses) or a type the allow-list does
 * not hold. The bytes and the name are unchanged.
 */
export function withAllowedType(file: File): File {
  const extension = extensionOf(file.name);
  if (!isAllowedExtension(extension)) {
    return file;
  }
  const allowed = ALLOWED_MIME_BY_EXTENSION[extension];
  if (allowed.includes(file.type.toLowerCase())) {
    return file;
  }
  return new File([file], file.name, { type: allowed[0], lastModified: file.lastModified });
}

/** Why a file cannot be added to a project already holding `existingBytes`, or null. */
export function projectCapacityError(
  existingBytes: number,
  file: { name: string; size: number },
): string | null {
  return existingBytes + file.size > MAX_PROJECT_SIZE_BYTES
    ? `"${file.name}" would take this project past its ${String(MAX_PROJECT_SIZE_MB)} MB storage limit.`
    : null;
}

export type UploadValidation = { ok: true } | { ok: false; error: string };

export function validateFileForUpload(file: { name: string; size: number }): UploadValidation {
  const extension = extensionOf(file.name);
  if (!isAllowedExtension(extension)) {
    return {
      ok: false,
      error: `"${file.name}" is not a supported type. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}.`,
    };
  }
  if (file.size === 0) {
    return { ok: false, error: `"${file.name}" is empty.` };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      ok: false,
      error: `"${file.name}" exceeds the ${String(MAX_FILE_SIZE_MB)} MB limit.`,
    };
  }
  return { ok: true };
}

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Whether `value` is a well-formed UUID (any version), the only id shape the backend accepts for a
 * project, run, document or report. Checked before a request is made, so a malformed id in a URL is
 * a not-found or an error on the page rather than pydantic's 422 text (FE-009, FE-011).
 */
export function isUuid(value: string): boolean {
  return UUID_PATTERN.test(value);
}
