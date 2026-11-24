import {
  type UploadOptions,
  postFormWithProgress,
  requestJson,
  requestVoid,
} from "@/lib/api/client";
import {
  type ProjectDocument,
  type ProjectDocumentDetail,
  documentDetailSchema,
  documentListSchema,
  documentSchema,
} from "@/lib/types/document";

export function listDocuments(projectId: string): Promise<ProjectDocument[]> {
  return requestJson(`/projects/${projectId}/documents`, documentListSchema);
}

/**
 * One file per request, as the endpoint accepts (FE-024 queues the rest). Sent with
 * XMLHttpRequest so the caller can show progress and cancel (FE-049).
 */
export function uploadDocument(
  projectId: string,
  file: File,
  options: UploadOptions = {},
): Promise<ProjectDocument> {
  const form = new FormData();
  form.append("file", file);
  return postFormWithProgress(`/projects/${projectId}/documents`, documentSchema, form, options);
}

export function getDocument(documentId: string): Promise<ProjectDocumentDetail> {
  return requestJson(`/documents/${documentId}`, documentDetailSchema);
}

export function deleteDocument(documentId: string): Promise<void> {
  return requestVoid(`/documents/${documentId}`, { method: "DELETE" });
}

export function reindexDocument(documentId: string): Promise<ProjectDocument> {
  return requestJson(`/documents/${documentId}/reindex`, documentSchema, { method: "POST" });
}
