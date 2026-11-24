"use client";

import { useQuery } from "@tanstack/react-query";

import { DocumentTable } from "@/components/documents/document-table";
import { UploadDropzone } from "@/components/documents/upload-dropzone";
import { IngestionPrivacyNotice } from "@/components/privacy/data-processing-notice";
import { Button } from "@/components/ui/button";
import { StatusAnnouncer } from "@/components/ui/status-announcer";
import { listDocuments } from "@/lib/api/documents";
import { getProject } from "@/lib/api/projects";
import { useLlmConfig } from "@/lib/hooks/use-llm-config";
import { profileOf } from "@/lib/types/pharma";

const PROCESSING_STATES = new Set(["uploaded", "processing"]);

export function DocumentsView({ projectId }: { projectId: string }) {
  // Uploading is the moment a user decides to hand over a document, so it is where the
  // destination of its text belongs — the chat notice arrives far too late.
  const capability = useLlmConfig();
  const { data, isLoading, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => listDocuments(projectId),
    // Poll while any document is still being processed so the table updates live.
    refetchInterval: (query) => {
      const documents = query.state.data;
      if (documents === undefined) {
        return false;
      }
      return documents.some((document) => PROCESSING_STATES.has(document.status)) ? 2000 : false;
    },
  });
  // REQ-UX-03 (PH-DEF-038): the project's profile decides the table's columns, read from the query
  // the project's guard has already loaded (`ProjectGuard`), never fetched here. A pharmaceutical
  // project's table has its controlled columns before any document's metadata is written.
  const project = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId),
    enabled: false,
  });
  const controlled =
    project.data !== undefined && profileOf(project.data) === "pharma_manufacturing";

  return (
    <section className="space-y-6">
      <IngestionPrivacyNotice llmConfig={capability.config} capabilityStatus={capability.status} />
      {/* Held shut only while the lookup is genuinely in flight. A *failed* lookup releases
          it: the notice hedges honestly in that state, and the query is never retried, so
          locking on error would disable uploads for the life of the page. */}
      <UploadDropzone projectId={projectId} disabled={capability.status === "pending"} />
      {isLoading && <p className="text-muted-foreground">Loading documents…</p>}
      {/* RE2E-DEF-016: the message says to try again, so the page offers the means. */}
      {isError && (
        <div role="alert" className="space-y-3">
          <p className="text-destructive">
            {error instanceof Error ? error.message : "Failed to load documents."}
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={isFetching}
            onClick={() => {
              void refetch();
            }}
          >
            {isFetching ? "Retrying…" : "Try again"}
          </Button>
        </div>
      )}
      {data !== undefined && (
        <DocumentTable projectId={projectId} documents={data} controlled={controlled} />
      )}
      {/* Where "Document deleted." and "Reindex started." are said (RE2E-DEF-017, DV-RMD-77). */}
      <StatusAnnouncer />
    </section>
  );
}
