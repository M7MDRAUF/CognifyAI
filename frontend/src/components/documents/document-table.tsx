"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { DeleteDocumentDialog } from "@/components/documents/delete-document-dialog";
import { DocumentDetailsDialog } from "@/components/documents/document-details-dialog";
import { StatusBadge } from "@/components/documents/status-badge";
import { LifecycleBadge } from "@/components/pharma/lifecycle-badge";
import { MetadataIssues } from "@/components/pharma/metadata-issues";
import { Button } from "@/components/ui/button";
import { announce } from "@/components/ui/status-announcer";
import { ApiError } from "@/lib/api/client";
import { reindexDocument } from "@/lib/api/documents";
import type { ProjectDocument } from "@/lib/types/document";
import type { ControlledMetadata } from "@/lib/types/pharma";
import { formatBytes } from "@/lib/utils/format";

interface DocumentTableProps {
  projectId: string;
  documents: ProjectDocument[];
  /** The project is pharmaceutical: the controlled columns show whatever the documents hold. */
  controlled?: boolean;
}

export function DocumentTable({ projectId, documents, controlled = false }: DocumentTableProps) {
  const queryClient = useQueryClient();
  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["documents", projectId] });

  // A failed reindex is shown in its own row (FE-026, Probe P-31); it was silent before.
  const [reindexError, setReindexError] = useState<{ documentId: string; message: string } | null>(
    null,
  );

  const reindex = useMutation({
    mutationFn: reindexDocument,
    onMutate: () => {
      setReindexError(null);
    },
    onSuccess: () => {
      announce("Reindex started.");
      return invalidate();
    },
    onError: (cause: unknown, documentId: string) => {
      setReindexError({
        documentId,
        message: cause instanceof ApiError ? cause.message : "The request failed.",
      });
    },
  });

  if (documents.length === 0) {
    return <p className="text-muted-foreground">No documents uploaded yet.</p>;
  }

  // REQ-UX-03: the controlled columns appear for a pharmaceutical project from its first upload
  // (PH-DEF-038), and whenever a document carries controlled metadata, which only a
  // pharmaceutical project's documents do; a General table is the baseline's.
  const controlledTable =
    controlled || documents.some((document) => document.controlled_metadata != null);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4 font-medium">Name</th>
            {controlledTable && (
              <>
                <th className="py-2 pr-4 font-medium">Number</th>
                <th className="py-2 pr-4 font-medium">Revision</th>
                <th className="py-2 pr-4 font-medium">Lifecycle</th>
              </>
            )}
            <th className="py-2 pr-4 font-medium hidden sm:table-cell">Size</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((document) => (
            <tr key={document.id} className="border-b">
              {/* RE2E-DEF-019: a long name without spaces breaks inside its cell, not under the next. */}
              <td className="py-2 pr-4 max-w-[200px] wrap-anywhere">
                <DocumentDetailsDialog document={document} />
                {document.error_message !== null && (
                  <span className="block text-xs text-destructive">{document.error_message}</span>
                )}
                {document.controlled_metadata != null && (
                  <MetadataIssues
                    status={document.controlled_metadata.metadata_status}
                    issues={document.controlled_metadata.metadata_issues}
                  />
                )}
              </td>
              {controlledTable && <ControlledCells metadata={document.controlled_metadata} />}
              <td className="py-2 pr-4 hidden sm:table-cell">
                {formatBytes(document.file_size_bytes)}
              </td>
              <td className="py-2 pr-4">
                <StatusBadge status={document.status} />
              </td>
              <td className="py-2">
                <div className="flex gap-2">
                  {/* RE2E-DEF-007: named for its document, as Delete is; the visible text stays. */}
                  <Button
                    variant="outline"
                    size="sm"
                    aria-label={`Reindex ${document.original_filename}`}
                    disabled={
                      (reindex.isPending && reindex.variables === document.id) ||
                      document.status === "processing"
                    }
                    onClick={() => {
                      reindex.mutate(document.id);
                    }}
                  >
                    Reindex
                  </Button>
                  <DeleteDocumentDialog
                    documentId={document.id}
                    documentName={document.original_filename}
                    projectId={projectId}
                  />
                </div>
                {reindexError?.documentId === document.id && (
                  <p role="alert" className="mt-1 text-xs text-destructive">
                    Could not reindex {document.original_filename}. {reindexError.message}
                  </p>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const NOT_RECORDED = <span className="text-muted-foreground">Not recorded</span>;

/** A document's number, revision and lifecycle; a document without metadata says so in each. */
function ControlledCells({ metadata }: { metadata: ControlledMetadata | null | undefined }) {
  if (metadata == null) {
    return (
      <>
        <td className="py-2 pr-4">{NOT_RECORDED}</td>
        <td className="py-2 pr-4">{NOT_RECORDED}</td>
        <td className="py-2 pr-4">{NOT_RECORDED}</td>
      </>
    );
  }
  return (
    <>
      <td className="py-2 pr-4 whitespace-nowrap">{metadata.document_number ?? NOT_RECORDED}</td>
      <td className="py-2 pr-4">{metadata.revision ?? NOT_RECORDED}</td>
      <td className="py-2 pr-4">
        <LifecycleBadge status={metadata.lifecycle_status} />
      </td>
    </>
  );
}
