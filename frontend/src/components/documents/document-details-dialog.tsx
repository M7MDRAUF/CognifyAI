"use client";

import { Fragment, type ReactNode, useState } from "react";

import { LifecycleBadge } from "@/components/pharma/lifecycle-badge";
import { MetadataIssues } from "@/components/pharma/metadata-issues";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/documents/status-badge";
import type { ProjectDocument } from "@/lib/types/document";
import type { ControlledMetadata } from "@/lib/types/pharma";
import { formatBytes } from "@/lib/utils/format";
import { LocalTime } from "@/components/ui/local-time";

interface DocumentDetailsDialogProps {
  document: ProjectDocument;
}

export function DocumentDetailsDialog({ document }: DocumentDetailsDialogProps) {
  const [open, setOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button
          type="button"
          className="font-medium text-left hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:rounded-sm"
        >
          {document.original_filename}
        </button>
      </DialogTrigger>
      {/* A controlled document's sixteen fields can outgrow a short viewport; the shared dialog bounds and scrolls. */}
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{document.original_filename}</DialogTitle>
        </DialogHeader>
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Status</dt>
            <dd>
              <StatusBadge status={document.status} />
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Size</dt>
            <dd>{formatBytes(document.file_size_bytes)}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Type</dt>
            <dd>{document.mime_type}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Uploaded</dt>
            <dd>
              <LocalTime iso={document.created_at} />
            </dd>
          </div>
          {document.error_message !== null && (
            <div className="flex flex-col gap-1">
              <dt className="text-muted-foreground">Error</dt>
              <dd className="text-destructive text-xs">{document.error_message}</dd>
            </div>
          )}
        </dl>
        {document.controlled_metadata != null && (
          <ControlledMetadataSection metadata={document.controlled_metadata} />
        )}
        <div className="flex justify-end">
          <Button
            variant="outline"
            onClick={() => {
              setOpen(false);
            }}
          >
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

const NOT_RECORDED = "Not recorded";

export const DOCUMENT_ASSERTED =
  "As stated in the document's own header. These values are not verified against a " +
  "document-control system.";

/** `sop` reads as an acronym; every other value reads as words. */
function words(value: string | null): string {
  if (value === null) {
    return NOT_RECORDED;
  }
  if (value === "sop") {
    return "SOP";
  }
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Every controlled field the server stored for this document (REQ-UX-03), each as text. */
function ControlledMetadataSection({ metadata }: { metadata: ControlledMetadata }) {
  const supersedes =
    metadata.supersedes_number === null
      ? NOT_RECORDED
      : `${metadata.supersedes_number} revision ${metadata.supersedes_revision ?? "unknown"}`;
  const fields: [string, ReactNode][] = [
    ["Document number", metadata.document_number ?? NOT_RECORDED],
    ["Title", metadata.title ?? NOT_RECORDED],
    ["Document type", words(metadata.document_type)],
    ["Revision", metadata.revision ?? NOT_RECORDED],
    ["Effective date", metadata.effective_date ?? NOT_RECORDED],
    ["Lifecycle", <LifecycleBadge key="lifecycle" status={metadata.lifecycle_status} />],
    ["Supersedes", supersedes],
    ["Site", metadata.site ?? NOT_RECORDED],
    ["Area", metadata.area ?? NOT_RECORDED],
    ["Equipment", metadata.equipment.length > 0 ? metadata.equipment.join(", ") : NOT_RECORDED],
    ["Process", metadata.process ?? NOT_RECORDED],
    ["Product family", metadata.product_family ?? NOT_RECORDED],
    ["Source authority", words(metadata.source_authority)],
    ["Control status", words(metadata.control_status)],
    ["Data classification", words(metadata.data_classification)],
    ["Metadata status", words(metadata.metadata_status)],
  ];
  return (
    <section aria-label="Controlled metadata" className="space-y-2 border-t pt-3">
      <h3 className="text-sm font-semibold">Controlled metadata</h3>
      {/* Threat T-04 (PH-DEF-037): the values are the document's own claims. */}
      <p className="text-xs text-muted-foreground">{DOCUMENT_ASSERTED}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        {fields.map(([term, value]) => (
          <Fragment key={term}>
            <dt className="text-muted-foreground">{term}</dt>
            <dd className="break-words">{value}</dd>
          </Fragment>
        ))}
      </dl>
      <MetadataIssues status={metadata.metadata_status} issues={metadata.metadata_issues} />
    </section>
  );
}
