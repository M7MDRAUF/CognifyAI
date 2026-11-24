"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { announce } from "@/components/ui/status-announcer";
import { ApiError } from "@/lib/api/client";
import { deleteDocument } from "@/lib/api/documents";

interface DeleteDocumentDialogProps {
  documentId: string;
  documentName: string;
  projectId: string;
}

export function DeleteDocumentDialog({
  documentId,
  documentName,
  projectId,
}: DeleteDocumentDialogProps) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => deleteDocument(documentId),
    onSuccess: () => {
      setOpen(false);
      announce("Document deleted.");
      return queryClient.invalidateQueries({ queryKey: ["documents", projectId] });
    },
    onError: (cause: unknown) => {
      // The vector store is cleaned before the row is deleted (RMD6-T01, RE2E-DEF-027): a 503
      // from it means nothing was deleted, which the generic wording ("the change may still have
      // been saved") does not say. The list is unchanged, so it is not re-read.
      if (cause instanceof ApiError && cause.code === "qdrant_operation_failed") {
        setError("The document was not deleted: the search index could not be reached. Try again.");
        return;
      }
      setError(cause instanceof ApiError ? cause.message : "Could not delete the document.");
    },
  });

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        // A pending delete keeps the dialog open: Escape, the close button and a click outside would
        // otherwise hide the request's outcome (RE2E-DEF-018).
        if (!next && mutation.isPending) {
          return;
        }
        // Every opening starts clean; a failure from an earlier attempt is not this one's (FE-027).
        if (next) {
          setError(null);
        }
        setOpen(next);
      }}
    >
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" aria-label={`Delete ${documentName}`}>
          Delete
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete document</DialogTitle>
          <DialogDescription>
            This permanently deletes &ldquo;{documentName}&rdquo; and its indexed chunks. This
            cannot be undone.
          </DialogDescription>
        </DialogHeader>
        {error !== null && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={() => {
              setOpen(false);
            }}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={mutation.isPending}
            onClick={() => {
              mutation.mutate();
            }}
          >
            {mutation.isPending ? "Deleting…" : "Delete permanently"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
