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
import { deleteProject } from "@/lib/api/projects";

interface DeleteProjectDialogProps {
  projectId: string;
  projectName: string;
}

export function DeleteProjectDialog({ projectId, projectName }: DeleteProjectDialogProps) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => deleteProject(projectId),
    onSuccess: () => {
      setOpen(false);
      announce("Project deleted.");
    },
    onError: (cause: unknown) => {
      setError(cause instanceof ApiError ? cause.message : "Could not delete the project.");
    },
    // A failure may follow a delete that already committed, so the list is re-read either way.
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["projects"] }),
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
        <Button
          variant="outline"
          size="sm"
          aria-label={`Delete ${projectName}`}
          className="text-destructive hover:text-destructive hover:bg-destructive/10"
        >
          Delete
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete project</DialogTitle>
          <DialogDescription>
            This permanently deletes “{projectName}”, its documents, chunks, and all query history.
            This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        {error !== null && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          {/* RE2E-DEF-006: a way out besides Escape and the corner button, as the document dialog has. */}
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
