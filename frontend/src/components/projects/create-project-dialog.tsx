"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { PROFILE_LABELS } from "@/components/pharma/profile-badge";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { announce } from "@/components/ui/status-announcer";
import { Textarea } from "@/components/ui/textarea";
import { createProject } from "@/lib/api/projects";
import { ApiError } from "@/lib/api/client";
import { useSubmitLock } from "@/lib/hooks/use-submit-lock";
import type { DomainProfile } from "@/lib/types/pharma";

const EMPTY_FORM = { name: "", description: "", domain: "", objective: "" };

const PROFILE_CHOICES: readonly DomainProfile[] = ["general", "pharma_manufacturing"];

export function CreateProjectDialog() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  // REQ-UX-01: General preselected; `createProject` omits the field for General.
  const [profile, setProfile] = useState<DomainProfile>("general");
  const [error, setError] = useState<string | null>(null);
  // FE-035: the name field itself is marked invalid, not only the form.
  const [nameInvalid, setNameInvalid] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const submitLock = useSubmitLock();

  const mutation = useMutation({
    mutationFn: createProject,
    onSuccess: async () => {
      setOpen(false);
      setForm(EMPTY_FORM);
      setProfile("general");
      setError(null);
      announce("Project created.");
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (cause: unknown) => {
      setError(cause instanceof ApiError ? cause.message : "Could not create the project.");
    },
    onSettled: submitLock.release,
  });

  const handleSubmit = (event: React.SyntheticEvent) => {
    event.preventDefault();
    if (form.name.trim() === "") {
      setError("Project name is required.");
      setNameInvalid(true);
      const field = formRef.current?.elements.namedItem("name");
      if (field instanceof HTMLInputElement) {
        field.focus();
      }
      return;
    }
    if (!submitLock.acquire()) {
      return;
    }
    mutation.mutate({ ...form, domain_profile: profile });
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        // Every opening starts clean; a failure from an earlier attempt is not this one's (FE-027).
        if (next) {
          setError(null);
          setNameInvalid(false);
        } else {
          // REQ-UX-01: General is preselected at every opening, after a cancel or a refusal too
          // (PH-DEF-040).
          setProfile("general");
        }
        setOpen(next);
      }}
    >
      <DialogTrigger asChild>
        <Button>New project</Button>
      </DialogTrigger>
      <DialogContent>
        <form ref={formRef} onSubmit={handleSubmit} noValidate className="grid gap-4">
          <DialogHeader>
            <DialogTitle>Create project</DialogTitle>
            <DialogDescription>
              Group documents and decision questions under a single project.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2">
            <Label htmlFor="project-name">Name</Label>
            <Input
              id="project-name"
              name="name"
              required
              aria-invalid={nameInvalid ? true : undefined}
              aria-describedby={nameInvalid ? "create-project-error" : undefined}
              value={form.name}
              onChange={(event) => {
                setForm((current) => ({ ...current, name: event.target.value }));
                if (error !== null) setError(null);
                setNameInvalid(false);
              }}
            />
          </div>
          <fieldset className="space-y-2" aria-describedby="project-profile-hint">
            <legend className="text-sm font-medium leading-none">Domain profile</legend>
            <div className="flex flex-wrap gap-x-6 gap-y-2 pt-1">
              {PROFILE_CHOICES.map((choice) => (
                <label key={choice} className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="domain_profile"
                    value={choice}
                    checked={profile === choice}
                    onChange={() => {
                      setProfile(choice);
                    }}
                    className="size-4"
                  />
                  {PROFILE_LABELS[choice]}
                </label>
              ))}
            </div>
            <p id="project-profile-hint" className="text-xs text-muted-foreground">
              Pharmaceutical manufacturing adds controlled-document metadata, version-aware evidence
              and human-review escalation. The profile cannot be changed after creation.
            </p>
          </fieldset>
          <div className="grid gap-2">
            <Label htmlFor="project-domain">Domain</Label>
            <Input
              id="project-domain"
              value={form.domain}
              onChange={(event) => {
                setForm((current) => ({ ...current, domain: event.target.value }));
              }}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="project-objective">Analysis objective</Label>
            <Input
              id="project-objective"
              value={form.objective}
              onChange={(event) => {
                setForm((current) => ({ ...current, objective: event.target.value }));
              }}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="project-description">Description</Label>
            <Textarea
              id="project-description"
              value={form.description}
              onChange={(event) => {
                setForm((current) => ({ ...current, description: event.target.value }));
              }}
            />
          </div>
          {error !== null && (
            <p id="create-project-error" role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? "Creating…" : "Create project"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
