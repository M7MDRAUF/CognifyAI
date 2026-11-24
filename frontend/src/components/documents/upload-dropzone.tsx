"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api/client";
import { uploadDocument } from "@/lib/api/documents";
import { useSubmitLock } from "@/lib/hooks/use-submit-lock";
import type { ProjectDocument } from "@/lib/types/document";
import {
  ALLOWED_EXTENSIONS,
  MAX_FILE_SIZE_MB,
  MAX_PROJECT_SIZE_MB,
  projectCapacityError,
  validateFileForUpload,
  withAllowedType,
} from "@/lib/utils/validation";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  projectId: string;
  /**
   * Holds the control shut while the surface above it cannot yet say where a document's text
   * is sent. Uploading is the moment the decision is made and it is irreversible once the
   * chunks have been embedded, so the disclosure has to land first (V6 §1).
   */
  disabled?: boolean;
}

/** What happened to one file of a drop or a selection; every file gets one (FE-024). */
interface FileResult {
  name: string;
  outcome: "uploaded" | "duplicate uploaded" | "failed" | "cancelled" | "not sent" | "skipped";
  /** Why a file was not uploaded, in the words the user sees. */
  reason?: string;
}

interface Progress {
  name: string;
  position: number;
  total: number;
  /** 0 to 100: the share of the file's bytes sent. */
  percent: number;
}

const OUTCOME_TEXT: Record<FileResult["outcome"], string> = {
  uploaded: "uploaded",
  "duplicate uploaded": "uploaded as another copy",
  failed: "failed",
  cancelled: "cancelled",
  "not sent": "not uploaded (cancelled)",
  skipped: "skipped",
};

function isAbort(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

export function UploadDropzone({ projectId, disabled = false }: UploadDropzoneProps) {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [results, setResults] = useState<FileResult[]>([]);
  const [rejections, setRejections] = useState<string[]>([]);
  const [progress, setProgress] = useState<Progress | null>(null);
  // OD-03: a name the project already has is uploaded again only when the user says so.
  const [duplicates, setDuplicates] = useState<{
    names: string[];
    decide: (upload: boolean) => void;
  } | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const cancelledRef = useRef(false);
  const submitLock = useSubmitLock();

  const listedDocuments = (): ProjectDocument[] =>
    queryClient.getQueryData<ProjectDocument[]>(["documents", projectId]) ?? [];

  /** Validates, asks about duplicates, then uploads one file at a time; reports every file. */
  const uploadAll = async (files: File[]) => {
    const rejected: string[] = [];
    const accepted: File[] = [];
    // The project cap, checked against the listed documents; the server stays the authority.
    let projectBytes = listedDocuments().reduce(
      (total, document) => total + document.file_size_bytes,
      0,
    );
    for (const file of files) {
      const validation = validateFileForUpload(file);
      const overCapacity = validation.ok ? projectCapacityError(projectBytes, file) : null;
      if (!validation.ok) {
        rejected.push(validation.error);
      } else if (overCapacity !== null) {
        rejected.push(overCapacity);
      } else {
        projectBytes += file.size;
        accepted.push(file);
      }
    }
    setRejections(rejected);
    const done: FileResult[] = [];
    setResults([]);

    const known = new Set(listedDocuments().map((document) => document.original_filename));
    const repeated = accepted.filter((file) => known.has(file.name));
    let toSend = accepted;
    if (repeated.length > 0) {
      const upload = await new Promise<boolean>((resolve) => {
        setDuplicates({ names: repeated.map((file) => file.name), decide: resolve });
      });
      setDuplicates(null);
      if (!upload) {
        toSend = accepted.filter((file) => !known.has(file.name));
        done.push(...repeated.map((file) => ({ name: file.name, outcome: "skipped" as const })));
        setResults([...done]);
      }
    }

    for (const [index, file] of toSend.entries()) {
      if (cancelledRef.current) {
        done.push({ name: file.name, outcome: "not sent" });
        continue;
      }
      const controller = new AbortController();
      controllerRef.current = controller;
      setProgress({ name: file.name, position: index + 1, total: toSend.length, percent: 0 });
      try {
        await uploadDocument(projectId, withAllowedType(file), {
          signal: controller.signal,
          onProgress: (fraction) => {
            setProgress((current) =>
              current === null ? current : { ...current, percent: Math.round(fraction * 100) },
            );
          },
        });
        done.push({
          name: file.name,
          outcome: known.has(file.name) ? "duplicate uploaded" : "uploaded",
        });
        void queryClient.invalidateQueries({ queryKey: ["documents", projectId] });
      } catch (cause) {
        if (isAbort(cause)) {
          done.push({ name: file.name, outcome: "cancelled" });
        } else {
          done.push({
            name: file.name,
            outcome: "failed",
            reason: cause instanceof ApiError ? cause.message : "Upload failed.",
          });
        }
      }
      setResults([...done]);
    }
    controllerRef.current = null;
    setProgress(null);
    setResults([...done]);
  };

  const handleFiles = (files: FileList | null) => {
    if (files === null || files.length === 0) {
      return;
    }
    // Checked here rather than only on the button: a drop lands on the region, which has no
    // disabled state of its own, so the button alone would leave the drag-and-drop path open.
    // The drop is refused out loud (RE2E-DEF-008): silence read as an upload that never came.
    if (disabled) {
      const names = Array.from(files, (file) => file.name);
      const one = names.length === 1;
      setNotice(
        `${names.join(", ")} ${one ? "was" : "were"} not uploaded: uploads are paused until this ` +
          `page can say where document text is sent. Add ${one ? "it" : "them"} again when ` +
          "Choose files is available.",
      );
      return;
    }
    // A drop lands on the region whatever the button's state, so the lock is what refuses a
    // second batch while one is being uploaded (FE-021).
    if (!submitLock.acquire()) {
      setNotice("An upload is already in progress. Wait for it to finish, then add the next file.");
      return;
    }
    setNotice(null);
    cancelledRef.current = false;
    void uploadAll(Array.from(files)).finally(submitLock.release);
  };

  const cancel = () => {
    cancelledRef.current = true;
    controllerRef.current?.abort();
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    handleFiles(event.dataTransfer.files);
  };

  const accept = ALLOWED_EXTENSIONS.map((extension) => `.${extension}`).join(",");
  const busy = progress !== null || duplicates !== null;
  const failures = [
    ...rejections,
    ...results
      .filter((result) => result.outcome === "failed")
      .map((result) => `"${result.name}" failed to upload. ${result.reason ?? ""}`.trim()),
  ];

  return (
    <div
      role="region"
      aria-label="Upload area"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "rounded-lg border border-dashed p-6 text-center transition-colors",
        isDragging && !disabled && "border-primary bg-primary/5",
        disabled && "opacity-60",
      )}
    >
      <p className="text-sm text-muted-foreground">
        {isDragging && !disabled
          ? "Drop your files here"
          : `Drag and drop files here, or use Choose files. PDF, TXT, Markdown, DOCX, CSV, or XLSX (max ${String(MAX_FILE_SIZE_MB)} MB each, ${String(MAX_PROJECT_SIZE_MB)} MB per project).`}
      </p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={accept}
        className="sr-only"
        // One control per action (FE-017): the visible button opens this picker, so the input
        // is neither a tab stop nor announced; drops reach it through the region.
        tabIndex={-1}
        aria-hidden="true"
        aria-label="Upload document"
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <Button
        className="mt-4"
        disabled={disabled || busy}
        onClick={() => inputRef.current?.click()}
      >
        {progress !== null ? "Uploading…" : "Choose files"}
      </Button>

      {duplicates !== null && (
        <div role="alert" className="mt-3 space-y-2 text-sm">
          <p>
            This project already has{" "}
            {duplicates.names.map((name) => `a document named "${name}"`).join(" and ")}. Uploading
            it again adds another copy.
          </p>
          <div className="flex justify-center gap-2">
            <Button
              size="sm"
              onClick={() => {
                duplicates.decide(true);
              }}
            >
              Upload another copy
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                duplicates.decide(false);
              }}
            >
              Skip it
            </Button>
          </div>
        </div>
      )}

      {progress !== null && (
        <div role="status" className="mt-3 space-y-1 text-sm">
          <p>
            Uploading {progress.name} ({progress.position} of {progress.total}): {progress.percent}%
          </p>
          <progress
            className="w-full"
            max={100}
            value={progress.percent}
            aria-label={`Uploading ${progress.name}`}
          />
          <Button size="sm" variant="outline" onClick={cancel}>
            Cancel upload
          </Button>
        </div>
      )}

      {notice !== null && (
        <p role="alert" className="mt-3 text-sm text-destructive">
          {notice}
        </p>
      )}
      {failures.length > 0 && (
        <div role="alert" className="mt-3 space-y-1 text-sm text-destructive">
          {failures.map((failure) => (
            <p key={failure}>{failure}</p>
          ))}
        </div>
      )}
      {results.length > 0 && (
        <ul aria-label="Upload results" className="mt-3 space-y-1 text-sm">
          {results.map((result, index) => (
            <li key={`${String(index)}-${result.name}`}>
              {result.name}: {OUTCOME_TEXT[result.outcome]}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
