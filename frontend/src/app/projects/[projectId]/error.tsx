"use client";

/**
 * Project error boundary (FE-048): a render error in one project page (a chart, the graph, the
 * Markdown renderer) is contained here, below the project's tabs, so the rest of the project stays
 * reachable. `retry` re-fetches and re-renders the page.
 */
export default function ProjectError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <div role="alert" className="flex flex-col items-center gap-4 py-16 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">Something went wrong</h1>
      <p className="text-sm text-muted-foreground">
        This part of the project hit an unexpected error. Other tabs still work.
        {error.digest !== undefined && <> Reference: {error.digest}</>}
      </p>
      <button
        type="button"
        onClick={retry}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2"
      >
        Try again
      </button>
    </div>
  );
}
