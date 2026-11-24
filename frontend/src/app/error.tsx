"use client";

/**
 * Root error boundary (FE-048): a render error in a page shows this instead of the framework's
 * default screen, keeping the header and a way to recover. `retry` re-fetches and re-renders the
 * segment. The error's own message is not shown: it may be a developer's text.
 */
export default function RootError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <div
      role="alert"
      className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-4 px-6 py-24 text-center"
    >
      <h1 className="text-2xl font-semibold tracking-tight">Something went wrong</h1>
      <p className="text-sm text-muted-foreground">
        This page hit an unexpected error. Try again; if it keeps happening, reload the page.
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
