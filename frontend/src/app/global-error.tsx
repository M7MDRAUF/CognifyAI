"use client";

import "./globals.css";

/**
 * Last-resort boundary (FE-048) for an error in the root layout itself. It replaces the root
 * layout, so it renders its own `<html>` and `<body>` and imports the global styles.
 */
export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en">
      <body className="flex min-h-full flex-col items-center justify-center gap-4 px-6 py-24 text-center">
        <title>Something went wrong · CognifyAI</title>
        <div role="alert" className="flex flex-col items-center gap-4">
          <h1 className="text-2xl font-semibold tracking-tight">Something went wrong</h1>
          <p className="text-sm text-muted-foreground">
            CognifyAI hit an unexpected error and could not show this page.
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
      </body>
    </html>
  );
}
