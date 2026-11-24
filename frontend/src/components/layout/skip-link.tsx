/** The id of the root layout's one `main` landmark: the skip link's target (SD-06). */
export const MAIN_CONTENT_ID = "main-content";

/**
 * The first tab stop on every page the root layout renders (SE-008, WCAG 2.4.1, technique G1):
 * it skips the site header to `main`. Visually hidden until it takes focus. A plain in-page anchor
 * rendered on the server, so it needs no client JavaScript and nothing from the CSP.
 * `global-error.tsx` replaces the layout and renders no header to skip (SD-07).
 */
export function SkipLink() {
  return (
    <a
      href={`#${MAIN_CONTENT_ID}`}
      className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-background focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:shadow focus-visible:outline-2 focus-visible:outline-offset-2"
    >
      Skip to main content
    </a>
  );
}
