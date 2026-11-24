/**
 * The boundary for `notFound()` thrown by `app/projects/[projectId]/layout.tsx`: a boundary in a
 * segment wraps its page and child segments, never the layout above them, so a layout's
 * `notFound()` lands one level up. It renders the same project not-found page (FE-009).
 */
import type { Metadata } from "next";

export { default } from "./[projectId]/not-found";

// FE-034: the head title of a malformed project URL (the page's own <title> repeats it).
export const metadata: Metadata = { title: "Project not found" };
