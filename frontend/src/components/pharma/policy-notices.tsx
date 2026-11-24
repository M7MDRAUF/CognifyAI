import type { PharmaRun } from "@/lib/types/pharma";

/**
 * What the regulatory-boundary check changed in this response (REQ-UX-05): one line per notice,
 * its code and the server's sentence. Nothing renders when the check changed nothing. The title
 * is a heading, so moving by headings reaches it before the analysis (PH-DEF-040).
 */
export function PolicyNotices({ notices }: { notices: PharmaRun["policy_notices"] }) {
  if (notices.length === 0) {
    return null;
  }
  return (
    <div className="space-y-1 rounded-md border p-3 text-sm">
      <h3 className="font-medium">The regulatory-boundary check changed this response</h3>
      <ul aria-label="Policy notices" className="list-disc space-y-1 pl-5">
        {notices.map((notice, index) => (
          <li key={`${String(index)}-${notice.code}`}>
            {notice.code}: {notice.notice}
          </li>
        ))}
      </ul>
    </div>
  );
}
