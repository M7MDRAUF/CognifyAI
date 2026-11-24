import type { Metadata } from "next";

import { ChatView } from "@/components/chat/chat-view";

export const metadata: Metadata = { title: "Chat" };

/**
 * The viewed run lives in the URL (`?run=<uuid>`, FE-011), so a refresh, back/forward or a shared
 * link restores it. The raw value is passed on; `ChatView` validates it before any request.
 */
export default async function ChatPage({
  params,
  searchParams,
}: {
  params: Promise<{ projectId: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { projectId } = await params;
  const { run } = await searchParams;
  const runParam = Array.isArray(run) ? (run[0] ?? null) : (run ?? null);
  return <ChatView projectId={projectId} initialRunParam={runParam} />;
}
