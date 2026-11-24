import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { connection } from "next/server";

import { AppHeader } from "@/components/layout/app-header";
import { MAIN_CONTENT_ID, SkipLink } from "@/components/layout/skip-link";
import { Providers } from "@/components/providers";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// FE-034: every page names itself; the template adds the product (WCAG 2.4.2).
export const metadata: Metadata = {
  title: { default: "CognifyAI", template: "%s · CognifyAI" },
  description:
    "Agentic RAG decision intelligence platform: source-grounded answers with citations, risk analysis, and evaluation metrics.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Every page renders per request: the policy's nonce exists only then, and Next.js puts it on
  // the scripts it writes (RE2E-DEF-045, src/proxy.ts). A page built ahead has no nonce to carry.
  await connection();
  return (
    // `suppressHydrationWarning` covers this element's OWN attributes only, one level deep --
    // children are still diffed normally, so a real mismatch anywhere inside the app still
    // reports. It is here because browser extensions inject attributes onto <html> before React
    // hydrates, which React cannot distinguish from a genuine server/client divergence: Katalon
    // Recorder adds `katalonextensionid`, and Grammarly, ColorZilla and others behave the same
    // way. The attributes this app sets here are two static font CSS variables and a constant
    // class string, so there is no server/client-varying value for this to mask.
    // <body> carries it for the same reason: some extensions inject onto <body> instead (one adds
    // `__processed_<uuid>__="true"`, seen 2026-11-24). Its only attribute here is a constant class.
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col" suppressHydrationWarning>
        <SkipLink />
        <AppHeader />
        <Providers>
          {/* The skip link's target: focusable by script only, and not a control, so no ring. */}
          <main
            id={MAIN_CONTENT_ID}
            tabIndex={-1}
            className="flex flex-1 flex-col focus:outline-none"
          >
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
