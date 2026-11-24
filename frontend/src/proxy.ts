import { type NextRequest, NextResponse } from "next/server";

/**
 * The Content-Security-Policy, built for each request (RE2E-DEF-045, RMD11-T01, CT-RMD-22). The
 * policy used to allow every inline script (`script-src 'self' 'unsafe-inline'`), because the
 * framework's bootstrap scripts are inline and the pages were built static. Now each request gets a
 * fresh nonce: Next.js reads it from the policy on the request and puts it on every script it
 * writes, and `'strict-dynamic'` lets those scripts load the rest. An injected script has no nonce,
 * so it does not run. The nonce needs a render per request; the root layout asks for one.
 *
 * Next.js 16 calls this file convention Proxy (it was Middleware); DV-RMD-78.
 */

// The backend origin the client calls, from the same build-time `NEXT_PUBLIC_API_BASE_URL` as the
// client's fetches, uploads and event stream.
const API_ORIGIN = new URL(process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1")
  .origin;

/**
 * The policy for one response. `dev` adds `'unsafe-eval'`, which only React's development build
 * (`next dev`) needs; no production build has it. `img-src` keeps rendered Markdown from loading
 * remote images (FE-020); `upgrade-insecure-requests` is left out because the stack runs on
 * plain-HTTP loopback and the directive would rewrite the backend calls to HTTPS.
 */
export function contentSecurityPolicy(
  nonce: string,
  options: { dev: boolean; apiOrigin: string },
): string {
  return [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${options.dev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src 'self' ${options.apiOrigin}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

export function proxy(request: NextRequest): NextResponse {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const policy = contentSecurityPolicy(nonce, {
    dev: process.env.NODE_ENV === "development",
    apiOrigin: API_ORIGIN,
  });
  // The render reads the nonce from the policy on the request; `x-nonce` is there for any server
  // component that has to set it on a script of its own.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", policy);
  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", policy);
  return response;
}

// Pages only: static assets need no policy, and a prefetch's payload is not a document.
export const config = {
  matcher: [
    {
      source: "/((?!api|_next/static|_next/image|favicon.ico).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};
