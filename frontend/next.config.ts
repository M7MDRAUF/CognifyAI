import path from "node:path";
import { fileURLToPath } from "node:url";

import type { NextConfig } from "next";

// FE-032: the response headers every route carries. The Content-Security-Policy is no longer set
// here: it carries a per-request nonce, so `src/proxy.ts` builds it for each page (RE2E-DEF-045,
// RMD11-T01).

const nextConfig: NextConfig = {
  // Without this, Next.js infers the workspace root by walking up for a lockfile and picks the
  // first one it finds -- which on a machine with a stray `package-lock.json` in the home
  // directory is that directory, not this app. Turbopack then resolves against a `node_modules`
  // tree that has nothing to do with this project. Pinning the root to the directory this config
  // lives in makes resolution independent of whatever sits above the checkout, which also keeps
  // the build reproducible across machines.
  //
  // `fileURLToPath` rather than `new URL(...).pathname`: on Windows the latter yields
  // `/C:/Users/...`, a leading slash that makes the path invalid.
  turbopack: { root: path.dirname(fileURLToPath(import.meta.url)) },
  poweredByHeader: false,
  headers: () => [
    {
      source: "/:path*",
      headers: [
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "no-referrer" },
      ],
    },
  ],
};

export default nextConfig;
