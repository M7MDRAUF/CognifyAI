import { z } from "zod";

// FE-032: zod 4 compiles object parsers with `Function` by default, and the first object schema it
// builds probes for that by calling `Function("")`. The Content-Security-Policy in
// `next.config.ts` allows no `'unsafe-eval'` outside `next dev`, so the probe is a CSP violation,
// and a console error, on every page load. Every schema module takes `z` from here, which puts
// this setting before the first schema by construction. Parsing results are unchanged; object
// parsers are interpreted instead of compiled.
z.config({ jitless: true });

export { z };
