import Link from "next/link";

const FEATURES = [
  {
    title: "Document ingestion",
    // This page is static and server-rendered: it never fetches `/llm/config`, so any vendor
    // it names is a guess. "OpenAI" was a guess, and a wrong one both under `LLM_PROVIDER=fake`
    // (nothing is sent anywhere) and wherever a base URL override points the embedding client
    // at another host. The disclosure that matters — the text leaves the server — survives
    // without the name; the surfaces that *can* ask the backend name it (privacy/*).
    description:
      "Upload PDFs, DOCX, and spreadsheets. Text is extracted and chunked on the server, then " +
      "each chunk is sent to the embedding service this deployment is configured with — which " +
      "may be an external API — to build the search index.",
  },
  {
    title: "Agentic RAG pipeline",
    description:
      "A multi-agent workflow plans, retrieves, synthesises, critiques, and assesses risk before answering.",
  },
  {
    title: "Citations and evidence",
    // OD-09: not every answer can be called grounded. A citation that names no retrieved excerpt
    // is dropped, but retrieval itself is not graded on recall or precision (known limitation 32).
    description:
      "Answers cite the document excerpts they draw on, with source snippets. Retrieval is not " +
      "independently graded, so check the cited sources before you rely on an answer.",
  },
  {
    title: "Evaluation metrics",
    // OD-09 and OD-02: the scores come from an agent under test that never gates (D-2); its
    // agreement with the deterministic graders is not computed (known limitations 14 and 31).
    description:
      "An evaluator agent scores each answer for relevance, faithfulness, citation accuracy and " +
      "hallucination risk. The scores are a model's judgement, not a verified measurement, and " +
      "they never gate an answer.",
  },
] as const;

export function HomeHero() {
  return (
    <div className="mx-auto w-full max-w-4xl flex-1 px-6 py-16 flex flex-col gap-12">
      <section className="flex flex-col items-center gap-6 text-center">
        <h1 className="text-5xl font-bold tracking-tight">CognifyAI</h1>
        <p className="max-w-xl text-lg text-muted-foreground">
          Decision intelligence over your own documents. Upload them, ask complex questions, and get
          source-grounded recommendations with citations, risk analysis, and evaluator scores.
          Document text and questions are processed by the AI providers this deployment is
          configured with.
        </p>
        <Link
          href="/projects"
          className="rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2"
        >
          Open projects
        </Link>
      </section>

      <section aria-label="Features">
        <div className="grid gap-4 sm:grid-cols-2">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="rounded-lg border p-5">
              <h2 className="mb-1 text-sm font-semibold">{feature.title}</h2>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
