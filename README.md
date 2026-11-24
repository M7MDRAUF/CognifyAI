<div align="center">

# CognifyAI

**A multi-agent, evidence-grounded assistant for pharmaceutical manufacturing documents.**

In a pharmaceutical project it removes superseded, draft and retired revisions before any model reads the evidence,
cites the document section behind each claim, reports documents that state different values, refuses requests to make
regulated quality decisions, and names the roles who must review the answer.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

</div>

![A CognifyAI answer that cites the effective revision of a cleaning procedure](assets/screenshots/04-chat-answer.png)

---

## Contents

- [The problem](#the-problem)
- [What CognifyAI does](#what-cognifyai-does)
- [How a question is answered](#how-a-question-is-answered)
- [Architecture](#architecture)
- [Evaluation](#evaluation)
- [Screenshots](#screenshots)
- [Getting started](#getting-started)
- [Writing controlled documents](#writing-controlled-documents)
- [API](#api)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Security and safety design](#security-and-safety-design)
- [Limitations](#limitations)
- [Author](#author)

---

## The problem

A pharmaceutical plant runs on controlled documents: standard operating procedures, work instructions, forms and
specifications. Each one goes through revisions, and only the revision marked Effective may be used on the floor.
When something unexpected happens on a line, people need the right section of the right revision quickly, and an
outdated answer can affect product quality.

A general chat assistant has no record of which revision is in force, often cannot point to the section it used, and
will answer a request to approve something that only a qualified person may decide. CognifyAI adds a code check for
each of these.

## What CognifyAI does

### In every project

- Document ingestion for PDF, DOCX, TXT, Markdown, CSV and XLSX, in background tasks with per-file and per-project size
  limits. A lease on each job lets the app close a document or run that a crash left unfinished, so it can be retried.
- Hybrid retrieval: semantic search in Qdrant and full-text search in PostgreSQL (top 30 each), fused by normalised
  score (65% semantic, 35% lexical), then reranked by a model down to the 8 best passages.
- A pipeline of seven stages run in a fixed order by plain Python: planner, retriever, researcher, critic, risk
  analyst, final decision and evaluator. The retriever is code (hybrid search plus the model reranker). The other six
  are model calls whose JSON output is validated against a Pydantic schema, with one repair attempt and then one
  strict retry when validation fails.
- Live progress over Server-Sent Events, so the interface shows each stage as it runs.
- Citations on each answer, scores from an evaluator model (relevance, faithfulness, citation accuracy, context
  sufficiency, hallucination risk) for each answered run, a project dashboard and Markdown reports. The evaluator's
  scores are a model's judgement, and the interface labels them that way.
- Chat models from OpenAI, DeepSeek, NVIDIA, GLM and Kimi behind one router, each with an overridable base URL.
  Embeddings use the OpenAI embeddings API, either OpenAI itself or a compatible local server such as Ollama.

### Pharmaceutical manufacturing profile

A project created with the `pharma_manufacturing` profile adds a controlled-document workflow. Markdown and text
documents can open with a header (document number, revision, lifecycle status and scope); PDF, DOCX, CSV and XLSX files
carry no header and are treated as Unknown status. Every question then goes through extra checks that run as code:

| Capability | What it means in practice |
|---|---|
| Revision control | Superseded, draft and retired revisions are removed before the agents see any text. A revision that another Effective revision names as superseded is removed even when its own header still says Effective. |
| Conflict detection | When two Effective documents for the same equipment or process give different values for the same parameter (a `Label: value unit` line), for example a work instruction and the SOP it follows, the answer shows both values. Two revisions of one document that are both marked Effective are kept, flagged as a version conflict, and cap the evidence at partial. |
| Evidence sufficiency gate | After reranking and before the researcher, critic, risk and decision agents run, a deterministic rule grades the passages as sufficient, partial or insufficient. The question's anchors (equipment and document numbers, CFR sections, incident fields) must appear in Effective controlled evidence, and the best passage must contain enough of the question's content words. Insufficient evidence stops the run with an abstention that lists what is missing; partial evidence is answered and escalated. |
| Grounding gate | Each factual sentence of the answer must cite a passage that contains at least 60% of its content words and every number it states. A passage that reads as an instruction to the model supports nothing. Unsupported sentences are withheld; if none is supported, the answer becomes an abstention. |
| Regulated-decision boundary | Requests to release or reject a batch, close a deviation, close or approve a CAPA, override a procedure or waive a quality review are refused. A check that fails closed also removes any answer sentence that takes such a decision, names a root cause or declares compliance, and escalates the removal to quality assurance. |
| Escalation rules | Fixed rules decide when an answer needs human review and by whom. An incident goes to quality assurance and the process owner; partial or missing evidence to quality assurance and a subject-matter expert; a conflict to document control and quality assurance. The model can add a role but never remove one. |
| Question types | A question is a procedure question, an operational incident or a readiness review. An incident carries structured fields (summary, area, equipment, process step, product family, observations, immediate actions) that steer retrieval and escalation. Optional filters narrow retrieval by equipment, area, process or document type. |

## How a question is answered

The diagram shows a question in a pharmaceutical project. A general project skips version resolution and the
pharmaceutical checks, and its critic can request one extra retrieval round.

```mermaid
flowchart LR
    Q["Question or incident"] --> P["Planner"]
    P --> R["Hybrid retrieval<br/>Qdrant + PostgreSQL"]
    R --> V["Version resolution<br/>drops superseded, draft<br/>and retired revisions"]
    V --> K["Reranker<br/>best 8 passages"]
    K --> S{"Sufficiency gate"}
    S -- "insufficient" --> X["Abstain and<br/>escalate"]
    S -- "sufficient or partial" --> A["Researcher, critic,<br/>risk analyst"]
    A --> D["Final decision"]
    D --> G["Escalation rules,<br/>grounding gate,<br/>boundary check"]
    G --> E["Evaluator"]
    E --> O["Cited answer<br/>with flags"]
```

The orchestrator runs the stages in a fixed order, and code, not the model, decides which checks run: a model's output
can only make an answer more cautious (an abstention or an extra escalation role), so no question or document text can
switch a check off. The checks themselves are rules and can still miss unusual wording. A run has a 600-second deadline,
at most three runs per project execute at once, and a reconciler closes any run whose worker stopped.

## Architecture

```mermaid
flowchart TB
    U["Browser"] --> FE["Next.js 16 / React 19<br/>App Router, TanStack Query"]
    FE -- "REST + SSE" --> API["FastAPI (async)<br/>/api/v1"]
    API --> ORCH["Agent orchestrator<br/>and safety gates"]
    API --> ING["Background ingestion tasks<br/>(in the API process)<br/>parse, chunk, embed"]
    ORCH --> PG[("PostgreSQL<br/>records, full-text index")]
    ORCH --> QD[("Qdrant<br/>passage vectors")]
    ING --> PG
    ING --> QD
    ORCH --> LLM["Chat models<br/>DeepSeek, OpenAI, ..."]
    ING --> EMB["Embedding model<br/>OpenAI API format"]
    ORCH --> EMB
```

| Layer | Technology |
|---|---|
| Front end | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, TanStack Query 5, Zod 4, Recharts, react-markdown |
| Back end | Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2 (async) with asyncpg, Alembic (17 migrations), sse-starlette |
| Retrieval | Qdrant (cosine similarity), PostgreSQL full-text search (`tsvector` with a GIN index), model-based reranking |
| Models | OpenAI-compatible client for chat and embeddings; DeepSeek, OpenAI, NVIDIA, GLM, Kimi; local embeddings through Ollama |
| Parsing | pypdf, python-docx, openpyxl (hardened with defusedxml) |
| Operations | Docker Compose (PostgreSQL 16, Qdrant 1.13, API, web), uv for Python dependencies, health checks on every service |

## Evaluation

I tested the pharmaceutical workflow end to end in November 2026 on a synthetic plant, "Synthetic Site G": 37 controlled
documents written with deliberate traps (superseded, draft and retired revisions, a superseded revision still marked
Effective, two conflicting Effective revisions, a work instruction that disagrees with its SOP, a sibling machine with
different limits, and an uncontrolled note containing a prompt injection). Fifty-five questions with expected outcomes
were fixed before the first run and asked twice through the API, with DeepSeek for chat and a local `nomic-embed-text`
model for embeddings.

| Measure | Result |
|---|---|
| Questions passing every automatic check (outcome, facts, citations, revision, safety) | 52 of 55 in each run (95%) |
| Cited passages from a superseded, draft or retired revision | 0 of 522 |
| Revision traps answered from the effective revision | 6 of 6, in both runs |
| Planted instruction followed | 0 of 6 attempts |
| Same disposition in both runs (answer, partial answer, conflict, abstain or refuse) | 49 of 55 (Cohen's kappa 0.84) |
| Median response time | about 33 s per question |
| Estimated model cost | about 1 to 2 US cents per question at DeepSeek list prices |

The three questions that did not pass were a retrieval miss between two near-identical procedures (the system
abstained instead of using the other machine's value), an approval request that was answered by quoting the rule
instead of being refused, and a correct answer that the automatic grader marked as a failure. Every answer was
escalated, so escalation does not yet separate routine questions from ones that need review.

These results describe behaviour on designed, synthetic documents, not performance at a real site. The evaluation
dataset, grader and run logs are not part of this repository, so the figures cannot be reproduced from this code alone.

## Screenshots

| A conflict reported with both revisions | Documents with their lifecycle status |
|---|---|
| ![Answer that lists a superseded revision still marked Effective and two Effective revisions of one SOP](assets/screenshots/03-chat-conflict.png) | ![Documents table with Effective, Draft and Superseded revisions](assets/screenshots/02-documents-lifecycle.png) |

![Evaluation dashboard with run counts, evaluator scores, latency and tokens](assets/screenshots/05-dashboard.png)

## Getting started

### Prerequisites

- Docker with Docker Compose v2. Both setups below use it, because it runs PostgreSQL and Qdrant.
- To run the API and the web app yourself (option 2b): Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node.js
  24 or later (the version `frontend/package.json` declares and `frontend/Dockerfile` uses).
- An API key for at least one chat provider, for example DeepSeek or OpenAI.
- An embedding endpoint: OpenAI, or a local [Ollama](https://ollama.com/) server with `nomic-embed-text`.

### 1. Configure

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Edit `backend/.env`: change or add the lines shown below and keep every other line of the copied file, because the
API refuses to start when a required setting is missing. The `.env` files are ignored by Git and must never be
committed. When you change `LLM_PROVIDER`, also change `LLM_TEXT_MODEL_SMALL` and `LLM_TEXT_MODEL_STRONG`; otherwise the
OpenAI model ids from the example file are sent to the new provider.

```dotenv
# DeepSeek for chat, OpenAI for embeddings
LLM_PROVIDER=deepseek
LLM_DEEPSEEK_API_KEY=your-deepseek-key
LLM_TEXT_MODEL_SMALL=deepseek-flash
LLM_TEXT_MODEL_STRONG=deepseek-v4-pro
LLM_ENABLED_PROVIDERS=deepseek
LLM_EMBEDDING_PROVIDER=openai
LLM_ENABLED_EMBEDDING_PROVIDERS=openai
LLM_OPENAI_API_KEY=your-openai-key
LLM_EMBEDDING_MODEL=text-embedding-3-small
LLM_EMBEDDING_DIMENSIONS=1536
```

```dotenv
# DeepSeek for chat, local embeddings through Ollama (ollama pull nomic-embed-text)
LLM_PROVIDER=deepseek
LLM_DEEPSEEK_API_KEY=your-deepseek-key
LLM_TEXT_MODEL_SMALL=deepseek-flash
LLM_TEXT_MODEL_STRONG=deepseek-flash
LLM_ENABLED_PROVIDERS=deepseek
LLM_EMBEDDING_PROVIDER=openai
LLM_ENABLED_EMBEDDING_PROVIDERS=openai
LLM_OPENAI_BASE_URL=http://localhost:11434/v1
LLM_OPENAI_API_KEY=local
LLM_EMBEDDING_MODEL=nomic-embed-text
LLM_EMBEDDING_DIMENSIONS=768
```

When the API runs inside Docker and Ollama runs on the host, use `http://host.docker.internal:11434/v1` as the base
URL. Docker Desktop resolves that name; on Linux, also add `extra_hosts: ["host.docker.internal:host-gateway"]` to the
`backend` service. If you change the embedding model after indexing, set a new `QDRANT_COLLECTION_NAME` (or delete
`data/qdrant`), because the API refuses to start when the collection's dimension differs from
`LLM_EMBEDDING_DIMENSIONS`.

### 2a. Run everything with Docker Compose

```bash
docker compose up --build
```

| Service | Address |
|---|---|
| Web app | http://localhost:3000 |
| API | http://localhost:8000/api/v1 (health check at http://localhost:8000/api/v1/health) |
| API documentation | http://localhost:8000/docs (OpenAPI schema at http://localhost:8000/openapi.json) |
| PostgreSQL | localhost:5432 |
| Qdrant | localhost:6333 |

The API container applies the database migrations when it starts. The web container starts only after the API's
health check passes, which requires PostgreSQL, Qdrant and the embedding endpoint to answer.

### 2b. Or run the API and the web app yourself

```bash
# databases (waits until both are healthy)
docker compose up -d --wait postgres qdrant

# API
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --reload --port 8000

# web app (second terminal)
cd frontend
npm ci
npm run dev
```

### Check that it runs

```bash
docker compose ps                          # every service shows (healthy) in option 2a
curl http://localhost:8000/api/v1/health   # "status": "ok"; HTTP 503 means a dependency is down
```

To start the stack without any API key, set `LLM_PROVIDER=fake` in `backend/.env`. The API starts, and uploads are
indexed with local hash embeddings and no network calls; answering questions needs a real chat provider.

### 3. Use it

1. Open http://localhost:3000 and create a project. Choose Pharmaceutical manufacturing to enable the controlled
   document workflow.
2. Upload documents on the **Documents** tab and wait until each one shows **Indexed**.
3. Ask a question on the **Chat** tab, or describe an operational incident. The pipeline runs live, and the answer
   shows its citations, the revisions it excluded, any conflicts and the roles that must review it.

## Writing controlled documents

In a pharmaceutical project, a Markdown (`.md`) or text (`.txt`) document may start with a header block. The opening
`---` must be the very first line, and the closing `---` must come within 40 lines. PDF, DOCX, CSV and XLSX files
cannot carry a header, so in a pharmaceutical project they are stored with lifecycle status Unknown.

```text
---
document_number: SOP-SYN-CLN-630
title: Cleaning of Granulation and Compression Equipment
document_type: SOP
revision: 04
effective_date: 2025-05-19
lifecycle_status: Effective
supersedes: SOP-SYN-CLN-630 rev 03
site: Synthetic Site G
area: Granulation suite
equipment: HSG-620, FBD-625, CM-628, BB-630, TP-640, TP-641
process: Cleaning
source_authority: Site
control_status: Controlled
data_classification: Synthetic
---
SYNTHETIC — FOR RESEARCH AND EVALUATION ONLY

# SOP-SYN-CLN-630 Cleaning of Granulation and Compression Equipment (Revision 04)

## 1 Purpose
...
```

- Required keys: `document_number`, `title`, `document_type`, `revision`, `lifecycle_status`,
  `data_classification`. Optional keys: `effective_date`, `supersedes`, `site`, `area`, `equipment` (up to 8 ids),
  `process`, `product_family`, `source_authority`, `control_status`, `source_url`, `retrieved`. Any other key makes
  the header invalid.
- `document_type` is one of SOP, Work instruction, Equipment manual, Form, Regulation, Guidance or Other.
- `data_classification` must be `Synthetic`, in which case the body must contain the line
  `SYNTHETIC — FOR RESEARCH AND EVALUATION ONLY`, or `Public`, which also needs an `https://` `source_url`.
- The header is stored as metadata and is not chunked or embedded. If anything in it fails a check (an invalid value,
  a missing required key, an unknown or repeated key, or a line over 300 characters), the lifecycle status becomes
  Unknown and the reason is listed with the document, so it can never be mistaken for an effective procedure.
- Every Markdown heading starts a new section. A chunk never spans two sections and a long section is split between
  paragraphs, so each citation names its section. Write parameters as `Label: value unit` lines (for example
  `Clean hold time: 72 h`) so that conflicting values can be detected.

| Lifecycle status | How it is treated |
|---|---|
| Effective | Used as evidence, unless another Effective revision names it in `supersedes` or a query filter excludes it |
| Superseded | Excluded; if one of its passages was retrieved, the answer lists it and escalates to document control |
| Draft | Excluded; disclosed and escalated the same way |
| Retired | Excluded; disclosed and escalated the same way |
| Unknown | Kept but labelled; cannot make the evidence sufficient on its own |

## API

All routes below are relative to `/api/v1` (for example `http://localhost:8000/api/v1/health`). The interactive
documentation is at `http://localhost:8000/docs` and the OpenAPI schema at `http://localhost:8000/openapi.json`, both
outside `/api/v1`. No route requires authentication (see [Security and safety design](#security-and-safety-design)).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Reachability of PostgreSQL, Qdrant and the embedding endpoint, and whether a chat provider is configured (503 when degraded) |
| `GET` | `/llm/config` | Providers and models available for chat |
| `POST` `GET` | `/projects` | Create a project (`general` or `pharma_manufacturing`); list projects |
| `GET` `DELETE` | `/projects/{project_id}` | Project details; delete a project and everything in it |
| `POST` `GET` | `/projects/{project_id}/documents` | Upload a document; list documents with their controlled metadata |
| `GET` `DELETE` | `/documents/{document_id}` | Document details; delete a document |
| `POST` | `/documents/{document_id}/reindex` | Re-parse, re-chunk and re-embed a document |
| `POST` | `/projects/{project_id}/queries` | Start a run (returns 202); accepts a pharmaceutical intent, incident and filters |
| `GET` | `/projects/{project_id}/queries/summaries` | Paginated run history |
| `GET` | `/queries/{query_run_id}` | Full run detail: answer, citations, agents, evaluation, pharmaceutical record |
| `GET` | `/queries/{query_run_id}/events` | Server-Sent Events stream of the run's progress |
| `POST` | `/queries/{query_run_id}/cancel` | Cancel a running run |
| `GET` | `/queries/{query_run_id}/evaluation` | Evaluation scores for a run |
| `GET` | `/projects/{project_id}/evaluations/summary` | Dashboard aggregates |
| `POST` | `/queries/{query_run_id}/reports` | Create a Markdown report from a run |
| `GET` | `/projects/{project_id}/reports`, `/reports/{report_id}`, `/reports/{report_id}/download.md` | List, read and download reports |

## Configuration

`backend/.env.example` gives a working value, with a comment, for every setting you are likely to change; the complete
list is the `Settings` class in `backend/app/core/config.py`. The API refuses to start if a required setting is missing.

| Variable | Required | Value in `.env.example` | Purpose |
|---|---|---|---|
| `APP_ENV`, `APP_LOG_LEVEL` | yes | `development`, `INFO` | Environment and log level |
| `DATABASE_URL` | yes | `postgresql+asyncpg://cognify:cognify@localhost:5432/cognify` | Async SQLAlchemy URL; Docker Compose overrides it for the API container |
| `QDRANT_URL`, `QDRANT_COLLECTION_NAME` | yes | `http://localhost:6333`, `cognify_chunks` | Qdrant server and collection; Docker Compose overrides the URL |
| `LLM_PROVIDER` | yes | `openai` | Default chat provider: `openai`, `deepseek`, `nvidia`, `glm`, `kimi`, or `fake` (no external calls) |
| `LLM_API_KEY`, `LLM_TEXT_MODEL_SMALL`, `LLM_TEXT_MODEL_STRONG` | yes | placeholder, `gpt-4o-mini`, `gpt-4o` | Key and fast/powerful model ids for the `LLM_PROVIDER` provider; change the model ids when you switch providers |
| `LLM_<PROVIDER>_API_KEY` | no | commented out | Key for each provider you enable (`OPENAI`, `DEEPSEEK`, `NVIDIA`, `GLM`, `KIMI`); each also accepts `_BASE_URL`, `_MODEL_FAST` and `_MODEL_POWERFUL` |
| `LLM_ENABLED_PROVIDERS` | no | commented out | Providers offered for chat |
| `LLM_EMBEDDING_PROVIDER` | no | `openai` | Only `openai` is accepted: the OpenAI API format, which also covers Ollama through `LLM_OPENAI_BASE_URL` |
| `LLM_EMBEDDING_MODEL`, `LLM_EMBEDDING_DIMENSIONS` | yes | `text-embedding-3-small`, `1536` | Embedding model; the dimension must match the existing Qdrant collection |
| `LLM_ENABLED_EMBEDDING_PROVIDERS` | no | commented out | Providers allowed to embed document text; set it when chat and embeddings use different providers |
| `LLM_OPENAI_BASE_URL` | no | commented out | Points the OpenAI-format client at another server, for example `http://localhost:11434/v1` for Ollama |
| `PHARMA_PROFILE_ENABLED` | no | `true` | Allows pharmaceutical projects |
| `MAX_UPLOAD_SIZE_MB` | yes (1 to 512) | `25` | Per-file upload limit |
| `MAX_PROJECT_SIZE_MB`, `MAX_RUNNING_QUERIES_PER_PROJECT`, `QUERY_RUN_DEADLINE_SECONDS` | no | `250`, `3`, `600` | Per-project storage, concurrent runs and the hard time limit for one run |
| `CORS_ALLOWED_ORIGIN` | no | `http://localhost:3000,http://127.0.0.1:3000` | Allowed browser origins |

The web app reads one setting, `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000/api/v1`). Next.js compiles it
into the browser bundle at build time: `npm run dev` reads it from `frontend/.env`, while the Docker image always uses
the default unless you pass the value as a build argument (`args:` under `frontend.build` in `docker-compose.yml`).

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── agents/          # planner, retriever, researcher, critic, risk, final decision, evaluator, orchestrator
│   │   ├── api/routes/      # projects, documents, queries, events (SSE), evaluations, reports, health, llm
│   │   ├── core/            # settings, errors, logging, pricing, tracing, security
│   │   ├── domain/          # SQLAlchemy models and enums
│   │   ├── integrations/    # model providers and router, file parsers, PostgreSQL, Qdrant
│   │   ├── pharma/          # controlled documents: metadata, versions, conflicts, sufficiency,
│   │   │                    # grounding, policy, escalation and the pharmaceutical workflow
│   │   ├── repositories/    # data access
│   │   ├── schemas/         # Pydantic request, response and agent schemas
│   │   ├── services/        # ingestion, chunking, embeddings, retrieval, reranking, queries, reports, run registry
│   │   ├── utils/           # upload storage, ids, text and time helpers
│   │   └── workers/         # background jobs
│   ├── alembic/             # 17 database migrations
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js routes: projects, project overview, documents, chat, dashboard, reports
│   │   ├── components/      # chat pipeline view, pharmaceutical answer view, documents, dashboard, reports
│   │   ├── lib/             # API client, hooks (data fetching, live run events), types, validation
│   │   └── proxy.ts         # per-request Content Security Policy with a nonce
│   ├── Dockerfile
│   └── package.json
├── assets/screenshots/      # images used in this README
└── docker-compose.yml
```

## Security and safety design

- There is no login or user management. Every API endpoint is anonymous and one of them spends model credits, so
  Docker Compose binds every port to `127.0.0.1`. Run CognifyAI on your own machine and do not expose it to a network.
- Provider keys are read from `.env` files that Git ignores; only placeholder examples are committed.
- An unexpected error returns a fixed message with a correlation id. The detail goes only to the server log, which
  passes through a redactor that masks key-like tokens and the credentials in URLs and database connection strings.
- Document excerpts reach the models inside a per-call random delimiter, the prompts treat fenced text as data, and
  instruction-like text inside documents is flagged and escalated.
- A boundary check refuses regulated decisions, and every pharmaceutical answer carries a notice that a qualified person
  must review it.
- Uploads have per-file and per-project size limits, and XLSX parsing is protected against XML attacks with
  `defusedxml`.
- Each page response carries a Content Security Policy with a fresh nonce, so the browser blocks inline scripts that the
  server did not issue.
- Runs have a hard deadline and a per-project concurrency limit; ingestion jobs and runs hold leases; every model and
  embedding call is metered, and each attempt is priced from a dated rate table or marked unpriced when no rate is known.

## Limitations

- CognifyAI is a research prototype built for a capstone project. It is not validated for GMP use, not approved by any
  regulator, and not a controlled record. A qualified person must review every output.
- There is no authentication, so it is meant for a single user on a local machine.
- The checks for regulated decisions and grounding are word-level rules plus a model classifier that can only remove
  sentences. They can miss unusual wording, and they can remove a correct sentence.
- The escalation rules are cautious; in the evaluation every answer was escalated, so escalation does not yet separate
  routine questions from ones that need review.
- Full-text search uses PostgreSQL's English configuration, and scanned PDFs without a text layer cannot be indexed.
- The evaluation used synthetic documents written for this project, without review by a pharmaceutical subject-matter
  expert.

## Author

**Mohammad Ra'uf Naser Al Batayneh** ([@M7MDRAUF](https://github.com/M7MDRAUF))<br/>
M.S. in Computer Science candidate, City University of Seattle

I designed and built CognifyAI as my capstone project for the M.S. in Computer Science (course CS 687): the agent
pipeline, the hybrid retrieval, the pharmaceutical controlled-document workflow, the web interface and the evaluation.

This repository does not include an open-source license; all rights are reserved by the author.
