import { z } from "@/lib/zod";

/**
 * The pharmaceutical manufacturing profile's contracts, mirroring `backend/app/schemas/pharma.py`
 * and `backend/app/pharma/enums.py` (PDP3-T05). Every field these add to the General types is
 * optional, so a General payload, or one from a backend that predates the profile, parses to the
 * same object as before.
 */

export const domainProfileSchema = z.enum(["general", "pharma_manufacturing"]);
export type DomainProfile = z.infer<typeof domainProfileSchema>;

/** A project's profile; a project read from a backend that sends none is General. */
export function profileOf(project: { domain_profile?: DomainProfile | undefined }): DomainProfile {
  return project.domain_profile ?? "general";
}

export const lifecycleStatusSchema = z.enum([
  "draft",
  "effective",
  "superseded",
  "retired",
  "unknown",
]);
export type LifecycleStatus = z.infer<typeof lifecycleStatusSchema>;

export const controlledDocumentTypeSchema = z.enum([
  "sop",
  "work_instruction",
  "equipment_manual",
  "form",
  "regulation",
  "guidance",
  "other",
]);
export type ControlledDocumentType = z.infer<typeof controlledDocumentTypeSchema>;

export const sourceAuthoritySchema = z.enum(["site", "vendor", "regulator", "other"]);
export const controlStatusSchema = z.enum(["controlled", "uncontrolled", "reference"]);
export const dataClassificationSchema = z.enum(["synthetic", "public"]);
export const metadataStatusSchema = z.enum(["valid", "missing", "invalid"]);

export const queryIntentSchema = z.enum([
  "procedure_question",
  "operational_incident",
  "readiness_review",
]);
export type PharmaQueryIntent = z.infer<typeof queryIntentSchema>;

export const sufficiencySchema = z.enum(["sufficient", "partial", "insufficient"]);
export type EvidenceSufficiency = z.infer<typeof sufficiencySchema>;

export const pharmaRiskCategorySchema = z.enum([
  "operational",
  "product_quality",
  "equipment",
  "documentation",
  "data_integrity",
  "escalation",
]);

export const escalationRoleSchema = z.enum([
  "quality_assurance",
  "process_owner",
  "subject_matter_expert",
  "document_control",
]);
export type EscalationRole = z.infer<typeof escalationRoleSchema>;

export const conflictKindSchema = z.enum([
  "superseded_match",
  "retired_match",
  "draft_match",
  "unknown_status",
  "multiple_effective_revisions",
  "superseded_marked_effective",
  "overlapping_effective_scope",
  "parameter_disagreement",
]);

export const exclusionReasonSchema = z.enum([
  "superseded",
  "retired",
  "draft",
  "lower_revision",
  "superseded_by_effective",
  "filter_mismatch",
]);

export const policyCodeSchema = z.enum([
  "PH-POL-01",
  "PH-POL-02",
  "PH-POL-03",
  "PH-POL-04",
  "PH-POL-05",
  "PH-POL-06",
  "PH-POL-07",
  "PH-POL-08",
  "PH-POL-09",
  "PH-POL-10",
]);

/** The incident form's bounds; the server's `IncidentInput` holds the same ones. */
export const INCIDENT_LIMITS = { summary: 500, label: 200, text: 2000 } as const;
const EQUIPMENT_ID = /^[A-Z]{1,5}-?[0-9]{1,5}[A-Z]?$/;

// The server's text rules (`backend/app/schemas/_text.py`; PH-DEF-040): a C0 or C1 control
// character other than tab, newline and carriage return is refused, and a value made only of
// white space, format (Cf) and control (Cc) characters is blank, U+200B alone included.
const CONTROL_CHARACTER = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/;
const VISUALLY_BLANK = /^[\s\p{Cf}\p{Cc}]*$/u;

/** Whether `value` holds a control character the server refuses. */
export function hasControlCharacter(value: string): boolean {
  return CONTROL_CHARACTER.test(value);
}

/** Whether `value` renders as nothing, as the server judges it (`is_visually_blank`). */
export function isVisuallyBlank(value: string): boolean {
  return VISUALLY_BLANK.test(value);
}

export const controlledMetadataSchema = z.object({
  document_number: z.string().nullable(),
  title: z.string().nullable(),
  document_type: controlledDocumentTypeSchema.nullable(),
  revision: z.string().nullable(),
  effective_date: z.string().nullable(),
  lifecycle_status: lifecycleStatusSchema,
  supersedes_number: z.string().nullable(),
  supersedes_revision: z.string().nullable(),
  site: z.string().nullable(),
  area: z.string().nullable(),
  equipment: z.array(z.string()),
  process: z.string().nullable(),
  product_family: z.string().nullable(),
  source_authority: sourceAuthoritySchema.nullable(),
  control_status: controlStatusSchema.nullable(),
  data_classification: dataClassificationSchema.nullable(),
  metadata_status: metadataStatusSchema,
  metadata_issues: z.array(z.string()),
});
export type ControlledMetadata = z.infer<typeof controlledMetadataSchema>;

export const incidentInputSchema = z.object({
  summary: z
    .string()
    .trim()
    .min(1)
    .max(INCIDENT_LIMITS.summary)
    .refine((value) => !isVisuallyBlank(value), "Describe the incident in a sentence or two"),
  area: z.string().max(INCIDENT_LIMITS.label).optional(),
  equipment_id: z
    .string()
    .max(INCIDENT_LIMITS.label)
    .refine((value) => value === "" || EQUIPMENT_ID.test(value), "Use an identifier such as TP-201")
    .optional(),
  process_step: z.string().max(INCIDENT_LIMITS.label).optional(),
  product_family: z.string().max(INCIDENT_LIMITS.label).optional(),
  observations: z.string().max(INCIDENT_LIMITS.text).optional(),
  immediate_actions: z.string().max(INCIDENT_LIMITS.text).optional(),
});
export type IncidentInput = z.infer<typeof incidentInputSchema>;

export const pharmaFiltersSchema = z.object({
  equipment_ids: z.array(z.string().regex(EQUIPMENT_ID)).max(8).default([]),
  areas: z.array(z.string().min(1).max(100)).max(8).default([]),
  processes: z.array(z.string().min(1).max(100)).max(8).default([]),
  document_types: z.array(controlledDocumentTypeSchema).max(7).default([]),
});
export type PharmaFilters = z.infer<typeof pharmaFiltersSchema>;

export const pharmaQueryInputSchema = z
  .object({
    intent: queryIntentSchema,
    incident: incidentInputSchema.nullish(),
    filters: pharmaFiltersSchema.optional(),
  })
  .refine(
    (input) => (input.intent === "operational_incident") === (input.incident != null),
    "An incident description goes with the operational-incident intent, and only with it",
  );
export type PharmaQueryInput = z.infer<typeof pharmaQueryInputSchema>;

export const evidenceAnchorSchema = z.object({
  chunk_id: z.uuid(),
  document_id: z.uuid(),
  chunk_index: z.number(),
  document_number: z.string(),
  revision: z.string(),
  title: z.string(),
  section: z.string(),
  page: z.number().nullable(),
  lifecycle_status: lifecycleStatusSchema,
});
export type EvidenceAnchor = z.infer<typeof evidenceAnchorSchema>;

export const evidenceItemSchema = z.object({
  citation_key: z.string(),
  anchor: evidenceAnchorSchema,
  excerpt: z.string(),
});

export const excludedEvidenceSchema = z.object({
  anchor: evidenceAnchorSchema,
  reason: exclusionReasonSchema,
});

export const conflictSchema = z.object({
  kind: conflictKindSchema,
  description: z.string(),
  document_numbers: z.array(z.string()),
  citation_keys: z.array(z.string()),
});

export const riskFlagSchema = z.object({
  category: pharmaRiskCategorySchema,
  severity: z.enum(["low", "medium", "high"]),
  description: z.string(),
  citation_keys: z.array(z.string()),
});

export const humanEscalationSchema = z.object({
  required: z.boolean(),
  roles: z.array(escalationRoleSchema),
  reasons: z.array(
    z.object({ code: z.string(), text: z.string(), source: z.enum(["rule", "model"]) }),
  ),
});
export type HumanEscalation = z.infer<typeof humanEscalationSchema>;

export const policyNoticeSchema = z.object({
  code: policyCodeSchema,
  field: z.string(),
  action: z.enum([
    "sentence_removed",
    "escalation_forced",
    "abstention_enforced",
    "disclosure_added",
  ]),
  notice: z.string(),
});

export const relevantProcedureSchema = z.object({
  document_number: z.string(),
  revision: z.string(),
  title: z.string(),
  lifecycle_status: lifecycleStatusSchema,
  sections: z.array(z.string()),
  summary: z.string(),
  citation_keys: z.array(z.string()),
});

/** The nine sections of the intended use, in order (REQ-RSP-01). */
export const RESPONSE_SECTION_KEYS = [
  "incident_or_question_summary",
  "relevant_procedures",
  "evidence_and_citations",
  "quality_and_operational_risks",
  "conflicts_or_ambiguities",
  "missing_information",
  "required_human_escalation",
  "suggested_investigation_questions_or_next_steps",
  "limitations_and_uncertainty",
] as const;

export const pharmaResponseSchema = z.object({
  incident_or_question_summary: z.string(),
  relevant_procedures: z.array(relevantProcedureSchema),
  evidence_and_citations: z.array(evidenceItemSchema),
  quality_and_operational_risks: z.array(riskFlagSchema),
  conflicts_or_ambiguities: z.object({
    deterministic: z.array(conflictSchema),
    model_reported: z.array(z.string()),
  }),
  missing_information: z.array(z.string()),
  required_human_escalation: humanEscalationSchema,
  suggested_investigation_questions_or_next_steps: z.array(z.string()),
  limitations_and_uncertainty: z.array(z.string()),
});
export type PharmaResponse = z.infer<typeof pharmaResponseSchema>;

/** What a result is, and the gate that decided it (`PharmaDisposition`, `DispositionGate`). */
export const dispositionSchema = z.enum([
  "answer",
  "partial_answer",
  "abstain_insufficient_evidence",
  "refuse_prohibited_decision",
  "out_of_scope",
  "conflict_requires_review",
  "system_failure",
]);
export type PharmaDisposition = z.infer<typeof dispositionSchema>;
export const dispositionGateSchema = z.enum(["sufficiency", "decision", "grounding", "policy"]);
export type DispositionGate = z.infer<typeof dispositionGateSchema>;

export const pharmaRunSchema = z.object({
  intent: queryIntentSchema,
  incident: incidentInputSchema.nullish(),
  filters: pharmaFiltersSchema,
  sufficiency: sufficiencySchema,
  abstained: z.boolean(),
  // Absent from a run recorded before the server stored them.
  disposition: dispositionSchema.nullish(),
  disposition_gate: dispositionGateSchema.nullish(),
  rule_version: z.string(),
  banner: z.string(),
  response: pharmaResponseSchema,
  excluded_evidence: z.array(excludedEvidenceSchema),
  policy_notices: z.array(policyNoticeSchema),
});
export type PharmaRun = z.infer<typeof pharmaRunSchema>;
