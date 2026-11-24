import {
  type ControlledDocumentType,
  INCIDENT_LIMITS,
  type IncidentInput,
  type PharmaFilters,
  type PharmaQueryInput,
  type PharmaQueryIntent,
  hasControlCharacter,
  incidentInputSchema,
  isVisuallyBlank,
} from "@/lib/types/pharma";

/**
 * The pharmaceutical inquiry's form state and the `pharma` object built from it (REQ-UX-04,
 * REQ-API-03). Every bound and rule here is the server's (`IncidentInput`, `PharmaFilters` and
 * `PharmaQueryInput` in `backend/app/schemas/pharma_api.py`), so a request this builds is never
 * refused for its shape; `pharma-inquiry-panel.test.tsx` reads the server's bounds to hold them.
 */

export type IncidentField = keyof Required<IncidentInput>;
export type FilterField = "equipment_ids" | "areas" | "processes";

export interface IncidentFieldSpec {
  name: IncidentField;
  label: string;
  max: number;
  multiline: boolean;
  hint?: string;
}

export const INCIDENT_FIELDS: readonly IncidentFieldSpec[] = [
  { name: "summary", label: "Incident summary", max: INCIDENT_LIMITS.summary, multiline: true },
  { name: "area", label: "Area", max: INCIDENT_LIMITS.label, multiline: false },
  {
    name: "equipment_id",
    label: "Equipment ID",
    max: INCIDENT_LIMITS.label,
    multiline: false,
    hint: "For example TP-201.",
  },
  { name: "process_step", label: "Process step", max: INCIDENT_LIMITS.label, multiline: false },
  { name: "product_family", label: "Product family", max: INCIDENT_LIMITS.label, multiline: false },
  { name: "observations", label: "Observations", max: INCIDENT_LIMITS.text, multiline: true },
  {
    name: "immediate_actions",
    label: "Immediate actions taken",
    max: INCIDENT_LIMITS.text,
    multiline: true,
  },
];

/** The server's `MAX_FILTER_VALUES` and `MAX_FILTER_VALUE_LENGTH`. */
export const MAX_FILTER_VALUES = 8;
export const MAX_FILTER_VALUE_LENGTH = 100;

export const FILTER_FIELDS: readonly { name: FilterField; label: string }[] = [
  { name: "equipment_ids", label: "Equipment IDs" },
  { name: "areas", label: "Areas" },
  { name: "processes", label: "Processes" },
];

export const INTENT_LABELS: Readonly<Record<PharmaQueryIntent, string>> = {
  procedure_question: "Procedure question",
  operational_incident: "Operational incident",
  readiness_review: "Readiness review",
};

export const DOCUMENT_TYPE_LABELS: Readonly<Record<ControlledDocumentType, string>> = {
  sop: "SOP",
  work_instruction: "Work instruction",
  equipment_manual: "Equipment manual",
  form: "Form",
  regulation: "Regulation",
  guidance: "Guidance",
  other: "Other",
};

export interface PharmaDraft {
  intent: PharmaQueryIntent;
  incident: Record<IncidentField, string>;
  /** Comma-separated values, as typed. */
  filters: Record<FilterField, string>;
  documentTypes: ControlledDocumentType[];
}

export const EMPTY_PHARMA_DRAFT: PharmaDraft = {
  intent: "procedure_question",
  incident: {
    summary: "",
    area: "",
    equipment_id: "",
    process_step: "",
    product_family: "",
    observations: "",
    immediate_actions: "",
  },
  filters: { equipment_ids: "", areas: "", processes: "" },
  documentTypes: [],
};

export type DraftErrors = Partial<Record<IncidentField | FilterField, string>>;

export type BuiltPharmaInput =
  { ok: true; input: PharmaQueryInput } | { ok: false; errors: DraftErrors };

const isEquipmentId = (value: string): boolean =>
  incidentInputSchema.shape.equipment_id.safeParse(value).success;

function splitValues(typed: string): string[] {
  return typed
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value !== "");
}

const CONTROL_CHARACTER_ERROR = "Remove the control character from this field.";

function incidentOf(draft: PharmaDraft, errors: DraftErrors): IncidentInput {
  const built: Partial<Record<IncidentField, string>> = {};
  for (const field of INCIDENT_FIELDS) {
    const value = draft.incident[field.name].trim();
    if (value.length > field.max) {
      errors[field.name] = `At most ${String(field.max)} characters.`;
    } else if (hasControlCharacter(value)) {
      errors[field.name] = CONTROL_CHARACTER_ERROR;
    } else if (value !== "" && !(field.name === "summary" && isVisuallyBlank(value))) {
      // A summary that renders as nothing is no summary (the server's `is_visually_blank`).
      built[field.name] = value;
    }
  }
  if (built.summary === undefined && errors.summary === undefined) {
    errors.summary = "Describe the incident in a sentence or two.";
  }
  if (built.equipment_id !== undefined && !isEquipmentId(built.equipment_id)) {
    errors.equipment_id = "Use an identifier such as TP-201.";
  }
  return { ...built, summary: built.summary ?? "" };
}

function filtersOf(draft: PharmaDraft, errors: DraftErrors): PharmaFilters | null {
  const lists = {} as Record<FilterField, string[]>;
  for (const { name } of FILTER_FIELDS) {
    const values = splitValues(draft.filters[name]);
    if (values.length > MAX_FILTER_VALUES) {
      errors[name] = `At most ${String(MAX_FILTER_VALUES)} values.`;
    } else if (name === "equipment_ids" && !values.every(isEquipmentId)) {
      errors[name] = "Each identifier must look like TP-201.";
    } else if (values.some(hasControlCharacter)) {
      errors[name] = CONTROL_CHARACTER_ERROR;
    } else if (values.some((value) => value.length > MAX_FILTER_VALUE_LENGTH)) {
      errors[name] = `Each value must be at most ${String(MAX_FILTER_VALUE_LENGTH)} characters.`;
    }
    lists[name] = values;
  }
  const empty =
    lists.equipment_ids.length === 0 &&
    lists.areas.length === 0 &&
    lists.processes.length === 0 &&
    draft.documentTypes.length === 0;
  return empty ? null : { ...lists, document_types: [...draft.documentTypes] };
}

/**
 * The `pharma` object for `draft`, or the field errors that stop it. Only the operational-incident
 * intent carries an incident (the server refuses one on any other intent), filters are sent only
 * when one is given, and an empty optional field is left out rather than sent empty.
 */
export function buildPharmaInput(draft: PharmaDraft): BuiltPharmaInput {
  const errors: DraftErrors = {};
  const input: PharmaQueryInput = { intent: draft.intent };
  if (draft.intent === "operational_incident") {
    input.incident = incidentOf(draft, errors);
  }
  const filters = filtersOf(draft, errors);
  if (filters !== null) {
    input.filters = filters;
  }
  return Object.keys(errors).length > 0 ? { ok: false, errors } : { ok: true, input };
}
