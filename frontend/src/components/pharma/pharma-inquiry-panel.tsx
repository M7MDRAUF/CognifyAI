"use client";

import { useEffect, useId, useRef } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  DOCUMENT_TYPE_LABELS,
  type DraftErrors,
  FILTER_FIELDS,
  type FilterField,
  INCIDENT_FIELDS,
  INTENT_LABELS,
  type IncidentField,
  MAX_FILTER_VALUES,
  type PharmaDraft,
} from "@/lib/pharma/incident-form";
import { type ControlledDocumentType, type PharmaQueryIntent } from "@/lib/types/pharma";

/** Threat T-12: synthetic or public examples only; incident text goes to the configured model. */
export const REAL_DATA_WARNING =
  "Enter synthetic or public examples only. Do not enter real batch, product, patient or " +
  "personal data: with a model provider configured, this text is sent to it.";

const INTENTS = Object.keys(INTENT_LABELS) as PharmaQueryIntent[];
const DOCUMENT_TYPES = Object.keys(DOCUMENT_TYPE_LABELS) as ControlledDocumentType[];
/** The order in which a refused submit looks for the field to focus. */
const FIELD_ORDER: readonly (IncidentField | FilterField)[] = [
  ...INCIDENT_FIELDS.map((field) => field.name),
  ...FILTER_FIELDS.map((field) => field.name),
];

interface PharmaInquiryPanelProps {
  draft: PharmaDraft;
  onChange: (draft: PharmaDraft) => void;
  errors: DraftErrors;
  /** Incremented by a refused submit: the first field in error takes focus (FE-035). */
  focusRequest: number;
  disabled: boolean;
}

/**
 * A pharmaceutical project's inquiry details (REQ-UX-04): the intent, the incident form for an
 * operational incident, and the optional retrieval filters. The question itself stays in the
 * question form; `ChatView` sends these beside it as the `pharma` object.
 */
export function PharmaInquiryPanel({
  draft,
  onChange,
  errors,
  focusRequest,
  disabled,
}: PharmaInquiryPanelProps) {
  const prefix = useId();
  const fieldId = (name: string) => `${prefix}-${name}`;
  const handled = useRef(0);

  useEffect(() => {
    if (focusRequest === handled.current) {
      return;
    }
    handled.current = focusRequest;
    const first = FIELD_ORDER.find((name) => errors[name] !== undefined);
    if (first !== undefined) {
      document.getElementById(`${prefix}-${first}`)?.focus();
    }
  }, [focusRequest, errors, prefix]);

  const describedBy = (name: IncidentField | FilterField, hint: boolean): string | undefined => {
    const ids = [
      ...(errors[name] !== undefined ? [fieldId(`${name}-error`)] : []),
      ...(hint ? [fieldId(`${name}-hint`)] : []),
    ];
    return ids.length > 0 ? ids.join(" ") : undefined;
  };

  const fieldError = (name: IncidentField | FilterField) =>
    errors[name] !== undefined && (
      <p id={fieldId(`${name}-error`)} className="text-xs text-destructive">
        {errors[name]}
      </p>
    );

  return (
    <section aria-labelledby={fieldId("heading")} className="space-y-4 rounded-lg border p-4">
      <h2 id={fieldId("heading")} className="text-base font-semibold">
        Inquiry details
      </h2>

      <fieldset className="space-y-2" disabled={disabled}>
        <legend className="text-sm font-medium leading-none">Inquiry type</legend>
        <div className="flex flex-wrap gap-x-6 gap-y-2 pt-1">
          {INTENTS.map((intent) => (
            <label key={intent} className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name={fieldId("intent")}
                value={intent}
                checked={draft.intent === intent}
                onChange={() => {
                  onChange({ ...draft, intent });
                }}
                className="size-4"
              />
              {INTENT_LABELS[intent]}
            </label>
          ))}
        </div>
      </fieldset>

      {draft.intent === "operational_incident" && (
        <fieldset
          className="grid gap-3 sm:grid-cols-2"
          disabled={disabled}
          aria-describedby={fieldId("real-data")}
        >
          <legend className="mb-2 text-sm font-medium leading-none">Incident</legend>
          {/* Threat T-12 (PH-DEF-037): the form itself says what not to type into it. */}
          <p id={fieldId("real-data")} className="text-xs text-muted-foreground sm:col-span-2">
            {REAL_DATA_WARNING}
          </p>
          {INCIDENT_FIELDS.map((field) => {
            const control = {
              id: fieldId(field.name),
              value: draft.incident[field.name],
              maxLength: field.max,
              required: field.name === "summary",
              "aria-invalid": errors[field.name] !== undefined ? true : undefined,
              "aria-describedby": describedBy(field.name, field.hint !== undefined),
            };
            const update = (value: string) => {
              onChange({ ...draft, incident: { ...draft.incident, [field.name]: value } });
            };
            return (
              <div
                key={field.name}
                // Top-aligned: a cell beside a taller one (a hint, an error) keeps its input level
                // with its neighbour's instead of stretching its rows (PH-DEF-015).
                className={
                  field.multiline
                    ? "grid content-start gap-1 sm:col-span-2"
                    : "grid content-start gap-1"
                }
              >
                <Label htmlFor={fieldId(field.name)}>
                  {field.label}
                  {/* Seen, not read: the control's `required` already says it (PH-DEF-040). */}
                  {field.name === "summary" && (
                    <span aria-hidden="true" className="font-normal text-muted-foreground">
                      {" "}
                      (required)
                    </span>
                  )}
                </Label>
                {field.multiline ? (
                  <Textarea
                    {...control}
                    rows={2}
                    onChange={(event) => {
                      update(event.target.value);
                    }}
                  />
                ) : (
                  <Input
                    {...control}
                    onChange={(event) => {
                      update(event.target.value);
                    }}
                  />
                )}
                {field.hint !== undefined && (
                  <p id={fieldId(`${field.name}-hint`)} className="text-xs text-muted-foreground">
                    {field.hint}
                  </p>
                )}
                {fieldError(field.name)}
              </div>
            );
          })}
        </fieldset>
      )}

      <fieldset className="grid gap-3 sm:grid-cols-3" disabled={disabled}>
        <legend className="mb-2 text-sm font-medium leading-none">Filters (optional)</legend>
        {FILTER_FIELDS.map((field) => (
          <div key={field.name} className="grid content-start gap-1">
            <Label htmlFor={fieldId(field.name)}>{field.label}</Label>
            <Input
              id={fieldId(field.name)}
              value={draft.filters[field.name]}
              aria-invalid={errors[field.name] !== undefined ? true : undefined}
              aria-describedby={describedBy(field.name, true)}
              onChange={(event) => {
                onChange({
                  ...draft,
                  filters: { ...draft.filters, [field.name]: event.target.value },
                });
              }}
            />
            <p id={fieldId(`${field.name}-hint`)} className="text-xs text-muted-foreground">
              Separate values with commas, up to {MAX_FILTER_VALUES}.
            </p>
            {fieldError(field.name)}
          </div>
        ))}
        <div className="grid gap-1 sm:col-span-3">
          <span id={fieldId("types")} className="text-sm font-medium leading-none">
            Document types
          </span>
          <div
            role="group"
            aria-labelledby={fieldId("types")}
            className="flex flex-wrap gap-x-4 gap-y-2 pt-1"
          >
            {DOCUMENT_TYPES.map((type) => (
              <label key={type} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={draft.documentTypes.includes(type)}
                  onChange={(event) => {
                    onChange({
                      ...draft,
                      documentTypes: event.target.checked
                        ? [...draft.documentTypes, type]
                        : draft.documentTypes.filter((chosen) => chosen !== type),
                    });
                  }}
                  className="size-4"
                />
                {DOCUMENT_TYPE_LABELS[type]}
              </label>
            ))}
          </div>
        </div>
      </fieldset>
    </section>
  );
}
