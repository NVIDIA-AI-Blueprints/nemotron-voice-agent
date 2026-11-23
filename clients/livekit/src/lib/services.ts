// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ClientConfig, CustomFieldSpec, CustomService, ServiceEntry, Slot } from "./types.ts";

export const SLOT_TITLES: Record<Slot, string> = { llm: "LLM Services", asr: "ASR Services", tts: "TTS Services" };

/** The custom endpoint field that the form shows as the display name. */
export const NAME_FIELD = "name";

const FIELD_LABELS: Record<string, string> = {
  base_url: "Server URL",
  model_id: "Model ID",
  extra_params: "Extra params JSON",
  server: "Server (gRPC endpoint)",
  model: "Model",
  voice_id: "Voice ID",
  function_id: "Function ID",
};

/** Fields kept in memory only, never in this browser's storage: they can hold API keys. */
const MEMORY_ONLY_FIELDS = new Set(["extra_params"]);

const GRPC_ADDRESS = /^(\[[\da-fA-F.]*:[\da-fA-F:.]*\]|[\w.-]+):(\d{1,5})$/;
const MAX_PORT = 65535;
const MAX_FIELD_CHARS = 512;

export const CUSTOM_ID_PREFIX = "custom-";

/** Return the custom endpoint fields of a slot, or undefined when the server does not accept custom endpoints. */
export function customFieldSpec(config: Partial<ClientConfig> | undefined, slot: Slot): CustomFieldSpec | undefined {
  return config?.custom_endpoints === true ? config.custom_fields?.[slot] : undefined;
}

/** Return the fields that the form asks for besides the display name. */
export function endpointFields(spec: CustomFieldSpec): string[] {
  return spec.fields.filter((field) => field !== NAME_FIELD);
}

export function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? field;
}

function isJsonObject(raw: string): boolean {
  try {
    const value: unknown = JSON.parse(raw);
    return typeof value === "object" && value !== null && !Array.isArray(value);
  } catch {
    return false;
  }
}

function parseHttpUrl(raw: string): URL | undefined {
  try {
    const url = new URL(raw);
    return (url.protocol === "http:" || url.protocol === "https:") && url.hostname ? url : undefined;
  } catch {
    return undefined;
  }
}

function isGrpcAddress(raw: string): boolean {
  const match = GRPC_ADDRESS.exec(raw);
  const port = Number(match?.[2]);
  return port >= 1 && port <= MAX_PORT;
}

/**
 * Return why a custom endpoint cannot be used, or "" when it can.
 *
 * The UI server applies the same rules when it mints the token. Checking here first lets the form say what is wrong.
 */
export function customServiceError(slot: Slot, spec: CustomFieldSpec, fields: Readonly<Record<string, string>>): string {
  const value = (key: string) => (fields[key] ?? "").trim();
  const missing = spec.required.filter((field) => !value(field));
  if (missing.length > 0) return `${missing.map(fieldLabel).join(" and ")} required`;
  const tooLong = endpointFields(spec).find((field) => value(field).length > MAX_FIELD_CHARS);
  if (tooLong) return `${fieldLabel(tooLong)} must be at most ${MAX_FIELD_CHARS} characters`;
  const address = slot === "llm" ? value("base_url") : value("server");
  if (/\s/.test(address)) return "The address cannot contain spaces";
  if (slot === "llm") {
    const url = parseHttpUrl(address);
    if (!url) return "Server URL must start with http:// or https://";
    if (url.username || url.password) return "Server URL cannot contain a user name or password";
    if (value("extra_params") && !isJsonObject(value("extra_params"))) return "Extra params must be a JSON object";
    return "";
  }
  return isGrpcAddress(address) ? "" : "Server must be host:port with a port from 1 to 65535, and IPv6 in brackets";
}

/** Return the trimmed, non-empty fields of a custom endpoint, with its display name, as the agent reads them. */
export function customPayload(service: CustomService, spec: CustomFieldSpec): Record<string, string> {
  const payload: Record<string, string> = { [NAME_FIELD]: service.name.trim() };
  for (const field of endpointFields(spec)) {
    const value = (service.fields[field] ?? "").trim();
    if (value) payload[field] = value;
  }
  return payload;
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    typeof value === "object" && value !== null && !Array.isArray(value) &&
    Object.values(value).every((item) => typeof item === "string")
  );
}

/** Return the well-formed custom endpoints of a slot from stored JSON. Malformed entries are dropped. */
export function parseStoredServices(value: unknown, slot: Slot): CustomService[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item: unknown) => {
    if (typeof item !== "object" || item === null) return [];
    const { id, name, fields } = item as Record<string, unknown>;
    if (typeof id !== "string" || !id.startsWith(CUSTOM_ID_PREFIX) || typeof name !== "string" || !isStringRecord(fields)) {
      return [];
    }
    return [{ id, slot, name, fields }];
  });
}

/** Return the custom endpoints as this browser stores them: without the memory-only fields. */
export function storedServices(services: readonly CustomService[]): CustomService[] {
  return services.map((service) => ({
    ...service,
    fields: Object.fromEntries(Object.entries(service.fields).filter(([key]) => !MEMORY_ONLY_FIELDS.has(key))),
  }));
}

/** Return the lines that a service card shows under its name: the endpoint first, then the model or voice. */
export function serviceDetails(slot: Slot, entry: Readonly<Partial<ServiceEntry>>): string[] {
  const details =
    slot === "llm"
      ? [entry.base_url, entry.model_id]
      : [entry.server, slot === "tts" && entry.voice_id ? `voice: ${entry.voice_id}` : entry.model];
  if (entry.function_id) details.push(`function_id: ${entry.function_id}`);
  return details.filter((detail): detail is string => Boolean(detail));
}

/** Return the id of the service to use: the chosen one while it is offered, else the server default, else the first. */
export function effectiveServiceId(chosen: string, entries: readonly ServiceEntry[], custom: readonly CustomService[]): string {
  if (chosen && (entries.some((entry) => entry.id === chosen) || custom.some((service) => service.id === chosen))) {
    return chosen;
  }
  return entries.find((entry) => entry.default)?.id ?? entries[0]?.id ?? "";
}

export function settingsKey(slot: Slot, id: string): string {
  return `${slot}:${id}`;
}
