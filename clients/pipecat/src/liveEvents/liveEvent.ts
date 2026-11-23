// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One row of the Event log: a request the voice agent sent to its backend, or an OpenAI-format event the backend streamed
// back. The server sends each as an RTVI server message `{"type": "live-event", "v": 1, ...}`. This file reads such a
// message strictly (anything else is ignored), sorts events into families for the filter, and writes the one line that a
// row shows. Pure functions, no React.

export const LIVE_EVENT_TYPE = "live-event";
export const LIVE_EVENT_VERSION = 1;

/** "out": sent to the backend (a request). "in": streamed back by it (an event). */
export type Direction = "out" | "in";

export interface LiveEvent {
  /** The event's number in this call, counting from 1. A gap means the server dropped events it could not send. */
  seq: number;
  /** Seconds since the bot started. */
  t: number;
  direction: Direction;
  /** The event's `type` (`response.created`), or the request (`responses.create`, `conversations.create`). */
  kind: string;
  /** The response the event belongs to in this call (`r1`, `r2`, ...), or null for a call of the conversation endpoints. */
  stream: string | null;
  /** The delegation the response works for, or null. */
  delegation: string | null;
  /** The event as the server kept it: long text is cut, and a prompt is shown as its size. */
  event: Record<string, unknown>;
}

const MAX_KIND = 120;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalString(value: unknown): string | null | undefined {
  if (value === null || value === undefined) return null;
  return typeof value === "string" ? value : undefined;
}

/** Reads one server message as a live event, or returns null when it is not a version 1 `live-event` message. */
export function parseLiveEvent(message: unknown): LiveEvent | null {
  if (!isRecord(message) || message.type !== LIVE_EVENT_TYPE || message.v !== LIVE_EVENT_VERSION) return null;
  const { seq, t, direction, kind, event } = message;
  if (typeof seq !== "number" || !Number.isInteger(seq) || seq < 1) return null;
  if (typeof t !== "number" || !Number.isFinite(t) || t < 0) return null;
  if (direction !== "out" && direction !== "in") return null;
  if (typeof kind !== "string" || kind === "" || kind.length > MAX_KIND) return null;
  if (!isRecord(event)) return null;
  const stream = optionalString(message.stream);
  const delegation = optionalString(message.delegation);
  if (stream === undefined || delegation === undefined) return null;
  return { seq, t, direction, kind, stream, delegation, event };
}

// ---- Families, for the filter ---------------------------------------------------------------------------------------

export type Family = "request" | "lifecycle" | "tool" | "text" | "error";

export const FAMILIES: readonly { id: Family; label: string }[] = [
  { id: "request", label: "Requests" },
  { id: "lifecycle", label: "Lifecycle" },
  { id: "tool", label: "Tools" },
  { id: "text", label: "Text" },
  { id: "error", label: "Errors" },
];

const TOOL_ITEMS = new Set(["mcp_call", "function_call", "custom_tool_call", "web_search_call", "file_search_call"]);

function item(event: Record<string, unknown>): Record<string, unknown> | null {
  return isRecord(event.item) ? event.item : null;
}

function responseStatus(event: Record<string, unknown>): string {
  return isRecord(event.response) && typeof event.response.status === "string" ? event.response.status : "";
}

/** The family of an event: what the filter hides or shows it with. */
export function familyOf(row: LiveEvent): Family {
  const { kind, event } = row;
  if (kind === "error" || kind === "response.failed" || kind.endsWith(".failed") || responseStatus(event) === "failed") {
    return "error";
  }
  if (row.direction === "out") return "request";
  if (kind.startsWith("response.output_text.") || kind.startsWith("response.refusal.")) return "text";
  if (kind.startsWith("response.mcp_call") || kind.startsWith("response.function_call") || kind.startsWith("response.custom_tool_call")) {
    return "tool";
  }
  if (kind.startsWith("response.output_item.")) {
    const type = item(event)?.type;
    return typeof type === "string" && TOOL_ITEMS.has(type) ? "tool" : "lifecycle";
  }
  return "lifecycle";
}

// ---- The line of a row ----------------------------------------------------------------------------------------------

const SUMMARY_CHARS = 160;

function oneLine(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > SUMMARY_CHARS ? `${flat.slice(0, SUMMARY_CHARS - 1)}…` : flat;
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** The shortest useful form of an id: `resp_a5eb2a9b77f7…` becomes `resp_a5eb2a9b`. */
export function shortId(id: string): string {
  const cut = id.indexOf("_");
  return cut > 0 && id.length > cut + 9 ? `${id.slice(0, cut + 9)}` : id;
}

/** The last line of a text: a request starts with the conversation so far, and its newest line is what is asked now. */
function lastLine(text: string): string {
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines.at(-1) ?? "";
}

function requestText(event: Record<string, unknown>): string {
  const input = Array.isArray(event.input) ? event.input : [];
  for (let index = input.length - 1; index >= 0; index -= 1) {
    const entry = input[index];
    if (!isRecord(entry)) continue;
    if (typeof entry.content === "string") return lastLine(entry.content);
    const parts = Array.isArray(entry.content) ? entry.content : [];
    const text = parts.map((part) => (isRecord(part) ? str(part.text) : "")).filter(Boolean).join("\n");
    if (text) return lastLine(text);
  }
  return "";
}

function itemLine(entry: Record<string, unknown> | null): string {
  if (!entry) return "";
  const type = str(entry.type);
  if (TOOL_ITEMS.has(type)) {
    const label = str(entry.server_label);
    const name = `${label ? `${label}.` : ""}${str(entry.name)}`;
    return oneLine(`${name} ${str(entry.arguments)}`);
  }
  return type;
}

function usageLine(event: Record<string, unknown>): string {
  const response = isRecord(event.response) ? event.response : null;
  const usage = response && isRecord(response.usage) ? response.usage : null;
  if (!usage) return "";
  const input = usage.input_tokens;
  const output = usage.output_tokens;
  return typeof input === "number" && typeof output === "number" ? `${input} in / ${output} out tokens` : "";
}

/** The one line a row shows after its type: the part of the event that tells what it is about. */
export function summarize(row: LiveEvent): string {
  const { kind, event } = row;
  if (kind === "responses.create") return oneLine(requestText(event));
  if (kind.startsWith("conversations.")) return shortId(str(event.id));
  if (kind === "error") return oneLine(str(event.message));
  if (kind === "response.created" || kind === "response.completed" || kind === "response.failed" || kind === "response.incomplete") {
    const response = isRecord(event.response) ? event.response : {};
    const parts = [str(response.status), kind === "response.created" ? str(response.model) : usageLine(event)];
    return parts.filter(Boolean).join(" · ");
  }
  if (kind === "response.queued" || kind === "response.in_progress") return shortId(str(isRecord(event.response) ? event.response.id : ""));
  if (kind === "response.output_item.added" || kind === "response.output_item.done") return itemLine(item(event));
  if (kind.startsWith("response.mcp_call") || kind.startsWith("response.function_call")) {
    return str(event.arguments) ? oneLine(str(event.arguments)) : shortId(str(event.item_id));
  }
  if (kind === "response.output_text.delta") return oneLine(str(event.delta));
  if (kind === "response.output_text.done") return oneLine(str(event.text));
  if (kind.startsWith("response.content_part.")) {
    const part = isRecord(event.part) ? event.part : {};
    return str(part.text) ? oneLine(str(part.text)) : str(part.type);
  }
  return "";
}

/** "+12.345 s": when the event happened, seconds since the bot started. */
export function formatTime(t: number): string {
  return `+${t.toFixed(3)} s`;
}
