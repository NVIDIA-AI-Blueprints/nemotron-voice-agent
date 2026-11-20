// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Model, Voice } from "./types.ts";

/** Data-channel topic of the events that the agent publishes: tool calls, barge-ins, errors, and session info. */
export const EVENTS_TOPIC = "agent.events";
/** Data-channel topic of the commands that this page sends to the agent. */
export const CONTROL_TOPIC = "agent.control";

export type AgentEvent = { ts?: number } & (
  | { type: "session"; scenario: string; recipe: string; models: Model[] }
  | { type: "tool_call"; name: string; args: unknown }
  | { type: "tool_result"; name: string; ok: boolean; result: unknown; ms: number }
  | { type: "barge_in" }
  | { type: "false_interruption"; resumed: boolean }
  | { type: "voices"; voices: Voice[]; current: string }
  | { type: "voice_changed"; voice: string; name: string }
  | { type: "error"; source: string; message: string; status: number | null; recoverable: boolean }
  | { type: "agent_state"; state: string }
);

export type EventTone = "info" | "ok" | "error" | "notice";

export interface EventEntry {
  icon: string;
  text: string;
  tone: EventTone;
}

const MAX_DETAIL_LENGTH = 400;

type Check = (value: unknown) => boolean;
type Shape = Record<string, Check>;

const isString: Check = (value) => typeof value === "string";
const isBoolean: Check = (value) => typeof value === "boolean";
const isNumber: Check = (value) => typeof value === "number" && Number.isFinite(value);
const isAnything: Check = () => true;
const optional = (check: Check): Check => (value) => value === undefined || check(value);
const nullable = (check: Check): Check => (value) => value === null || check(value);

function hasShape(value: unknown, shape: Shape): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return Object.entries(shape).every(([key, check]) => check(record[key]));
}

const listOf = (shape: Shape): Check => (value) => Array.isArray(value) && value.every((item) => hasShape(item, shape));

const EVENT_SHAPES: Record<AgentEvent["type"], Shape> = {
  session: {
    scenario: isString,
    recipe: isString,
    models: listOf({ kind: isString, name: isString, runs: isString, id: optional(nullable(isString)) }),
  },
  tool_call: { name: isString, args: isAnything },
  tool_result: { name: isString, ok: isBoolean, result: isAnything, ms: isNumber },
  barge_in: {},
  false_interruption: { resumed: isBoolean },
  voices: { voices: listOf({ id: isString, name: isString }), current: isString },
  voice_changed: { voice: isString, name: isString },
  error: { source: isString, message: isString, status: nullable(isNumber), recoverable: isBoolean },
  agent_state: { state: isString },
};

const isEventType = (value: unknown): value is AgentEvent["type"] =>
  typeof value === "string" && Object.hasOwn(EVENT_SHAPES, value);

/** Decode an agent event. Returns null for anything that is not JSON, of an unknown type, or missing a field. */
export function decodeAgentEvent(payload: Uint8Array): AgentEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(payload));
  } catch {
    return null;
  }
  if (!hasShape(value, { type: isEventType, ts: optional(isNumber) })) return null;
  return hasShape(value, EVENT_SHAPES[value.type as AgentEvent["type"]]) ? (value as AgentEvent) : null;
}

function shorten(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value) ?? "";
  return text.length > MAX_DETAIL_LENGTH ? `${text.slice(0, MAX_DETAIL_LENGTH)}…` : text;
}

function isRateLimited(status: number | null, message: string): boolean {
  return status === 429 || /too many requests|rate.?limit/i.test(message);
}

/** Return the transcript line for an agent event, or null for events that only update state. */
export function describeAgentEvent(event: AgentEvent): EventEntry | null {
  switch (event.type) {
    case "session":
      return { icon: "🔊", text: `Session started · ${event.scenario} · ${event.recipe}`, tone: "info" };
    case "tool_call":
      return { icon: "🔧", text: `Tool call: ${event.name}(${shorten(event.args)})`, tone: "info" };
    case "tool_result":
      return {
        icon: event.ok ? "✅" : "❌",
        text: `Tool result: ${event.name} → ${shorten(event.result)} (${event.ms} ms)`,
        tone: event.ok ? "ok" : "error",
      };
    case "barge_in":
      return { icon: "✋", text: "Barge-in: you interrupted the agent mid-reply", tone: "notice" };
    case "false_interruption":
      return {
        icon: "↩️",
        text: `False interruption${event.resumed ? ": agent resumed speaking" : ""}`,
        tone: "notice",
      };
    case "voice_changed":
      return { icon: "🗣️", text: `Voice changed to ${event.name}`, tone: "info" };
    case "error": {
      const limited = isRateLimited(event.status, event.message);
      if (event.recoverable) {
        return {
          icon: "🔁",
          text: `${event.source} request failed${limited ? " (rate limited, HTTP 429)" : ""}. Retrying.`,
          tone: "notice",
        };
      }
      const hint = limited ? " The hosted model is rate limited. Wait a minute, or choose another model." : "";
      return { icon: "⚠️", text: `${event.source} error: ${event.message}${hint}`, tone: "error" };
    }
    default:
      return null;
  }
}
