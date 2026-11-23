// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { decodeAgentEvent, describeAgentEvent } from "../src/lib/agentEvents.ts";

const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));

test("agent events decode, and payloads that are not an event object are ignored", () => {
  assert.deepEqual(decodeAgentEvent(encode({ type: "barge_in" })), { type: "barge_in" });
  assert.equal(decodeAgentEvent(encode([1])), null);
  assert.equal(decodeAgentEvent(encode(null)), null);
  assert.equal(decodeAgentEvent(new TextEncoder().encode("{not json")), null);
  assert.equal(decodeAgentEvent(encode({ type: "unknown" })), null);
  assert.equal(decodeAgentEvent(encode({ type: "barge_in", ts: "now" })), null);
});

test("each event type is checked field by field as the agent publishes it", () => {
  const session = {
    type: "session", ts: 1.5, scenario: "generic", recipe: "cloud",
    models: [{ kind: "LLM", name: "Nemotron", runs: "cloud", id: null }, { kind: "TTS", name: "Magpie", runs: "local" }],
  };
  assert.deepEqual(decodeAgentEvent(encode(session)), session);
  const voices = { type: "voices", voices: [{ id: "Aria", name: "Aria" }], current: "Aria" };
  assert.deepEqual(decodeAgentEvent(encode(voices)), voices);
  const error = { type: "error", source: "LLM", message: "x", status: null, recoverable: false };
  assert.deepEqual(decodeAgentEvent(encode(error)), error);
  assert.ok(decodeAgentEvent(encode({ type: "tool_result", name: "t", ok: true, result: { a: 1 }, ms: 12.5 })));
  assert.ok(decodeAgentEvent(encode({ type: "tool_call", name: "t" })));
});

test("a malformed event of a known type is dropped", () => {
  const malformed = [
    { type: "voices", voices: null, current: "Aria" },
    { type: "voices", voices: [{ id: 1, name: "Aria" }], current: "Aria" },
    { type: "voices", voices: [] },
    { type: "session", scenario: "generic", recipe: "cloud", models: [{ kind: "LLM" }] },
    { type: "tool_call", name: 3 },
    { type: "tool_result", name: "t", ok: "yes", result: 1, ms: 1 },
    { type: "false_interruption" },
    { type: "voice_changed", voice: "Aria" },
    { type: "error", source: "LLM", message: "x", status: "429", recoverable: true },
    { type: "agent_state", state: null },
  ];
  for (const event of malformed) assert.equal(decodeAgentEvent(encode(event)), null, JSON.stringify(event));
});

test("a recoverable error reads as a retry, and a hard rate limit as an error with a hint", () => {
  const busy = describeAgentEvent({ type: "error", source: "LLM", message: "429", status: 429, recoverable: true });
  assert.equal(busy?.tone, "notice");
  assert.equal(busy?.text, "LLM request failed (rate limited, HTTP 429). Retrying.");
  const refused = describeAgentEvent({ type: "error", source: "TTS", message: "Connection refused", status: null, recoverable: true });
  assert.equal(refused?.text, "TTS request failed. Retrying.");
  const failed = describeAgentEvent({ type: "error", source: "LLM", message: "Too Many Requests", status: null, recoverable: false });
  assert.equal(failed?.tone, "error");
  assert.match(failed?.text ?? "", /choose another model/);
  assert.equal(describeAgentEvent({ type: "agent_state", state: "thinking" }), null);
});
