// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { customPrompt, fromCatalog } from "../src/lib/prompts.ts";
import { buildTokenRequest, effectiveVoiceId } from "../src/lib/session.ts";
import type { ServiceEntry } from "../src/lib/types.ts";

const bank = fromCatalog({
  id: "bank",
  name: "Bank",
  description: "",
  examples: [],
  content: "You are a bank agent.",
  tools: ["get_balance"],
  default: false,
});
const entry = (id: string): ServiceEntry => ({ id, name: id, source: "cloud-nim", default: true, settings: {} });

test("a built-in prompt is sent as its use case, with the voice and the offered services", () => {
  const request = buildTokenRequest({
    prompt: bank,
    defaultScenario: "generic",
    voice: "Aria",
    services: {
      llm: { kind: "builtin", entry: entry("nemotron-super"), settings: { temperature: 0.3, top_p: undefined } },
      tts: { kind: "builtin", entry: entry("magpie"), settings: {} },
    },
  });
  assert.deepEqual(request, {
    scenario: "bank",
    voice: "Aria",
    services: { llm: { key: "nemotron-super", settings: { temperature: 0.3 } }, tts: { key: "magpie" } },
  });
});

test("a custom prompt runs on the default use case and sends its text", () => {
  const request = buildTokenRequest({
    prompt: customPrompt("brief", "", "Answer briefly."),
    defaultScenario: "generic",
    voice: "",
    services: {},
  });
  assert.deepEqual(request, { scenario: "generic", prompt: "Answer briefly.", services: {} });
});

test("a custom TTS sends the server's fields and no listed voice", () => {
  const service = { id: "custom-tts-1", slot: "tts" as const, name: "Mine", fields: { server: "t:1", voice_id: "Sofia" } };
  const spec = { fields: ["name", "server", "voice_id", "function_id"], required: ["server"] };
  const request = buildTokenRequest({
    prompt: bank,
    defaultScenario: "generic",
    voice: "Aria",
    services: { tts: { kind: "custom", service, spec } },
  });
  assert.deepEqual(request.services.tts, { custom: { name: "Mine", server: "t:1", voice_id: "Sofia" } });
  assert.equal(request.voice, undefined);
});

test("the voice falls back to the list default, then the first voice", () => {
  const list = { voices: [{ id: "Aria", name: "Aria" }, { id: "Sofia", name: "Sofia" }], default: "Sofia" };
  assert.equal(effectiveVoiceId("Aria", list), "Aria");
  assert.equal(effectiveVoiceId("Gone", list), "Sofia");
  assert.equal(effectiveVoiceId("", { voices: list.voices, default: "Gone" }), "Aria");
  assert.equal(effectiveVoiceId("Aria", undefined), "");
});
