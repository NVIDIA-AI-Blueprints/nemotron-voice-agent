// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  customFieldSpec,
  customPayload,
  customServiceError,
  effectiveServiceId,
  endpointFields,
  parseStoredServices,
  serviceDetails,
  storedServices,
} from "../src/lib/services.ts";
import type { ClientConfig, CustomFieldSpec, ServiceEntry } from "../src/lib/types.ts";

const SPECS: Record<"llm" | "asr" | "tts", CustomFieldSpec> = {
  llm: { fields: ["name", "base_url", "model_id", "extra_params"], required: ["base_url", "model_id"] },
  asr: { fields: ["name", "server", "model", "function_id"], required: ["server"] },
  tts: { fields: ["name", "server", "voice_id", "function_id"], required: ["server"] },
};

const entry = (id: string, isDefault = false): ServiceEntry => ({
  id,
  name: id,
  source: "self-hosted",
  default: isDefault,
  settings: {},
});

test("custom endpoints are offered only when the server opts in, with its fields", () => {
  const config: ClientConfig = { ws_url: "", custom_endpoints: true, custom_fields: SPECS };
  assert.equal(customFieldSpec(config, "llm"), SPECS.llm);
  assert.equal(customFieldSpec({ ...config, custom_endpoints: false }, "llm"), undefined);
  assert.equal(customFieldSpec({ ws_url: "", custom_fields: SPECS }, "llm"), undefined);
  assert.equal(customFieldSpec({ ws_url: "", custom_endpoints: true, custom_fields: {} }, "tts"), undefined);
  assert.equal(customFieldSpec(undefined, "asr"), undefined);
  assert.deepEqual(endpointFields(SPECS.tts), ["server", "voice_id", "function_id"]);
});

test("a custom LLM needs an http(s) URL without credentials, a model, and JSON object extra params", () => {
  const llm = (fields: Record<string, string>) => customServiceError("llm", SPECS.llm, fields);
  assert.equal(llm({ base_url: "http://my-llm:8000/v1", model_id: "m" }), "");
  assert.match(llm({ base_url: "http://my-llm:8000/v1" }), /Model ID/);
  assert.match(llm({ base_url: "grpc://x:1", model_id: "m" }), /http/);
  assert.match(llm({ base_url: "http://x/ v1", model_id: "m" }), /spaces/);
  assert.match(llm({ base_url: "https://user:secret@x/v1", model_id: "m" }), /user name or password/);
  assert.match(llm({ base_url: "https://token@x/v1", model_id: "m" }), /user name or password/);
  const extra = { base_url: "https://x/v1", model_id: "m", extra_params: "[1]" };
  assert.match(llm(extra), /JSON object/);
  assert.equal(llm({ ...extra, extra_params: '{"top_k": 5}' }), "");
  assert.match(llm({ ...extra, extra_params: `{"stop": "${"x".repeat(520)}"}` }), /at most 512 characters/);
});

test("a custom speech service needs a gRPC host:port with a valid port, no user, and IPv6 in brackets", () => {
  const asr = (server: string) => customServiceError("asr", SPECS.asr, { server });
  assert.equal(asr("my-asr:50051"), "");
  assert.equal(asr("my_asr.local:1"), "");
  assert.equal(asr("10.0.0.5:65535"), "");
  assert.equal(asr("[::1]:50051"), "");
  assert.equal(asr("[2001:db8::7]:443"), "");
  assert.equal(customServiceError("tts", SPECS.tts, { server: "grpc.nvcf.nvidia.com:443", function_id: "f" }), "");
  for (const bad of ["my-tts", "tts:0", "tts:99999", "tts:", "user@tts:50051", "::1:50051", "::1", "[::1]", "[]:1",
    "http://my-asr:50051", "tts:50051/path"]) {
    assert.match(asr(bad), /host:port/, bad);
  }
  assert.match(customServiceError("asr", SPECS.asr, {}), /required/);
});

test("the payload keeps the server's non-empty fields, trimmed, with the name", () => {
  const service = { id: "custom-1", slot: "tts" as const, name: " Mine ", fields: { server: " t:1 ", voice_id: "", base_url: "x" } };
  assert.deepEqual(customPayload(service, SPECS.tts), { name: "Mine", server: "t:1" });
});

test("stored custom endpoints keep only well-formed entries", () => {
  const good = { id: "custom-llm-1", slot: "tts", name: "Mine", fields: { base_url: "http://x/v1" } };
  const stored = [
    good,
    null,
    "custom-llm-2",
    { id: 7, name: "x", fields: {} },
    { id: "other-1", name: "x", fields: {} },
    { id: "custom-llm-3", name: null, fields: {} },
    { id: "custom-llm-4", name: "x", fields: null },
    { id: "custom-llm-5", name: "x", fields: [] },
    { id: "custom-llm-6", name: "x", fields: { base_url: 5 } },
  ];
  assert.deepEqual(parseStoredServices(stored, "llm"), [{ ...good, slot: "llm" }]);
  assert.deepEqual(parseStoredServices({ not: "a list" }, "llm"), []);
  assert.deepEqual(parseStoredServices(undefined, "llm"), []);
});

test("extra params stay in memory and are never written to storage", () => {
  const service = { id: "custom-llm-1", slot: "llm" as const, name: "x", fields: { base_url: "http://x/v1", extra_params: '{"api_key":"k"}' } };
  assert.deepEqual(storedServices([service]), [{ ...service, fields: { base_url: "http://x/v1" } }]);
  assert.equal(service.fields.extra_params, '{"api_key":"k"}');
});

test("details show the endpoint first, then the model or the voice", () => {
  assert.deepEqual(serviceDetails("llm", { base_url: "http://l/v1", model_id: "m" }), ["http://l/v1", "m"]);
  assert.deepEqual(serviceDetails("tts", { server: "t:1", voice_id: "Aria", model: "x" }), ["t:1", "voice: Aria"]);
  assert.deepEqual(serviceDetails("asr", { server: "a:1", function_id: "f" }), ["a:1", "function_id: f"]);
});

test("the chosen service is kept while offered, else the server default is used", () => {
  const entries = [entry("a"), entry("b", true)];
  const custom = [{ id: "custom-x", slot: "llm" as const, name: "x", fields: {} }];
  assert.equal(effectiveServiceId("a", entries, custom), "a");
  assert.equal(effectiveServiceId("custom-x", entries, custom), "custom-x");
  assert.equal(effectiveServiceId("custom-x", entries, []), "b");
  assert.equal(effectiveServiceId("gone", entries, custom), "b");
  assert.equal(effectiveServiceId("", [entry("a")], []), "a");
  assert.equal(effectiveServiceId("", [], []), "");
});
