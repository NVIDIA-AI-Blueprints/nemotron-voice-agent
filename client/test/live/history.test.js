// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The history panel's reading of /history (client/public/live/history.js). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  backendEntry,
  contentText,
  historyView,
  pendingEntry,
  promptEntries,
  promptOrigin,
  shouldPoll,
  snapshotKey,
  talkerEntry,
} from "../../public/live/history.js";

test("the panel polls only while it is open and the page visible", () => {
  assert.equal(shouldPoll(true, "visible"), true);
  assert.equal(shouldPoll(false, "visible"), false);
  assert.equal(shouldPoll(true, "hidden"), false);
});

test("an answer is compared by its own text, plus the client history's length in client mode", () => {
  assert.equal(snapshotKey("{\"a\":1}", null), "{\"a\":1}");
  const local = { delegationId: "item_1", history: [{}, {}] };
  const before = snapshotKey("{}", local);
  local.history.push({});
  assert.notEqual(snapshotKey("{}", local), before);
});

test("each pending line is styled by its kind and shown as the backend reads it", () => {
  assert.deepEqual(pendingEntry({ kind: "caller", text: "two lattes" }), {
    cls: "user pending", role: "next delegation · caller", text: "two lattes",
  });
  assert.deepEqual(pendingEntry({ kind: "agent", text: "One moment." }), {
    cls: "assistant pending", role: "next delegation · agent", text: "One moment.",
  });
  assert.equal(pendingEntry({ kind: "context", text: "cart has 2 items" }).cls, "context pending");
  assert.equal(pendingEntry({ kind: "instruction", text: "speak slowly" }).role, "next delegation · instruction");
  // A line with no kind shows as the application's.
  assert.deepEqual(pendingEntry({ text: "plain" }), { cls: "context pending", role: "next delegation · line", text: "plain" });
});

test("a prompt part says whose it is: the client's, the config's or the server's", () => {
  assert.equal(promptOrigin("session.instructions"), "client");
  assert.equal(promptOrigin("delegation.responses.instructions"), "client");
  assert.equal(promptOrigin("prompts.yaml: frontend"), "config");
  assert.equal(promptOrigin("the core: the newest request's instructions"), "server");
  const [entry] = promptEntries([{ source: "session.instructions", text: "Line one\nLine two" }]);
  assert.equal(entry.cls, "prompt client");
  assert.equal(entry.collapsed, "Line one … (17 characters)");
  assert.deepEqual(promptEntries(undefined), []);
});

test("talker messages: system collapsed, application context marked by its kind, others by role", () => {
  assert.equal(talkerEntry({ role: "system", content: "rules" }).collapsed, "5 characters");
  for (const kind of ["app_context", "typed"]) {
    assert.deepEqual(talkerEntry({ role: "user", kind, content: "store is busy" }), {
      cls: "context", role: "application context", text: "store is busy",
    });
  }
  assert.deepEqual(talkerEntry({ role: "assistant", content: [{ type: "text", text: "Hi" }] }), {
    cls: "assistant", role: "assistant", text: "Hi",
  });
});

test("a backend result in the talker's memory is marked as one, not as the caller's words", () => {
  assert.deepEqual(talkerEntry({ role: "user", kind: "result", content: "We have chai." }), {
    cls: "result", role: "backend result (reference data)", text: "We have chai.",
  });
  assert.deepEqual(talkerEntry({ role: "user", kind: "caller", content: "We have chai." }), {
    cls: "user", role: "user", text: "We have chai.",
  });
});

test("backend messages show their phase; an mcp_call is a backend step with its outcome", () => {
  const message = (phase) => ({ type: "message", role: "assistant", phase, content: [{ type: "output_text", text: "x" }] });
  assert.equal(backendEntry(message("commentary")).role, "assistant · commentary");
  assert.equal(backendEntry(message("final_answer")).role, "assistant · final_answer");
  assert.equal(backendEntry(message(undefined)).role, "assistant");
  assert.deepEqual(
    backendEntry({ type: "mcp_call", server_label: "delegation_core", name: "show", arguments: "{}", output: "Latte $5", status: "completed" }),
    { cls: "call", role: "mcp_call · delegation_core · completed", text: "show({}) → Latte $5" },
  );
  assert.equal(
    backendEntry({ type: "mcp_call", server_label: "cafe", name: "get_menu", arguments: "{}", output: null, error: { content: "boom" }, status: "failed" }).text,
    "get_menu({}) → boom",
  );
});

test("backend items by type", () => {
  assert.equal(backendEntry({ type: "function_call", call_id: "c1", name: "get_menu", arguments: "{}" }).text, "get_menu({})");
  assert.equal(backendEntry({ type: "function_call_output", call_id: "c1", output: "{\"ok\":1}" }).role, "function_call_output · c1");
  assert.equal(backendEntry({ type: "reasoning", summary: [] }).text, "(hidden)");
  assert.equal(backendEntry({ role: "user", content: [{ type: "input_text", text: "hi" }] }).text, "hi");
  assert.equal(backendEntry({ type: "other", x: 1 }).role, "other");
  assert.equal(contentText(["a", { text: "b" }, { type: "image" }]), "a\nb\n{\"type\":\"image\"}");
});

test("a client-mode snapshot has no server backend", () => {
  const view = historyView({ frontend: { model: "talker", messages: [{ role: "system", content: "s" }] }, backend: null });
  assert.equal(view.frontend.count, "1 messages");
  assert.deepEqual(view.frontend.entries, []);  // the first system message is counted, not shown
  assert.deepEqual(view.backend, { model: "", count: "", entries: [], empty: "No server backend in this session." });
});

test("a responses-mode snapshot: prompt, items, then the pending lines", () => {
  const view = historyView({
    frontend: { model: "talker", prompt: [], messages: [] },
    backend: {
      model: "backend",
      busy: true,
      prompt: [{ source: "delegation.responses.instructions", text: "Be brief" }],
      items: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] }],
      pending: [{ kind: "caller", text: "and tea?" }],
    },
  });
  assert.equal(view.backend.model, "backend");
  assert.equal(view.backend.count, "1 items · working · 1 waiting");
  assert.deepEqual(view.backend.entries.map((entry) => entry.cls), ["prompt client", "user", "user pending"]);
});
