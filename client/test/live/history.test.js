// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The history panel's reading of /history (client/public/live/history.js). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CONTEXT_PREFIX,
  RESULT_PREFIX,
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

test("each pending line is labeled by its prefix, as the heard log writes it", () => {
  assert.deepEqual(pendingEntry("Caller: two lattes"), {
    cls: "user pending", role: "next delegation · Caller", text: "two lattes",
  });
  assert.deepEqual(pendingEntry("Agent: One moment."), {
    cls: "assistant pending", role: "next delegation · Agent", text: "One moment.",
  });
  assert.equal(pendingEntry("App context: cart has 2 items").cls, "context pending");
  assert.equal(pendingEntry("App instruction: speak slowly").role, "next delegation · App instruction");
  // A line with no label is shown whole.
  assert.deepEqual(pendingEntry("plain"), { cls: "context pending", role: "next delegation · line", text: "plain" });
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

test("talker messages: system collapsed, application context marked, others by role", () => {
  assert.equal(talkerEntry({ role: "system", content: "rules" }).collapsed, "5 characters");
  assert.deepEqual(talkerEntry({ role: "user", content: `${CONTEXT_PREFIX}store is busy` }), {
    cls: "context", role: "application context", text: "store is busy",
  });
  assert.deepEqual(talkerEntry({ role: "assistant", content: [{ type: "text", text: "Hi" }] }), {
    cls: "assistant", role: "assistant", text: "Hi",
  });
});

test("a backend result in the talker's memory is marked as one, not as the caller's words", () => {
  assert.deepEqual(talkerEntry({ role: "user", content: `${RESULT_PREFIX}We have chai.` }), {
    cls: "result", role: "backend result (reference data)", text: "We have chai.",
  });
});

test("backend messages show their phase; an mcp_call is a backend step with its outcome", () => {
  const message = (phase) => ({ type: "message", role: "assistant", phase, content: [{ type: "output_text", text: "x" }] });
  assert.equal(backendEntry(message("commentary")).role, "assistant · commentary");
  assert.equal(backendEntry(message("final_answer")).role, "assistant · final_answer");
  assert.equal(backendEntry(message(undefined)).role, "assistant");
  assert.deepEqual(
    backendEntry({ type: "mcp_call", server_label: "voiceclaw", name: "show", arguments: "{}", output: "Latte $5", status: "completed" }),
    { cls: "call", role: "mcp_call · voiceclaw · completed", text: "show({}) → Latte $5" },
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
      items: [{ type: "message", role: "user", content: [{ type: "input_text", text: "Caller: hi" }] }],
      pending: ["Caller: and tea?"],
    },
  });
  assert.equal(view.backend.model, "backend");
  assert.equal(view.backend.count, "1 items · working · 1 waiting");
  assert.deepEqual(view.backend.entries.map((entry) => entry.cls), ["prompt client", "user", "user pending"]);
});
