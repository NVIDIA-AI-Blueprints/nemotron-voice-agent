// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The bounded event log (client/public/live/log.js). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import { EventLog, LOG_LIMIT, payloadText, searchText } from "../../public/live/log.js";

test("the log keeps the newest entries and says which it dropped", () => {
  const log = new EventLog(3);
  const entries = Array.from({ length: 5 }, (_, i) => ({ type: `e${i}` }));
  assert.deepEqual(log.add(entries[0]), []);
  log.add(entries[1]);
  log.add(entries[2]);
  assert.deepEqual(log.add(entries[3]), [entries[0]]);
  assert.deepEqual(log.add(entries[4]), [entries[1]]);
  assert.deepEqual(log.entries.map((entry) => entry.type), ["e2", "e3", "e4"]);
  log.clear();
  assert.deepEqual(log.entries, []);
  assert.equal(LOG_LIMIT, 2000);
});

test("the export holds each entry's own fields, not the page's", () => {
  const log = new EventLog();
  log.add({ at: "t", cat: "app", dir: "in", type: "x", text: "y", payload: { a: 1 }, actor: "client", delta: false, row: {}, search: "x" });
  assert.deepEqual(log.exported(), [{ at: "t", cat: "app", dir: "in", type: "x", text: "y", payload: { a: 1 }, actor: "client" }]);
});

test("the search text is made when a filter first needs it, then kept", () => {
  const payload = { Model: "Nemotron" };
  const entry = { type: "Session.Started", text: "Hello", payload };
  assert.equal(entry.search, undefined);
  assert.equal(searchText(entry), `session.started hello ${JSON.stringify(payload).toLowerCase()}`);
  payload.Model = "changed";
  assert.match(searchText(entry), /nemotron/);  // kept, not made again
  assert.equal(searchText({ type: "t" }), "t  \"\"");
});

test("a payload shows as indented JSON", () => {
  assert.equal(payloadText({ a: [1] }), "{\n  \"a\": [\n    1\n  ]\n}");
});
