// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The delegation cards' reading of the events (client/public/live/delegations.js), shaped as the delegation core
// writes them (src/delegation_core/im/stream.py). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DelegationBook,
  PROGRESS_AFTER_MS,
  finalAnswers,
  isDisplay,
  messageText,
  progressLines,
  runBatch,
} from "../../public/live/delegations.js";

/** A view that records what the book draws, per card. */
function recorder() {
  const cards = new Map();
  return {
    created(card) { cards.set(card.id, { stages: {}, status: null, tone: null, flow: card.responseId }); },
    stage(card, key, spec) {
      const view = cards.get(card.id);
      if (spec === null) delete view.stages[key];
      else view.stages[key] = { ...(view.stages[key] || {}), ...spec };
    },
    status(card, text, tone) { Object.assign(cards.get(card.id), { status: text, tone }); },
    flow(card, responseId) { cards.get(card.id).flow = responseId; },
    of: (id) => cards.get(id),
  };
}

const announce = (book, id, responseId) =>
  book.announce({ type: "session.delegation.created", offset_ms: 5, delegation: { id, type: "delegation", target: "responses", response_id: responseId } });
const envelope = (id, event) => ({ type: "response.event", delegation_id: id, event });
const created = (id) => ({ type: "response.created", response: { id, status: "queued", output: [] } });
const queued = (id) => ({ type: "response.queued", response: { id, status: "queued", output: [] } });
const running = (id, previous) => ({
  type: "response.in_progress",
  response: { id, status: "in_progress", output: [], ...(previous ? { previous_response_id: previous } : {}) },
});
const done = (item, index = 0) => ({ type: "response.output_item.done", output_index: index, item });
const message = (text, phase) => ({
  id: `msg_${text.length}`, type: "message", status: "completed", role: "assistant", phase,
  content: [{ type: "output_text", text, annotations: [], logprobs: [] }],
});
const call = (callId, name, args = "{}") => ({ id: `fc_${callId}`, type: "function_call", status: "completed", call_id: callId, name, arguments: args });
const mcp = (id, name, status, output, label = "cafe") => ({
  id, type: "mcp_call", server_label: label, name, arguments: "{}", output, error: null, status,
});
const completed = (id) => ({ type: "response.completed", response: { id, status: "completed", output: [], usage: { total_tokens: 9 } } });

/** Read `events` of delegation `id` in order; return the last read's result. */
function play(book, id, events) {
  let last = null;
  for (const event of events) last = book.read(envelope(id, event));
  return last;
}

test("the answer is the final_answer; commentary is progress; queued and running show first", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), queued("r1")]);
  assert.equal(view.of("d1").tone, "waiting");
  play(book, "d1", [running("r1")]);
  assert.equal(view.of("d1").status, "running");
  const result = play(book, "d1", [done(message("Checking the menu.", "commentary")), done(message("We have chai.", "final_answer")), completed("r1")]);
  assert.deepEqual(result, { answered: true });
  const card = view.of("d1");
  assert.equal(card.stages.answer.text, "We have chai.");
  assert.equal(card.stages.progress.text, "Checking the menu.");
  assert.deepEqual([card.status, card.tone], ["answered", "done"]);
  assert.equal(card.flow, "r1");
});

test("streamed text and a message with no phase: only a completed message is an answer", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), running("r1"), { type: "response.output_text.delta", delta: "partial" }, completed("r1")]);
  assert.equal(view.of("d1").stages.answer, undefined);
  assert.equal(view.of("d1").status, "ended with no answer");
  const other = new DelegationBook(recorder(), []);
  announce(other, "d2", "r2");
  play(other, "d2", [created("r2"), running("r2"), done(message("No phase is an answer."))]);
  assert.deepEqual(other.cards.get("d2").answers, ["No phase is an answer."]);
});

test("each task's answer is its own final_answer; the card shows them all", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), running("r1"), done(message("Flight booked.", "final_answer")), done(message("Hotel booked.", "final_answer")), completed("r1")]);
  assert.equal(view.of("d1").stages.answer.text, "Flight booked.\n\nHotel booked.");
  assert.match(view.of("d1").stages.answer.label, /2/);
});

test("the show call is the display, at once; another mcp_call is a backend step", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), running("r1"), { type: "response.output_item.added", output_index: 0, item: mcp("m1", "show", "in_progress", null, "delegation_core") }]);
  assert.ok(view.of("d1").stages.mcp_m1);
  play(book, "d1", [done(mcp("m1", "show", "completed", "Latte, medium: $5.00", "delegation_core"))]);
  assert.equal(view.of("d1").stages.mcp_m1, undefined);
  assert.equal(view.of("d1").stages.display.text, "Latte, medium: $5.00");
  assert.equal(book.cards.get("d1").toolRan, false);
  play(book, "d1", [done(mcp("m2", "get_menu", "completed", "{\"items\":[]}"), 1), done(mcp("m3", "manage_cart", "failed", "boom"), 2)]);
  assert.match(view.of("d1").stages.mcp_m2.label, /backend ran cafe · get_menu \(completed\)/);
  assert.equal(view.of("d1").stages.mcp_m3.cls, "warn");
  assert.equal(book.cards.get("d1").toolRan, true);
  assert.equal(view.of("d1").stages.answer, undefined);
});

test("a show that did not complete is no display: nothing reaches the screen, and it is no backend step", () => {
  // As in the frontend adapter, a failed show carries its error as its output, which the screen never shows.
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), running("r1"), done(mcp("m1", "show", "failed", "show: display must be non-empty text", "delegation_core"))]);
  assert.equal(view.of("d1").stages.display, undefined);
  assert.equal(view.of("d1").stages.mcp_m1.cls, "warn");
  assert.match(view.of("d1").stages.mcp_m1.label, /display \(show\) failed/);
  assert.equal(book.cards.get("d1").toolRan, false);
});

test("a response that ends with the app's calls asks the page to run them, then a continuation answers", () => {
  const view = recorder();
  const book = new DelegationBook(view, ["get_menu"]);
  announce(book, "d1", "r1");
  play(book, "d1", [created("r1"), running("r1"), { type: "response.output_item.added", output_index: 0, item: call("c1", "get_menu", "") }]);
  assert.ok(view.of("d1").stages.call_pending_0);
  const calls = [call("c1", "get_menu"), call("c2", "teleport")];
  const result = play(book, "d1", [done(calls[0], 0), done(calls[1], 1), completed("r1")]);
  assert.deepEqual(result, { run: calls });
  assert.equal(view.of("d1").stages.call_pending_0, undefined);
  assert.equal(view.of("d1").stages.call_c2.cls, "warn");
  assert.equal(view.of("d1").tone, "waiting");
  // The continuation: a new response of the same delegation.
  assert.deepEqual(play(book, "d1", [created("r2"), running("r2"), done(message("Chai.", "final_answer")), completed("r2")]), { answered: true });
  assert.equal(view.of("d1").flow, "r2");
});

test("the taker the IM names hands the older work over; the older's commentary is the handover's line", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1", "r1");
  announce(book, "d2", "r2");
  play(book, "d1", [created("r1"), running("r1")]);
  play(book, "d2", [created("r2"), running("r2", "r1")]);
  assert.equal(book.cards.get("d1").taker, "d2");
  assert.match(view.of("d2").stages.took.text, /d1/);
  play(book, "d1", [done(message("Handing over.", "commentary")), completed("r1")]);
  assert.equal(view.of("d1").stages.progress, undefined);
  assert.equal(view.of("d1").stages.handover_line.text, "Handing over.");
  assert.deepEqual([view.of("d1").status, view.of("d1").tone], ["handed over → d2", "ended"]);
  play(book, "d2", [done(message("Both done.", "final_answer")), completed("r2")]);
  assert.equal(view.of("d2").status, "answered");
});

test("a naming that comes before the response it names, or after it ended, still hands it over", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  announce(book, "d1");
  announce(book, "d2");
  play(book, "d2", [created("r2"), running("r2", "r1")]);  // r1 has not reached this page yet
  play(book, "d1", [created("r1")]);
  assert.equal(book.cards.get("d1").taker, "d2");

  const late = recorder();
  const other = new DelegationBook(late, []);
  announce(other, "d1");
  announce(other, "d2");
  play(other, "d1", [created("r1"), running("r1"), completed("r1")]);
  assert.equal(late.of("d1").status, "ended with no answer");
  play(other, "d2", [created("r2"), running("r2", "r1")]);
  assert.equal(late.of("d1").status, "handed over → d2");
});

test("the endings: failed with its code, cancelled at close, incomplete with a reason", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  for (const id of ["d1", "d2", "d3"]) announce(book, id);
  play(book, "d1", [created("r1"), { type: "response.failed", response: { id: "r1", status: "failed", error: { code: "server_error", message: "The backend failed" } } }]);
  assert.deepEqual([view.of("d1").status, view.of("d1").tone], ["failed: server_error", "failed"]);
  assert.equal(view.of("d1").stages.ending.text, "The backend failed");
  play(book, "d2", [created("r2"), { type: "response.incomplete", response: { id: "r2", status: "cancelled", incomplete_details: null } }]);
  assert.deepEqual([view.of("d2").status, view.of("d2").tone], ["cancelled: the session closed", "ended"]);
  play(book, "d3", [created("r3"), { type: "response.incomplete", response: { id: "r3", status: "incomplete", incomplete_details: { reason: "max_output_tokens" } } }]);
  assert.equal(view.of("d3").status, "incomplete: max_output_tokens");
});

test("an event of a delegation this call never announced is not read", () => {
  const view = recorder();
  const book = new DelegationBook(view, []);
  assert.equal(book.read(envelope("unknown", created("r9"))), null);
  assert.equal(book.cards.size, 0);
});

test("the item helpers read phase and the display the same way", () => {
  const items = [message("note", "commentary"), message("answer", "final_answer"), message("", "final_answer"), call("c", "x"), mcp("m", "show", "completed", "d")];
  assert.deepEqual(finalAnswers(items), ["answer"]);
  assert.deepEqual(progressLines(items), ["note"]);
  assert.equal(isDisplay(items[4]), true);
  assert.equal(isDisplay(mcp("m", "get_menu", "completed", "d")), false);
  assert.equal(messageText({ content: "plain" }), "plain");
  assert.equal(messageText({ content: [{ type: "output_text", text: "a" }, { type: "refusal", refusal: "no" }] }), "a");
});

test("a batch of function calls still running after PROGRESS_AFTER_MS gets one progress line, whatever its tools", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const said = [];
  let finish;
  const running = runBatch([call("c1", "any_tool")], () => new Promise((resolve) => { finish = resolve; }), {
    slow: () => said.push("progress"),
    signal: new AbortController().signal,
  });
  t.mock.timers.tick(PROGRESS_AFTER_MS - 1);
  assert.deepEqual(said, []);
  t.mock.timers.tick(1);
  assert.deepEqual(said, ["progress"]);
  t.mock.timers.tick(10 * PROGRESS_AFTER_MS);
  finish("output");
  assert.deepEqual(await running, ["output"]);
  assert.deepEqual(said, ["progress"]);  // one line for the batch
});

test("a quicker batch, a batch with no line to say and a stopped call get no progress line", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const said = [];
  const slow = () => said.push("progress");
  const signal = new AbortController().signal;
  // The batch ends first: its timer is cleared before runBatch returns, so nothing follows the outputs.
  const quick = runBatch([call("c1"), call("c2")], (fc) => new Promise((resolve) => setTimeout(() => resolve(fc.call_id), PROGRESS_AFTER_MS - 1)), { slow, signal });
  t.mock.timers.tick(PROGRESS_AFTER_MS - 1);
  assert.deepEqual(await quick, ["c1", "c2"]);
  t.mock.timers.tick(2 * PROGRESS_AFTER_MS);
  // No line to say (the box unchecked, or an empty line): no timer.
  let finish;
  const unchecked = runBatch([call("c3")], () => new Promise((resolve) => { finish = resolve; }), { slow: null, signal });
  t.mock.timers.tick(2 * PROGRESS_AFTER_MS);
  finish("x");
  await unchecked;
  // End or a new Start while the tools run: nothing more is sent for the call.
  const scope = new AbortController();
  const stopped = runBatch([call("c4")], () => new Promise((resolve) => { finish = resolve; }), { slow, signal: scope.signal });
  scope.abort();
  t.mock.timers.tick(2 * PROGRESS_AFTER_MS);
  finish("y");
  await stopped;
  assert.deepEqual(said, []);
});
