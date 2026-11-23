// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Needs Node 22.6 or later with type stripping turned on.
import assert from "node:assert/strict";
import test from "node:test";
import { RECORDED_CALL } from "./airline-recorded-call.ts";

// The recorded preview is a call of the real tools: every message in it is one the server relays.
const ALL_LIVE_EVENT_MESSAGES = RECORDED_CALL.flatMap((step) => step.events);
import {
  FAMILIES,
  familyOf,
  formatTime,
  parseLiveEvent,
  shortId,
  summarize,
  type LiveEvent,
} from "../src/liveEvents/liveEvent.ts";
import {
  EMPTY_LOG,
  MAX_ROWS,
  filterRows,
  logReducer,
  missingCount,
  toJsonLines,
  visibleRows,
  type LogState,
} from "../src/liveEvents/log.ts";
import { CallLogStore, EMPTY_CALL_LOG, callLogReducer, type CallLog, type CallLogAction } from "../src/liveEvents/store.ts";
import { LOG_CAPABILITY, logEnabled } from "../src/liveEvents/gate.ts";

function rows(): LiveEvent[] {
  return ALL_LIVE_EVENT_MESSAGES.map((message) => {
    const row = parseLiveEvent(message);
    assert.ok(row, "every sample message parses");
    return row;
  });
}

function row(seq: number, kind = "response.created", extra: Partial<LiveEvent> = {}): LiveEvent {
  return { seq, t: seq, direction: "in", kind, stream: "r1", delegation: null, event: { type: kind }, ...extra };
}

function find(kind: string, from = rows()): LiveEvent {
  const found = from.find((item) => item.kind === kind);
  assert.ok(found, `a ${kind} sample`);
  return found;
}

// ---- Reading a message ---------------------------------------------------------------------------------------------

test("reads every sample message into a row", () => {
  const all = rows();
  assert.equal(all.length, ALL_LIVE_EVENT_MESSAGES.length);
  assert.deepEqual(all.map((item) => item.seq), all.map((_, index) => index + 1));
  assert.equal(all[0].kind, "conversations.create");
  assert.equal(all[0].stream, null);
  assert.equal(all[1].direction, "out");
  assert.equal(all[1].stream, "r1");
  assert.match(all[1].delegation ?? "", /^item_/);
});

test("ignores messages that are not version 1 live-event messages", () => {
  const good = ALL_LIVE_EVENT_MESSAGES[0];
  assert.ok(parseLiveEvent(good));
  assert.equal(parseLiveEvent(null), null);
  assert.equal(parseLiveEvent("live-event"), null);
  assert.equal(parseLiveEvent([good]), null);
  assert.equal(parseLiveEvent({ ...good, type: "agent-task-update" }), null);
  assert.equal(parseLiveEvent({ ...good, v: 2 }), null);
  assert.equal(parseLiveEvent({ ...good, seq: 0 }), null);
  assert.equal(parseLiveEvent({ ...good, seq: 1.5 }), null);
  assert.equal(parseLiveEvent({ ...good, seq: "1" }), null);
  assert.equal(parseLiveEvent({ ...good, t: -1 }), null);
  assert.equal(parseLiveEvent({ ...good, t: Number.NaN }), null);
  assert.equal(parseLiveEvent({ ...good, direction: "sideways" }), null);
  assert.equal(parseLiveEvent({ ...good, kind: "" }), null);
  assert.equal(parseLiveEvent({ ...good, kind: "k".repeat(200) }), null);
  assert.equal(parseLiveEvent({ ...good, event: null }), null);
  assert.equal(parseLiveEvent({ ...good, event: [] }), null);
  assert.equal(parseLiveEvent({ ...good, stream: 7 }), null);
  assert.equal(parseLiveEvent({ ...good, delegation: {} }), null);
});

test("a missing stream or delegation reads as null", () => {
  const without = Object.entries(ALL_LIVE_EVENT_MESSAGES[0]).filter(([key]) => key !== "stream" && key !== "delegation");
  const parsed = parseLiveEvent(Object.fromEntries(without));
  assert.ok(parsed);
  assert.equal(parsed.stream, null);
  assert.equal(parsed.delegation, null);
});

// ---- Families -------------------------------------------------------------------------------------------------------

test("sorts events into the families the filter offers", () => {
  const all = rows();
  assert.equal(familyOf(find("responses.create", all)), "request");
  assert.equal(familyOf(find("conversations.create", all)), "request");
  assert.equal(familyOf(find("response.created", all)), "lifecycle");
  assert.equal(familyOf(find("response.in_progress", all)), "lifecycle");
  assert.equal(familyOf(find("response.completed", all)), "lifecycle");
  assert.equal(familyOf(find("response.output_text.done", all)), "text");
  assert.equal(familyOf(find("response.mcp_call.completed", all)), "tool");
  // A tool call's item is a tool event, and the answer's item is lifecycle.
  assert.equal(familyOf(find("response.output_item.added", all)), "tool");
  assert.equal(familyOf(row(1, "response.output_item.done", { event: { item: { type: "message" } } })), "lifecycle");
  assert.equal(familyOf(row(1, "response.function_call_arguments.delta")), "tool");
  assert.ok(FAMILIES.every((family) => typeof family.label === "string"));
});

test("a failure is an error whatever else it is", () => {
  assert.equal(familyOf(row(1, "error")), "error");
  assert.equal(familyOf(row(1, "response.failed")), "error");
  assert.equal(familyOf(row(1, "response.mcp_call.failed")), "error");
  assert.equal(familyOf(row(1, "response.completed", { event: { response: { status: "failed" } } })), "error");
  assert.equal(familyOf(row(1, "responses.create", { direction: "out", event: {} })), "request");
});

// ---- The line of a row --------------------------------------------------------------------------------------------

test("a request shows the newest line of its input: what the caller said", () => {
  const request = find("responses.create");
  assert.equal(summarize(request), "I need a flight from Boston to Denver tomorrow.");
  const plain = row(1, "responses.create", { direction: "out", event: { input: [{ role: "user", content: "one\n\ntwo " }] } });
  assert.equal(summarize(plain), "two");
});

test("a tool call shows the tool and its arguments", () => {
  const added = find("response.output_item.added");
  assert.equal(summarize(added), 'airline.flights_search {"origin":"Boston","destination":"Denver","date":"2026-11-05"}');
  const text = find("response.output_text.done");
  assert.match(summarize(text), /^Four flights match\. One at /);
});

test("a response shows its status, model and tokens", () => {
  const created = find("response.created");
  assert.equal(summarize(created), "queued · nvidia/nemotron-3.5-super-vl-120b-a12b");
  assert.equal(summarize(find("response.completed")), "completed · 4100 in / 216 out tokens");
  assert.equal(summarize(find("response.in_progress")), "resp_01b6e2c4");
});

test("shows what it can and never throws on an event it does not know", () => {
  assert.equal(summarize(row(1, "response.something.new", { event: { odd: [1, 2] } })), "");
  assert.equal(summarize(row(1, "responses.create", { direction: "out", event: {} })), "");
  assert.equal(summarize(row(1, "response.output_item.added", { event: { item: "not an object" } })), "");
  assert.equal(summarize(row(1, "error", { event: { message: "a repeated result" } })), "a repeated result");
  assert.equal(summarize(row(1, "conversations.create", { event: { id: "conv_46966796915a4159" } })), "conv_46966796");
});

test("a long line is cut to one line", () => {
  const long = row(1, "response.output_text.delta", { event: { delta: "word ".repeat(100) } });
  const line = summarize(long);
  assert.ok(line.length <= 160);
  assert.ok(line.endsWith("…"));
  assert.equal(line.includes("\n"), false);
  assert.equal(summarize(row(1, "response.output_text.delta", { event: { delta: "a\n\n b" } })), "a b");
});

test("ids and times are written short", () => {
  assert.equal(shortId("resp_a5eb2a9b77f74b3d"), "resp_a5eb2a9b");
  assert.equal(shortId("conv_1"), "conv_1");
  assert.equal(shortId("plain"), "plain");
  assert.equal(formatTime(0), "+0.000 s");
  assert.equal(formatTime(158.8404), "+158.840 s");
});

// ---- The log -------------------------------------------------------------------------------------------------------

function fill(session: string, count: number, from = 1): LogState {
  let state: LogState = logReducer(EMPTY_LOG, { type: "started", session });
  for (let seq = from; seq < from + count; seq += 1) state = logReducer(state, { type: "event", session, event: row(seq) });
  return state;
}

test("a call starts an empty log and its events are kept in order", () => {
  const state = fill("s1", 3);
  assert.equal(state.phase, "live");
  assert.deepEqual(state.rows.map((item) => item.seq), [1, 2, 3]);
});

test("an event of a new call starts that call's log", () => {
  const old = fill("s1", 3);
  const next = logReducer(old, { type: "event", session: "s2", event: row(1) });
  assert.equal(next.session, "s2");
  assert.deepEqual(next.rows.map((item) => item.seq), [1]);
});

test("events that arrive before the connection is reported are kept", () => {
  let state = logReducer(EMPTY_LOG, { type: "event", session: "s1", event: row(1) });
  state = logReducer(state, { type: "started", session: "s1" });
  assert.equal(state.rows.length, 1);
  assert.equal(state.phase, "live");
});

test("a different call's start forgets the old log", () => {
  const state = logReducer(fill("s1", 3), { type: "started", session: "s2" });
  assert.equal(state.rows.length, 0);
  assert.equal(state.session, "s2");
});

test("a repeated or out-of-order event is ignored", () => {
  const state = fill("s1", 3);
  assert.equal(logReducer(state, { type: "event", session: "s1", event: row(3) }), state);
  assert.equal(logReducer(state, { type: "event", session: "s1", event: row(2) }), state);
});

test("a hang up leaves the log on screen, and clearing empties it", () => {
  const ended = logReducer(fill("s1", 3), { type: "ended" });
  assert.equal(ended.phase, "ended");
  assert.equal(ended.rows.length, 3);
  assert.equal(logReducer(EMPTY_LOG, { type: "ended" }), EMPTY_LOG);
  const cleared = logReducer(ended, { type: "clear" });
  assert.deepEqual([cleared.rows.length, cleared.dropped], [0, 0]);
  assert.equal(logReducer(cleared, { type: "clear" }), cleared);
});

test("keeps the newest events and counts the ones it dropped", () => {
  let state = fill("s1", MAX_ROWS);
  state = logReducer(state, { type: "event", session: "s1", event: row(MAX_ROWS + 1) });
  state = logReducer(state, { type: "event", session: "s1", event: row(MAX_ROWS + 2) });
  assert.equal(state.rows.length, MAX_ROWS);
  assert.equal(state.dropped, 2);
  assert.equal(state.rows[0].seq, 3);
  assert.equal(state.rows.at(-1)?.seq, MAX_ROWS + 2);
});

test("shows the log of the current call, or the last one between calls", () => {
  const state = fill("s1", 2);
  assert.equal(visibleRows(state, "s1").length, 2);
  assert.equal(visibleRows(state, "").length, 2);
  assert.equal(visibleRows(state, "s2").length, 0);
});

test("the filter hides a family and keeps the order", () => {
  const all = rows();
  const noRequests = filterRows(all, new Set(["request"]));
  assert.ok(noRequests.every((item) => familyOf(item) !== "request"));
  assert.deepEqual(noRequests.map((item) => item.seq), [...noRequests.map((item) => item.seq)].sort((a, b) => a - b));
  assert.equal(filterRows(all, new Set()).length, all.length);
  assert.equal(filterRows(all, new Set(FAMILIES.map((family) => family.id))).length, 0);
});

test("copying writes one JSON object per line, and gaps in seq are counted", () => {
  const all = rows();
  const lines = toJsonLines(all).split("\n");
  assert.equal(lines.length, all.length);
  assert.deepEqual(JSON.parse(lines[2]), all[2]);
  assert.equal(missingCount(all), 0);
  assert.equal(missingCount([row(1), row(2), row(6)]), 3);
  assert.equal(missingCount([row(5)]), 0);
});

test("the recorded call's events cover each step and stay in order", () => {
  const seqs = ALL_LIVE_EVENT_MESSAGES.map((message) => (message as { seq: number }).seq);
  assert.deepEqual(seqs, [...seqs].sort((a, b) => a - b));
  assert.equal(new Set(seqs).size, seqs.length);
  assert.ok(RECORDED_CALL.every((step) => step.id === "connect" || step.events.length > 0));
});

// ---- The call's store -----------------------------------------------------------------------------------------------

function play(...actions: CallLogAction[]): CallLog {
  return actions.reduce(callLogReducer, EMPTY_CALL_LOG);
}

const say = (message: unknown): CallLogAction => ({ type: "message", message });

test("a call starts at its connection and its events fill the log", () => {
  let state = play({ type: "connected" });
  assert.deepEqual([state.call, state.phase, state.log.rows.length], [1, "live", 0]);
  for (const message of ALL_LIVE_EVENT_MESSAGES.slice(0, 9)) state = callLogReducer(state, say(message));
  assert.equal(state.log.rows.length, 9);
  // A second connection report in the same call changes nothing.
  assert.equal(callLogReducer(state, { type: "connected" }), state);
});

test("events that arrive before the connection is reported are kept, and the report starts nothing new", () => {
  let state = play(say(ALL_LIVE_EVENT_MESSAGES[0]), say(ALL_LIVE_EVENT_MESSAGES[1]));
  assert.deepEqual([state.call, state.phase, state.log.rows.length], [1, "live", 2]);
  state = callLogReducer(state, { type: "connected" });
  assert.deepEqual([state.call, state.log.rows.length], [1, 2]);
});

test("a hang up leaves the log on screen, and the next call starts empty", () => {
  let state = play({ type: "connected" }, ...ALL_LIVE_EVENT_MESSAGES.slice(0, 9).map(say), { type: "disconnected" });
  assert.deepEqual([state.phase, state.log.phase, state.log.rows.length], ["ended", "ended", 9]);
  assert.equal(callLogReducer(state, { type: "disconnected" }), state);
  state = callLogReducer(state, { type: "connected" });
  assert.deepEqual([state.call, state.phase, state.log.rows.length], [2, "live", 0]);
  // The first event of a call starts it even when the connection was not reported.
  const unreported = callLogReducer(play({ type: "connected" }, { type: "disconnected" }), say(ALL_LIVE_EVENT_MESSAGES[0]));
  assert.deepEqual([unreported.call, unreported.phase, unreported.log.rows.length], [2, "live", 1]);
  // A new bot's first event starts a new call even when neither the hang up nor the next connection was reported.
  const missed = play({ type: "connected" }, ...ALL_LIVE_EVENT_MESSAGES.slice(0, 9).map(say), say(ALL_LIVE_EVENT_MESSAGES[0]));
  assert.deepEqual([missed.call, missed.phase, missed.log.rows.length], [2, "live", 1]);
});

test("a repeated, out-of-order or foreign message changes nothing, and clearing keeps the call", () => {
  const state = play({ type: "connected" }, ...ALL_LIVE_EVENT_MESSAGES.slice(0, 9).map(say));
  assert.equal(callLogReducer(state, say(ALL_LIVE_EVENT_MESSAGES[8])), state);
  assert.equal(callLogReducer(state, say(ALL_LIVE_EVENT_MESSAGES[2])), state);
  for (const message of [{ type: "agent-task-update", task_id: "t1" }, { type: "live-event", v: 2 }, null, "text", 5, []]) {
    assert.equal(callLogReducer(state, say(message)), state);
  }
  const cleared = callLogReducer(state, { type: "clear" });
  assert.deepEqual([cleared.log.rows.length, cleared.phase, cleared.call], [0, "live", 1]);
  assert.equal(callLogReducer(cleared, { type: "clear" }), cleared);
});

test("the store tells its listeners of a change, and not of a message that changes nothing", () => {
  const store = new CallLogStore();
  let calls = 0;
  const stop = store.subscribe(() => {
    calls += 1;
  });
  store.dispatch({ type: "disconnected" });
  assert.equal(calls, 0);
  store.dispatch({ type: "connected" });
  assert.equal(calls, 1);
  store.dispatch({ type: "message", message: { type: "agent-task-update" } });
  assert.equal(calls, 1);
  store.dispatch({ type: "message", message: ALL_LIVE_EVENT_MESSAGES[0] });
  assert.equal(calls, 2);
  stop();
  store.dispatch({ type: "disconnected" });
  assert.equal(calls, 2);
  assert.equal(store.getSnapshot().phase, "ended");
});

// ---- When the tab shows ----------------------------------------------------------------------------------------------

test("the Event log applies only to an example that lists the capability", () => {
  assert.equal(LOG_CAPABILITY, "event_log");
  assert.equal(logEnabled({ capabilities: ["event_log"] }), true);
  for (const other of [{ capabilities: ["webcam", "attachments"] }, { capabilities: [] }, {}, null, undefined]) {
    assert.equal(logEnabled(other), false);
  }
});
