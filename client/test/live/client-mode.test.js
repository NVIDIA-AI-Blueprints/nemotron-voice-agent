// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The client-mode loop (client/public/live/client-mode.js), with route answers shaped as POST /v1/live/delegate
// gives them (src/live/extras.py). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ClientQueue,
  FAILED_LINE,
  PARTLY_DONE_LINE,
  STARTED_LINE,
  TOOL_ROUNDS,
  Transcript,
  rebuiltRequest,
  roundPoster,
  runClientDelegation,
} from "../../public/live/client-mode.js";

const message = (text, phase) => ({
  id: `msg_${text.length}`, type: "message", status: "completed", role: "assistant", phase,
  content: [{ type: "output_text", text, annotations: [], logprobs: [] }],
});
const call = (callId, name = "get_menu", args = "{}") => ({ id: `fc_${callId}`, type: "function_call", status: "completed", call_id: callId, name, arguments: args });
const answer = (output) => ({ response: { id: "resp_x", output }, sent_request: { model: "m", tools: ["get_menu"] } });

/**
 * The loop's ports, recorded. `rounds`: what each round answers, in order: an output list, an Error to throw, or a
 * function of (input, signal) returning either.
 */
function harness(rounds, { sendResult = true, delegationId = "item_1" } = {}) {
  const record = { inputs: [], tools: [], sent: [], stages: {}, statuses: [], logs: [] };
  const scope = new AbortController();
  let index = 0;
  const port = {
    delegationId,
    transcript: "user: what teas do you have?",
    history: [],
    model: "backend-model",
    cart: () => ({ lines: [], total: 0 }),
    round: async (input, signal) => {
      record.inputs.push(structuredClone(input));
      const next = rounds[index++];
      const value = typeof next === "function" ? await next(input, signal) : next;
      if (value instanceof Error) throw value;
      return answer(value);
    },
    runTool: async (fc) => {
      record.tools.push(fc.name);
      return { type: "function_call_output", call_id: fc.call_id, output: JSON.stringify({ ran: fc.name }) };
    },
    send: (event) => { record.sent.push(event); return event.type === "session.commentary.append" ? sendResult : true; },
    nextEventId: (prefix) => `${prefix}_1`,
    signal: scope.signal,
    view: {
      stage: (key, spec) => { record.stages[key] = spec; },
      status: (text, tone) => record.statuses.push([text, tone]),
      log: (entry) => record.logs.push(entry),
    },
  };
  return { port, record, scope };
}

const commentary = (record) => record.sent.filter((event) => event.type === "session.commentary.append");

test("a delegation: its own history, the tools it asks for, then its final_answer as commentary with its id", async () => {
  const { port, record } = harness([
    [message("Looking at the menu.", "commentary"), call("c1")],
    [message("We have chai and green tea.", "final_answer")],
  ]);
  const outcome = await runClientDelegation(port);
  assert.deepEqual(outcome, { status: "answered", answer: "We have chai and green tea." });
  assert.deepEqual(record.sent[0], { type: "session.thinking.append", event_id: "progress_1", delegation_id: "item_1", content: STARTED_LINE });
  assert.deepEqual(commentary(record), [
    { type: "session.commentary.append", event_id: "result_1", delegation_id: "item_1", content: "We have chai and green tea." },
  ]);
  // Round 1 sends only the rebuilt request; round 2 adds the round's items and the tool's output.
  assert.deepEqual(record.inputs.map((input) => input.map((item) => item.type)), [
    ["message"],
    ["message", "message", "function_call", "function_call_output"],
  ]);
  assert.match(record.inputs[0][0].content[0].text, /user: what teas do you have\?/);
  assert.equal(record.stages.progress.text, "Looking at the menu.");  // commentary is progress, never the answer
  assert.equal(record.stages.answer.text, "We have chai and green tea.");
  assert.deepEqual(record.statuses.at(-1), ["complete", "done"]);
  assert.equal(port.history.length, 5);
});

test("the answer round: round 6's tool outputs are sent once more, and no tool it asks for runs", async () => {
  const rounds = Array.from({ length: TOOL_ROUNDS }, (_, i) => [call(`c${i + 1}`, "manage_cart")]);
  const { port, record } = harness([...rounds, [call("c7", "place_order")]]);
  const outcome = await runClientDelegation(port);
  assert.equal(record.tools.length, TOOL_ROUNDS);
  assert.equal(record.inputs.length, TOOL_ROUNDS + 1);
  assert.equal(record.inputs.at(-1).at(-1).call_id, `c${TOOL_ROUNDS}`);  // round 6's output reached round 7
  assert.ok(!record.tools.includes("place_order"));
  assert.match(record.stages.notrun.label, /not run/);
  // A tool ran and no answer came: "partly finished", never "could not complete".
  assert.deepEqual(outcome, { status: "no answer", answer: PARTLY_DONE_LINE });
  assert.equal(commentary(record)[0].content, PARTLY_DONE_LINE);
  assert.doesNotMatch(PARTLY_DONE_LINE, /could not complete/i);

  const answered = harness([...rounds, [message("Your order is placed.", "final_answer")]]);
  assert.deepEqual(await runClientDelegation(answered.port), { status: "answered", answer: "Your order is placed." });
});

test("a failed round speaks a failure line with the delegation's id instead of leaving silence", async () => {
  const before = harness([new Error("POST /v1/live/delegate answered HTTP 502: The backend could not complete the round")]);
  assert.deepEqual(await runClientDelegation(before.port), { status: "failed", answer: FAILED_LINE });
  assert.deepEqual(commentary(before.record).map((event) => [event.delegation_id, event.content]), [["item_1", FAILED_LINE]]);
  assert.match(before.record.stages.bkerr.text, /HTTP 502/);
  assert.deepEqual(before.record.statuses.at(-1), ["failed", "failed"]);

  const after = harness([[call("c1", "place_order")], new Error("POST /v1/live/delegate answered HTTP 429: Too many console rounds")]);
  assert.deepEqual(await runClientDelegation(after.port), { status: "failed", answer: PARTLY_DONE_LINE });

  const empty = harness([[]]);
  assert.deepEqual(await runClientDelegation(empty.port), { status: "no answer", answer: FAILED_LINE });
});

test("the session's stop signal: nothing starts, a round in flight ends the loop, tool outputs are not sent", async () => {
  const early = harness([[message("never", "final_answer")]]);
  early.scope.abort();
  assert.equal((await runClientDelegation(early.port)).status, "stopped");
  assert.deepEqual(early.record.sent, []);
  assert.deepEqual(early.record.inputs, []);

  const inFlight = harness([
    (input, signal) => new Promise((resolve, reject) => signal.addEventListener("abort", () => reject(new Error("AbortError")))),
    [message("never", "final_answer")],
  ]);
  const running = runClientDelegation(inFlight.port);
  await new Promise((resolve) => setTimeout(resolve, 5));
  inFlight.scope.abort();
  assert.equal((await running).status, "stopped");
  assert.equal(inFlight.record.inputs.length, 1);
  assert.deepEqual(commentary(inFlight.record), []);
  assert.deepEqual(inFlight.record.statuses.at(-1), ["stopped: the session ended", "ended"]);

  const duringTools = harness([[call("c1", "place_order")], [message("never", "final_answer")]]);
  duringTools.port.runTool = async (fc) => {
    duringTools.scope.abort();  // End while the order is placed
    return { type: "function_call_output", call_id: fc.call_id, output: "{}" };
  };
  assert.equal((await runClientDelegation(duringTools.port)).status, "stopped");
  assert.equal(duringTools.record.inputs.length, 1);
  assert.match(duringTools.record.stages.stopped.text, /Tools that already ran stay done/);
});

test("an answer the data channel could not take marks the card not sent", async () => {
  const { port, record } = harness([[message("Done.", "final_answer")]], { sendResult: false });
  assert.deepEqual(await runClientDelegation(port), { status: "unsent", answer: "Done." });
  assert.deepEqual(record.statuses.at(-1), ["not sent: the data channel is closed", "failed"]);
  assert.equal(record.stages.handoff, undefined);
});

test("a typed request names no delegation in its appends", async () => {
  const { port, record } = harness([[message("Hi.", "final_answer")]], { delegationId: null });
  await runClientDelegation(port);
  assert.deepEqual(record.sent.map((event) => event.delegation_id), [null, null]);
});

test("delegations run one at a time, each with its own history", async () => {
  const queue = new ClientQueue();
  const order = [];
  let release;
  const first = harness([() => new Promise((resolve) => { release = () => resolve([message("First.", "final_answer")]); })]);
  const second = harness([[call("c9")], [message("Second.", "final_answer")]]);
  const one = queue.run(async () => { order.push("first starts"); const outcome = await runClientDelegation(first.port); order.push("first ends"); return outcome; });
  assert.equal(queue.busy, true);
  const two = queue.run(async () => { order.push("second starts"); return runClientDelegation(second.port); });
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.deepEqual(order, ["first starts"]);  // the second waits for the first
  release();
  assert.equal((await one).answer, "First.");
  assert.equal((await two).answer, "Second.");
  assert.deepEqual(order, ["first starts", "first ends", "second starts"]);
  assert.equal(queue.busy, false);
  // The second delegation's first round carries only its own request, not the first's items.
  assert.equal(second.record.inputs[0].length, 1);
  assert.equal(first.port.history.length, 2);
  // A task that throws does not stop the next.
  const failing = queue.run(async () => { throw new Error("boom"); });
  await assert.rejects(failing, /boom/);
  assert.equal(await queue.run(async () => "next"), "next");
});

test("the transcript joins a speaker's deltas within 1.5 s and keeps the newest lines", () => {
  let clock = 0;
  const transcript = new Transcript(() => clock);
  transcript.record("user", "What teas ");
  clock = 1000;
  transcript.record("user", "do you have?");
  clock = 3000;
  transcript.record("assistant", "One moment.");
  clock = 3100;
  transcript.record("user", "k");  // "user: k", seven characters: left out of the context
  assert.equal(transcript.context(), "user: What teas do you have?\nassistant: One moment.");
  for (let i = 0; i < 45; i += 1) { clock += 2000; transcript.record(i % 2 ? "user" : "assistant", `line number ${i}`); }
  assert.equal(transcript.lines.length, 40);
  assert.equal(transcript.context(2), "user: line number 43\nassistant: line number 44");
});

test("the rebuilt request holds the transcript and the cart", () => {
  const text = rebuiltRequest("user: two lattes", { lines: [], total: 0 });
  assert.match(text, /^Recent voice transcript \(may contain errors or corrections\):\nuser: two lattes\n\n/);
  assert.match(text, /Current cart state: \{"lines":\[\],"total":0\}/);
  assert.match(rebuiltRequest("", {}), /\(nothing transcribed yet\)/);
});

test("a round posts the delegation's history with the page's settings and reads the route's answer", async () => {
  const requests = [];
  const reply = (status, body) => async (url, init) => {
    requests.push({ url, init });
    return { ok: status < 300, status, json: async () => body };
  };
  const scope = new AbortController();
  const ok = roundPoster({ fetch: reply(200, answer([message("Hi.", "final_answer")])), headers: () => ({ Authorization: "Bearer t+1" }), model: "m", instructions: "", tools: [{ type: "function", name: "get_menu" }] });
  const data = await ok([{ type: "message", role: "user", content: [] }], scope.signal);
  assert.equal(data.response.output.length, 1);
  const [{ url, init }] = requests;
  assert.equal(url, "/v1/live/delegate");
  assert.equal(init.method, "POST");
  assert.equal(init.headers.Authorization, "Bearer t+1");
  assert.equal(init.signal, scope.signal);
  assert.deepEqual(JSON.parse(init.body), { input: [{ type: "message", role: "user", content: [] }], model: "m", tools: [{ type: "function", name: "get_menu" }] });

  const refused = roundPoster({ fetch: reply(429, { detail: "Too many console rounds: at most 2 at once" }), headers: () => ({}), model: "m", tools: [] });
  await assert.rejects(refused([], scope.signal), /HTTP 429: Too many console rounds/);
  const hollow = roundPoster({ fetch: reply(200, { response: {} }), headers: () => ({}), model: "m", tools: [] });
  await assert.rejects(hollow([], scope.signal), /no response\.output/);
});
