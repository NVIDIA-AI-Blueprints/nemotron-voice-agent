// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client mode: this page does each delegated piece of work itself. `session.delegation.created` says only that the
// frontend wants help, never what with, so the page rebuilds the request from its transcript and the cart, runs
// stateless rounds through `POST /v1/live/delegate` with the tools in this browser, and sends the final answer as one
// `session.commentary.append`. Delegations run one at a time (ClientQueue), so two loops never change the cart at
// once, and every step checks the call's stop signal so a loop never outlives its session.

import { finalAnswers, progressLines } from "./delegations.js";

/** Rounds that may run tools; one more round then answers with the last tools' outputs. */
export const TOOL_ROUNDS = 6;
/** The silent note that work started; the cart demo's own words. */
export const STARTED_LINE = "Working on that now. Nothing has been ordered yet.";
/** Said when no answer came and no tool ran. */
export const FAILED_LINE = "I could not get that done, and nothing was changed. Please ask me again.";
/** Said when no answer came after a tool ran: never "could not complete", since an order may have been placed. */
export const PARTLY_DONE_LINE = "I may have only partly finished that, and some of it may already be done. " +
  "Please check before asking again.";
/** An append holds at most 500 tokens; this many characters stay well inside it. */
export const MAX_APPEND_CHARS = 1200;

/**
 * The page's own record of the call: what the caller said and what the frontend spoke, from the transcript deltas.
 * In client mode the announcement carries no task text, so this and the application's state are all the page has.
 */
export class Transcript {
  constructor(now = () => performance.now()) {
    this.lines = [];  // [{role, text, at}]
    this.now = now;
  }

  /** A delta joins the last line when it is the same speaker's within 1.5 s; at most 40 lines are kept. */
  record(role, text) {
    const last = this.lines[this.lines.length - 1];
    if (last && last.role === role && this.now() - last.at < 1500) {
      last.text += text;
      last.at = this.now();
    } else {
      this.lines.push({ role, text, at: this.now() });
    }
    if (this.lines.length > 40) this.lines.shift();
  }

  /** The newest `turns` lines as "role: text", leaving out lines of seven characters or fewer. */
  context(turns = 8) {
    return this.lines.slice(-turns)
      .map((line) => `${line.role}: ${line.text.trim()}`)
      .filter((line) => line.length > 7)
      .join("\n");
  }
}

/** The request the page rebuilds for one delegation, from its transcript and the cart. */
export function rebuiltRequest(transcript, cart) {
  return `Recent voice transcript (may contain errors or corrections):\n${transcript || "(nothing transcribed yet)"}\n\n` +
    `Current cart state: ${JSON.stringify(cart)}\n\n` +
    "The voice assistant delegated the user's latest request to you. Handle it and return " +
    "a short result suitable for reading aloud.";
}

/** Runs tasks one at a time, in the order they were given; a task that fails does not stop the next. */
export class ClientQueue {
  constructor() {
    this.tail = Promise.resolve();
    this.size = 0;  // tasks running or waiting
  }

  /** Whether a task is running or waiting, so a new one waits. */
  get busy() {
    return this.size > 0;
  }

  /** Run `task` after every task given before it; returns its result. */
  run(task) {
    this.size += 1;
    const result = this.tail.then(task).finally(() => { this.size -= 1; });
    this.tail = result.catch(() => {});
    return result;
  }
}

/**
 * One backend round through the console route: the returned function sends the delegation's history and returns
 * `{response: {id, output}, sent_request}`. It throws an Error with the route's status and detail, a network error,
 * or for an answer with no output.
 */
export function roundPoster({ fetch, headers, model, instructions, tools }) {
  return async (input, signal) => {
    const response = await fetch("/v1/live/delegate", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers() },
      body: JSON.stringify({ input, model, instructions: instructions || undefined, tools }),
      signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = data?.detail || data?.error?.message || "no detail";
      throw new Error(`POST /v1/live/delegate answered HTTP ${response.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    if (!Array.isArray(data?.response?.output)) throw new Error("POST /v1/live/delegate answered with no response.output");
    return data;
  };
}

/**
 * One client delegation, start to end. Returns `{status, answer}`: status `answered`, `no answer`, `failed`,
 * `stopped` (the session ended) or `unsent` (the channel could not take the answer).
 *
 * `delegationId`: the delegation's id, or null for the page's own typed request. `history`: an empty list the loop
 * fills with the delegation's items. `round(input, signal)`: one route round (roundPoster). `runTool(call)`: runs one
 * function call in this browser and returns its `function_call_output` item. `send(event)`: false when the data
 * channel is not open. `view`: `stage(key, spec)`, `status(text, tone)` and `log(entry)` draw the card and the log.
 */
export async function runClientDelegation({
  delegationId, transcript, history, cart, round, runTool, send, nextEventId, signal, view, model,
}) {
  let toolRan = false;
  const stopped = () => {
    view.stage("stopped", {
      cls: "warn",
      label: "stopped: the session ended",
      text: toolRan
        ? "End or a new Start stopped this delegation. Tools that already ran stay done; nothing more is sent."
        : "End or a new Start stopped this delegation before it finished; nothing more is sent.",
    });
    view.status("stopped: the session ended", "ended");
    return { status: "stopped", answer: "" };
  };
  if (signal.aborted) return stopped();
  view.status("running", "running");

  send({ type: "session.thinking.append", event_id: nextEventId("progress"), delegation_id: delegationId, content: STARTED_LINE });
  const request = rebuiltRequest(transcript, cart());
  history.push({ type: "message", role: "user", content: [{ type: "input_text", text: request }] });
  view.stage("reconstruct", {
    cls: "warn",
    label: "client reconstructs the request",
    text: "session.delegation.created carried NO task text — only an id, target and offset.\n" +
      "We rebuild intent from our own transcript plus the cart, as this delegation's own history.",
    payload: request,
  });

  let answer = "";
  let failure = null;
  for (let number = 1; number <= TOOL_ROUNDS + 1; number += 1) {
    if (signal.aborted) return stopped();
    let data;
    try {
      data = await round(history, signal);
    } catch (error) {
      if (signal.aborted) return stopped();
      failure = error;
      break;
    }
    if (signal.aborted) return stopped();
    const output = data.response.output;
    history.push(...output);
    view.log({
      cat: "delegation", dir: "out", actor: "client",
      type: `POST /v1/live/delegate (round ${number})`,
      text: `our own backend call → ${model} · the frontend sees none of this`,
      payload: { sent_request: data.sent_request, response_id: data.response.id },
    });
    if (number === 1) {
      view.stage("ourcall", {
        cls: "tool",
        label: `client → our backend · POST /v1/live/delegate (${model})`,
        text: "We chose this model, prompt and toolset. None of it was declared to the session.\n" +
          "This delegation keeps its own history and sends it whole each round.",
        payload: { model: data.sent_request?.model, tools: data.sent_request?.tools },
      });
    }
    const notes = progressLines(output);
    if (notes.length) view.stage("progress", { cls: "hosted", label: "progress (commentary, not the answer)", text: notes.join("\n") });

    const calls = output.filter((item) => item.type === "function_call");
    if (!calls.length || number > TOOL_ROUNDS) {
      answer = finalAnswers(output).join("\n\n");
      if (calls.length) {
        view.stage("notrun", {
          cls: "warn",
          label: `the answer round asked for tools: not run (round ${number})`,
          text: `${calls.map((call) => `${call.name}(${call.arguments})`).join("\n")}\n\n` +
            `Rounds 1-${TOOL_ROUNDS} may run tools; this one only answers with their outputs.`,
        });
      }
      break;
    }

    view.stage(`ocall_${number}`, {
      cls: "tool",
      label: `our backend asks for tools (round ${number})`,
      text: `${calls.map((call) => `${call.name}(${call.arguments})`).join("\n")}\n\n` +
        "These never reach the session. There is no response.item.create here.",
    });
    const outputs = await Promise.all(calls.map((call) => runTool(call)));
    toolRan = true;
    if (signal.aborted) return stopped();
    history.push(...outputs);
    view.stage(`oresult_${number}`, {
      cls: "tool",
      label: `client → our backend · function_call_output (round ${number})`,
      text: number === TOOL_ROUNDS
        ? "Added to this delegation's history; the next round is the last and only answers with them."
        : "Added to this delegation's history and sent with the next round.",
      payload: outputs.map((item) => ({ call_id: item.call_id, output: item.output })),
    });
  }

  let status = "answered";
  if (failure) {
    status = "failed";
    view.stage("bkerr", { cls: "warn", label: "our backend failed", text: String(failure.message || failure) });
  } else if (!answer) {
    status = "no answer";
  }
  const line = status === "answered" ? answer : (toolRan ? PARTLY_DONE_LINE : FAILED_LINE);
  view.stage("answer", {
    cls: status === "answered" ? "answer" : "warn",
    label: status === "answered" ? `our backend's verified result (${model})` : "no answer: the caller hears instead",
    text: line,
  });

  // The only thing the frontend receives; client delegation lets the page validate, redact or drop it first.
  if (signal.aborted) return stopped();
  const sent = send({
    type: "session.commentary.append",
    event_id: nextEventId("result"),
    delegation_id: delegationId,
    content: line.slice(0, MAX_APPEND_CHARS),
  });
  if (!sent) {
    view.status("not sent: the data channel is closed", "failed");
    return { status: "unsent", answer: line };
  }
  view.stage("handoff", {
    cls: "spoken",
    label: "client → frontend · session.commentary.append",
    text: `delegation_id: ${delegationId ?? "null (locally triggered, not a server delegation)"}\n` +
      "Commentary is spoken aloud; the frontend phrases it in its own voice.\n" +
      "session.thinking.append would deliver the same facts silently instead.",
  });
  view.status(status === "answered" ? "complete" : status, status === "answered" ? "done" : "failed");
  return { status, answer: line };
}
