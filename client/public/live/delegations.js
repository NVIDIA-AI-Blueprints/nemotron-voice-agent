// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The delegation cards' reading of the `response.event` envelopes, by structure and never by text, as the frontend
// adapter reads them (src/examples/frontend_backend_live/cascade/delegations.py). A `message` with phase `commentary`
// is progress; `final_answer`, or no phase, is an answer. A completed `mcp_call` named `show` is the display. A
// `previous_response_id` on `response.in_progress` names the older response whose work this delegation takes over.
// Streamed text (`response.output_text.delta`) is never taken as the answer.

/** The IM's display tool (src/delegation_core/im/tools.py): its `mcp_call` carries what the screen shows. */
export const DISPLAY_TOOL = "show";

/** The text of a message item: its `output_text` parts, joined; a plain string content is taken whole. */
export function messageText(item) {
  const content = item.content;
  if (typeof content === "string") return content;
  return (content || []).filter((part) => part?.type === "output_text").map((part) => part.text || "").join("");
}

/** Whether a message item is progress: phase `commentary`. A message with no phase is an answer. */
export const isProgress = (item) => item.phase === "commentary";

/** Whether an item is the `show` call's `mcp_call`, completed or not. */
const isShow = (item) => item.type === "mcp_call" && item.name === DISPLAY_TOOL;
/** Whether an item is the display: the `show` call's `mcp_call`, once it completed (a failed one carries its error). */
export const isDisplay = (item) => isShow(item) && item.status === "completed";

/** The answers among `items`: each completed message that is not progress, as text. */
export function finalAnswers(items) {
  return items
    .filter((item) => item.type === "message" && (item.status ?? "completed") === "completed" && !isProgress(item))
    .map(messageText)
    .filter((text) => text.trim());
}

/** The progress lines among `items`: each completed `commentary` message, as text. */
export function progressLines(items) {
  return items
    .filter((item) => item.type === "message" && (item.status ?? "completed") === "completed" && isProgress(item))
    .map(messageText)
    .filter((text) => text.trim());
}

const safeParse = (text) => { try { return JSON.parse(text); } catch { return text; } };

/**
 * Every delegation of one call, read from its events.
 *
 * `view` draws: `created(card)`, `stage(card, key, {cls, label, text, payload})` (a key drawn again replaces its
 * stage), `status(card, text, tone)` with tone `running`, `waiting`, `done`, `ended` or `failed`, and
 * `flow(card, responseId)`. `clientTools` are the function names this page runs.
 */
export class DelegationBook {
  constructor(view, clientTools = []) {
    this.view = view;
    this.clientTools = new Set(clientTools);
    this.cards = new Map();       // delegation id -> card
    this.byResponse = new Map();  // response id -> card
    this.named = new Map();       // a response named as taken over before it reached this page -> the taker's id
  }

  /** `session.delegation.created`: the delegation's card. `typed`: the page's own typed request. */
  announce(event, { typed = false } = {}) {
    const delegation = event.delegation || {};
    const card = {
      id: delegation.id,
      target: delegation.target,
      offsetMs: event.offset_ms,
      typed,
      responseId: delegation.response_id || null,
      response: null,     // the newest response: {id, finals, calls}
      open: true,         // no ending yet
      answers: [],        // every final_answer, in order
      progress: [],       // every commentary line, in order
      displays: [],       // every display
      toolRan: false,     // a function call or a backend tool ran for it
      taker: null,        // the delegation the IM named as taking over its work
      tookOver: [],       // the delegations whose work it took over
      awaitsTaker: false, // it ended with no answer; a taker's naming may still come
    };
    this.cards.set(card.id, card);
    this.view.created(card);
    if (card.responseId) this.byResponse.set(card.responseId, card);
    return card;
  }

  /**
   * One `response.event` envelope. Returns what the page does next: `{run: calls}` when the response ended waiting
   * for the page's function calls, `{answered: true}` when the delegation answered, else null. An event of a
   * delegation this call never announced is not read.
   */
  read(envelope) {
    const card = this.cards.get(envelope.delegation_id);
    const event = envelope.event || {};
    if (!card) return null;
    const response = event.response || {};
    switch (event.type) {
      case "response.created":
        this.opened(card, response);
        return null;
      case "response.queued":
        if (card.open) this.view.status(card, "queued: waits for its turn", "waiting");
        return null;
      case "response.in_progress":
        this.runs(card, response);
        return null;
      case "response.output_item.added":
        this.added(card, event);
        return null;
      case "response.function_call_arguments.delta": {
        const key = `call_pending_${event.output_index}`;
        card.arguments = card.arguments || {};
        card.arguments[key] = (card.arguments[key] || "") + (event.delta || "");
        this.view.stage(card, key, { cls: "tool", text: card.arguments[key] });
        return null;
      }
      case "response.reasoning_summary_text.delta":
        card.reasoning = (card.reasoning || "") + (event.delta || "");
        this.view.stage(card, "reason", { cls: "reason", label: "backend reasoning (summary)", text: card.reasoning });
        return null;
      case "response.output_item.done":
        this.item(card, event.item || {}, event.output_index);
        return null;
      case "response.completed":
      case "response.failed":
      case "response.incomplete":
        return this.terminal(card, event.type, response);
      default:
        return null;  // deltas and the rest: the event log shows them
    }
  }

  opened(card, response) {
    card.response = { id: response.id || null, finals: 0, calls: [] };
    if (response.id) {
      card.responseId = response.id;
      this.byResponse.set(response.id, card);
      this.view.flow(card, response.id);
      const taker = this.named.get(response.id);  // its taker's naming came first, on the taker's own stream
      if (taker !== undefined && taker !== card.id) {
        this.named.delete(response.id);
        this.handOver(card, taker);
      }
    }
    card.open = true;
    card.awaitsTaker = false;
    this.view.status(card, response.status === "in_progress" ? "running" : "queued: waits for its turn",
      response.status === "in_progress" ? "running" : "waiting");
  }

  runs(card, response) {
    if (!card.response) this.opened(card, response);
    this.view.status(card, "running", "running");
    const previous = response.previous_response_id;
    if (!previous) return;
    const older = this.byResponse.get(previous);
    if (older === undefined) {
      this.named.set(previous, card.id);  // the response it names has not reached this page yet
    } else if (older !== card) {
      this.handOver(older, card.id);
    }
  }

  handOver(older, takerId) {
    if (older.taker !== null) return;
    older.taker = takerId;
    const taker = this.cards.get(takerId);
    if (taker) {
      taker.tookOver.push(older.id);
      this.view.stage(taker, "took", {
        cls: "hosted",
        label: "takes over earlier work",
        text: `The IM named this request as carrying the work of ${taker.tookOver.join(", ")} too (previous_response_id).`,
      });
    }
    this.view.stage(older, "handover", {
      cls: "hosted",
      label: `handed over → ${takerId}`,
      text: `A newer request joined this work; it continues there (named by the IM in previous_response_id).\n` +
        "This delegation's own response ends with the handover's line, which is never said.",
    });
    if (older.awaitsTaker) {  // it already ended with no answer: the naming says why
      older.awaitsTaker = false;
      this.view.status(older, `handed over → ${takerId}`, "ended");
    }
  }

  added(card, event) {
    const item = event.item || {};
    if (item.type === "function_call") {
      this.view.stage(card, `call_pending_${event.output_index}`, {
        cls: "tool", label: `backend is composing a call to ${item.name}`, text: "streaming arguments…",
      });
    } else if (item.type === "mcp_call") {
      this.view.stage(card, `mcp_${item.id}`, {
        cls: "hosted",
        label: isShow(item) ? "display (show) …" : `backend runs ${item.server_label} · ${item.name} …`,
        text: `${item.name}(${item.arguments || ""})`,
      });
    }
  }

  item(card, item, outputIndex) {
    if (item.type === "message" && (item.status ?? "completed") === "completed") {
      const text = messageText(item);
      if (isProgress(item)) {
        card.progress.push(text);
        this.view.stage(card, card.taker ? "handover_line" : "progress", card.taker
          ? { cls: "hosted", label: "the handover's line (commentary, never said)", text }
          : { cls: "hosted", label: "progress (commentary, not the answer)", text: card.progress.slice(-3).join("\n") });
      } else {
        card.answers.push(text);
        if (card.response) card.response.finals += 1;
        this.view.stage(card, "answer", {
          cls: "answer",
          label: card.answers.length > 1
            ? `answers (final_answer, one per task: ${card.answers.length})`
            : "answer (final_answer)",
          text: card.answers.join("\n\n"),
        });
      }
    } else if (item.type === "function_call") {
      card.toolRan = true;
      if (card.response) card.response.calls.push(item);
      const known = this.clientTools.has(item.name);
      this.view.stage(card, `call_pending_${outputIndex}`, null);  // the composing stage gives way to the call
      this.view.stage(card, `call_${item.call_id}`, known
        ? {
          cls: "tool",
          label: `backend → client · function_call ${item.name}`,
          text: `call_id: ${item.call_id}\nThe backend asked this page to run it; it came inside a response.event envelope.`,
          payload: { call_id: item.call_id, name: item.name, arguments: safeParse(item.arguments) },
        }
        : { cls: "warn", label: `unknown function ${item.name}`, text: "This page has no implementation for this name.", payload: item });
    } else if (isDisplay(item)) {
      card.displays.push(String(item.output ?? ""));
      this.view.stage(card, `mcp_${item.id}`, null);
      this.view.stage(card, "display", {
        cls: "hosted", label: "display (show) → the screen at once, never said", text: card.displays.join("\n\n"),
      });
    } else if (isShow(item)) {
      this.view.stage(card, `mcp_${item.id}`, {
        cls: "warn", label: `display (show) failed (${item.status}): nothing reaches the screen`,
        text: String(item.output ?? ""),
      });
    } else if (item.type === "mcp_call") {
      card.toolRan = true;
      this.view.stage(card, `mcp_${item.id}`, {
        cls: item.status === "completed" ? "hosted" : "warn",
        label: `backend ran ${item.server_label} · ${item.name} (${item.status})`,
        text: `${item.name}(${item.arguments || ""})`,
        payload: item.error ? { output: safeParse(item.output), error: item.error } : safeParse(item.output),
      });
    } else if (item.type === "reasoning") {
      const summary = (item.summary || []).map((part) => part.text || "").join("\n");
      if (summary) this.view.stage(card, "reason", { cls: "reason", label: "backend reasoning (summary)", text: summary });
    }
  }

  terminal(card, kind, response) {
    const calls = card.response?.calls || [];
    if (kind === "response.completed" && calls.length) {
      this.view.status(card, `waits for the app: ${calls.length} function call${calls.length > 1 ? "s" : ""}`, "waiting");
      return { run: calls };
    }
    card.open = false;
    if (kind === "response.completed" && card.response?.finals) {
      this.view.status(card, "answered", "done");
      this.view.stage(card, "handoff", {
        cls: "spoken",
        label: "what the frontend gets",
        text: "The answer goes to the frontend, which says it when the caller is idle, together with any other ready " +
          "result; a display went to the screen at once. It words the answer in its own voice.\n" +
          `Backend tokens: ${JSON.stringify(response.usage || {})}`,
      });
      return { answered: true };
    }
    if (kind === "response.completed" && card.taker !== null) {
      this.view.status(card, `handed over → ${card.taker}`, "ended");
    } else if (kind === "response.completed") {
      card.awaitsTaker = true;  // a taker's naming may still come, on its own stream
      this.view.status(card, "ended with no answer", "ended");
    } else if (kind === "response.incomplete" && response.status === "cancelled") {
      this.view.status(card, "cancelled: the session closed", "ended");
    } else if (kind === "response.incomplete") {
      this.view.status(card, `incomplete: ${response.incomplete_details?.reason || "no reason"}`, "failed");
      this.view.stage(card, "ending", { cls: "warn", label: "the backend did not finish", text: kind, payload: response });
    } else {
      const error = response.error || {};
      this.view.status(card, `failed: ${error.code || "error"}`, "failed");
      this.view.stage(card, "ending", { cls: "warn", label: "the backend failed", text: error.message || kind, payload: response });
    }
    return null;
  }
}
