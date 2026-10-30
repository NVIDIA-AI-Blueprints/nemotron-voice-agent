// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Live Console — front end.
//
// Connects the browser to a live session over WebRTC, then instruments every frame on the "oai-events" data
// channel so you can watch the frontend/backend/client handoffs.
//
// Who does what:
//   frontend = the model the caller talks to. Hears the mic, speaks, decides when to delegate.
//   backend  = the delegate. In `responses` delegation the server runs it; in `client` delegation this page does.
//   client   = this file + cafe.js. Executes function tools and returns results.
//
// This file is the page: its DOM, the WebRTC session and the wiring. Its logic lives in the sibling modules, which
// have no DOM and are tested in client/test/live/. A value from the server, a model or a tool reaches the page only
// through html.js or as textContent.

import { CafeTools, CLIENT_TOOL_NAMES, MENU, MODIFIERS, SIZES } from "./cafe.js";
import { ClientQueue, Transcript, roundPoster, runClientDelegation } from "./client-mode.js";
import {
  ICE_GATHER_MS,
  REASONING_EFFORTS,
  SESSION_START_MS,
  TOOL_CHOICES,
  buildConfig as buildSessionConfig,
  hasTurn,
  iceServerList,
  parseTools,
  sessionMode,
  tokenFromHash,
  waitForIceGathering,
} from "./config.js";
import { DelegationBook, PROGRESS_AFTER_MS, runBatch } from "./delegations.js";
import { HISTORY_POLL_MS, historyView, shouldPoll, snapshotKey } from "./history.js";
import { cartHtml, escapeHtml, menuHtml, optionsHtml } from "./html.js";
import { EventLog, payloadText, searchText } from "./log.js";

// ---------------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const els = {
  statusDot: $("statusDot"), statusText: $("statusText"),
  badgeFrontend: $("badgeFrontend"), badgeBackend: $("badgeBackend"),
  noteFrontend: $("noteFrontend"), noteBackend: $("noteBackend"),
  mSession: $("mSession"), mUsage: $("mUsage"), mCtx: $("mCtx"),
  mDelegations: $("mDelegations"), mTools: $("mTools"),
  optMode: $("optMode"), optBackend: $("optBackend"), backendModels: $("backendModels"),
  optVoice: $("optVoice"), optEffort: $("optEffort"), optToolChoice: $("optToolChoice"), optToken: $("optToken"),
  optAutoCommentary: $("optAutoCommentary"),
  optInstructions: $("optInstructions"), optBackendInstructions: $("optBackendInstructions"),
  optTools: $("optTools"), optHistory: $("optHistory"),
  btnStart: $("btnStart"), btnStop: $("btnStop"), btnMute: $("btnMute"),
  btnShowConfig: $("btnShowConfig"), btnShowPrompts: $("btnShowPrompts"),
  player: $("player"), setupOptions: $("setupOptions"), advanced: $("advanced"), actionHint: $("actionHint"),
  panelHistory: $("panelHistory"), btnHistoryToggle: $("btnHistoryToggle"),
  captions: $("captions"), typedInput: $("typedInput"), btnTyped: $("btnTyped"),
  cartView: $("cartView"), menuView: $("menuView"),
  filters: $("filters"), hideDeltas: $("hideDeltas"), search: $("search"), log: $("log"),
  btnAutoscroll: $("btnAutoscroll"), btnClear: $("btnClear"), btnExport: $("btnExport"),
  delegations: $("delegations"),
  histFrontend: $("histFrontend"), histBackend: $("histBackend"),
  histFrontendModel: $("histFrontendModel"), histBackendModel: $("histBackendModel"),
  histFrontendCount: $("histFrontendCount"), histBackendCount: $("histBackendCount"),
  ctlKind: $("ctlKind"), ctlPreset: $("ctlPreset"), ctlContent: $("ctlContent"), btnAppend: $("btnAppend"),
  ctlUpdModel: $("ctlUpdModel"), ctlUpdEffort: $("ctlUpdEffort"), ctlUpdToolChoice: $("ctlUpdToolChoice"),
  btnUpdate: $("btnUpdate"), rawEvent: $("rawEvent"), btnRaw: $("btnRaw"),
  modal: $("modal"), modalTitle: $("modalTitle"), modalBody: $("modalBody"), modalClose: $("modalClose"),
  tokenModal: $("tokenModal"), tokenForm: $("tokenForm"), tokenInput: $("tokenInput"), tokenError: $("tokenError"),
};

// ---------------------------------------------------------------------------------
// Runtime state: the page's, and the call's, which each Start makes anew
// ---------------------------------------------------------------------------------
const app = {
  info: null,            // GET /v1/live/info
  eventSeq: 0,
  log: new EventLog(),   // the newest LOG_LIMIT events
  autoscroll: true,
  filters: new Set(["lifecycle", "transcript", "delegation", "backend", "tool", "context", "app", "error"]),
  call: null,            // the current call (newCall); the last one stays readable after it ends
};

// The application's own state: one cafe cart, which outlives a call, as a kiosk's cart would.
const cafe = new CafeTools();

/** Run one function call in this browser's cafe: its output is the JSON text the server's tools.py would give. */
async function runCafe(fc) {
  const started = performance.now();
  const output = await cafe.execute(fc.call_id, fc.name, fc.arguments);
  return { args: safeParse(fc.arguments), result: safeParse(output), output, latencyMs: Math.round(performance.now() - started) };
}

/**
 * One call's state, made anew at each Start. Work still running for an earlier call holds that call's object, never
 * the new one, and stops on its `scope` signal, which End, a new Start and every cleanup fire.
 */
function newCall() {
  return {
    sentConfig: null,             // the session config the server started with (session.started)
    pc: null, dc: null, mic: null,
    sessionId: null, sessionSnapshot: null,
    ready: false, finalized: false, muted: false,
    ended: false,                 // cleanup ran: the page let go of the connection itself
    closeTimer: null,
    startTimer: null,             // session.started must come within SESSION_START_MS of the answer
    scope: new AbortController(), // the call's stop signal: End, a new Start, cleanup
    book: new DelegationBook(cardView, CLIENT_TOOL_NAMES), // every delegation's card, read from its events
    typedRuns: new Set(),         // event ids of the response.create of typed requests, to label their cards
    toolCallCount: 0,
    lastOutputAt: 0,
    lastBargeAt: 0,
    speechTarget: null,           // delegation currently being spoken by the frontend
    speechTimer: null,
    captionBubbles: { user: null, assistant: null },
    // --- client delegation only -------------------------------------------------
    mode: "responses",
    tools: [],                    // the backend tools the setup form held at Start
    transcript: new Transcript(), // our own record of the call: the announcement carries no task text
    clientQueue: new ClientQueue(), // client delegations, one at a time, in order
    clientView: null,             // the newest client delegation's own history, for the history panel
    historyTimer: null,           // polls GET /v1/live/sessions/{id}/history while connected
    historyShown: "",             // the last snapshot rendered, so an unchanged one is not redrawn
  };
}

const nextEventId = (prefix) => `${prefix}_${String(++app.eventSeq).padStart(3, "0")}`;
const nowMs = () => performance.now();
const clock = () => new Date().toLocaleTimeString("en-US", { hour12: false }) + "." +
  String(Date.now() % 1000).padStart(3, "0");
const authHeaders = () => (els.optToken.value ? { Authorization: `Bearer ${els.optToken.value}` } : {});
const frontendModel = () => app.info?.frontend?.model || "frontend";
const backendModel = () => els.optBackend.value.trim() || app.info?.backend?.model || "backend";

// ---------------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------------
const ACTOR_LABEL = { talker: "frontend", thinker: "backend", client: "client" };

function logEvent({ cat, dir, type, text, payload, actor, delta = false }) {
  const entry = { at: clock(), cat, dir, type, text, payload, actor, delta };
  for (const dropped of app.log.add(entry)) dropped.row?.remove();  // the newest LOG_LIMIT rows stay

  const row = document.createElement("div");
  row.className = `ev ${dir} cat-${cat}`;
  entry.row = row;
  rowEntries.set(row, entry);

  const tag = actor ? `<span class="who-tag ${actor}">${ACTOR_LABEL[actor]}</span>` : "";
  row.innerHTML =
    `<span class="t">${entry.at}</span>` +
    `<span class="dir">${dir === "out" ? "▲" : "▼"}</span>` +
    `<span class="body"><span class="ty">${escapeHtml(type)}</span>${tag} ` +
    `<span class="txt">${text ? escapeHtml(text) : ""}</span></span>`;

  if (payload !== undefined) {
    // The payload's text is made when the row is first opened, not for every event that arrives.
    row.querySelector(".ty").addEventListener("click", () => {
      let pre = row.querySelector("pre");
      if (!pre) {
        pre = document.createElement("pre");
        pre.hidden = true;
        pre.textContent = payloadText(payload);
        row.querySelector(".body").appendChild(pre);
      }
      pre.hidden = !pre.hidden;
    });
  }

  applyFilter(row);
  els.log.appendChild(row);
  if (app.autoscroll) els.log.scrollTop = els.log.scrollHeight;
}

/** Each row's log entry. */
const rowEntries = new WeakMap();

function applyFilter(row) {
  const entry = rowEntries.get(row);
  const q = els.search.value.trim().toLowerCase();
  const visible = app.filters.has(entry.cat)
    && (!q || searchText(entry).includes(q))
    && !(els.hideDeltas.checked && entry.delta);
  row.style.display = visible ? "" : "none";
}
const refilter = () => els.log.querySelectorAll(".ev").forEach(applyFilter);

// ---------------------------------------------------------------------------------
// Sending client events
// ---------------------------------------------------------------------------------
/** Send on `call`'s own data channel; false (and an error row) when it is not open, so nothing reaches a later call. */
function sendOn(call, event, { cat = "app", text = "", actor = "client" } = {}) {
  if (!call.dc || call.dc.readyState !== "open") {
    logEvent({ cat: "error", dir: "out", type: event.type, text: "data channel not open — not sent", payload: event });
    return false;
  }
  call.dc.send(JSON.stringify(event));
  logEvent({ cat, dir: "out", type: event.type, text, payload: event, actor });
  return true;
}

/** Send on the current call's data channel. */
const send = (event, options) => sendOn(app.call, event, options);

// ---------------------------------------------------------------------------------
// Captions
// ---------------------------------------------------------------------------------
function caption(role, delta, startMs, endMs) {
  const who = role === "user" ? "user (mic)" : `frontend · ${frontendModel()}`;
  let bubble = app.call.captionBubbles[role];
  if (!bubble || bubble.dataset.closed === "1") {
    bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;
    bubble.innerHTML = `<span class="meta">${escapeHtml(who)} · ${escapeHtml(startMs)}ms</span><span class="body"></span>`;
    els.captions.appendChild(bubble);
    app.call.captionBubbles[role] = bubble;
  }
  bubble.querySelector(".body").textContent += delta;
  bubble.querySelector(".meta").textContent = `${who} · ${bubble.dataset.start || startMs}–${endMs}ms`;
  if (!bubble.dataset.start) bubble.dataset.start = startMs;
  els.captions.scrollTop = els.captions.scrollHeight;

  clearTimeout(bubble._idle);
  bubble._idle = setTimeout(() => { bubble.dataset.closed = "1"; }, 1800);
}

function captionNote(text) {
  const el = document.createElement("div");
  el.className = "bubble barge";
  el.textContent = text;
  els.captions.appendChild(el);
  els.captions.scrollTop = els.captions.scrollHeight;
  app.call.captionBubbles.user = null;
  app.call.captionBubbles.assistant = null;
}

// ---------------------------------------------------------------------------------
// Delegation cards: delegations.js reads each delegation's events by structure; cardView draws what it reads
// ---------------------------------------------------------------------------------
const TONES = ["running", "waiting", "done", "ended", "failed"];

/** What delegations.js draws through: a card, its stages, its status and its response id. */
const cardView = {
  created(d) {
    const toClient = d.target === "client";
    const card = document.createElement("div");
    card.className = "dcard";
    const [title, text] = d.typed
      ? [`1 · this page → ${toClient ? "our backend" : "backend"} (typed)`,
        `You typed this request. ${toClient
          ? "This page runs it with its own backend rounds."
          : "The server runs it as a request of the app's own; its events come back here like a delegation's."}`]
      : toClient
        ? ["1 · frontend → client", `The frontend decided it needs help and handed it to US.
This event carries metadata ONLY — no task text, no transcript, no arguments.
The server calls no backend here; nothing happens until we act.`]
        : ["1 · frontend → backend", `The frontend decided this needs the backend and delegated it.
The server calls the backend model and supplies the conversation context itself.
The exact request payload (instructions + input) is never sent to the client.`];
    card.innerHTML = `
    <header>
      <span class="did">${escapeHtml(d.id)}</span>
      <span class="dstat">target: ${escapeHtml(d.target)} · +${escapeHtml(d.offsetMs)}ms</span>
    </header>
    <div class="stage">
      <span class="slabel">${escapeHtml(title)}</span>
      <div class="stext">${escapeHtml(text)}</div>
      <div class="flow"><span>${escapeHtml(frontendModel())}</span><span>→</span><span>${
        toClient ? "this browser" : escapeHtml(app.call.sentConfig?.delegation?.responses?.model || "backend")
      }</span><span>${toClient ? "no response_id" : `response_id: ${escapeHtml(d.responseId || "pending")}`}</span></div>
    </div>`;
    d.el = card;
    d.stages = {};
    const empty = els.delegations.querySelector(".empty");
    if (empty) empty.remove();
    els.delegations.prepend(card);
    els.mDelegations.textContent = app.call.book.cards.size;
  },
  stage: (d, key, spec) => stage(d, key, spec),
  status: (d, text, tone) => setStatus(d, text, tone),
  flow(d, responseId) {
    d.el.querySelector(".flow").lastElementChild.textContent = `response_id: ${responseId}`;
  },
};

/** Draw a card's stage `key` (drawn again: replaced); `spec` null removes it. */
function stage(d, key, spec) {
  let node = d.stages[key];
  if (spec === null) {
    node?.remove();
    delete d.stages[key];
    return null;
  }
  const { cls = "", label, text = "", payload } = spec;
  if (!node) {
    node = document.createElement("div");
    node.innerHTML = `<span class="slabel"></span><div class="stext"></div>`;
    d.el.appendChild(node);
    d.stages[key] = node;
  }
  node.className = `stage ${cls}`;
  if (label) node.querySelector(".slabel").textContent = label;
  node.querySelector(".stext").textContent = text;
  if (payload !== undefined) {
    let pre = node.querySelector("pre");
    if (!pre) { pre = document.createElement("pre"); node.appendChild(pre); }
    pre.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  }
  return node;
}

/** A card's status line and its edge's tone (running, waiting, done, ended or failed); a barge-in mark goes. */
function setStatus(d, text, tone = "running") {
  d.status = text;
  for (const name of TONES) d.el.classList.toggle(name, name === tone);
  d.el.classList.remove("interrupted");
  d.el.querySelector(".dstat").textContent = `target: ${d.target} · +${d.offsetMs}ms · ${text}`;
}

// ---------------------------------------------------------------------------------
// Backend (nested Responses) events
// ---------------------------------------------------------------------------------
function onBackendEvent(envelope) {
  const e = envelope.event || {};
  logEvent({
    cat: "backend", dir: "in", actor: "thinker",
    type: `response.event ▸ ${e.type || "?"}`,
    text: e.delta ? JSON.stringify(e.delta).slice(0, 80) : (envelope.delegation_id || ""),
    payload: envelope,
    delta: /\.delta$|in_progress$/.test(e.type || ""),
  });
  if (e.type === "response.output_item.done" && e.item?.type === "function_call") {
    logEvent({ cat: "tool", dir: "in", actor: "thinker", type: `function_call → ${e.item.name}`, text: `call_id ${e.item.call_id}`, payload: e.item });
  }
  const next = app.call.book.read(envelope);
  const d = app.call.book.cards.get(envelope.delegation_id);
  if (next?.run) runToolCalls(d, next.run);
  if (next?.answered) {
    app.call.speechTarget = d;
    clearTimeout(app.call.speechTimer);
    app.call.speechTimer = setTimeout(() => { app.call.speechTarget = null; }, 12000);
  }
}

/**
 * Run the function calls a response ended with, then send their outputs and one `response.create`, together: no other
 * event goes out between them, so each `response.create` continues exactly the response whose outputs it follows.
 */
async function runToolCalls(d, calls) {
  const call = app.call;
  const backend = backendModel();
  stage(d, "exec", {
    cls: "tool",
    label: "client executes the function calls",
    text: calls.map((fc) => `${fc.name}(${fc.arguments})`).join("\n"),
  });
  // Optional spoken progress while the batch runs: the example's still-working line, once, when the batch is still
  // running after PROGRESS_AFTER_MS. This is a client-authored update to the frontend, independent of the backend
  // response.
  const progress = els.optAutoCommentary.checked ? app.info?.lines?.still_working : "";
  const slow = progress ? () => sendOn(call, {
    type: "session.commentary.append",
    event_id: nextEventId("commentary"),
    delegation_id: null, // responses delegation rejects non-null IDs
    content: progress,
  }, { cat: "context", text: `auto progress update: the tools still run after ${PROGRESS_AFTER_MS} ms`, actor: "client" }) : null;

  const t0 = nowMs();
  const results = await runBatch(calls, async (fc) => {
    const { args, result, output, latencyMs } = await runCafe(fc);
    countTool(call);
    stage(d, `result_${fc.call_id}`, {
      cls: "tool",
      label: `client → ${backend} · function_call_output (${latencyMs}ms)`,
      text: `The client sends this back on the data channel as response.item.create.`,
      payload: { call_id: fc.call_id, name: fc.name, args, output: result },
    });
    return { fc, output };
  }, { slow, signal: call.scope.signal });
  if (call.scope.signal.aborted) {  // End or a new Start came while the tools ran: their outputs are not sent
    stage(d, "continue", { cls: "warn", label: "not sent: the session ended", text: "The tools ran; their outputs were not sent." });
    return;
  }

  for (const { fc, output } of results) {
    sendOn(call, {
      type: "response.item.create",
      event_id: nextEventId("tool_result"),
      item: { type: "function_call_output", call_id: fc.call_id, output },
    }, { cat: "tool", text: `${fc.name} result (${Math.round(nowMs() - t0)}ms)`, actor: "client" });
  }
  // One continuation for the whole batch, right after its outputs.
  sendOn(call, { type: "response.create", event_id: nextEventId("continue") },
    { cat: "tool", text: "continue the delegated response with the tool results", actor: "client" });
  stage(d, "continue", {
    cls: "tool",
    label: "client → backend · response.create",
    text: "Resumes the delegated response now that every function result is in.",
  });
}

/** One more tool run on `call`'s meter (the cart view shows what it changed). */
function countTool(call) {
  call.toolCallCount += 1;
  if (call === app.call) els.mTools.textContent = call.toolCallCount;
  renderCart();
}

const safeParse = (s) => { try { return JSON.parse(s); } catch { return s; } };

// ---------------------------------------------------------------------------------
// CLIENT DELEGATION (the loop is client-mode.js)
//
// The contrast with responses delegation is the whole point:
//   - The server tells us *that* the frontend wants help, never *what with*. No task text.
//   - We reconstruct the request from our own transcript buffer + application state.
//   - We pick the model, own the prompt, and run the ENTIRE tool loop ourselves (one backend round at a time
//     through POST /v1/live/delegate). The frontend never sees a single one of those tool calls.
//   - The only thing that goes back to the frontend is plain text in a commentary append, which it speaks.
// ---------------------------------------------------------------------------------
/** Queue client delegation `d`; `delegationId` is what its appends name: its id, or null for a typed request. */
function startClientDelegation(d, delegationId) {
  const call = app.call;
  const model = backendModel();
  const transcript = call.transcript.context();  // what was said up to the announcement
  const history = [];
  if (call.clientQueue.busy) setStatus(d, "queued: waits for the delegation before it", "waiting");
  const view = {
    stage: (key, spec) => stage(d, key, spec),
    status: (text, tone) => {
      d.open = tone === "running" || tone === "waiting";
      setStatus(d, text, tone);
    },
    log: logEvent,
  };
  call.clientQueue.run(async () => {
    call.clientView = { model, delegationId: d.id, history };
    const outcome = await runClientDelegation({
      delegationId, transcript, history, model, view,
      lines: app.info?.lines,
      cart: () => cafe.snapshot(),
      round: roundPoster({
        fetch: (...args) => fetch(...args),
        headers: authHeaders,
        model,
        instructions: els.optBackendInstructions.value.trim(),
        tools: call.tools,
      }),
      runTool: (fc) => runClientTool(call, fc),
      send: (event) => sendOn(call, event, {
        cat: "context",
        actor: "client",
        text: event.type === "session.thinking.append" ? `quiet progress for ${d.id}` : `result → frontend for ${d.id}`,
      }),
      nextEventId,
      signal: call.scope.signal,
    });
    if (outcome.status !== "stopped" && outcome.status !== "unsent" && call === app.call) {
      call.speechTarget = d;
      clearTimeout(call.speechTimer);
      call.speechTimer = setTimeout(() => { call.speechTarget = null; }, 12000);
    }
  }).catch((error) => {  // a fault of this page's own code: shown, and the next delegation still runs
    logEvent({ cat: "error", dir: "in", type: "client delegation", text: String(error?.message || error), payload: String(error?.stack || error) });
    setStatus(d, "failed: the console's own error (see the event log)", "failed");
  });
}

/** Run one function call of a client delegation in this browser; its output goes to our backend, not the session. */
async function runClientTool(call, fc) {
  const { args, result, output, latencyMs } = await runCafe(fc);
  countTool(call);
  logEvent({
    cat: "tool", dir: "out", actor: "client",
    type: `client tool ${fc.name} (client delegation)`,
    text: `${latencyMs}ms — result goes to OUR backend, not to the session`,
    payload: { args, result },
  });
  return { type: "function_call_output", call_id: fc.call_id, output };
}

// ---------------------------------------------------------------------------------
// Live conversation history: what the talker and the backend hold (read by history.js)
// ---------------------------------------------------------------------------------
/** Whether the history panel is open and the page visible: only then does the poll fetch. */
const historyWanted = () => shouldPoll(!els.panelHistory.classList.contains("collapsed"), document.visibilityState);

function startHistory(call = app.call) {
  stopHistory(call);
  call.historyShown = "";
  call.historyTimer = setInterval(() => { if (historyWanted()) refreshHistory(call); }, HISTORY_POLL_MS);
  if (historyWanted()) refreshHistory(call);
}

function stopHistory(call = app.call) {
  clearInterval(call.historyTimer);
  call.historyTimer = null;
}

async function refreshHistory(call = app.call) {
  if (!call.sessionId || call.historyTimer === null) return;  // no session, or its poll stopped
  let text;
  try {
    const response = await fetch(`/v1/live/sessions/${encodeURIComponent(call.sessionId)}/history`, { headers: authHeaders() });
    if (!response.ok) {
      // The session is gone (closed) or its engine does not report history: keep what is shown.
      if (response.status === 404) stopHistory(call);
      return;
    }
    text = await response.text();
  } catch {
    return;
  }
  if (call !== app.call) return;  // an earlier call's answer: the panel shows the new call
  // In client delegation the server runs no backend; this page's own backend conversation is the history.
  const local = call.mode === "client" ? call.clientView : null;  // the newest client delegation's own history
  const key = snapshotKey(text, local);
  if (key === call.historyShown) return;
  let snapshot;
  try { snapshot = JSON.parse(text); } catch { return; }
  call.historyShown = key;
  if (!snapshot.backend && local) {
    snapshot.backend = { model: `${local.model} (ours) · ${local.delegationId}`, items: local.history, local: true };
  }
  renderHistory(snapshot);
}

function renderHistory(snapshot) {
  const { frontend, backend } = historyView(snapshot);
  els.histFrontendModel.textContent = frontend.model;
  els.histFrontendCount.textContent = frontend.count;
  fillHistory(els.histFrontend, frontend.entries, frontend.empty);
  els.histBackendModel.textContent = backend.model;
  els.histBackendCount.textContent = backend.count;
  fillHistory(els.histBackend, backend.entries, backend.empty);
}

function fillHistory(container, entries, emptyText) {
  const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
  const open = new Set([...container.querySelectorAll("details[open]")].map((d) => d.dataset.index));
  container.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  entries.forEach((entry, index) => {
    const row = document.createElement("div");
    row.className = `hmsg ${entry.cls}`;
    const role = document.createElement("span");
    role.className = "hrole";
    role.textContent = entry.role;
    const text = document.createElement("div");
    text.className = "htext";
    text.textContent = entry.text;
    row.appendChild(role);
    if (entry.collapsed) {
      const details = document.createElement("details");
      details.dataset.index = String(index);
      details.open = open.has(String(index));
      const summary = document.createElement("summary");
      summary.textContent = entry.collapsed;
      details.append(summary, text);
      row.appendChild(details);
    } else {
      row.appendChild(text);
    }
    container.appendChild(row);
  });
  if (atBottom) container.scrollTop = container.scrollHeight;
}

// ---------------------------------------------------------------------------------
// Server event router
// ---------------------------------------------------------------------------------
function onServerEvent(ev) {
  switch (ev.type) {
    case "session.started": {
      app.call.ready = true;
      clearTimeout(app.call.startTimer);
      app.call.sessionId = ev.session?.id;
      app.call.sessionSnapshot = ev.session;
      // The session runs with the resolved settings session.started shows, not the form's.
      app.call.sentConfig = ev.session;
      app.call.mode = sessionMode(ev.session);
      updateBackendBadge();
      els.mSession.textContent = app.call.sessionId || "—";
      setConnState("live", "Connected");
      els.mSession.title = app.call.sessionId || "";
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: ev.type, text: `session ${app.call.sessionId} active`, payload: ev });
      enableControls(true);
      startHistory();
      break;
    }

    case "session.updated":
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: ev.type, text: "backend settings applied", payload: ev });
      if (ev.session) {  // the whole resolved session
        app.call.sentConfig = ev.session;
        updateBackendBadge();
      }
      break;

    case "session.input_transcript.delta":
      detectBargeIn(ev);
      app.call.transcript.record("user", ev.delta);
      caption("user", ev.delta, ev.start_ms, ev.end_ms);
      logEvent({ cat: "transcript", dir: "in", type: ev.type, text: ev.delta, payload: ev });
      break;

    case "session.output_transcript.delta":
      app.call.lastOutputAt = nowMs();
      app.call.transcript.record("assistant", ev.delta);
      caption("assistant", ev.delta, ev.start_ms, ev.end_ms);
      logEvent({ cat: "transcript", dir: "in", actor: "talker", type: ev.type, text: ev.delta, payload: ev });
      if (app.call.speechTarget) {
        const d = app.call.speechTarget;
        d.spoken = (d.spoken || "") + ev.delta;
        stage(d, "spoken", {
          cls: "spoken", label: "6 · what the frontend actually said", text: d.spoken,
        });
      }
      break;

    case "session.delegation.created": {
      const d = app.call.book.announce(ev, { typed: app.call.typedRuns.has(ev.client_event_id) });
      logEvent({
        cat: "delegation", dir: "in", actor: "talker", type: ev.type,
        text: `${ev.delegation.id} → ${ev.delegation.target}`, payload: ev,
      });
      if (ev.delegation.target === "client") startClientDelegation(d, d.id);
      break;
    }

    case "response.event":
      onBackendEvent(ev);
      break;

    case "session.instructions.appended":
    case "session.thinking.appended":
    case "session.commentary.appended":
      logEvent({
        cat: "context", dir: "in", actor: "talker", type: ev.type,
        text: `ack for ${ev.client_event_id} · timeline ${ev.start_ms}–${ev.end_ms}ms (context injected, not spoken yet)`,
        payload: ev,
      });
      break;

    case "session.input_audio.muted":
    case "session.input_audio.unmuted":
      app.call.muted = ev.type.endsWith("muted") && !ev.type.endsWith("unmuted");
      els.btnMute.textContent = app.call.muted ? "Unmute mic" : "Mute mic";
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: "mic input state changed (model output is unaffected)", payload: ev });
      break;

    case "session.usage.updated":
      els.mUsage.textContent = ev.usage?.seconds ?? "—";
      els.mCtx.textContent = ev.context_window?.usage_ratio != null
        ? `${Math.round(ev.context_window.usage_ratio * 100)}%` : "—";
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: `${ev.usage?.seconds}s voice`, payload: ev });
      break;

    case "session.closed":
      app.call.finalized = true;
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: `reason: ${ev.reason} · final usage ${ev.usage?.seconds}s`, payload: ev });
      els.mUsage.textContent = ev.usage?.seconds ?? els.mUsage.textContent;
      setConnState("idle", `Closed (${ev.reason})`);
      cleanup();
      break;

    case "error":
      logEvent({ cat: "error", dir: "in", type: ev.type, text: `${ev.error?.code || ""} ${ev.error?.message || ""}`, payload: ev });
      break;

    default:
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: "", payload: ev });
  }
}

function detectBargeIn(ev) {
  const sinceOutput = nowMs() - app.call.lastOutputAt;
  if (app.call.lastOutputAt === 0 || sinceOutput > 1500) return;
  if (nowMs() - app.call.lastBargeAt < 3000) return;
  app.call.lastBargeAt = nowMs();

  captionNote("⟂ barge-in — the user started speaking while the frontend was still speaking (full duplex).");
  logEvent({
    cat: "app", dir: "in", type: "app.barge_in_detected",
    text: `user speech at ${ev.start_ms}ms overlapped assistant speech ${Math.round(sinceOutput)}ms ago`,
    payload: { user_start_ms: ev.start_ms, ms_since_assistant_delta: Math.round(sinceOutput), delta: ev.delta },
  });

  for (const d of app.call.book.cards.values()) {
    if (!d.open) continue;
    stage(d, "interrupt", {
      cls: "warn",
      label: "interruption during backend work",
      text:
        "The user interrupted the frontend while this delegation was still running.\n" +
        "Interrupting speech does not cancel backend work — it keeps going and the result still comes back.\n" +
        "Cancelling is the application's decision, tracked in application state.",
    });
    d.el.classList.add("interrupted");
  }
}

// ---------------------------------------------------------------------------------
// Configuration (built by config.js)
// ---------------------------------------------------------------------------------
/** The setup form's values, as config.js reads them. */
function formValues() {
  return {
    mode: els.optMode.value,
    frontendModel: frontendModel(),
    voice: els.optVoice.value,
    instructions: els.optInstructions.value,
    history: els.optHistory.value,
    tools: els.optTools.value,
    backendModel: backendModel(),
    backendInstructions: els.optBackendInstructions.value,
    toolChoice: els.optToolChoice.value,
    effort: els.optEffort.value,
  };
}

/** The `session` object of the session start: everything on the setup form. Throws a readable Error. */
function buildConfig() {
  return buildSessionConfig(formValues());
}

function fillSelect(el, values, selected) {
  el.innerHTML = optionsHtml(values, selected);
}

/** Fill the form from GET /v1/live/info. */
function applyInfo() {
  const info = app.info;
  fillSelect(els.optVoice, info.voices, info.voices.includes("marin") ? "marin" : info.voices[0]);
  fillSelect(els.optEffort, REASONING_EFFORTS, "default");
  fillSelect(els.ctlUpdEffort, REASONING_EFFORTS, "default");
  fillSelect(els.optToolChoice, TOOL_CHOICES, "auto");
  fillSelect(els.ctlUpdToolChoice, ["unchanged", ...TOOL_CHOICES], "unchanged");
  els.backendModels.innerHTML = optionsHtml(info.backend_models || [], null, []);
  els.optBackend.value = info.backend?.model || "";
  els.ctlUpdModel.value = info.backend?.model || "";
  els.optInstructions.value = info.instructions?.frontend || "";
  els.optBackendInstructions.value = info.instructions?.backend || "";
  els.optTools.value = JSON.stringify(info.sample_tools || [], null, 2);
  els.badgeFrontend.textContent = info.frontend?.model || "—";
  els.noteFrontend.textContent = `${info.frontend?.provider || "frontend"} · prompts ${info.prompt_version || ""}`;
  updateBackendBadge();
  refreshPresets();
  for (const warning of info.warnings || []) {
    logEvent({ cat: "error", dir: "in", type: "server.warning", text: warning });
  }
}

/** The BACKEND badge: the running session's resolved settings while it runs, the form's otherwise. */
function updateBackendBadge() {
  const live = Boolean(app.call?.ready);
  const client = live ? app.call.mode === "client" : els.optMode.value === "client";
  const model = live ? app.call.sentConfig?.delegation?.responses?.model : null;
  els.badgeBackend.textContent = client ? `${backendModel()} (ours)` : model || backendModel();
  els.noteBackend.textContent = client
    ? "run by this browser · client delegation"
    : `${app.info?.backend?.provider || "backend"} · responses delegation`;
}

// ---------------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------------
function setConnState(kind, text) {
  els.statusDot.className = "dot" + (kind === "live" ? " live" : kind === "err" ? " err" : "");
  els.statusText.textContent = text;
  document.body.dataset.conn = kind || "idle";
}

function enableControls(on) {
  // The session's settings are fixed once it starts; lock them so they cannot look editable.
  document.body.dataset.session = on ? "on" : "off";
  els.setupOptions.disabled = on;
  if (on) els.advanced.open = false;
  els.actionHint.textContent = on
    ? "Speak any time; you can interrupt the assistant. Steer it from the bar at the bottom."
    : "Locks these settings for the session. Allow the microphone, then speak.";
  els.btnStop.disabled = !on;
  els.btnMute.disabled = !on;
  els.btnAppend.disabled = !on;
  els.btnUpdate.disabled = !on;
  els.btnTyped.disabled = !on;
  els.btnRaw.disabled = !on;
}

/** The page's view of a call, emptied for a new one: cards, captions, meters, history and the mute button. */
function resetCallView() {
  const note = document.createElement("p");
  note.className = "empty";
  note.textContent = "No delegation yet. Ask about the menu once connected.";
  els.delegations.replaceChildren(note);
  els.captions.replaceChildren();
  els.mSession.textContent = "—";
  els.mSession.title = "";
  els.mUsage.textContent = "0";
  els.mCtx.textContent = "0%";
  els.mDelegations.textContent = "0";
  els.mTools.textContent = "0";
  els.btnMute.textContent = "Mute mic";
  for (const label of [els.histFrontendModel, els.histFrontendCount, els.histBackendModel, els.histBackendCount]) {
    label.textContent = "";
  }
  fillHistory(els.histFrontend, [], "Start a conversation to see the talker's context.");
  fillHistory(els.histBackend, [], "Start a conversation to see the backend's history.");
}

/** The page's ICE servers from GET /api/ice-servers (host candidates only without them); credentials are not logged. */
async function loadIceServers() {
  try {
    const response = await fetch("/api/ice-servers");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const servers = iceServerList(await response.json());
    logEvent({
      cat: "lifecycle", dir: "in", type: "GET /api/ice-servers",
      text: servers.length
        ? `${servers.length} ICE server(s)${hasTurn(servers) ? ", TURN included" : ""}`
        : "none: host candidates only (enough on this machine or a LAN)",
      payload: { urls: servers.map((server) => server.urls) },
    });
    return servers;
  } catch (error) {
    logEvent({ cat: "lifecycle", dir: "in", type: "GET /api/ice-servers", text: `not available (${error.message}): host candidates only` });
    return [];
  }
}

/** What a failed connection most likely needs, for its message. */
function turnHint(iceServers) {
  return hasTurn(iceServers)
    ? "The TURN server was offered; check that it is reachable."
    : "No TURN server was offered: a server on another network needs TURN_USERNAME and TURN_PASSWORD (and TURN_URL).";
}

/** End `call` with a clear message instead of a hang: the microphone stops and the page can start again. */
function failCall(call, message) {
  logEvent({ cat: "error", dir: "in", type: "connect.failed", text: message });
  setConnState("err", message);
  cleanup(call);
}

async function connect() {
  els.btnStart.disabled = true;
  const previous = app.call;
  const call = app.call = newCall();
  cleanup(previous);  // an earlier call still holding a connection lets it go
  resetCallView();
  setConnState("", "Connecting…");

  try {
    const config = buildConfig();  // refuses bad backend tools JSON in both modes
    call.mode = els.optMode.value;
    call.tools = parseTools(els.optTools.value);  // client mode sends these with each of its rounds
    const iceServers = await loadIceServers();
    const pc = new RTCPeerConnection({ iceServers });
    call.pc = pc;

    pc.addEventListener("track", (event) => {
      els.player.srcObject = new MediaStream([event.track]);
      els.player.play().catch(() => {
        els.player.classList.add("needs-play");
        setConnState("live", "Press play on the audio control to hear the assistant.");
      });
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: "webrtc.track", text: "remote audio track attached (frontend speech)" });
    });
    pc.addEventListener("connectionstatechange", () => {
      logEvent({ cat: "lifecycle", dir: "in", type: "webrtc.connectionstate", text: pc.connectionState });
      if (pc.connectionState === "failed" && call === app.call && !call.finalized) {
        failCall(call, `WebRTC could not connect (ICE failed). ${turnHint(iceServers)}`);
      }
    });

    call.mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    for (const track of call.mic.getAudioTracks()) pc.addTrack(track, call.mic);
    logEvent({ cat: "lifecycle", dir: "out", type: "webrtc.microphone", text: "mic track added — audio goes to the server over WebRTC" });

    // Register handlers before creating the offer.
    const dc = pc.createDataChannel("oai-events");
    call.dc = dc;
    dc.addEventListener("message", ({ data }) => {
      if (call !== app.call) return;  // a late event of an earlier call
      let ev;
      try { ev = JSON.parse(data); } catch { return; }
      onServerEvent(ev);
    });
    dc.addEventListener("close", () => {
      if (call === app.call && !call.finalized && !call.ended) {  // not the page's own close in cleanup
        logEvent({ cat: "error", dir: "in", type: "datachannel.close", text: "closed before session.closed — final usage unconfirmed" });
        setConnState("err", "Disconnected without final usage.");
        cleanup(call);
      }
    });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    if (!(await waitForIceGathering(pc))) {
      logEvent({
        cat: "lifecycle", dir: "out", type: "webrtc.ice",
        text: `ICE gathering did not finish in ${ICE_GATHER_MS / 1000} s (a TURN server this browser cannot reach?): ` +
          "the offer carries the candidates gathered so far",
      });
    }

    logEvent({
      cat: "lifecycle", dir: "out", type: "POST /v1/live/sessions",
      text: "SDP offer plus the session config below", payload: { session: config },
    });
    const response = await fetch("/v1/live/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ transport: { type: "webrtc", sdp: pc.localDescription.sdp }, session: config }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error?.message || result.detail || `HTTP ${response.status}`);

    updateBackendBadge();
    logEvent({
      cat: "lifecycle", dir: "in", type: "POST /v1/live/sessions → 201",
      text: `session ${result.session?.id} created; its configuration comes with session.started`,
      payload: { session: result.session },
    });

    await pc.setRemoteDescription({ type: "answer", sdp: result.transport.sdp });
    // The HTTP request started the session; do not send session.start here.
    setConnState("", "Negotiated — waiting for session.started…");
    call.startTimer = setTimeout(() => {
      if (call === app.call && !call.ready) {
        failCall(call, `No session.started within ${SESSION_START_MS / 1000} s: the connection did not come up. ${turnHint(iceServers)}`);
      }
    }, SESSION_START_MS);
  } catch (error) {
    logEvent({ cat: "error", dir: "in", type: "connect.failed", text: String(error.message || error).slice(0, 400), payload: String(error) });
    setConnState("err", "Connection failed — see console");
    cleanup(call);
  }
}

function gracefulClose() {
  if (!app.call.ready || !app.call.dc || app.call.dc.readyState !== "open") return;
  els.btnStop.disabled = true;
  setConnState("", "Closing — draining pending events…");
  const call = app.call;
  call.scope.abort();  // stops this page's own work for the call: client rounds, tool outputs
  // The session.closed handler is already registered; keep everything alive until it arrives.
  send({ type: "session.close", event_id: nextEventId("close") },
    { cat: "lifecycle", text: "queued backend work is cancelled; an active reply may finish" });
  call.closeTimer = setTimeout(() => {
    logEvent({ cat: "error", dir: "in", type: "close.timeout", text: "no session.closed within 15s — finalization unconfirmed" });
    cleanup(call);
  }, 15000);
}

/** Let go of `call`'s connection, microphone and timers; the page's controls follow only the current call. */
function cleanup(call = app.call) {
  call.ended = true;
  call.scope.abort();
  clearTimeout(call.closeTimer);
  clearTimeout(call.startTimer);
  clearTimeout(call.speechTimer);
  stopHistory(call);
  call.mic?.getTracks().forEach((t) => t.stop());
  try { call.dc?.close(); } catch {}
  try { call.pc?.close(); } catch {}
  call.ready = false;
  call.dc = null;
  if (call !== app.call) return;
  els.player.srcObject = null;
  els.player.classList.remove("needs-play");
  els.btnStart.disabled = false;
  enableControls(false);
}

// ---------------------------------------------------------------------------------
// Client state rendering
// ---------------------------------------------------------------------------------
function renderCart() {
  els.cartView.innerHTML = cartHtml(cafe.snapshot(), cafe.lastOrder);
}

function renderMenu() {
  els.menuView.innerHTML = menuHtml(MENU, SIZES, MODIFIERS);
}

// ---------------------------------------------------------------------------------
// Append presets, from GET /v1/live/info: the example's lines that suit each kind of append, then its presets
// ---------------------------------------------------------------------------------
const PRESET_LINES = {
  "session.instructions.append": [["Greet first", "intro"]],
  "session.commentary.append": [["Still working", "still_working"]],
};

/** The chosen kind's presets as [label, text]: its lines, then the example's presets (`presets`), unless empty. */
function presets() {
  const kind = els.ctlKind.value;
  return [
    ...(PRESET_LINES[kind] || []).map(([label, name]) => [label, app.info?.lines?.[name] || ""]),
    ...(app.info?.presets?.[kind] || []),
  ].filter(([, text]) => text);
}

function refreshPresets() {
  const list = presets();
  els.ctlPreset.innerHTML = optionsHtml(list.map((_, i) => String(i)), "0", list.map((p) => p[0]));
  els.ctlContent.value = list[0]?.[1] || "";
}

// ---------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------
async function loadInfo() {
  const response = await fetch("/v1/live/info", { headers: authHeaders() });
  if (!response.ok) {
    throw Object.assign(new Error(response.status === 401
      ? "The server requires a bearer token: enter it above, then reload the defaults."
      : `GET /v1/live/info failed (HTTP ${response.status}). Is the live example enabled on this server?`),
    { status: response.status });
  }
  app.info = await response.json();
  applyInfo();
  logEvent({
    cat: "app", dir: "in", type: "app.ready",
    text: "press Start conversation",
    payload: {
      frontend: app.info.frontend, backend: app.info.backend,
      backend_models: app.info.backend_models, client_tools: CLIENT_TOOL_NAMES,
    },
  });
}

/** Take a bearer token from `live.html#token=…` (the fragment never reaches the server), then drop it from the URL. */
function tokenFromUrl() {
  const token = tokenFromHash(location.hash);  // keeps a "+"
  if (!token) return;
  els.optToken.value = token;
  history.replaceState(null, "", location.pathname + location.search);
}

/** Ask for the bearer token in a dialog; resolves once the user submits one. */
function askToken(message) {
  els.tokenError.textContent = message;
  els.tokenInput.value = "";
  els.tokenModal.hidden = false;
  els.tokenInput.focus();
  return new Promise((resolve) => {
    els.tokenForm.addEventListener("submit", function submit(event) {
      event.preventDefault();
      const token = els.tokenInput.value.trim();
      if (!token) return;
      els.tokenForm.removeEventListener("submit", submit);
      els.tokenModal.hidden = true;
      resolve(token);
    });
  });
}

/** Load the server defaults, asking for the bearer token for as long as the server refuses it. */
async function loadInfoWithToken() {
  for (;;) {
    try {
      return await loadInfo();
    } catch (error) {
      if (error.status !== 401) throw error;
      els.optToken.value = await askToken(els.optToken.value ? "That token was refused. Try again." : "");
    }
  }
}

async function boot() {
  app.call = newCall();
  tokenFromUrl();
  refreshPresets();
  renderMenu();
  renderCart();
  enableControls(false);
  els.btnStart.disabled = true;
  try {
    await loadInfoWithToken();
  } catch (error) {
    logEvent({ cat: "error", dir: "in", type: "app.info", text: String(error.message || error) });
    setConnState("err", String(error.message || error));
  }
  els.btnStart.disabled = false;

  // --- wiring ------------------------------------------------------------------
  els.btnStart.addEventListener("click", connect);
  els.btnStop.addEventListener("click", gracefulClose);
  els.optToken.addEventListener("change", () => loadInfo().then(() => setConnState("", "Idle")).catch((e) => setConnState("err", e.message)));

  els.btnMute.addEventListener("click", () => {
    send({ type: app.call.muted ? "session.input_audio.unmute" : "session.input_audio.mute", event_id: nextEventId("mic") },
      { cat: "lifecycle", text: app.call.muted ? "resume mic input" : "mute mic input (does not stop the frontend speaking)" });
  });

  els.btnAppend.addEventListener("click", () => {
    const content = els.ctlContent.value.trim();
    if (!content) return;
    send({
      type: els.ctlKind.value,
      event_id: nextEventId("ctx"),
      delegation_id: null, // responses delegation rejects non-null delegation IDs
      content,
    }, { cat: "context", text: content.slice(0, 70), actor: "client" });
  });

  els.btnUpdate.addEventListener("click", () => {
    if (app.call.mode === "client") {
      // There is nothing for the session to update: it holds no backend configuration at all.
      // Swapping our own model is a purely local change and emits no session event.
      logEvent({
        cat: "app", dir: "out", actor: "client",
        type: "local backend swap (no session event)",
        text: `client delegation → ${els.ctlUpdModel.value}. session.update would be rejected: there is no delegation.responses to change.`,
        payload: { model: els.ctlUpdModel.value },
      });
      els.optBackend.value = els.ctlUpdModel.value;
      updateBackendBadge();
      return;
    }
    const responses = {};
    if (els.ctlUpdModel.value.trim()) responses.model = els.ctlUpdModel.value.trim();
    if (els.ctlUpdEffort.value !== "default") responses.reasoning = { effort: els.ctlUpdEffort.value };
    if (els.ctlUpdToolChoice.value !== "unchanged") responses.tool_choice = els.ctlUpdToolChoice.value;
    send({
      type: "session.update",
      event_id: nextEventId("update"),
      session: { delegation: { type: "responses", responses } },
    }, { cat: "lifecycle", text: `update backend → ${JSON.stringify(responses)}`, actor: "client" });
  });

  els.btnRaw.addEventListener("click", () => {
    let event;
    try { event = JSON.parse(els.rawEvent.value); } catch (e) {
      logEvent({ cat: "error", dir: "out", type: "raw event", text: `not valid JSON: ${e.message}` });
      return;
    }
    send(event, { cat: "app", text: "raw event, sent exactly as typed", actor: "client" });
  });

  els.btnTyped.addEventListener("click", () => {
    const text = els.typedInput.value.trim();
    if (!text) return;
    if (app.call.mode === "client") {
      // response.item.create / response.create require responses delegation. In client mode typed text goes
      // straight to our own backend instead.
      app.call.transcript.record("user", text);
      logEvent({
        cat: "delegation", dir: "out", actor: "client",
        type: "typed input (client delegation)",
        text: "no session command exists for this — sent to our own backend",
        payload: { text },
      });
      const ev = { delegation: { id: `local_${Date.now()}`, target: "client" }, offset_ms: 0 };
      const d = app.call.book.announce(ev, { typed: true });
      startClientDelegation(d, null);  // not a delegation the session announced: its appends name none
      els.typedInput.value = "";
      return;
    }
    send({
      type: "response.item.create",
      event_id: nextEventId("typed"),
      item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
    }, { cat: "delegation", text: "queue exact typed text for the backend", actor: "client" });
    const run = nextEventId("typed_run");
    app.call.typedRuns.add(run);  // its delegation's announcement names this event: its card says "typed"
    send({ type: "response.create", event_id: run },
      { cat: "delegation", text: "run the backend on the queued item", actor: "client" });
    els.typedInput.value = "";
  });
  els.typedInput.addEventListener("keydown", (e) => { if (e.key === "Enter") els.btnTyped.click(); });

  els.optMode.addEventListener("change", () => {
    const client = els.optMode.value === "client";
    updateBackendBadge();
    logEvent({
      cat: "app", dir: "in", type: "delegation mode",
      text: client
        ? "client — the session gets NO backend config; we reconstruct intent, run the model, own the tool loop, and hand back text via commentary"
        : "responses — the server owns the backend call, supplies context, forwards the stream; we only execute function calls",
      payload: { mode: els.optMode.value, note: "fixed at session creation; switching needs a new session" },
    });
  });
  els.optBackend.addEventListener("input", updateBackendBadge);

  els.ctlKind.addEventListener("change", refreshPresets);
  els.ctlPreset.addEventListener("change", () => {
    els.ctlContent.value = presets()[els.ctlPreset.value]?.[1] || "";
  });

  els.filters.addEventListener("change", (e) => {
    const cat = e.target.dataset.cat;
    if (e.target === els.hideDeltas) return refilter();
    if (!cat) return;
    e.target.checked ? app.filters.add(cat) : app.filters.delete(cat);
    refilter();
  });
  els.search.addEventListener("input", refilter);
  els.btnAutoscroll.addEventListener("click", () => {
    app.autoscroll = !app.autoscroll;
    els.btnAutoscroll.classList.toggle("on", app.autoscroll);
  });
  els.btnClear.addEventListener("click", () => { els.log.innerHTML = ""; app.log.clear(); });
  els.btnExport.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(app.log.exported(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `live-session-${app.call.sessionId || "log"}.json`;
    a.click();
  });

  const showModal = (title, body) => {
    els.modalTitle.textContent = title;
    els.modalBody.textContent = body;
    els.modal.hidden = false;
  };
  els.modalClose.addEventListener("click", () => { els.modal.hidden = true; });
  els.modal.addEventListener("click", (e) => { if (e.target === els.modal) els.modal.hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") els.modal.hidden = true; });
  els.player.addEventListener("playing", () => els.player.classList.remove("needs-play"));

  const setHistoryOpen = (open) => {
    els.panelHistory.classList.toggle("collapsed", !open);
    document.body.classList.toggle("history-collapsed", !open);
    els.btnHistoryToggle.textContent = open ? "hide" : "show";
    els.btnHistoryToggle.setAttribute("aria-expanded", String(open));
    try { localStorage.setItem("live.historyOpen", open ? "1" : "0"); } catch {}
    if (historyWanted()) refreshHistory(app.call);  // opened: show it now, not at the next tick
  };
  document.addEventListener("visibilitychange", () => { if (historyWanted()) refreshHistory(app.call); });
  let historyOpen = true;
  try { historyOpen = localStorage.getItem("live.historyOpen") !== "0"; } catch {}
  setHistoryOpen(historyOpen);
  els.btnHistoryToggle.addEventListener("click", () => setHistoryOpen(els.panelHistory.classList.contains("collapsed")));
  els.btnShowConfig.addEventListener("click", () => {
    let config = app.call.sentConfig;
    let title = "Session config (as the server started it)";
    if (!config) {
      title = "Session config (preview — start a conversation to send it)";
      try { config = buildConfig(); } catch (error) { config = { error: error.message }; }
    }
    showModal(title, JSON.stringify(config, null, 2));
  });
  els.btnShowPrompts.addEventListener("click", () =>
    showModal("Prompts",
      `===== FRONTEND (session.instructions) — ${frontendModel()} =====\n\n${els.optInstructions.value}` +
      `\n\n===== BACKEND (delegation.responses.instructions) — ${backendModel()} =====\n\n${els.optBackendInstructions.value}`));
}

boot();
