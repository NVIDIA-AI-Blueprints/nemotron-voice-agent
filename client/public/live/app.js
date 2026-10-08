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

import { MENU, SIZES, MODIFIERS, state, cartSnapshot, executeTool, CLIENT_TOOL_NAMES } from "./cafe.js";

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
  player: $("player"),
  captions: $("captions"), typedInput: $("typedInput"), btnTyped: $("btnTyped"),
  cartView: $("cartView"), menuView: $("menuView"),
  filters: $("filters"), hideDeltas: $("hideDeltas"), search: $("search"), log: $("log"),
  btnAutoscroll: $("btnAutoscroll"), btnClear: $("btnClear"), btnExport: $("btnExport"),
  delegations: $("delegations"),
  ctlKind: $("ctlKind"), ctlPreset: $("ctlPreset"), ctlContent: $("ctlContent"), btnAppend: $("btnAppend"),
  ctlUpdModel: $("ctlUpdModel"), ctlUpdEffort: $("ctlUpdEffort"), ctlUpdToolChoice: $("ctlUpdToolChoice"),
  btnUpdate: $("btnUpdate"), rawEvent: $("rawEvent"), btnRaw: $("btnRaw"),
  modal: $("modal"), modalTitle: $("modalTitle"), modalBody: $("modalBody"), modalClose: $("modalClose"),
};

const REASONING_EFFORTS = ["default", "minimal", "low", "medium", "high"];
const TOOL_CHOICES = ["auto", "none", "required"];

// ---------------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------------
const app = {
  info: null,            // GET /v1/live/info
  sentConfig: null,      // the session config the server accepted
  pc: null, dc: null, mic: null,
  sessionId: null, sessionSnapshot: null,
  ready: false, finalized: false, muted: false,
  closeTimer: null,
  eventSeq: 0,
  logEntries: [],
  autoscroll: true,
  filters: new Set(["lifecycle", "transcript", "delegation", "backend", "tool", "context", "app", "error"]),
  delegations: new Map(),       // delegation_id -> record
  byResponseId: new Map(),      // backend response id -> delegation record
  toolCallCount: 0,
  lastOutputAt: 0,
  lastBargeAt: 0,
  speechTarget: null,           // delegation currently being spoken by the frontend
  speechTimer: null,
  captionBubbles: { user: null, assistant: null },
  // --- client delegation only ---------------------------------------------------
  mode: "responses",
  transcript: [],               // rolling [{role, text, at}] we must keep ourselves
  clientHistory: [],            // the conversation we send to the backend, round after round
};

/**
 * Keep our own conversation record. In client delegation this is mandatory: the delegation event carries
 * metadata only, never the task text, so this buffer plus application state is the ONLY way to work out what
 * the user actually wants.
 */
function recordTranscript(role, text) {
  const last = app.transcript[app.transcript.length - 1];
  if (last && last.role === role && nowMs() - last.at < 1500) {
    last.text += text;
    last.at = nowMs();
  } else {
    app.transcript.push({ role, text, at: nowMs() });
  }
  if (app.transcript.length > 40) app.transcript.shift();
}

function transcriptContext(turns = 8) {
  return app.transcript.slice(-turns)
    .map((t) => `${t.role}: ${t.text.trim()}`)
    .filter((l) => l.length > 7)
    .join("\n");
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
  const entry = { at: clock(), cat, dir, type, text, payload, actor };
  app.logEntries.push(entry);

  const row = document.createElement("div");
  row.className = `ev ${dir} cat-${cat}`;
  row.dataset.cat = cat;
  row.dataset.delta = delta ? "1" : "";
  row.dataset.search = `${type} ${text || ""} ${JSON.stringify(payload || "")}`.toLowerCase();

  const tag = actor ? `<span class="who-tag ${actor}">${ACTOR_LABEL[actor]}</span>` : "";
  row.innerHTML =
    `<span class="t">${entry.at}</span>` +
    `<span class="dir">${dir === "out" ? "▲" : "▼"}</span>` +
    `<span class="body"><span class="ty">${escapeHtml(type)}</span>${tag} ` +
    `<span class="txt">${text ? escapeHtml(text) : ""}</span></span>`;

  if (payload !== undefined) {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(payload, null, 2);
    pre.hidden = true;
    row.querySelector(".body").appendChild(pre);
    row.querySelector(".ty").addEventListener("click", () => { pre.hidden = !pre.hidden; });
  }

  applyFilter(row);
  els.log.appendChild(row);
  if (app.autoscroll) els.log.scrollTop = els.log.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function applyFilter(row) {
  const q = els.search.value.trim().toLowerCase();
  const visible = app.filters.has(row.dataset.cat)
    && (!q || row.dataset.search.includes(q))
    && !(els.hideDeltas.checked && row.dataset.delta === "1");
  row.style.display = visible ? "" : "none";
}
const refilter = () => els.log.querySelectorAll(".ev").forEach(applyFilter);

// ---------------------------------------------------------------------------------
// Sending client events
// ---------------------------------------------------------------------------------
function send(event, { cat = "app", text = "", actor = "client" } = {}) {
  if (!app.dc || app.dc.readyState !== "open") {
    logEvent({ cat: "error", dir: "out", type: event.type, text: "data channel not open — not sent", payload: event });
    return false;
  }
  app.dc.send(JSON.stringify(event));
  logEvent({ cat, dir: "out", type: event.type, text, payload: event, actor });
  return true;
}

// ---------------------------------------------------------------------------------
// Captions
// ---------------------------------------------------------------------------------
function caption(role, delta, startMs, endMs) {
  const who = role === "user" ? "user (mic)" : `frontend · ${frontendModel()}`;
  let bubble = app.captionBubbles[role];
  if (!bubble || bubble.dataset.closed === "1") {
    bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;
    bubble.innerHTML = `<span class="meta">${escapeHtml(who)} · ${startMs}ms</span><span class="body"></span>`;
    els.captions.appendChild(bubble);
    app.captionBubbles[role] = bubble;
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
  app.captionBubbles.user = null;
  app.captionBubbles.assistant = null;
}

// ---------------------------------------------------------------------------------
// Delegation cards
// ---------------------------------------------------------------------------------
function createDelegationCard(ev) {
  const d = {
    id: ev.delegation.id,
    target: ev.delegation.target,
    responseId: ev.delegation.response_id || null,
    offsetMs: ev.offset_ms,
    startedAt: nowMs(),
    status: "running",
    pendingCalls: [],
    reasoning: "",
    answer: "",
    el: null,
    stages: {},
  };
  app.delegations.set(d.id, d);
  if (d.responseId) app.byResponseId.set(d.responseId, d);

  const toClient = d.target === "client";
  const card = document.createElement("div");
  card.className = "dcard";
  card.innerHTML = `
    <header>
      <span class="did">${escapeHtml(d.id)}</span>
      <span class="dstat">target: ${d.target} · +${d.offsetMs}ms</span>
    </header>
    <div class="stage">
      <span class="slabel">1 · frontend → ${toClient ? "client" : "backend"}</span>
      <div class="stext">${toClient
        ? `The frontend decided it needs help and handed it to US.
This event carries metadata ONLY — no task text, no transcript, no arguments.
The server calls no backend here; nothing happens until we act.`
        : `The frontend decided this needs the backend and delegated it.
The server calls the backend model and supplies the conversation context itself.
The exact request payload (instructions + input) is never sent to the client.`}</div>
      <div class="flow"><span>${escapeHtml(frontendModel())}</span><span>→</span><span>${
        toClient ? "this browser" : escapeHtml(app.sentConfig?.delegation?.responses?.model || "backend")
      }</span><span>${toClient ? "no response_id" : `response_id: ${d.responseId || "pending"}`}</span></div>
    </div>`;
  d.el = card;
  const empty = els.delegations.querySelector(".empty");
  if (empty) empty.remove();
  els.delegations.prepend(card);
  els.mDelegations.textContent = app.delegations.size;
  return d;
}

function stage(d, key, { cls = "", label, text = "", payload }) {
  let node = d.stages[key];
  if (!node) {
    node = document.createElement("div");
    node.className = `stage ${cls}`;
    node.innerHTML = `<span class="slabel"></span><div class="stext"></div>`;
    d.el.appendChild(node);
    d.stages[key] = node;
  }
  if (label) node.querySelector(".slabel").textContent = label;
  node.querySelector(".stext").textContent = text;
  if (payload !== undefined) {
    let pre = node.querySelector("pre");
    if (!pre) { pre = document.createElement("pre"); node.appendChild(pre); }
    pre.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  }
  return node;
}

function setStatus(d, status) {
  d.status = status;
  d.el.classList.toggle("done", status === "complete");
  d.el.classList.toggle("interrupted", status === "interrupted");
  d.el.querySelector(".dstat").textContent =
    `target: ${d.target} · +${d.offsetMs}ms · ${status}`;
}

// ---------------------------------------------------------------------------------
// Backend (nested Responses) events
// ---------------------------------------------------------------------------------
function resolveDelegation(envelope) {
  if (envelope.delegation_id && app.delegations.has(envelope.delegation_id)) {
    return app.delegations.get(envelope.delegation_id);
  }
  const rid = envelope.event?.response?.id || envelope.event?.item_id;
  if (rid && app.byResponseId.has(rid)) return app.byResponseId.get(rid);
  // Fall back to the most recent running delegation.
  return [...app.delegations.values()].reverse().find((d) => d.status === "running") || null;
}

function onBackendEvent(envelope) {
  const e = envelope.event || {};
  const d = resolveDelegation(envelope);
  const backend = backendModel();

  logEvent({
    cat: "backend", dir: "in", actor: "thinker",
    type: `response.event ▸ ${e.type || "?"}`,
    text: e.delta ? JSON.stringify(e.delta).slice(0, 80) : (envelope.delegation_id || ""),
    payload: envelope,
    delta: /\.delta$|in_progress$/.test(e.type || ""),
  });
  if (!d) return;

  switch (e.type) {
    case "response.created":
      if (e.response?.id) {
        d.responseId = e.response.id;
        app.byResponseId.set(e.response.id, d);
        d.el.querySelector(".flow").lastElementChild.textContent = `response_id: ${e.response.id}`;
      }
      break;

    case "response.reasoning_summary_text.delta":
      d.reasoning += e.delta || "";
      stage(d, "reason", { cls: "reason", label: `2 · ${backend} reasoning (summary)`, text: d.reasoning });
      break;

    case "response.reasoning_summary_part.done":
      d.reasoning += "\n";
      break;

    case "response.output_text.delta":
      d.answer += e.delta || "";
      stage(d, "answer", { cls: "answer", label: `${backend} result text → returned to the frontend`, text: d.answer });
      break;

    case "response.output_item.added":
      if (e.item?.type === "function_call") {
        stage(d, `call_pending_${e.output_index}`, {
          cls: "tool", label: `backend is composing a call to ${e.item.name}`, text: "streaming arguments…",
        });
      }
      break;

    case "response.function_call_arguments.delta": {
      const node = d.stages[`call_pending_${e.output_index}`];
      if (node) node.querySelector(".stext").textContent = (node.querySelector(".stext").textContent.replace("streaming arguments…", "")) + (e.delta || "");
      break;
    }

    case "response.output_item.done":
      onOutputItemDone(d, e, backend);
      break;

    case "response.completed":
    case "response.incomplete":
    case "response.failed":
      onResponseFinished(d, e, backend);
      break;
  }
}

function onOutputItemDone(d, e, backend) {
  const item = e.item || {};

  if (item.type === "function_call" && CLIENT_TOOL_NAMES.includes(item.name)) {
    d.pendingCalls.push(item);
    delete d.stages[`call_pending_${e.output_index}`];
    stage(d, `call_${item.call_id}`, {
      cls: "tool",
      label: `3 · ${backend} → client · function_call ${item.name}`,
      text: `call_id: ${item.call_id}\nThe backend asked the client to run this. It reaches the browser inside a response.event envelope.`,
      payload: { call_id: item.call_id, name: item.name, arguments: safeParse(item.arguments) },
    });
    logEvent({
      cat: "tool", dir: "in", actor: "thinker",
      type: `function_call → ${item.name}`,
      text: `call_id ${item.call_id}`,
      payload: item,
    });
  } else if (item.type === "function_call") {
    stage(d, `call_${item.call_id || e.output_index}`, {
      cls: "warn", label: `unknown function ${item.name}`,
      text: "No client implementation registered for this name.", payload: item,
    });
    d.pendingCalls.push(item);
  } else if (item.type === "message") {
    const text = messageText(item);
    if (text) {
      d.answer = text;
      stage(d, "answer", { cls: "answer", label: `${backend} result text → returned to the frontend`, text });
    }
  }
}

async function onResponseFinished(d, e, backend) {
  if (d.pendingCalls.length) {
    const calls = d.pendingCalls.splice(0);
    stage(d, "exec", {
      cls: "tool",
      label: "4 · client executes the function calls",
      text: calls.map((c) => `${c.name}(${c.arguments})`).join("\n"),
    });
    await runToolCalls(d, calls, backend);
    return;
  }

  if (e.type === "response.completed") {
    setStatus(d, "complete");
    stage(d, "handoff", {
      cls: "spoken",
      label: "5 · what the frontend sees",
      text:
        `The backend's result is handed back to the frontend as the delegation's answer.\n` +
        `It phrases it in its own voice — the exact wording is not guaranteed.\n` +
        `Backend tokens: ${JSON.stringify(e.response?.usage || {})}`,
    });
    app.speechTarget = d;
    clearTimeout(app.speechTimer);
    app.speechTimer = setTimeout(() => { app.speechTarget = null; }, 12000);
  } else {
    setStatus(d, e.type.replace("response.", ""));
    stage(d, "handoff", { cls: "warn", label: "backend did not complete", text: e.type, payload: e.response || e });
  }
}

async function runToolCalls(d, calls, backend) {
  // Optional spoken progress while a slow tool runs. This is a client-authored update to the frontend,
  // independent of the backend response.
  const slow = calls.find((c) => c.name === "place_order" && safeParse(c.arguments)?.confirmed);
  if (slow && els.optAutoCommentary.checked) {
    send({
      type: "session.commentary.append",
      event_id: nextEventId("commentary"),
      delegation_id: null, // responses delegation rejects non-null IDs
      content: "One moment — I'm sending that order to the counter now.",
    }, { cat: "context", text: "auto progress update while place_order runs", actor: "client" });
  }

  await Promise.all(calls.map(async (call) => {
    const t0 = nowMs();
    const { args, result, latencyMs } = await executeTool(call.name, call.arguments);
    app.toolCallCount += 1;
    els.mTools.textContent = app.toolCallCount;
    renderCart();

    stage(d, `result_${call.call_id}`, {
      cls: "tool",
      label: `client → ${backend} · function_call_output (${latencyMs}ms)`,
      text: `The client sends this back on the data channel as response.item.create.`,
      payload: { call_id: call.call_id, name: call.name, args, output: result },
    });

    send({
      type: "response.item.create",
      event_id: nextEventId("tool_result"),
      item: {
        type: "function_call_output",
        call_id: call.call_id,
        output: JSON.stringify(result),
      },
    }, { cat: "tool", text: `${call.name} result (${Math.round(nowMs() - t0)}ms)`, actor: "client" });
  }));

  // One continuation for the whole batch, after every pending result is queued.
  send({ type: "response.create", event_id: nextEventId("continue") },
    { cat: "tool", text: "continue the delegated response with the tool results", actor: "client" });
  stage(d, "continue", {
    cls: "tool",
    label: "client → backend · response.create",
    text: "Explicitly resumes the delegated response now that every function result is in.",
  });
}

const safeParse = (s) => { try { return JSON.parse(s); } catch { return s; } };
const messageText = (item) =>
  (typeof item.content === "string" ? item.content : (item.content || []).map((c) => c.text || "").join(""));

// ---------------------------------------------------------------------------------
// CLIENT DELEGATION
//
// The contrast with responses delegation is the whole point:
//   - The server tells us *that* the frontend wants help, never *what with*. No task text.
//   - We reconstruct the request from our own transcript buffer + application state.
//   - We pick the model, own the prompt, and run the ENTIRE tool loop ourselves (one backend round at a time
//     through POST /v1/live/delegate). The frontend never sees a single one of those tool calls.
//   - The only thing that goes back to the frontend is plain text in a commentary append, which it speaks.
// ---------------------------------------------------------------------------------
async function runClientDelegation(d) {
  const model = backendModel();
  // A non-null delegation_id is legal only for a delegation the server actually created.
  // Our locally-triggered ones must use null or the append is rejected.
  const delegationId = d.synthetic ? null : d.id;

  // 1. Quiet progress, tied to THIS delegation. A non-null delegation_id is legal here — and only here.
  send({
    type: "session.thinking.append",
    event_id: nextEventId("progress"),
    delegation_id: delegationId,
    content: "Working on that now. Nothing has been ordered yet.",
  }, { cat: "context", text: `quiet progress for ${d.id}`, actor: "client" });

  // 2. Rebuild the request. This is the work responses delegation does for you.
  const reconstructed =
    `Recent voice transcript (may contain errors or corrections):\n${transcriptContext() || "(nothing transcribed yet)"}\n\n` +
    `Current cart state: ${JSON.stringify(cartSnapshot())}\n\n` +
    `The voice assistant delegated the user's latest request to you. Handle it and return ` +
    `a short result suitable for reading aloud.`;

  stage(d, "reconstruct", {
    cls: "warn",
    label: "2 · client reconstructs the request",
    text:
      "session.delegation.created carried NO task text — only an id, target and offset.\n" +
      "We rebuild intent from our own transcript buffer plus application state.",
    payload: reconstructed,
  });

  app.clientHistory.push({ type: "message", role: "user", content: [{ type: "input_text", text: reconstructed }] });
  const tools = (() => { try { return JSON.parse(els.optTools.value || "[]"); } catch { return []; } })();
  let finalText = "";

  for (let round = 1; round <= 6; round += 1) {
    let data;
    try {
      const res = await fetch("/v1/live/delegate", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          input: app.clientHistory,
          model,
          instructions: els.optBackendInstructions.value.trim() || undefined,
          tools,
        }),
      });
      data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data).slice(0, 400));
    } catch (error) {
      stage(d, "bkerr", { cls: "warn", label: "our backend failed", text: String(error) });
      setStatus(d, "failed");
      return;
    }

    const output = data.response.output || [];
    app.clientHistory.push(...output);
    logEvent({
      cat: "delegation", dir: "out", actor: "client",
      type: `POST /v1/live/delegate (round ${round})`,
      text: `our own backend call → ${model} · the frontend sees none of this`,
      payload: { sent_request: data.sent_request, response_id: data.response.id },
    });

    if (round === 1) {
      stage(d, "ourcall", {
        cls: "tool",
        label: `3 · client → our backend · POST /v1/live/delegate (${model})`,
        text:
          "We chose this model, prompt and toolset. None of it was declared to the session.\n" +
          "We keep the conversation and send it whole each round.",
        payload: { model: data.sent_request.model, tools: data.sent_request.tools },
      });
    }

    const calls = output.filter((o) => o.type === "function_call");
    if (!calls.length) {
      finalText = output.filter((o) => o.type === "message").map(messageText).join("");
      break;
    }

    // 3. OUR tool loop: we run these ourselves, in-process. The session is not involved at any point.
    stage(d, `ocall_${round}`, {
      cls: "tool",
      label: `4 · our backend asks for tools (round ${round})`,
      text: calls.map((c) => `${c.name}(${c.arguments})`).join("\n") +
        "\n\nThese never reach the session. There is no response.item.create here.",
    });

    const outputs = await Promise.all(calls.map(async (call) => {
      const { args, result, latencyMs } = await executeTool(call.name, call.arguments);
      app.toolCallCount += 1;
      els.mTools.textContent = app.toolCallCount;
      renderCart();
      logEvent({
        cat: "tool", dir: "out", actor: "client",
        type: `client tool ${call.name} (client delegation)`,
        text: `${latencyMs}ms — result goes to OUR backend, not to the session`,
        payload: { args, result },
      });
      return { type: "function_call_output", call_id: call.call_id, output: JSON.stringify(result) };
    }));

    stage(d, `oresult_${round}`, {
      cls: "tool",
      label: `5 · client → our backend · function_call_output (round ${round})`,
      text: "Added to our conversation and sent with the next round.",
      payload: outputs.map((o) => ({ call_id: o.call_id, output: safeParse(o.output) })),
    });
    app.clientHistory.push(...outputs);
  }

  if (!finalText) finalText = "I could not complete that. Let me try again.";

  stage(d, "answer", {
    cls: "answer",
    label: `our backend's verified result (${model})`,
    text: finalText,
  });

  // 4. The one and only thing the frontend receives. We could validate, redact, or drop it first — that review
  //    step is the reason to choose client delegation.
  send({
    type: "session.commentary.append",
    event_id: nextEventId("result"),
    delegation_id: delegationId,
    content: finalText.slice(0, 1200),
  }, { cat: "context", text: `result → frontend for ${d.id}`, actor: "client" });

  stage(d, "handoff", {
    cls: "spoken",
    label: "6 · client → frontend · session.commentary.append",
    text:
      `delegation_id: ${delegationId ?? "null (locally triggered, not a server delegation)"}\n` +
      "Commentary is spoken aloud; the frontend phrases it in its own voice.\n" +
      "session.thinking.append would deliver the same facts silently instead.",
  });

  setStatus(d, "complete");
  app.speechTarget = d;
  clearTimeout(app.speechTimer);
  app.speechTimer = setTimeout(() => { app.speechTarget = null; }, 12000);
}

// ---------------------------------------------------------------------------------
// Server event router
// ---------------------------------------------------------------------------------
function onServerEvent(ev) {
  switch (ev.type) {
    case "session.started": {
      app.ready = true;
      app.sessionId = ev.session?.id;
      app.sessionSnapshot = ev.session;
      els.mSession.textContent = app.sessionId || "—";
      setConnState("live", `Connected · ${app.sessionId}`);
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: ev.type, text: `session ${app.sessionId} active`, payload: ev });
      enableControls(true);
      break;
    }

    case "session.updated":
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: ev.type, text: "backend settings applied", payload: ev });
      if (ev.session?.delegation?.responses?.model) {
        els.badgeBackend.textContent = ev.session.delegation.responses.model;
        if (app.sentConfig?.delegation?.responses) app.sentConfig.delegation.responses.model = ev.session.delegation.responses.model;
      }
      break;

    case "session.input_transcript.delta":
      detectBargeIn(ev);
      recordTranscript("user", ev.delta);
      caption("user", ev.delta, ev.start_ms, ev.end_ms);
      logEvent({ cat: "transcript", dir: "in", type: ev.type, text: ev.delta, payload: ev });
      break;

    case "session.output_transcript.delta":
      app.lastOutputAt = nowMs();
      recordTranscript("assistant", ev.delta);
      caption("assistant", ev.delta, ev.start_ms, ev.end_ms);
      logEvent({ cat: "transcript", dir: "in", actor: "talker", type: ev.type, text: ev.delta, payload: ev });
      if (app.speechTarget) {
        const d = app.speechTarget;
        d.spoken = (d.spoken || "") + ev.delta;
        stage(d, "spoken", {
          cls: "spoken", label: "6 · what the frontend actually said", text: d.spoken,
        });
      }
      break;

    case "session.delegation.created": {
      const d = createDelegationCard(ev);
      logEvent({
        cat: "delegation", dir: "in", actor: "talker", type: ev.type,
        text: `${ev.delegation.id} → ${ev.delegation.target}`, payload: ev,
      });
      if (ev.delegation.target === "client") runClientDelegation(d);
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
      app.muted = ev.type.endsWith("muted") && !ev.type.endsWith("unmuted");
      els.btnMute.textContent = app.muted ? "Unmute mic" : "Mute mic";
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: "mic input state changed (model output is unaffected)", payload: ev });
      break;

    case "session.usage.updated":
      els.mUsage.textContent = ev.usage?.seconds ?? "—";
      els.mCtx.textContent = ev.context_window?.usage_ratio != null
        ? `${Math.round(ev.context_window.usage_ratio * 100)}%` : "—";
      logEvent({ cat: "lifecycle", dir: "in", type: ev.type, text: `${ev.usage?.seconds}s voice`, payload: ev });
      break;

    case "session.closed":
      app.finalized = true;
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
  const sinceOutput = nowMs() - app.lastOutputAt;
  if (app.lastOutputAt === 0 || sinceOutput > 1500) return;
  if (nowMs() - app.lastBargeAt < 3000) return;
  app.lastBargeAt = nowMs();

  captionNote("⟂ barge-in — the user started speaking while the frontend was still speaking (full duplex).");
  logEvent({
    cat: "app", dir: "in", type: "app.barge_in_detected",
    text: `user speech at ${ev.start_ms}ms overlapped assistant speech ${Math.round(sinceOutput)}ms ago`,
    payload: { user_start_ms: ev.start_ms, ms_since_assistant_delta: Math.round(sinceOutput), delta: ev.delta },
  });

  for (const d of app.delegations.values()) {
    if (d.status !== "running") continue;
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
// Configuration
// ---------------------------------------------------------------------------------
function parseHistory(text) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((line) => {
    const at = line.indexOf(":");
    const role = at < 0 ? "" : line.slice(0, at).trim();
    if (!["developer", "user", "assistant"].includes(role)) {
      throw new Error(`Earlier messages: "${line}" must start with developer:, user: or assistant:`);
    }
    return {
      type: "message", role,
      content: [{ type: role === "assistant" ? "output_text" : "input_text", text: line.slice(at + 1).trim() }],
    };
  });
}

/** The `session` object of the session start: everything on the setup form. Throws a readable Error. */
function buildConfig() {
  const config = { model: frontendModel(), audio: { output: { voice: els.optVoice.value } } };
  const instructions = els.optInstructions.value.trim();
  if (instructions) config.instructions = instructions;
  const input = parseHistory(els.optHistory.value);
  if (input.length) config.input = input;

  if (els.optMode.value === "client") {
    config.delegation = { type: "client" };
    return config;
  }
  let tools;
  try { tools = JSON.parse(els.optTools.value || "[]"); } catch (e) { throw new Error(`Backend tools are not valid JSON: ${e.message}`); }
  const responses = { model: backendModel() };
  const backendInstructions = els.optBackendInstructions.value.trim();
  if (backendInstructions) responses.instructions = backendInstructions;
  if (tools.length) { responses.tools = tools; responses.tool_choice = els.optToolChoice.value; }
  if (els.optEffort.value !== "default") responses.reasoning = { effort: els.optEffort.value };
  config.delegation = { type: "responses", responses };
  return config;
}

function fillSelect(el, values, selected) {
  el.innerHTML = values.map((v) => `<option value="${v}"${v === selected ? " selected" : ""}>${v}</option>`).join("");
}

/** Fill the form from GET /v1/live/info. */
function applyInfo() {
  const info = app.info;
  fillSelect(els.optVoice, info.voices, info.voices.includes("marin") ? "marin" : info.voices[0]);
  fillSelect(els.optEffort, REASONING_EFFORTS, "default");
  fillSelect(els.ctlUpdEffort, REASONING_EFFORTS, "default");
  fillSelect(els.optToolChoice, TOOL_CHOICES, "auto");
  fillSelect(els.ctlUpdToolChoice, ["unchanged", ...TOOL_CHOICES], "unchanged");
  els.backendModels.innerHTML = (info.backend_models || []).map((m) => `<option value="${m}"></option>`).join("");
  els.optBackend.value = info.backend?.model || "";
  els.ctlUpdModel.value = info.backend?.model || "";
  els.optInstructions.value = info.instructions?.frontend || "";
  els.optBackendInstructions.value = info.instructions?.backend || "";
  els.optTools.value = JSON.stringify(info.sample_tools || [], null, 2);
  els.badgeFrontend.textContent = info.frontend?.model || "—";
  els.noteFrontend.textContent = `${info.frontend?.provider || "frontend"} · prompts ${info.prompt_version || ""}`;
  updateBackendBadge();
  for (const warning of info.warnings || []) {
    logEvent({ cat: "error", dir: "in", type: "server.warning", text: warning });
  }
}

function updateBackendBadge() {
  const client = els.optMode.value === "client";
  els.badgeBackend.textContent = client ? `${backendModel()} (ours)` : backendModel();
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
}

function enableControls(on) {
  els.btnStop.disabled = !on;
  els.btnMute.disabled = !on;
  els.btnAppend.disabled = !on;
  els.btnUpdate.disabled = !on;
  els.btnTyped.disabled = !on;
  els.btnRaw.disabled = !on;
}

async function connect() {
  els.btnStart.disabled = true;
  app.finalized = false;
  setConnState("", "Connecting…");

  try {
    const config = buildConfig();
    app.mode = els.optMode.value;
    app.clientHistory = [];
    const pc = new RTCPeerConnection();
    app.pc = pc;

    pc.addEventListener("track", (event) => {
      els.player.srcObject = new MediaStream([event.track]);
      els.player.play().catch(() => setConnState("live", "Press play on the audio control to hear the assistant."));
      logEvent({ cat: "lifecycle", dir: "in", actor: "talker", type: "webrtc.track", text: "remote audio track attached (frontend speech)" });
    });
    pc.addEventListener("connectionstatechange", () => {
      logEvent({ cat: "lifecycle", dir: "in", type: "webrtc.connectionstate", text: pc.connectionState });
    });

    app.mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    for (const track of app.mic.getAudioTracks()) pc.addTrack(track, app.mic);
    logEvent({ cat: "lifecycle", dir: "out", type: "webrtc.microphone", text: "mic track added — audio goes to the server over WebRTC" });

    // Register handlers before creating the offer.
    const dc = pc.createDataChannel("oai-events");
    app.dc = dc;
    dc.addEventListener("message", ({ data }) => {
      let ev;
      try { ev = JSON.parse(data); } catch { return; }
      onServerEvent(ev);
    });
    dc.addEventListener("close", () => {
      if (!app.finalized) {
        logEvent({ cat: "error", dir: "in", type: "datachannel.close", text: "closed before session.closed — final usage unconfirmed" });
        setConnState("err", "Disconnected without final usage.");
        cleanup();
      }
    });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIce(pc);

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

    app.sentConfig = result.sent_session_config;
    updateBackendBadge();
    logEvent({
      cat: "lifecycle", dir: "in", type: "POST /v1/live/sessions → 201",
      text: `session ${result.session?.id} created with the config below`,
      payload: { session: result.session, sent_session_config: app.sentConfig },
    });

    await pc.setRemoteDescription({ type: "answer", sdp: result.transport.sdp });
    // The HTTP request started the session; do not send session.start here.
    setConnState("", "Negotiated — waiting for session.started…");
  } catch (error) {
    logEvent({ cat: "error", dir: "in", type: "connect.failed", text: String(error.message || error).slice(0, 400), payload: String(error) });
    setConnState("err", "Connection failed — see console");
    cleanup();
  }
}

function waitForIce(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", check);
      reject(new Error("Timed out gathering ICE candidates"));
    }, 10000);
    function check() {
      if (pc.iceGatheringState !== "complete") return;
      clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    }
    pc.addEventListener("icegatheringstatechange", check);
    check();
  });
}

function gracefulClose() {
  if (!app.ready || !app.dc || app.dc.readyState !== "open") return;
  els.btnStop.disabled = true;
  setConnState("", "Closing — draining pending events…");
  // The session.closed handler is already registered; keep everything alive until it arrives.
  send({ type: "session.close", event_id: nextEventId("close") },
    { cat: "lifecycle", text: "queued backend work is cancelled; an active reply may finish" });
  app.closeTimer = setTimeout(() => {
    logEvent({ cat: "error", dir: "in", type: "close.timeout", text: "no session.closed within 15s — finalization unconfirmed" });
    cleanup();
  }, 15000);
}

function cleanup() {
  clearTimeout(app.closeTimer);
  app.mic?.getTracks().forEach((t) => t.stop());
  try { app.dc?.close(); } catch {}
  try { app.pc?.close(); } catch {}
  els.player.srcObject = null;
  app.ready = false;
  app.dc = null;
  els.btnStart.disabled = false;
  enableControls(false);
}

// ---------------------------------------------------------------------------------
// Client state rendering
// ---------------------------------------------------------------------------------
function renderCart() {
  const cart = cartSnapshot();
  const lines = cart.lines.map((l) =>
    `<div class="line"><span>${l.quantity}× ${l.size ? l.size + " " : ""}${l.name}` +
    `${l.modifiers.length ? " (" + l.modifiers.join(", ") + ")" : ""} <span style="color:#626262">${l.line_id}</span></span>` +
    `<span>$${l.line_total.toFixed(2)}</span></div>`).join("");
  els.cartView.innerHTML =
    (lines || `<div class="line" style="color:#626262">cart is empty</div>`) +
    `<div class="line tot"><span>subtotal</span><span>$${cart.subtotal.toFixed(2)}</span></div>` +
    `<div class="line"><span>tax</span><span>$${cart.tax.toFixed(2)}</span></div>` +
    `<div class="line"><span><b>total</b></span><span><b>$${cart.total.toFixed(2)}</b></span></div>` +
    (state.lastOrder
      ? `<div class="order">order ${state.lastOrder.order_id} for ${state.lastOrder.customer_name} · ready in ${state.lastOrder.ready_in_minutes} min</div>`
      : "");
}

function renderMenu() {
  els.menuView.innerHTML = MENU.map((m) =>
    `<div class="mitem"><span>${m.id} · ${m.name} <span style="color:#626262">${m.category}</span></span><span>$${m.price.toFixed(2)}</span></div>`
  ).join("") +
    `<div class="mitem" style="margin-top:6px"><span>sizes</span><span>${Object.entries(SIZES).map(([k, v]) => `${k} +$${v.toFixed(2)}`).join(" · ")}</span></div>` +
    `<div class="mitem"><span>modifiers</span><span>${Object.entries(MODIFIERS).map(([k, v]) => `${k} +$${v.price.toFixed(2)}`).join(" · ")}</span></div>`;
}

// ---------------------------------------------------------------------------------
// Append presets
// ---------------------------------------------------------------------------------
const PRESETS = {
  "session.instructions.append": [
    ["Greet first", "Greet the caller now in English. Introduce yourself as the Bluebird Cafe assistant and ask what they would like. Then pause and listen."],
    ["Recording disclosure", "Immediately say the following disclosure exactly and in full before responding to the caller: This call may be recorded for quality and training purposes."],
    ["Guardrail redirect", "Stop speaking about that request. Briefly explain that you cannot help with it, then wait for the user."],
    ["Slow down", "Speak more slowly and leave a short pause after each item you read back."],
  ],
  "session.thinking.append": [
    ["UI context", "The user is looking at the pastry section of the kiosk screen. 'this one' most likely refers to the butter croissant."],
    ["Loyalty status", "This caller is a loyalty member; the counter applies their discount at pickup. Do not quote a discounted price."],
    ["Store state", "The espresso machine is busy; drink orders are running about three minutes behind."],
  ],
  "session.commentary.append": [
    ["Still working", "I'm still checking that for you."],
    ["Taking longer", "This is taking a little longer than usual — I'm still on it."],
    ["Pickup counter", "Your order will be at the far end of the counter when it's ready."],
  ],
};

function refreshPresets() {
  const list = PRESETS[els.ctlKind.value] || [];
  els.ctlPreset.innerHTML = list.map((p, i) => `<option value="${i}">${p[0]}</option>`).join("");
  els.ctlContent.value = list[0]?.[1] || "";
}

// ---------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------
async function loadInfo() {
  const response = await fetch("/v1/live/info", { headers: authHeaders() });
  if (!response.ok) {
    throw new Error(response.status === 401
      ? "The server requires a bearer token: enter it above, then reload the defaults."
      : `GET /v1/live/info failed (HTTP ${response.status}). Is the live example enabled on this server?`);
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

async function boot() {
  refreshPresets();
  renderMenu();
  renderCart();
  enableControls(false);
  els.btnStart.disabled = true;
  try {
    await loadInfo();
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
    send({ type: app.muted ? "session.input_audio.unmute" : "session.input_audio.mute", event_id: nextEventId("mic") },
      { cat: "lifecycle", text: app.muted ? "resume mic input" : "mute mic input (does not stop the frontend speaking)" });
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
    if (app.mode === "client") {
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
    if (app.mode === "client") {
      // response.item.create / response.create require responses delegation. In client mode typed text goes
      // straight to our own backend instead.
      recordTranscript("user", text);
      logEvent({
        cat: "delegation", dir: "out", actor: "client",
        type: "typed input (client delegation)",
        text: "no session command exists for this — sent to our own backend",
        payload: { text },
      });
      const ev = { delegation: { id: `local_${Date.now()}`, target: "client" }, offset_ms: 0 };
      const d = createDelegationCard(ev);
      d.synthetic = true;  // not a real session delegation -> appends must use null
      runClientDelegation(d);
      els.typedInput.value = "";
      return;
    }
    send({
      type: "response.item.create",
      event_id: nextEventId("typed"),
      item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
    }, { cat: "delegation", text: "queue exact typed text for the backend", actor: "client" });
    send({ type: "response.create", event_id: nextEventId("typed_run") },
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
    els.ctlContent.value = PRESETS[els.ctlKind.value][els.ctlPreset.value][1];
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
  els.btnClear.addEventListener("click", () => { els.log.innerHTML = ""; app.logEntries = []; });
  els.btnExport.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(app.logEntries, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `live-session-${app.sessionId || "log"}.json`;
    a.click();
  });

  const showModal = (title, body) => {
    els.modalTitle.textContent = title;
    els.modalBody.textContent = body;
    els.modal.hidden = false;
  };
  els.modalClose.addEventListener("click", () => { els.modal.hidden = true; });
  els.btnShowConfig.addEventListener("click", () => {
    let config = app.sentConfig;
    let title = "Session config (as sent)";
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
