// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The history panel's entries from GET /v1/live/sessions/{id}/history, a {frontend, backend} snapshot of what the
// talker and the backend hold. `backend` is null in client mode, where the server runs no backend.

export const HISTORY_POLL_MS = 1000;
// The prefixes the talker's memory marks reference data with (src/examples/frontend_backend_live/cascade/memory.py).
export const RESULT_PREFIX = "Backend result (reference data): ";
export const CONTEXT_PREFIX = "Application context (reference data): ";

/** Whether the panel's poll fetches: only while the panel is open and the page is visible. */
export const shouldPoll = (panelOpen, visibility) => Boolean(panelOpen) && visibility === "visible";

/** What tells one answer from the last: its text and, in client mode, the page's own delegation and history length. */
export const snapshotKey = (text, local) => (local ? `${text}\n${local.delegationId}:${local.history.length}` : text);

/** A line the next delegation carries, as the server labels it ("Caller: …", "Agent: …", "App context: …"). */
export function pendingEntry(line) {
  const at = line.indexOf(": ");
  const label = at > 0 ? line.slice(0, at) : "";
  const text = at > 0 ? line.slice(at + 2) : line;
  const cls = { Caller: "user", Agent: "assistant" }[label] || "context";
  return { cls: `${cls} pending`, role: `next delegation · ${label || "line"}`, text };
}

/** Where a prompt part comes from: the client over the session, the server's config, or the server's code. */
export function promptOrigin(source) {
  if (source.startsWith("session.") || source.startsWith("delegation.")) return "client";
  if (source.startsWith("prompts.yaml")) return "config";
  return "server";
}

/** The system prompt of the role's next call, one collapsed entry per part. */
export function promptEntries(parts) {
  return (parts || []).map((part) => {
    const origin = promptOrigin(part.source);
    return {
      cls: `prompt ${origin}`,
      role: `system prompt · ${origin} · ${part.source}`,
      text: part.text,
      collapsed: `${part.text.split("\n")[0].slice(0, 90)} … (${part.text.length} characters)`,
    };
  });
}

export function talkerEntry(message) {
  const text = contentText(message.content);
  if (message.role === "system" || message.role === "developer") {
    return { cls: "system", role: `${message.role} message`, text, collapsed: `${text.length} characters` };
  }
  if (message.role === "user" && text.startsWith(CONTEXT_PREFIX)) {
    return { cls: "context", role: "application context", text: text.slice(CONTEXT_PREFIX.length) };
  }
  if (message.role === "user" && text.startsWith(RESULT_PREFIX)) {
    return { cls: "result", role: "backend result (reference data)", text: text.slice(RESULT_PREFIX.length) };
  }
  return { cls: message.role, role: message.role, text };
}

/**
 * One item of the backend's history. A message shows its phase (`commentary` is progress, `final_answer` the answer);
 * an `mcp_call` is a tool the backend ran itself (or, named `show`, the display), with its outcome.
 */
export function backendEntry(item) {
  switch (item.type) {
    case "function_call":
      return { cls: "call", role: `function_call · ${item.call_id || ""}`, text: `${item.name}(${item.arguments || ""})` };
    case "function_call_output":
      return { cls: "output", role: `function_call_output · ${item.call_id || ""}`, text: String(item.output ?? "") };
    case "mcp_call":
      return {
        cls: "call",
        role: `mcp_call · ${item.server_label || ""} · ${item.status || ""}`,
        text: `${item.name}(${item.arguments || ""}) → ${item.output ?? item.error?.content ?? ""}`,
      };
    case "reasoning":
      return { cls: "reasoning", role: "reasoning", text: (item.summary || []).map((s) => s.text || "").join("\n") || "(hidden)" };
    case "message":
    case undefined: {
      const role = item.role || "message";
      return { cls: role, role: item.phase ? `${role} · ${item.phase}` : role, text: contentText(item.content) };
    }
    default:
      return { cls: "message", role: item.type, text: JSON.stringify(item) };
  }
}

export function contentText(content) {
  if (typeof content === "string") return content;
  return (content || []).map((part) => (typeof part === "string" ? part : part.text ?? JSON.stringify(part))).join("\n");
}

/** For each column of the panel: its model, its count line, its entries in order and its text when there are none. */
export function historyView({ frontend, backend }) {
  const talker = frontend?.messages || [];
  const view = {
    frontend: {
      model: frontend?.model || "",
      count: `${talker.length} messages`,
      entries: [
        ...promptEntries(frontend?.prompt),
        ...talker.filter((m, i) => !(i === 0 && m.role === "system")).map(talkerEntry),
      ],
      empty: "The talker holds no messages yet.",
    },
  };
  if (!backend) {
    view.backend = { model: "", count: "", entries: [], empty: "No server backend in this session." };
    return view;
  }
  const items = backend.items || [];
  const pending = (Array.isArray(backend.pending) ? backend.pending : []).map(pendingEntry);
  view.backend = {
    model: backend.model || "",
    count: `${items.length} items${backend.busy ? " · working" : ""}${pending.length ? ` · ${pending.length} waiting` : ""}`,
    entries: [...promptEntries(backend.prompt), ...items.map(backendEntry), ...pending],
    empty: "The backend has not been delegated anything yet.",
  };
  return view;
}
