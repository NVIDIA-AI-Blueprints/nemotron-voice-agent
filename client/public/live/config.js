// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The session's settings, built from the setup form's values.

export const REASONING_EFFORTS = ["default", "minimal", "low", "medium", "high"];
export const TOOL_CHOICES = ["auto", "none", "required"];

/** How long the page waits for `session.started`: UNCONNECTED_TIMEOUT_SECONDS in src/live/gateway/webrtc.py. */
export const SESSION_START_MS = 30000;

/** The longest the page waits to gather ICE candidates before it sends the offer (it does not trickle candidates). */
export const ICE_GATHER_MS = 10000;

/**
 * The page's ICE servers from GET /api/ice-servers (`{iceServers: [...]}`), without entries that lack a URL. Any other
 * reply gives an empty list: host candidates only, enough on one machine or a LAN.
 */
export function iceServerList(data) {
  const servers = Array.isArray(data?.iceServers) ? data.iceServers : [];
  return servers
    .filter((server) => {
      const urls = Array.isArray(server?.urls) ? server.urls : [server?.urls];
      return urls.length > 0 && urls.every((url) => typeof url === "string" && url);
    })
    .map(({ urls, username, credential }) => ({
      urls, ...(username ? { username } : {}), ...(credential ? { credential } : {}),
    }));
}

/**
 * The bearer token in `live.html#token=…` (a fragment never reaches the server), or null. decodeURIComponent keeps the
 * `+` of a base64 token, which a query-string parser would turn into a space.
 */
export function tokenFromHash(hash) {
  for (const part of String(hash || "").replace(/^#/, "").split("&")) {
    if (!part.startsWith("token=")) continue;
    const raw = part.slice("token=".length);
    try { return decodeURIComponent(raw); } catch { return raw; }
  }
  return null;
}

/** The session's delegation mode from `session.started`; an omitted, `null` or `{}` delegation is client mode. */
export const sessionMode = (session) => session?.delegation?.type || "client";

/** Whether `servers` offer a TURN relay. */
export const hasTurn = (servers) => servers.some((server) => [server.urls].flat().some((url) => /^turns?:/i.test(url)));

/**
 * Wait at most `limitMs` for `pc` to gather its ICE candidates; true when gathering finished. An unreachable TURN
 * server can keep gathering going for 40 s or more, so at the limit the offer goes with the candidates so far, as
 * Pipecat's transport does.
 */
export function waitForIceGathering(pc, limitMs = ICE_GATHER_MS) {
  if (pc.iceGatheringState === "complete") return Promise.resolve(true);
  return new Promise((resolve) => {
    const finish = (complete) => {
      clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", check);
      resolve(complete);
    };
    const check = () => { if (pc.iceGatheringState === "complete") finish(true); };
    const timer = setTimeout(() => finish(false), limitMs);
    pc.addEventListener("icegatheringstatechange", check);
  });
}

/** The **Earlier messages** box: one `role: text` message per line. Throws a readable Error. */
export function parseHistory(text) {
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

/**
 * The **Backend tools** box: a JSON list of tools. Throws a readable Error. Responses mode sends the tools with the
 * session; client mode sends them with each of the page's own rounds.
 */
export function parseTools(text) {
  let tools;
  try { tools = JSON.parse(text || "[]"); } catch (e) { throw new Error(`Backend tools are not valid JSON: ${e.message}`); }
  if (!Array.isArray(tools)) throw new Error("Backend tools must be a JSON list of tools");
  return tools;
}

/**
 * The `session` object of the session start, from the setup form's values. Throws a readable Error.
 *
 * `form`: {mode, frontendModel, voice, instructions, history, tools, backendModel, backendInstructions, toolChoice,
 * effort}, each as the form holds it (text, or the selected value).
 */
export function buildConfig(form) {
  const config = { model: form.frontendModel, audio: { output: { voice: form.voice } } };
  const instructions = form.instructions.trim();
  if (instructions) config.instructions = instructions;
  const input = parseHistory(form.history);
  if (input.length) config.input = input;
  const tools = parseTools(form.tools);

  if (form.mode === "client") {
    config.delegation = { type: "client" };
    return config;
  }
  const responses = { model: form.backendModel };
  const backendInstructions = form.backendInstructions.trim();
  if (backendInstructions) responses.instructions = backendInstructions;
  if (tools.length) { responses.tools = tools; responses.tool_choice = form.toolChoice; }
  if (form.effort !== "default") responses.reasoning = { effort: form.effort };
  config.delegation = { type: "responses", responses };
  return config;
}
