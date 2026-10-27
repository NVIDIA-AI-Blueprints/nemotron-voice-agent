// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The session's settings from the setup form (client/public/live/config.js). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ICE_GATHER_MS,
  buildConfig,
  hasTurn,
  iceServerList,
  parseHistory,
  parseTools,
  sessionMode,
  tokenFromHash,
  waitForIceGathering,
} from "../../public/live/config.js";

const FORM = {
  mode: "responses",
  frontendModel: "talker-model",
  voice: "marin",
  instructions: "  You are a kiosk.  ",
  history: "",
  tools: "[]",
  backendModel: "backend-model",
  backendInstructions: "",
  toolChoice: "auto",
  effort: "default",
};
const TOOL = { type: "function", name: "get_menu", parameters: { type: "object" } };

test("earlier messages: one role: text per line", () => {
  assert.deepEqual(parseHistory("user: hi\n\n  assistant: Welcome.  "), [
    { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
    { type: "message", role: "assistant", content: [{ type: "output_text", text: "Welcome." }] },
  ]);
  assert.throws(() => parseHistory("bot: hi"), /must start with developer:, user: or assistant:/);
});

test("responses mode: the backend's settings go into delegation.responses", () => {
  const config = buildConfig({
    ...FORM, tools: JSON.stringify([TOOL]), backendInstructions: " Be brief ", toolChoice: "required", effort: "low",
  });
  assert.deepEqual(config, {
    model: "talker-model",
    audio: { output: { voice: "marin" } },
    instructions: "You are a kiosk.",
    delegation: {
      type: "responses",
      responses: {
        model: "backend-model", instructions: "Be brief", tools: [TOOL], tool_choice: "required",
        reasoning: { effort: "low" },
      },
    },
  });
});

test("responses mode: no tools sends no tool_choice; bad tools JSON stops the start", () => {
  assert.deepEqual(buildConfig(FORM).delegation, { type: "responses", responses: { model: "backend-model" } });
  assert.throws(() => buildConfig({ ...FORM, tools: "[{" }), /Backend tools are not valid JSON/);
});

test("client mode: the session gets no backend settings", () => {
  const config = buildConfig({ ...FORM, mode: "client", history: "user: hi" });
  assert.deepEqual(config.delegation, { type: "client" });
  assert.equal(config.input.length, 1);
});

test("the token in the link keeps '+' and is decoded once", () => {
  assert.equal(tokenFromHash("#token=a+b%2Bc"), "a+b+c");
  assert.equal(tokenFromHash("#token=YWJj+ZGVm/Z2hp=="), "YWJj+ZGVm/Z2hp==");
  assert.equal(tokenFromHash("#x=1&token=abc%3D%3D&y=2"), "abc==");
  assert.equal(tokenFromHash("#token=100%"), "100%");  // not valid percent-encoding: taken as it is
  assert.equal(tokenFromHash("#token="), "");
  assert.equal(tokenFromHash("#other=1"), null);
  assert.equal(tokenFromHash(""), null);
});

test("the mode is the resolved session's; an omitted delegation is client mode", () => {
  assert.equal(sessionMode({ delegation: { type: "responses", responses: { model: "m" } } }), "responses");
  assert.equal(sessionMode({ delegation: { type: "client" } }), "client");
  assert.equal(sessionMode({ delegation: null }), "client");
  assert.equal(sessionMode({ delegation: {} }), "client");
  assert.equal(sessionMode({}), "client");
});

test("the peer's ICE servers are the server's list, TURN credentials included", () => {
  const turn = { urls: ["turn:example.test:3478?transport=udp", "turn:example.test:3478?transport=tcp"], username: "u", credential: "p" };
  assert.deepEqual(iceServerList({ iceServers: [turn, { urls: "stun:stun.example.test" }, { urls: [] }, { username: "x" }, null] }), [
    turn,
    { urls: "stun:stun.example.test" },
  ]);
  assert.deepEqual(iceServerList({ iceServers: [] }), []);
  assert.deepEqual(iceServerList(null), []);
  assert.deepEqual(iceServerList({ detail: "Not Found" }), []);
  assert.equal(hasTurn([turn]), true);
  assert.equal(hasTurn([{ urls: "stun:stun.example.test" }]), false);
  assert.equal(hasTurn([{ urls: "turns:secure.example.test" }]), true);
});

/** A peer whose ICE gathering is in `state`, with a count of its gathering listeners. */
function gatheringPeer(state) {
  const pc = new EventTarget();
  pc.iceGatheringState = state;
  pc.listeners = 0;
  const add = pc.addEventListener.bind(pc);
  const remove = pc.removeEventListener.bind(pc);
  pc.addEventListener = (...args) => { pc.listeners += 1; add(...args); };
  pc.removeEventListener = (...args) => { pc.listeners -= 1; remove(...args); };
  pc.gathered = () => { pc.iceGatheringState = "complete"; pc.dispatchEvent(new Event("icegatheringstatechange")); };
  return pc;
}

test("the offer waits for ICE gathering, and at the limit goes with what was gathered", async () => {
  assert.equal(await waitForIceGathering(gatheringPeer("complete")), true);

  const finishes = gatheringPeer("gathering");
  const waiting = waitForIceGathering(finishes, 1000);
  finishes.gathered();
  assert.equal(await waiting, true);
  assert.equal(finishes.listeners, 0);

  // A TURN server the browser cannot reach keeps gathering going: the wait ends at its limit and does not fail.
  const stuck = gatheringPeer("gathering");
  assert.equal(await waitForIceGathering(stuck, 20), false);
  assert.equal(stuck.listeners, 0);
  assert.equal(ICE_GATHER_MS, 10000);
});

test("bad tools JSON stops the start in client mode too, with the same message", () => {
  for (const mode of ["responses", "client"]) {
    assert.throws(() => buildConfig({ ...FORM, mode, tools: "[{" }), /^Error: Backend tools are not valid JSON/);
    assert.throws(() => buildConfig({ ...FORM, mode, tools: "{\"type\": \"function\"}" }), /must be a JSON list/);
  }
  assert.deepEqual(parseTools(JSON.stringify([TOOL])), [TOOL]);
  assert.deepEqual(parseTools(""), []);
});
