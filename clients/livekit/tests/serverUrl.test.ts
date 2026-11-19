// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveServerUrl } from "../src/lib/serverUrl.ts";

test("the signaling address follows the page host and protocol on port 7880", () => {
  assert.equal(resolveServerUrl({ protocol: "http:", hostname: "gpu-box" }, ""), "ws://gpu-box:7880");
  assert.equal(resolveServerUrl({ protocol: "https:", hostname: "voice.example.org" }, ""), "wss://voice.example.org:7880");
});

test("a public LiveKit address is used only on its own host", () => {
  const configured = "wss://voice.example.org/livekit";
  assert.equal(resolveServerUrl({ protocol: "https:", hostname: "voice.example.org" }, configured), configured);
  assert.equal(resolveServerUrl({ protocol: "http:", hostname: "localhost" }, configured), "ws://localhost:7880");
  assert.equal(resolveServerUrl({ protocol: "http:", hostname: "localhost" }, "not a url"), "ws://localhost:7880");
});
