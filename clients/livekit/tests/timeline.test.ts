// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { buildTimeline, isStreaming, type EventItem, type TranscriptItem } from "../src/lib/timeline.ts";

test("the timeline merges transcripts and events in time order, without empty segments", () => {
  const transcript = (id: string, at: number, text = "hi"): TranscriptItem => ({
    kind: "transcript", id, speaker: "user", text, streaming: false, at,
  });
  const event: EventItem = { kind: "event", id: "e", entry: { icon: "", text: "x", tone: "info" }, at: 2 };
  const timeline = buildTimeline([transcript("b", 3), transcript("a", 1), transcript("blank", 0, " ")], [event]);
  assert.deepEqual(timeline.map((item) => item.id), ["a", "e", "b"]);
});

test("an agent segment is done once final or once the agent stops speaking", () => {
  assert.equal(isStreaming("agent", "true", true), false);
  assert.equal(isStreaming("agent", undefined, false), false);
  assert.equal(isStreaming("agent", undefined, true), true);
  assert.equal(isStreaming("user", undefined, false), true);
});
