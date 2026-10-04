// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Needs Node 22.6 or later: node --experimental-strip-types --test tests/conversation.test.ts
import assert from "node:assert/strict";
import test from "node:test";
import {
  hasContainingFinalizedTurn,
  isSpeakerLabeledTurn,
  resolveChronologicalAnchor,
  resolveSuppressionAnchor,
  speakerDisplayLabel,
  speakerTurnMessageId,
  splitAssistantMessagesAtTurnBoundaries,
  stripSilenceSentinel,
} from "../src/lib/conversation.ts";

test("suppresses every raw fragment contained by a later anchored turn", () => {
  const messages = [
    { text: "assistant summarize what we have covered", createdAt: "2026-09-22T06:51:27.627+00:00" },
    { text: "and list anything outstanding", createdAt: "2026-09-22T06:51:29.284+00:00" },
  ];
  const turns = [{
    text: "assistant summarize what we have covered and list anything outstanding",
    anchorCreatedAt: messages[1].createdAt,
  }];
  assert.equal(hasContainingFinalizedTurn(messages[0].text, messages[0].createdAt, turns), true);
  assert.equal(hasContainingFinalizedTurn(messages[1].text, messages[1].createdAt, turns), true);
});

test("advances the suppression anchor so a finalized turn covers a late raw bubble", () => {
  // Progress fires before the raw RTVI user bubble exists; the anchor lands on
  // whatever was last visible at that (too-early) moment.
  const beforeRaw = "2026-09-23T03:30:00.000Z";
  const progressAnchor = resolveSuppressionAnchor(undefined, beforeRaw, "2026-09-23T03:30:00.500Z");
  assert.equal(progressAnchor, beforeRaw);

  // The raw bubble now renders as the newest visible message, which the
  // progress anchor cannot cover.
  const rawBubbleCreatedAt = "2026-09-23T03:30:02.000Z";
  assert.equal(
    hasContainingFinalizedTurn("hello there", rawBubbleCreatedAt, [
      { text: "hello there world", anchorCreatedAt: progressAnchor },
    ]),
    false,
  );

  // Finalization upserts again while the raw bubble is newest, advancing the
  // anchor forward to cover the raw message time so the duplicate is suppressed.
  const finalizedAnchor = resolveSuppressionAnchor(
    progressAnchor,
    rawBubbleCreatedAt,
    "2026-09-23T03:30:02.500Z",
  );
  assert.equal(finalizedAnchor, rawBubbleCreatedAt);
  assert.equal(
    hasContainingFinalizedTurn("hello there", rawBubbleCreatedAt, [
      { text: "hello there world", anchorCreatedAt: finalizedAnchor },
    ]),
    true,
  );
});

test("never rewinds the suppression anchor and falls back when nothing is visible", () => {
  // Anchor only moves forward, so an older visible message cannot pull it back.
  assert.equal(
    resolveSuppressionAnchor(
      "2026-09-23T03:30:05.000Z",
      "2026-09-23T03:30:01.000Z",
      "2026-09-23T03:30:00.000Z",
    ),
    "2026-09-23T03:30:05.000Z",
  );
  // With no prior anchor and no visible message, the event timestamp is used.
  assert.equal(
    resolveSuppressionAnchor(undefined, undefined, "2026-09-23T03:30:00.000Z"),
    "2026-09-23T03:30:00.000Z",
  );
});

test("inserts late progress before an assistant message with the same timestamp", () => {
  const previousAssistant = "2026-09-23T10:11:42.394Z";
  const answeringAssistant = "2026-09-23T10:11:46.778Z";
  const alreadyRenderedMessages = [previousAssistant, answeringAssistant];

  assert.equal(
    resolveChronologicalAnchor(answeringAssistant, alreadyRenderedMessages),
    previousAssistant,
  );
});

test("inserts a late finalized turn before an already-rendered answer", () => {
  const previousAssistant = "2026-09-23T10:11:42.394Z";
  const userTurn = "2026-09-23T10:11:46.776Z";
  const answeringAssistant = "2026-09-23T10:11:47.005Z";
  const alreadyRenderedMessages = [previousAssistant, answeringAssistant];

  assert.equal(
    resolveChronologicalAnchor(userTurn, alreadyRenderedMessages),
    previousAssistant,
  );
});

test("does not suppress a distinct later repetition", () => {
  const turns = [{
    text: "No, wait, we ask people from New York",
    anchorCreatedAt: "2026-09-16T15:55:36.398+00:00",
  }];
  assert.equal(
    hasContainingFinalizedTurn(
      "No, wait",
      "2026-09-16T15:55:43.850+00:00",
      turns,
    ),
    false,
  );
});

test("labels an untagged multi-speaker turn Unknown", () => {
  assert.equal(speakerDisplayLabel(true, null, ""), "Unknown");
});

test("leaves an untagged single-speaker turn unlabeled", () => {
  assert.equal(speakerDisplayLabel(false, null, ""), undefined);
});

test("renders a finalized user turn only when the server labeled it", () => {
  assert.equal(isSpeakerLabeledTurn({ transcript: "hello", speaker_labeled: true }), true);
  assert.equal(isSpeakerLabeledTurn({ transcript: "hello", speaker_labeled: false }), false);
  assert.equal(isSpeakerLabeledTurn({ transcript: "hello", speaker_labeled: "true" }), false);
  assert.equal(isSpeakerLabeledTurn({ transcript: "hello" }), false);
});

test("uses one stable key for progress and final events from the same speaker run", () => {
  const progress = { turn_id: 3, run_index: 1, transcript: "hello" };
  const finalized = { turn_id: 3, run_index: 1, transcript: "hello there" };
  assert.equal(speakerTurnMessageId(progress, "progress"), "speaker-turn-3-1");
  assert.equal(speakerTurnMessageId(finalized, "final"), "speaker-turn-3-1");
  assert.equal(speakerTurnMessageId({ transcript: "legacy" }, "fallback"), "fallback");
});

test("drops a bot turn that is only silence sentinels", () => {
  assert.equal(stripSilenceSentinel("..."), "");
  assert.equal(stripSilenceSentinel(".. . .. . .. ."), "");
  assert.equal(stripSilenceSentinel("   "), "");
});

test("keeps only the prose after a sentinel prefix", () => {
  assert.equal(
    stripSilenceSentinel(".. . .. . .. . The recording captures a NASA crew member."),
    "The recording captures a NASA crew member.",
  );
  assert.equal(stripSilenceSentinel("... Wow this is excellent .. ."), "Wow this is excellent");
});

test("keeps a leading period that belongs to the word", () => {
  assert.equal(stripSilenceSentinel(".env stores configuration."), ".env stores configuration.");
  assert.equal(stripSilenceSentinel("...and then it stopped."), "...and then it stopped.");
  assert.equal(stripSilenceSentinel(".. . .env stores configuration."), ".env stores configuration.");
});

test("leaves ordinary sentence punctuation alone", () => {
  assert.equal(stripSilenceSentinel("Hello."), "Hello.");
  assert.equal(stripSilenceSentinel("Wow this is excellent isn't it."), "Wow this is excellent isn't it.");
  assert.equal(stripSilenceSentinel("One. Two. Three."), "One. Two. Three.");
});

test("splits assistant parts separated by a labeled user turn", () => {
  const messages = [{
    role: "assistant",
    final: false,
    createdAt: "2026-09-23T03:30:00.000Z",
    parts: [
      { text: "Earlier answer.", createdAt: "2026-09-23T03:30:01.000Z" },
      { text: "New answer.", createdAt: "2026-09-23T03:30:05.000Z" },
    ],
  }];

  const split = splitAssistantMessagesAtTurnBoundaries(
    messages,
    ["2026-09-23T03:30:03.000Z"],
  );

  assert.equal(split.length, 2);
  assert.equal(split[0].parts[0].text, "Earlier answer.");
  assert.equal(split[0].final, true);
  assert.equal(split[1].parts[0].text, "New answer.");
  assert.equal(split[1].createdAt, "2026-09-23T03:30:05.000Z");
  assert.equal(split[1].final, false);
});

test("puts an equal-timestamp assistant part after a late user boundary", () => {
  const messages = [{
    role: "assistant",
    final: false,
    createdAt: "2026-09-23T10:11:42.394Z",
    parts: [
      { text: "Earlier answer.", createdAt: "2026-09-23T10:11:42.394Z" },
      { text: "Answering this turn.", createdAt: "2026-09-23T10:11:46.778Z" },
    ],
  }];

  const split = splitAssistantMessagesAtTurnBoundaries(
    messages,
    ["2026-09-23T10:11:46.778Z"],
  );

  assert.equal(split.length, 2);
  assert.equal(split[0].parts[0].text, "Earlier answer.");
  assert.equal(split[1].parts[0].text, "Answering this turn.");
});

test("keeps a hidden sentinel separate from the next assistant answer", () => {
  const messages = [{
    role: "assistant",
    final: false,
    createdAt: "2026-09-23T03:30:00.000Z",
    parts: [
      { text: "...", createdAt: "2026-09-23T03:30:01.000Z" },
      { text: "The new response.", createdAt: "2026-09-23T03:30:05.000Z" },
    ],
  }];

  const split = splitAssistantMessagesAtTurnBoundaries(
    messages,
    ["2026-09-23T03:30:03.000Z"],
  );

  assert.equal(stripSilenceSentinel(split[0].parts[0].text), "");
  assert.equal(stripSilenceSentinel(split[1].parts[0].text), "The new response.");
});
