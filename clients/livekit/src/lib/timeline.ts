// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EventEntry } from "./agentEvents.ts";

export type Speaker = "user" | "agent";

export interface TranscriptItem {
  kind: "transcript";
  id: string;
  speaker: Speaker;
  text: string;
  streaming: boolean;
  at: number;
}

export interface EventItem {
  kind: "event";
  id: string;
  entry: EventEntry;
  at: number;
}

export type TimelineItem = TranscriptItem | EventItem;

/** Merge the non-empty transcripts with the agent events, in time order. */
export function buildTimeline(transcripts: TranscriptItem[], events: EventItem[]): TimelineItem[] {
  return [...transcripts.filter((item) => item.text.trim()), ...events].sort((a, b) => a.at - b.at);
}

/**
 * Return whether a transcript is still being written.
 *
 * A user segment is final once the speech recognizer marks it so. An agent segment carries the final mark only in its
 * stream trailer, so it also counts as done once the agent stops speaking.
 */
export function isStreaming(speaker: Speaker, finalAttribute: string | undefined, agentSpeaking: boolean): boolean {
  if (finalAttribute === "true") return false;
  return speaker === "user" || agentSpeaking;
}
