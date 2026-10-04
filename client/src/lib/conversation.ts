// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const NO_SPACE_BEFORE = /^[,.;:!?…)\]}]/;
const NO_SPACE_AFTER = /[([{]$/;
const SENTINEL_PREFIX = /^\s*(?:\.+(?:\s+|$))+/;
const SENTINEL_SUFFIX = /\s[.\s]*\.\s*$/;

export type AnchoredTranscript = {
  text: string;
  anchorCreatedAt: string;
};

type TimestampedPart = {
  createdAt?: string;
};

type SplittableMessage<TPart extends TimestampedPart> = {
  role: string;
  final?: boolean;
  parts: TPart[];
  createdAt: string;
  updatedAt?: string;
};

export const normalizeTranscript = (text?: string | null) =>
  (text ?? "").trim().replace(/\s+/g, " ");

/** Silent bot turns arrive as stray period runs; drop whole runs of periods, never a period that starts a word. */
export function stripSilenceSentinel(text: string): string {
  return text.replace(SENTINEL_PREFIX, "").replace(SENTINEL_SUFFIX, "").trim();
}

/** Restore assistant boundaries hidden by Pipecat's same-role message merging. */
export function splitAssistantMessagesAtTurnBoundaries<
  TPart extends TimestampedPart,
  TMessage extends SplittableMessage<TPart>,
>(messages: TMessage[], boundaries: string[]): TMessage[] {
  if (!boundaries.length) return messages;
  const sortedBoundaries = [...boundaries].filter(Boolean).sort();

  return messages.flatMap((message) => {
    if (message.role !== "assistant" || message.parts.length < 2) return [message];

    const groups: TPart[][] = [];
    let previousBucket = -1;
    for (const part of message.parts) {
      const partCreatedAt = part.createdAt || message.createdAt;
      const bucket = sortedBoundaries.filter((boundary) => boundary <= partCreatedAt).length;
      if (bucket !== previousBucket) {
        groups.push([]);
        previousBucket = bucket;
      }
      groups.at(-1)?.push(part);
    }
    if (groups.length < 2) return [message];

    return groups.map((parts, index) => ({
      ...message,
      parts,
      createdAt: index === 0 ? message.createdAt : (parts[0]?.createdAt || message.createdAt),
      updatedAt: parts.at(-1)?.createdAt || message.updatedAt,
      final: index < groups.length - 1 ? true : message.final,
    }));
  });
}

/** Join independently emitted transcript fragments without gluing words together. */
export function joinTranscriptParts(parts: string[]): string {
  return parts.reduce((joined, part) => {
    if (!part) return joined;
    if (
      !joined
      || joined.at(-1)?.match(/\s/)
      || part[0]?.match(/\s/)
      || NO_SPACE_BEFORE.test(part)
      || NO_SPACE_AFTER.test(joined)
    ) {
      return `${joined}${part}`;
    }
    return `${joined} ${part}`;
  }, "");
}

/** Match only containing turns that are not earlier than the raw message. */
export function hasContainingFinalizedTurn(
  messageText: string,
  messageCreatedAt: string,
  turns: AnchoredTranscript[],
): boolean {
  const normalizedMessage = normalizeTranscript(messageText);
  return Boolean(
    normalizedMessage
    && turns.some(
      (turn) =>
        turn.anchorCreatedAt >= messageCreatedAt
        && normalizeTranscript(turn.text).includes(normalizedMessage),
    )
  );
}

/**
 * Advance a labeled turn's suppression anchor (forward only) so it covers the raw bubble.
 */
export function resolveSuppressionAnchor(
  existingAnchor: string | undefined,
  latestVisibleCreatedAt: string | undefined,
  fallback: string,
): string {
  const candidates = [existingAnchor, latestVisibleCreatedAt].filter(
    (value): value is string => Boolean(value),
  );
  if (!candidates.length) return fallback;
  return candidates.reduce((latest, value) => (value > latest ? value : latest));
}

/** Only a diarized session labels its turns and stops publishing raw user transcripts. */
export function isSpeakerLabeledTurn(message: Record<string, unknown>): boolean {
  return message.speaker_labeled === true;
}

/** Give progress and final events for the same speaker run one stable client key. */
export function speakerTurnMessageId(
  message: Record<string, unknown>,
  fallback: string,
): string {
  const turnId = message.turn_id;
  const runIndex = message.run_index;
  if (
    (typeof turnId === "string" || typeof turnId === "number")
    && typeof runIndex === "number"
    && Number.isInteger(runIndex)
  ) {
    return `speaker-turn-${turnId}-${runIndex}`;
  }
  return fallback;
}

/** Keep the default prompt labeled "You"; expose diarized identities only in multi-speaker mode. */
export function speakerDisplayLabel(
  multiSpeakerSupport: boolean,
  speakerId: number | null,
  displayName: string,
): string | undefined {
  if (!multiSpeakerSupport) return undefined;
  if (displayName) return displayName;
  return speakerId === null ? "Unknown" : `Speaker ${speakerId + 1}`;
}

/**
 * Find the latest conversation message strictly older than a user turn.
 *
 * Strict comparison keeps a late user turn ahead of a same-timestamp assistant answer.
 */
export function resolveChronologicalAnchor(
  eventCreatedAt: string,
  messageCreatedAts: string[],
): string | undefined {
  if (!eventCreatedAt) return undefined;

  let preceding: string | undefined;
  for (const createdAt of messageCreatedAts) {
    if (createdAt < eventCreatedAt && (!preceding || createdAt > preceding)) {
      preceding = createdAt;
    }
  }
  return preceding;
}
