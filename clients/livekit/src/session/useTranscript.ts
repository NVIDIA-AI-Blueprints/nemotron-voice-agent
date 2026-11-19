// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { useLocalParticipant } from "@livekit/components-react";
import { DataStreamError, DataStreamErrorReason, type Room, type TextStreamInfo } from "livekit-client";
import { isStreaming, type TranscriptItem } from "../lib/timeline";
import { useSession } from "./session-context";

const TRANSCRIPTION_TOPIC = "lk.transcription";
const SEGMENT_ID = "lk.segment_id";
const TRANSCRIPTION_FINAL = "lk.transcription_final";

export interface TranscriptSegment {
  identity: string;
  info: TextStreamInfo;
  text: string;
}

const NO_SEGMENTS: TranscriptSegment[] = [];

const segmentId = (info: TextStreamInfo) => info.attributes?.[SEGMENT_ID] ?? info.id;

function isCutShort(error: unknown): boolean {
  return error instanceof DataStreamError && error.reason === DataStreamErrorReason.AbnormalEnd;
}

/**
 * Return the transcription segments of a room, each with the text received so far.
 *
 * The handler is registered as soon as the room exists, before it connects, so the first greeting is kept. Every
 * update of a segment arrives as a new text stream with the same segment id. A stream that is cut short, because the
 * room or the agent left mid-sentence, keeps the text received so far.
 */
export function useTranscriptionSegments(room: Room): TranscriptSegment[] {
  const [state, setState] = useState<{ room: Room; segments: TranscriptSegment[] }>({ room, segments: NO_SEGMENTS });

  useEffect(() => {
    let active = true;
    let current: TranscriptSegment[] = [];
    const update = (segment: TranscriptSegment) => {
      if (!active) return;
      const index = current.findIndex((known) => segmentId(known.info) === segmentId(segment.info));
      current = index === -1 ? [...current, segment] : current.with(index, segment);
      setState({ room, segments: current });
    };

    room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, async (reader, { identity }) => {
      let text = "";
      try {
        for await (const chunk of reader) {
          text += chunk;
          update({ identity, info: reader.info, text });
        }
      } catch (error) {
        if (!isCutShort(error)) throw error;
      }
    });
    return () => {
      active = false;
      room.unregisterTextStreamHandler(TRANSCRIPTION_TOPIC);
    };
  }, [room]);

  return state.room === room ? state.segments : NO_SEGMENTS;
}

/**
 * Return the live user and agent transcripts of the session, one item per segment.
 *
 * The agent publishes both sides, so their timestamps share the agent's clock with the agent events.
 */
export function useTranscript(agentSpeaking: boolean): TranscriptItem[] {
  const { segments } = useSession();
  const { localParticipant } = useLocalParticipant();
  const lastAgentIndex = segments.findLastIndex((segment) => segment.identity !== localParticipant.identity);

  return segments.map((segment, index) => {
    const attributes = segment.info.attributes ?? {};
    const speaker = segment.identity === localParticipant.identity ? "user" : "agent";
    return {
      kind: "transcript",
      id: segmentId(segment.info),
      speaker,
      text: segment.text,
      streaming: isStreaming(speaker, attributes[TRANSCRIPTION_FINAL], agentSpeaking && index === lastAgentIndex),
      at: segment.info.timestamp,
    };
  });
}
