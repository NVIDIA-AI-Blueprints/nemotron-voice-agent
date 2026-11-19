// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useVoiceAssistant } from "@livekit/components-react";
import { useStickToBottom } from "../../hooks/useStickToBottom";
import { buildTimeline, type EventItem, type TranscriptItem } from "../../lib/timeline";
import { useSession } from "../../session/session-context";
import { useTranscript } from "../../session/useTranscript";

function formatTime(at: number) {
  return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function EventMessage({ item }: Readonly<{ item: EventItem }>) {
  return (
    <li className={`transcript-message message-event message-event--${item.entry.tone}`}>
      <span className="message-timestamp">{formatTime(item.at)}</span>
      <div className="message-content">
        <span aria-hidden="true">{item.entry.icon}</span> <span>{item.entry.text}</span>
      </div>
    </li>
  );
}

function TranscriptMessage({ item }: Readonly<{ item: TranscriptItem }>) {
  const user = item.speaker === "user";
  return (
    <li className={`transcript-message ${user ? "message-user" : "message-bot"}${item.streaming ? " message-streaming" : ""}`}>
      <span className="message-timestamp">{formatTime(item.at)}</span>
      <div className="message-content">
        <span className="message-role">{user ? "You" : "Bot"}:</span> <span>{item.text}</span>
      </div>
    </li>
  );
}

/** The live transcripts of both sides and the agent events (tool calls, barge-ins, errors), in time order. */
export function ConversationPanel() {
  const { events } = useSession();
  const { state } = useVoiceAssistant();
  const transcripts = useTranscript(state === "speaking");
  const timeline = useMemo(() => buildTimeline(transcripts, events), [transcripts, events]);
  const bottomAnchorRef = useStickToBottom(timeline);

  return (
    <div className="p-4">
      <ul className="d-flex flex-col gap-2" style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {timeline.map((item) =>
          item.kind === "event" ? <EventMessage key={item.id} item={item} /> : <TranscriptMessage key={item.id} item={item} />,
        )}
      </ul>
      <div ref={bottomAnchorRef} className="conversation-bottom-spacer" aria-hidden="true" />
    </div>
  );
}
