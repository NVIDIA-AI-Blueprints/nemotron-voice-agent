// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useEffect, useRef } from "react";

import { useState } from "react";

interface Props {
  dataChannel: RTCDataChannel | null;
}
interface IncomingDataChannelMessage {
  label: string;
  type: string;
  data: string;
}

interface DataChannelMessage {
  actor: string;
  message_id: string;
  text: string;
}

interface AugmentedMessage extends DataChannelMessage {
  timestamp: Date;
}

export function Transcripts(props: Props) {
  const [transcripts, setTranscripts] = useState<AugmentedMessage[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Clear transcripts when a new connection is established
  useEffect(() => {
    if (props.dataChannel) {
      console.log(
        "New DataChannel connection detected, clearing old transcripts",
      );
      setTranscripts([]);
    }
  }, [props.dataChannel]);

  useEffect(() => {
    function onDataChannelMessage(event: MessageEvent) {
      try {
        const envelope = JSON.parse(event.data) as
          | IncomingDataChannelMessage
          | any;
        const payload =
          typeof (envelope as any)?.data === "string"
            ? JSON.parse((envelope as any).data)
            : (envelope as any)?.data;

        // Ignore control/config messages (e.g., { type: "tts_voices" | "tts_voice_current", ... })
        if (payload && typeof payload === "object" && (payload as any).type) {
          return;
        }
        console.log("Received DataChannel message:", payload);

        const message = (payload || {}) as Partial<DataChannelMessage>;
        // Only process transcript-like messages that have required fields
        if (
          typeof message.actor !== "string" ||
          typeof message.message_id !== "string" ||
          typeof message.text !== "string"
        ) {
          return;
        }

        // Create a fully typed, non-optional copy for use in closures
        const validMessage: DataChannelMessage = {
          actor: message.actor as string,
          message_id: message.message_id as string,
          text: message.text as string,
        };

        setTranscripts((prev) => {
          const existingMessage = prev.find(
            (t) =>
              t.actor === validMessage.actor &&
              t.message_id === validMessage.message_id,
          );
          if (existingMessage) {
            existingMessage.text = validMessage.text;
          } else {
            prev.push({ ...validMessage, timestamp: new Date() });
          }
          return [...prev];
        });
      } catch {
        // Ignore non-JSON or unexpected payloads
      }
    }
    props.dataChannel?.addEventListener("message", onDataChannelMessage);
    return () => {
      props.dataChannel?.removeEventListener("message", onDataChannelMessage);
    };
  }, [props.dataChannel]);

  useEffect(() => {
    // Scroll to the bottom of the transcripts every time a new transcript is added
    bottomRef.current?.scrollIntoView();
  }, [transcripts]);

  const filteredTranscripts = transcripts.filter((transcript) => {
    const actor = transcript.actor?.toLowerCase()?.trim();
    const isSystem = actor === "system";
    return !isSystem;
  });

  return filteredTranscripts.map((transcript) => (
    <div
      key={transcript.message_id}
      className={`flex gap-3 mb-4 ${
        transcript.actor === "bot" ? "" : "flex-row-reverse"
      }`}
    >
      <div
        className={`w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-semibold ${
          transcript.actor === "bot" ? "bg-[#FB4E0B]" : "bg-[#0EA5E9]"
        }`}
      >
        {transcript.actor === "bot" ? "A" : "N"}
      </div>

      <div className="flex-1 max-w-[80%]">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-gray-700">
            {transcript.actor === "bot" ? "Aria" : "Nathan Reeves"}
          </span>
          <span className="text-xs text-gray-400">
            {transcript.timestamp.toTimeString().slice(0, 8)}
          </span>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
          {transcript.text}
        </div>
      </div>
    </div>
  ));
}
