// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";
import { useDataChannel } from "@livekit/components-react";
import type { Model, VoiceList } from "../api";
import { EVENTS_TOPIC, decodeAgentEvent, describeAgentEvent } from "../lib/agentEvents";
import type { EventItem } from "../lib/timeline";

export interface AgentEventState {
  events: EventItem[];
  models: Model[] | null;
  voices: VoiceList | null;
}

const EMPTY: AgentEventState = { events: [], models: null, voices: null };

/** Collect the events that the agent publishes on its data-channel topic. `clear` starts over for a new session. */
export function useAgentEvents(): AgentEventState & { clear: () => void } {
  const [state, setState] = useState<AgentEventState>(EMPTY);

  const onMessage = useCallback((message: { payload: Uint8Array }) => {
    const event = decodeAgentEvent(message.payload);
    if (!event) return;
    const at = typeof event.ts === "number" ? event.ts * 1000 : Date.now();
    setState((previous) => {
      const entry = describeAgentEvent(event);
      const next = { ...previous };
      if (entry) {
        next.events = [...previous.events, { kind: "event", id: `event-${previous.events.length}`, entry, at }];
      }
      if (event.type === "session") next.models = event.models;
      if (event.type === "voices") next.voices = { voices: event.voices, default: event.current };
      return next;
    });
  }, []);

  useDataChannel(EVENTS_TOPIC, onMessage);
  const clear = useCallback(() => setState(EMPTY), []);
  return { ...state, clear };
}
