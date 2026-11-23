// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { liveEventsStore } from "../../liveEvents/store";
import { EventLogPanel } from "./EventLogPanel";

/**
 * The EVENT LOG tab's panel and the feed that fills it from the call's `live-event` server messages, in one piece that the
 * center panel loads only for an example that lists `event_log`. The feed renders nothing; the store lives outside React,
 * so it outlives the client that the app replaces after a hang up.
 */
export default function EventLogScreen({ shown }: Readonly<{ shown: boolean }>) {
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((message: unknown) => liveEventsStore.dispatch({ type: "message", message }), []),
  );
  useRTVIClientEvent(RTVIEvent.Connected, useCallback(() => liveEventsStore.dispatch({ type: "connected" }), []));
  useRTVIClientEvent(RTVIEvent.Disconnected, useCallback(() => liveEventsStore.dispatch({ type: "disconnected" }), []));
  return (
    <div
      id="center-panel-events"
      role="tabpanel"
      aria-labelledby="center-tab-events"
      className={`flex-1 min-h-0 overflow-hidden ${shown ? "" : "hidden"}`.trim()}
    >
      <EventLogPanel active={shown} />
    </div>
  );
}
