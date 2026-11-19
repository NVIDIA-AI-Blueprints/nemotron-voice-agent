// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLocalParticipant, useVoiceAssistant } from "@livekit/components-react";
import { Track } from "livekit-client";
import { useSession } from "../session/session-context";
import { CollapsibleCard } from "./CollapsibleCard";
import { PipelineSection } from "./status-panel/PipelineSection";
import { SessionSection } from "./status-panel/SessionSection";
import { StatusSection } from "./status-panel/StatusSection";
import { VoiceLevelVisualizer } from "./VoiceLevelVisualizer";

function AudioCards() {
  const { audioTrack: botTrack } = useVoiceAssistant();
  const { localParticipant, microphoneTrack } = useLocalParticipant();
  const userTrack = { participant: localParticipant, source: Track.Source.Microphone, publication: microphoneTrack };

  return (
    <>
      <CollapsibleCard label="BOT AUDIO" className="sidebar-audio-card" storageKey="bot-audio">
        <div className="sidebar-voice-visualizer">
          <VoiceLevelVisualizer track={botTrack} ariaLabel="Bot audio level" />
        </div>
      </CollapsibleCard>

      <CollapsibleCard label="USER AUDIO" className="sidebar-audio-card" storageKey="user-audio">
        <div className="sidebar-voice-visualizer">
          <VoiceLevelVisualizer track={userTrack} ariaLabel="User audio level" />
        </div>
      </CollapsibleCard>
    </>
  );
}

export function Sidebar() {
  const { isConnected } = useSession();

  return (
    <aside className="sidebar-panel d-flex flex-col">
      <StatusSection />
      <SessionSection />
      <PipelineSection />
      {isConnected && <AudioCards />}
    </aside>
  );
}
