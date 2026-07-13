// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useApp } from "../context/useApp";
import { useConnectionState } from "../hooks/useConnectionState";
import { CollapsibleCard } from "./CollapsibleCard";
import { StatusSection } from "./status-panel/StatusSection";
import { SessionSection } from "./status-panel/SessionSection";
import { SubagentsPanel } from "./SubagentsPanel";
import { VoiceLevelVisualizer } from "./VoiceLevelVisualizer";
import { WebcamVisionPanel } from "./WebcamVisionPanel";

const VISUALIZER_PROPS = {
  backgroundColor: "#0a0a0a",
  barColor: "#76b900",
  barCount: 20,
  barGap: 4,
  barWidth: 8,
  barMaxHeight: 44,
  barLineCap: "round" as const,
};

export function Sidebar() {
  const { isConnected } = useConnectionState();
  const { currentSessionId, selectedExample } = useApp();
  const canUseWebcam = selectedExample?.capabilities?.includes("webcam") ?? false;

  return (
    <aside className="sidebar-panel d-flex flex-col" style={{ width: "300px" }}>
      <StatusSection />
      <SessionSection />

      {isConnected && (
        <>
          <CollapsibleCard label="BOT AUDIO" className="sidebar-audio-card" storageKey="bot-audio">
            <div className="sidebar-voice-visualizer">
              <VoiceLevelVisualizer participantType="bot" ariaLabel="Bot audio level" {...VISUALIZER_PROPS} />
            </div>
          </CollapsibleCard>

          <CollapsibleCard label="USER AUDIO" className="sidebar-audio-card" storageKey="user-audio">
            <div className="sidebar-voice-visualizer">
              <VoiceLevelVisualizer participantType="local" ariaLabel="User audio level" {...VISUALIZER_PROPS} />
            </div>
          </CollapsibleCard>

          {currentSessionId && <SubagentsPanel key={currentSessionId} />}

          {canUseWebcam && currentSessionId && (
            <CollapsibleCard
              label="WEBCAM VISION"
              className="sidebar-webcam-card"
              storageKey="webcam-vision"
              keepMounted
            >
              <WebcamVisionPanel key={currentSessionId} sessionId={currentSessionId} />
            </CollapsibleCard>
          )}
        </>
      )}
    </aside>
  );
}
