// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useVoiceAssistant } from "@livekit/components-react";
import { useSession, type SessionPhase } from "../../session/session-context";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";
import { InlineSpinner } from "./Spinner";

const CLIENT_STATUS: Record<SessionPhase, string> = {
  idle: "READY",
  requesting: "STARTING",
  connecting: "CONNECTING",
  connected: "CONNECTED",
};

export function StatusSection() {
  const { phase, isConnecting, isConnected } = useSession();
  const { state: agentState } = useVoiceAssistant();
  const agentStatus = isConnected ? agentState.toUpperCase() : "---";

  return (
    <PanelSection label="STATUS">
      <div role="status">
        <StatusRow label="Client" value={CLIENT_STATUS[phase]}>
          {isConnecting && <InlineSpinner />}
        </StatusRow>
        <StatusRow label="Agent" value={agentStatus}>
          {(isConnecting || agentState === "connecting" || agentState === "initializing") && <InlineSpinner />}
        </StatusRow>
      </div>
    </PanelSection>
  );
}
