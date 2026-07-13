// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { usePipecatClientTransportState } from "@pipecat-ai/client-react";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";
import { InlineSpinner } from "./Spinner";

export function StatusSection() {
  const transportState = usePipecatClientTransportState();

  const isLoading =
    transportState === "authenticating" ||
    transportState === "authenticated" ||
    transportState === "connecting" ||
    transportState === "disconnecting";

  const getClientStatus = () => {
    switch (transportState) {
      case "authenticating": return "STARTING";
      case "authenticated": return "STARTING";
      case "ready": return "READY";
      case "connected": return "CONNECTED";
      case "connecting": return "CONNECTING";
      case "disconnecting": return "DISCONNECTING";
      case "error": return "ERROR";
      default: return "INITIALIZED";
    }
  };

  const getAgentStatus = () => {
    switch (transportState) {
      case "ready":
      case "connected": return "READY";
      case "authenticating":
      case "authenticated": return "STARTING";
      case "connecting": return "CONNECTING";
      case "disconnecting": return "DISCONNECTING";
      default: return "---";
    }
  };

  return (
    <PanelSection label="STATUS">
      <StatusRow label="Client" value={getClientStatus()}>
        {isLoading && <InlineSpinner />}
      </StatusRow>
      <StatusRow label="Agent" value={getAgentStatus()}>
        {isLoading && <InlineSpinner />}
      </StatusRow>
    </PanelSection>
  );
}
