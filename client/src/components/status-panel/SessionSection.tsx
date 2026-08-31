// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useState } from "react";
import { RTVIEvent, type BotReadyData } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useApp } from "../../context/useApp";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";

export function SessionSection() {
  const { availableTransports, selectedTransport } = useApp();
  const selectedTransportLabel =
    availableTransports.find((transport) => transport.id === selectedTransport)?.label ?? selectedTransport;

  const [protocolVersion, setProtocolVersion] = useState<string>("");

  useRTVIClientEvent(
    RTVIEvent.BotReady,
    useCallback((data: BotReadyData) => {
      setProtocolVersion(data.version ?? "");
    }, [])
  );

  useRTVIClientEvent(
    RTVIEvent.Disconnected,
    useCallback(() => {
      setProtocolVersion("");
    }, [])
  );

  return (
    <PanelSection label="SESSION">
      <StatusRow label="Transport" value={selectedTransportLabel} />
      <StatusRow label="RTVI protocol" value={protocolVersion || "---"} />
    </PanelSection>
  );
}
