// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { usePipecatClient } from "@pipecat-ai/client-react";
import { useApp } from "../../context/useApp";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";

export function SessionSection() {
  const client = usePipecatClient();
  const { availableTransports, selectedTransport } = useApp();
  const selectedTransportLabel =
    availableTransports.find((transport) => transport.id === selectedTransport)?.label ?? selectedTransport;

  return (
    <PanelSection label="SESSION">
      <StatusRow label="Transport" value={selectedTransportLabel} />
      <StatusRow label="RTVI" value={client?.version ?? "---"} />
    </PanelSection>
  );
}
