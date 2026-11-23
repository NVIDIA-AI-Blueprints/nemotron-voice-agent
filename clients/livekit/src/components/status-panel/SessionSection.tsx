// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDeployment } from "../../api";
import { useSession } from "../../session/session-context";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";

export function SessionSection() {
  const { roomName, serverUrl } = useSession();
  const { data: deployment } = useDeployment();

  return (
    <PanelSection label="SESSION">
      <StatusRow label="Framework" value="LiveKit" />
      <StatusRow label="Deployment" value={deployment?.label || "---"} title={deployment?.recipe} />
      <StatusRow label="Server" value={serverUrl} valueClassName="status-value--wrap" />
      <StatusRow label="Room" value={roomName || "---"} />
    </PanelSection>
  );
}
