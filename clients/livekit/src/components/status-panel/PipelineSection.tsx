// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDeployment } from "../../api";
import { useSession } from "../../session/session-context";
import { PanelSection } from "../PanelSection";
import { StatusRow } from "./StatusRow";

/** The models of the cascaded pipeline. A session shows what the agent reports, which includes its own choices. */
export function PipelineSection() {
  const { models: sessionModels } = useSession();
  const { data: deployment } = useDeployment();
  const models = sessionModels ?? deployment?.models ?? [];
  if (models.length === 0) return null;

  return (
    <PanelSection label={sessionModels ? "PIPELINE IN USE" : "DEFAULT PIPELINE"} storageKey="PIPELINE">
      {models.map((model) => (
        <StatusRow
          key={model.kind}
          label={model.kind}
          value={model.name}
          title={[model.id, model.runs].filter(Boolean).join(" · ")}
          valueClassName="status-value--wrap"
        />
      ))}
    </PanelSection>
  );
}
