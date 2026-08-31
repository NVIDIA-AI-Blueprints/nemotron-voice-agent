// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useConnectionState } from "../hooks/useConnectionState";
import { useApp } from "../context/useApp";
import { PanelSection } from "./PanelSection";

export function PipelineExampleSelector() {
  const { isLocked } = useConnectionState();
  const { selectedExample, selectExample, deploymentOptions, deploymentSelectable } = useApp();
  const canSwitch = deploymentSelectable && deploymentOptions.length > 1;

  if (!selectedExample || !canSwitch) return null;

  return (
    <PanelSection label="EXAMPLE">
      <select
        className="select-dark select-full"
        value={selectedExample.key}
        onChange={(e) => selectExample(e.target.value)}
        disabled={isLocked}
        aria-label="Example"
      >
        {deploymentOptions.map((option) => (
          <option key={option.key} value={option.key}>
            {option.label}
          </option>
        ))}
      </select>
    </PanelSection>
  );
}
