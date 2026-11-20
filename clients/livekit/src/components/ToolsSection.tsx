// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useApp } from "../context/useApp";
import { PanelSection } from "./PanelSection";

export function ToolsSection() {
  const { tools, toolsLoading, selectedPrompt } = useApp();

  const enabled = useMemo(() => new Set(selectedPrompt?.tools ?? []), [selectedPrompt]);

  if (toolsLoading) {
    return <PanelSection label="TOOLS" loading />;
  }
  if (tools.length === 0) return null;
  if (enabled.size === 0) return null;

  return (
    <PanelSection label="TOOLS">
      <ul className="tools-list">
        {tools.filter((t) => enabled.has(t.name)).map((t) => (
          <li key={t.name} className="tools-row is-active" title={t.description}>
            <span className="tools-indicator" aria-hidden>●</span>
            <span className="tools-name">{t.name}</span>
          </li>
        ))}
      </ul>
    </PanelSection>
  );
}
