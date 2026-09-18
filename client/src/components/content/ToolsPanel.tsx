// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useApp } from "../../context/useApp";
import type { Tool } from "../../api";

function ToolCard({ tool, isActive }: Readonly<{ tool: Tool; isActive: boolean }>) {
  const params = JSON.stringify(tool.parameters ?? {}, null, 2);
  return (
    <div className={`prompt-card${isActive ? " prompt-card--selected" : ""}`}>
      <div className="prompt-card__header">
        <span className="prompt-card__name">{tool.name}</span>
        <div className="prompt-card__actions">
          {isActive && <span className="prompt-card__badge">Enabled</span>}
        </div>
      </div>
      {tool.description && <p className="prompt-card__desc">{tool.description}</p>}
      <p className="prompts-section-label" style={{ marginTop: "var(--space-2)" }}>Parameters (JSON Schema)</p>
      <pre className="prompt-card__content">{params}</pre>
    </div>
  );
}

export function ToolsPanel() {
  const { tools, toolsLoading, selectedPrompt } = useApp();
  const enabled = useMemo(() => new Set(selectedPrompt?.tools ?? []), [selectedPrompt]);

  if (toolsLoading) {
    return (
      <div className="prompts-panel p-4">
        <div className="services-header"><h3 className="metrics-title">Tools</h3></div>
        <p className="prompts-status">Loading tools...</p>
      </div>
    );
  }

  if (tools.length === 0) {
    return (
      <div className="prompts-panel p-4">
        <div className="services-header"><h3 className="metrics-title">Tools</h3></div>
        <p className="prompts-status">This example does not register any tools.</p>
      </div>
    );
  }

  return (
    <div className="prompts-panel p-4">
      <div className="services-header">
        <h3 className="metrics-title">Tools</h3>
      </div>
      <p className="prompts-section-label">
        {selectedPrompt
          ? `Enabled tools highlighted for prompt: ${selectedPrompt.key}`
          : "All tools available in this example"}
      </p>
      <div className="prompts-list">
        {tools.map((t) => (
          <ToolCard key={t.name} tool={t} isActive={enabled.has(t.name)} />
        ))}
      </div>
    </div>
  );
}
