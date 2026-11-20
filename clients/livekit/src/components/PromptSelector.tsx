// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useApp } from "../context/useApp";
import { useSession } from "../session/session-context";
import { PanelSection } from "./PanelSection";

export function PromptSelector() {
  const { isLocked } = useSession();
  const { prompts, promptsLoading, selectedPromptKey, selectPrompt, selectedPrompt } = useApp();

  if (promptsLoading) {
    return <PanelSection label="PROMPT" loading />;
  }

  if (prompts.length === 0) return null;

  return (
    <PanelSection label="PROMPT">
      <select
        className="select-dark select-full"
        aria-label="Prompt"
        value={selectedPromptKey}
        onChange={(e) => selectPrompt(e.target.value)}
        title={selectedPrompt?.description || selectedPrompt?.content || ""}
        disabled={isLocked}
      >
        {prompts.map((p) => (
          <option key={p.key} value={p.key} title={p.description || p.content}>
            {p.label}
          </option>
        ))}
      </select>
      {selectedPrompt?.description && <p className="settings-hint">{selectedPrompt.description}</p>}
    </PanelSection>
  );
}
