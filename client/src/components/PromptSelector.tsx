// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo } from "react";
import { useConnectionState } from "../hooks/useConnectionState";
import { useApp } from "../context/useApp";
import { isSelectablePrompt } from "../utils";
import { PanelSection } from "./PanelSection";

const MULTILINGUAL_PROMPT_KEY = "multilingual_voice_assistant";

function hasMultilingualTerm(value: string): boolean {
  const normalized = value.toLowerCase();
  return normalized.includes("rnnt") || normalized.includes("multilingual");
}

export function PromptSelector() {
  const { isLocked } = useConnectionState();
  const { prompts, promptsLoading, selectedPromptKey, selectPrompt, selectedPrompt, selectedASR, selectedTTS } = useApp();
  const asrDescriptor = [selectedASR?.id, selectedASR?.name, selectedASR?.model].filter(Boolean).join(" ");
  const ttsDescriptor = [selectedTTS?.id, selectedTTS?.name, selectedTTS?.voiceId].filter(Boolean).join(" ");
  const multilingualReady = hasMultilingualTerm(asrDescriptor) && hasMultilingualTerm(ttsDescriptor);
  const selectablePrompts = useMemo(() => prompts.filter(isSelectablePrompt), [prompts]);
  const visiblePrompts = useMemo(
    () =>
      multilingualReady
        ? selectablePrompts
        : selectablePrompts.filter((p) => p.key !== MULTILINGUAL_PROMPT_KEY),
    [multilingualReady, selectablePrompts],
  );

  useEffect(() => {
    if (selectedPromptKey === MULTILINGUAL_PROMPT_KEY && !multilingualReady) {
      const fallback = selectablePrompts.find((p) => p.key !== MULTILINGUAL_PROMPT_KEY);
      if (fallback) selectPrompt(fallback.key);
    }
  }, [multilingualReady, selectablePrompts, selectPrompt, selectedPromptKey]);

  if (promptsLoading) {
    return <PanelSection label="PROMPT" loading loadingText="Loading..." />;
  }

  if (visiblePrompts.length === 0) return null;

  return (
    <PanelSection label="PROMPT">
      <select
        className="select-dark select-full"
        value={selectedPromptKey}
        onChange={(e) => {
          if (e.target.value === MULTILINGUAL_PROMPT_KEY && !multilingualReady) return;
          selectPrompt(e.target.value);
        }}
        title={selectedPrompt?.description || selectedPrompt?.content || ""}
        disabled={isLocked}
      >
        {visiblePrompts.map((p) => (
          <option key={p.key} value={p.key} title={p.description || p.content}>
            {p.key}
          </option>
        ))}
      </select>
    </PanelSection>
  );
}
