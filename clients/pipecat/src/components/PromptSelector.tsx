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
  const {
    prompts,
    promptsLoading,
    selectedPromptKey,
    selectPrompt,
    selectedPrompt,
    selectedASR,
    selectedTTS,
    selectedExample,
    speakerDiarizationEnabled,
  } = useApp();
  const asrDescriptor = [selectedASR?.id, selectedASR?.name, selectedASR?.model].filter(Boolean).join(" ");
  const ttsDescriptor = [selectedTTS?.id, selectedTTS?.name, selectedTTS?.voiceId].filter(Boolean).join(" ");
  const multilingualReady = hasMultilingualTerm(asrDescriptor) && hasMultilingualTerm(ttsDescriptor);
  const diarizationReady = speakerDiarizationEnabled && selectedASR?.speakerDiarizationSupported === true;
  const selectablePrompts = useMemo(() => prompts.filter(isSelectablePrompt), [prompts]);
  const visiblePrompts = useMemo(
    () => selectablePrompts.filter((prompt) => (
      (multilingualReady || prompt.key !== MULTILINGUAL_PROMPT_KEY)
      && (diarizationReady || prompt.multiSpeakerSupport !== true)
    )),
    [diarizationReady, multilingualReady, selectablePrompts],
  );

  useEffect(() => {
    const selectedIsHidden = !visiblePrompts.some((prompt) => prompt.key === selectedPromptKey);
    if (selectedPromptKey && selectedIsHidden) {
      const registryDefault = selectedExample?.defaults?.prompt?.[0];
      const registryDefaultKey = registryDefault && "key" in registryDefault ? registryDefault.key : "";
      const fallback = visiblePrompts.find((prompt) => prompt.key === registryDefaultKey)
        ?? visiblePrompts.find((prompt) => prompt.default)
        ?? visiblePrompts[0];
      if (fallback) selectPrompt(fallback.key);
    }
  }, [selectPrompt, selectedExample, selectedPromptKey, visiblePrompts]);

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
