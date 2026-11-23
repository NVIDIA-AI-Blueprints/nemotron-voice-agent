// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from "react";
import { usePromptCatalog, type Prompt } from "../api";
import {
  customPrompt,
  customPromptError,
  effectivePromptKey,
  fromCatalog,
  parseStoredPrompts,
  promptContentError,
  promptKey,
} from "../lib/prompts";
import { readLSJson, writeLSJson } from "../utils";

const STORAGE_KEY = "nvidia-voice-agent-livekit-prompts-custom";
const LOCKED = "Prompts are locked during a session";

export interface PromptState {
  prompts: Prompt[];
  promptsLoading: boolean;
  /** Whether the server's prompts arrived, which a session needs for its use case. */
  promptsLoaded: boolean;
  selectedPromptKey: string;
  selectedPrompt: Prompt | undefined;
  /** The use case that a custom prompt runs on: the server default. */
  defaultScenario: string;
  selectPrompt: (key: string) => void;
  /** Add a custom prompt. Returns why it was refused, or "" when it was added. */
  addPrompt: (name: string, description: string, content: string) => string;
  /** Update a custom prompt. Returns why it was refused, or "" when it was saved. */
  updatePrompt: (key: string, description: string, content: string) => string;
  removePrompt: (key: string) => void;
}

/** The use-case prompts from the server, the custom ones stored in this browser, and the choice. Fixed while `locked`. */
export function usePrompts(locked: boolean): PromptState {
  const { data: entries, isLoading: promptsLoading } = usePromptCatalog();
  const [custom, setCustom] = useState<Prompt[]>(() => parseStoredPrompts(readLSJson(STORAGE_KEY)));
  const [chosenKey, setChosenKey] = useState("");

  const builtIn = useMemo(() => (entries ?? []).map(fromCatalog), [entries]);
  const prompts = useMemo(() => [...builtIn, ...custom], [builtIn, custom]);
  const selectedPromptKey = effectivePromptKey(chosenKey, prompts);
  const selectedPrompt = prompts.find((prompt) => prompt.key === selectedPromptKey);
  const defaultScenario = builtIn.find((prompt) => prompt.default)?.key ?? builtIn[0]?.key ?? "";

  const persist = useCallback((next: Prompt[]) => {
    setCustom(next);
    writeLSJson(STORAGE_KEY, next);
  }, []);

  const selectPrompt = useCallback((key: string) => {
    if (!locked) setChosenKey(key);
  }, [locked]);

  const addPrompt = useCallback((name: string, description: string, content: string) => {
    if (locked) return LOCKED;
    const key = promptKey(name);
    const error = customPromptError(key, content, prompts);
    if (!error) persist([...custom, customPrompt(key, description.trim(), content.trim())]);
    return error;
  }, [custom, locked, persist, prompts]);

  const updatePrompt = useCallback((key: string, description: string, content: string) => {
    if (locked) return LOCKED;
    const error = promptContentError(content);
    if (!error) persist(custom.map((prompt) => (prompt.key === key ? customPrompt(key, description, content) : prompt)));
    return error;
  }, [custom, locked, persist]);

  const removePrompt = useCallback((key: string) => {
    if (!locked) persist(custom.filter((prompt) => prompt.key !== key));
  }, [custom, locked, persist]);

  const promptsLoaded = entries !== undefined;
  return useMemo(
    () => ({
      prompts, promptsLoading, promptsLoaded, selectedPromptKey, selectedPrompt, defaultScenario,
      selectPrompt, addPrompt, updatePrompt, removePrompt,
    }),
    [prompts, promptsLoading, promptsLoaded, selectedPromptKey, selectedPrompt, defaultScenario, selectPrompt, addPrompt,
      updatePrompt, removePrompt],
  );
}
