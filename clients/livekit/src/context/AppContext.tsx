// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState, type ReactNode } from "react";
import { useClientConfig, useServices, useTools, type ServiceSettingValues, type Slot, type Tool } from "../api";
import { customFieldSpec } from "../lib/services";
import type { SessionChoices } from "../lib/session";
import { AppContext } from "./app-context";
import { usePrompts, type PromptState } from "./usePrompts";
import { useServiceSlot, type SlotState } from "./useServiceSlot";

export interface AppState extends PromptState {
  services: Record<Slot, SlotState>;
  servicesLoading: boolean;
  serviceSettings: Record<string, ServiceSettingValues>;
  setServiceSetting: (settingsKey: string, name: string, value: unknown) => void;
  selectService: (slot: Slot, id: string) => void;

  selectedVoiceId: string;
  setSelectedVoiceId: (id: string) => void;

  tools: Tool[];
  toolsLoading: boolean;

  /** Whether the client config, the service catalog, and the prompts arrived, which a session needs. */
  ready: boolean;
  /** Fix the choices while a session uses them. */
  setLocked: (locked: boolean) => void;
  /** Everything that the next session is started with. */
  choices: SessionChoices;
}

export function AppProvider({ children }: Readonly<{ children: ReactNode }>) {
  const { data: clientConfig } = useClientConfig();
  const { data: catalog, isLoading: servicesLoading } = useServices();
  const { data: tools = [], isLoading: toolsLoading } = useTools();
  const [locked, setLocked] = useState(false);
  const prompts = usePrompts(locked);

  const [serviceSettings, setSettings] = useState<Record<string, ServiceSettingValues>>({});
  const setServiceSetting = useCallback((settingsKey: string, name: string, value: unknown) => {
    if (locked) return;
    setSettings((previous) => {
      const next = { ...previous[settingsKey] };
      if (value === undefined) delete next[name];
      else next[name] = value;
      return { ...previous, [settingsKey]: next };
    });
  }, [locked]);

  const asr = useServiceSlot("asr", catalog?.asr, serviceSettings, customFieldSpec(clientConfig, "asr"), locked);
  const llm = useServiceSlot("llm", catalog?.llm, serviceSettings, customFieldSpec(clientConfig, "llm"), locked);
  const tts = useServiceSlot("tts", catalog?.tts, serviceSettings, customFieldSpec(clientConfig, "tts"), locked);
  const services = useMemo(() => ({ asr, llm, tts }), [asr, llm, tts]);

  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const selectService = useCallback((slot: Slot, id: string) => {
    if (locked) return;
    services[slot].select(id);
    if (slot === "tts") setSelectedVoiceId("");
  }, [locked, services]);

  const ready = clientConfig !== undefined && catalog !== undefined && prompts.promptsLoaded;

  const choices = useMemo<SessionChoices>(() => ({
    prompt: prompts.selectedPrompt,
    defaultScenario: prompts.defaultScenario,
    voice: selectedVoiceId,
    services: Object.fromEntries(
      Object.entries(services).flatMap(([slot, state]) => (state.selection ? [[slot, state.selection]] : [])),
    ),
  }), [prompts.selectedPrompt, prompts.defaultScenario, selectedVoiceId, services]);

  const value = useMemo<AppState>(() => ({
    ...prompts,
    services, servicesLoading, serviceSettings, setServiceSetting, selectService,
    selectedVoiceId, setSelectedVoiceId,
    tools, toolsLoading,
    ready, setLocked,
    choices,
  }), [prompts, services, servicesLoading, serviceSettings, setServiceSetting, selectService,
       selectedVoiceId, tools, toolsLoading, ready, choices]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
