// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customPayload } from "./services.ts";
import type {
  CustomFieldSpec,
  CustomService,
  Prompt,
  ServiceEntry,
  ServiceSettingValues,
  Slot,
  SlotChoice,
  TokenRequest,
  VoiceList,
} from "./types.ts";

/**
 * The service that the UI uses for a slot: an offered entry with the values the user set, or a custom endpoint with
 * the fields that the server accepts for it.
 */
export type SlotSelection =
  | { kind: "builtin"; entry: ServiceEntry; settings: ServiceSettingValues }
  | { kind: "custom"; service: CustomService; spec: CustomFieldSpec };

export interface SessionChoices {
  prompt: Prompt | undefined;
  defaultScenario: string;
  voice: string;
  services: Partial<Record<Slot, SlotSelection>>;
}

function definedValues(values: ServiceSettingValues): ServiceSettingValues {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
}

function slotChoice(selection: SlotSelection): SlotChoice {
  if (selection.kind === "custom") return { custom: customPayload(selection.service, selection.spec) };
  const settings = definedValues(selection.settings);
  return Object.keys(settings).length > 0 ? { key: selection.entry.id, settings } : { key: selection.entry.id };
}

/** Build the body of `POST /token` from the choices in the UI. A custom prompt runs on the default use case. */
export function buildTokenRequest({ prompt, defaultScenario, voice, services }: SessionChoices): TokenRequest {
  const request: TokenRequest = {
    scenario: prompt?.builtIn ? prompt.key : defaultScenario,
    services: Object.fromEntries(
      Object.entries(services).map(([slot, selection]) => [slot, slotChoice(selection)]),
    ) as TokenRequest["services"],
  };
  if (prompt && !prompt.builtIn) request.prompt = prompt.content;
  if (voice && services.tts?.kind !== "custom") request.voice = voice;
  return request;
}

/** Return the voice to use: the chosen one while it is listed, else the list's default, else the first one. */
export function effectiveVoiceId(chosen: string, list: VoiceList | undefined): string {
  const voices = list?.voices ?? [];
  if (chosen && voices.some((voice) => voice.id === chosen)) return chosen;
  if (list?.default && voices.some((voice) => voice.id === list.default)) return list.default;
  return voices[0]?.id ?? "";
}
