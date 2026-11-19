// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The pipeline slots that a session can choose a service for, in the registry order that the Services tab uses. */
export const SLOTS = ["llm", "asr", "tts"] as const;
export type Slot = (typeof SLOTS)[number];

export type ServiceSource = "cloud-nim" | "self-hosted";
export type ServiceStatus = "ready" | "starting" | "error" | "offline";

export interface ServiceSettingSpec {
  label?: string;
  type: "bool" | "int" | "float" | "enum";
  default?: unknown;
  min?: number;
  max?: number;
  options?: string[];
  requires?: string;
  below?: string;
  reserve?: number;
  path?: string;
}

export type ServiceSettingsSchema = Record<string, ServiceSettingSpec>;
export type ServiceSettingValues = Record<string, unknown>;

/** A service that the agent worker offers for a slot, as `/services` lists it. The first default one is used. */
export interface ServiceEntry {
  id: string;
  name: string;
  source: ServiceSource;
  default: boolean;
  status?: ServiceStatus;
  base_url?: string;
  model_id?: string;
  server?: string;
  model?: string;
  voice_id?: string;
  function_id?: string;
  settings: ServiceSettingsSchema;
}

export type ServiceCatalog = Partial<Record<Slot, ServiceEntry[]>>;

/** A custom endpoint that the user added in the Services tab. It is stored in this browser only. */
export interface CustomService {
  id: string;
  slot: Slot;
  name: string;
  fields: Record<string, string>;
}

export interface Voice {
  id: string;
  name: string;
}

export interface VoiceList {
  voices: Voice[];
  default: string;
}

/** A use-case prompt from `/prompts`. */
export interface PromptEntry {
  id: string;
  name: string;
  description: string;
  examples: string[];
  content: string;
  tools: string[];
  default: boolean;
}

/** A prompt that the UI offers: a use case from the server, or a custom one stored in this browser. */
export interface Prompt {
  key: string;
  label: string;
  description: string;
  content: string;
  examples: string[];
  tools: string[];
  default: boolean;
  builtIn: boolean;
}

export interface Tool {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

export interface Model {
  kind: string;
  name: string;
  runs: string;
  id?: string | null;
}

export interface Deployment {
  recipe: string;
  label: string;
  models: Model[];
}

/** The fields of a custom endpoint for one slot, in form order, as the agent reads them. */
export interface CustomFieldSpec {
  fields: string[];
  required: string[];
}

/** What `/client-config` tells the page before it signs in or connects. */
export interface ClientConfig {
  ws_url: string;
  /** Whether the server accepts custom endpoints in the join token. */
  custom_endpoints: boolean;
  custom_fields: Partial<Record<Slot, CustomFieldSpec>>;
}

export interface AuthStatus {
  required: boolean;
  username: boolean;
  authenticated: boolean;
}

export interface PhoneNumber {
  number: string;
  scenario: string;
  name: string;
}

/** What the agent receives for one slot: an offered service with its settings, or a custom endpoint. */
export type SlotChoice = { key: string; settings?: ServiceSettingValues } | { custom: Record<string, string> };

/** The body of `POST /token`. The server keeps only the choices that it offered. */
export interface TokenRequest {
  scenario: string;
  prompt?: string;
  voice?: string;
  services: Partial<Record<Slot, SlotChoice>>;
}

export interface TokenResponse {
  token: string;
  room: string;
}
