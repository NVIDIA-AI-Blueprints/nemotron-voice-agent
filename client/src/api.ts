// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useQuery, QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, retry: 1 } },
});

export type BuiltInServiceSource = "cloud-nim" | "self-hosted";

export interface LLMService {
  id: string;
  name: string;
  modelId: string;
  baseUrl: string;
  systemPrompt: string;
  extraParams: string;
  builtIn: boolean;
  source?: BuiltInServiceSource;
}

export interface Prompt {
  key: string;
  description: string;
  content: string;
  default?: boolean;
  builtIn: boolean;
  selectable?: boolean;
  scope?: "session" | "agent";
  agent?: string;
  promptName?: string;
  tools?: string[];
}

export interface Tool {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

export type SubagentReasoning = "on" | "off" | "on_demand";

export interface Subagent {
  key: string;
  label: string;
  capability: string;
  delegatable: boolean;
  reasoning: SubagentReasoning;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }
  return res.json();
}

function normalizeServiceSource(source: unknown): BuiltInServiceSource | undefined {
  return source === "cloud-nim" || source === "self-hosted" ? source : undefined;
}

function serviceCatalogQueryOptions(pipelineMode = "") {
  const qs = pipelineMode ? `?pipeline_mode=${encodeURIComponent(pipelineMode)}` : "";
  return {
    queryKey: ["services", pipelineMode] as const,
    queryFn: () => fetchJson<ServiceCatalog>(`/api/services${qs}`),
  };
}

export function useDefaultLLMs(pipelineMode = "") {
  return useQuery({
    ...serviceCatalogQueryOptions(pipelineMode),
    select: (catalog): LLMService[] =>
      (catalog.llm ?? []).map((e) => ({
        id: e.id,
        name: e.name,
        modelId: String(e.model_id ?? ""),
        baseUrl: String(e.base_url ?? ""),
        systemPrompt: String(e.system_prompt ?? ""),
        extraParams: String(e.extra_params ?? ""),
        builtIn: true,
        source: normalizeServiceSource(e.source),
      })),
  });
}

export function useDefaultPrompts(pipelineMode = "") {
  const qs = pipelineMode ? `?pipeline_mode=${encodeURIComponent(pipelineMode)}` : "";
  return useQuery<Prompt[]>({
    queryKey: ["prompts", pipelineMode],
    queryFn: () => fetchJson<Prompt[]>(`/api/prompts${qs}`),
    select: (data) => data.map((p) => ({ ...p, builtIn: true, tools: p.tools ?? [] })),
  });
}

export function useDefaultTools(pipelineMode = "") {
  const qs = pipelineMode ? `?pipeline_mode=${encodeURIComponent(pipelineMode)}` : "";
  return useQuery<Tool[]>({
    queryKey: ["tools", pipelineMode],
    queryFn: () => fetchJson<Tool[]>(`/api/tools${qs}`),
  });
}

export function useSubagents(pipelineMode = "") {
  const qs = pipelineMode ? `?pipeline_mode=${encodeURIComponent(pipelineMode)}` : "";
  return useQuery<Subagent[]>({
    queryKey: ["subagents", pipelineMode],
    queryFn: () => fetchJson<Subagent[]>(`/api/subagents${qs}`),
  });
}

export interface SimpleService {
  id: string;
  name: string;
  server: string;
  model?: string;
  voiceId?: string;
  functionId?: string;
  builtIn: boolean;
  source?: BuiltInServiceSource;
}

export function useDefaultASR(pipelineMode = "") {
  return useQuery({
    ...serviceCatalogQueryOptions(pipelineMode),
    select: (catalog): SimpleService[] =>
      (catalog.asr ?? []).map((e) => ({
        id: e.id,
        name: e.name,
        server: String(e.server ?? ""),
        model: e.model ? String(e.model) : undefined,
        functionId: e.function_id ? String(e.function_id) : undefined,
        builtIn: true,
        source: normalizeServiceSource(e.source),
      })),
  });
}

export function useDefaultTTS(pipelineMode = "") {
  return useQuery({
    ...serviceCatalogQueryOptions(pipelineMode),
    select: (catalog): SimpleService[] =>
      (catalog.tts ?? []).map((e) => ({
        id: e.id,
        name: e.name,
        server: String(e.server ?? ""),
        model: e.model ? String(e.model) : undefined,
        voiceId: e.voice_id ? String(e.voice_id) : undefined,
        functionId: e.function_id ? String(e.function_id) : undefined,
        builtIn: true,
        source: normalizeServiceSource(e.source),
      })),
  });
}

export interface TTSVoice {
  id: string;
  name: string;
  language: string;
}

export interface TTSConfig {
  languages: string[];
  voices: TTSVoice[];
  defaultVoiceId: string;
}

export interface ServiceEntry {
  id: string;
  name: string;
  builtIn: boolean;
  source?: BuiltInServiceSource;
  [key: string]: unknown;
}

export type ServiceCatalog = Record<string, ServiceEntry[]>;
export type DeploymentDefaults = Record<string, Array<ServiceEntry | Prompt> | undefined>;

export interface DeploymentOption {
  id: string;
  key: string;
  label: string;
  slots: string[];
  capabilities?: string[];
  default_session_language?: string;
  defaults?: DeploymentDefaults;
}

export type TransportType = "webrtc" | "websocket";

export interface TransportOption {
  id: TransportType;
  label: string;
}

export interface DeploymentResponse {
  active: DeploymentOption;
  selectable: boolean;
  options: DeploymentOption[];
  transports: TransportOption[];
}

export interface IceServersResponse {
  iceServers?: RTCIceServer[];
}

export interface IceConfig {
  iceServers: RTCIceServer[];
  hasTurnServer: boolean;
}

export interface WebcamConfig {
  sample_interval_seconds?: number;
  frame_max_width?: number;
  jpeg_quality?: number;
  initial_upload_enabled?: boolean;
  initial_upload_delay_ms?: number;
}

function iceServerHasTurn(server: RTCIceServer): boolean {
  const urls = Array.isArray(server.urls) ? server.urls : [server.urls];
  return urls.some((url) => typeof url === "string" && url.trim().toLowerCase().startsWith("turn"));
}

export function useDeployment() {
  return useQuery<DeploymentResponse>({
    queryKey: ["deployment"],
    queryFn: () => fetchJson<DeploymentResponse>("/api/deployment"),
  });
}

export function useIceServers() {
  return useQuery<IceConfig>({
    queryKey: ["ice-servers"],
    queryFn: async () => {
      try {
        const data = await fetchJson<IceServersResponse>("/api/ice-servers");
        const iceServers = data.iceServers ?? [];
        return {
          iceServers,
          hasTurnServer: iceServers.some(iceServerHasTurn),
        };
      } catch {
        return { iceServers: [], hasTurnServer: false };
      }
    },
    retry: false,
  });
}

export function useServiceCatalog(pipelineMode = "") {
  return useQuery(serviceCatalogQueryOptions(pipelineMode));
}

export function useVoiceCatalog(
  server?: string,
  voiceId?: string,
  asrServer?: string,
  asrModel?: string,
  asrFunctionId?: string,
  functionId?: string,
  model?: string,
) {
  return useQuery<TTSConfig>({
    queryKey: [
      "tts-config",
      server || "default",
      voiceId || "",
      functionId || "",
      model || "",
      asrServer || "",
      asrModel || "",
      asrFunctionId || "",
    ],
    queryFn: () => {
      const params = new URLSearchParams();
      if (server) params.set("server", server);
      if (voiceId) params.set("voice_id", voiceId);
      if (functionId) params.set("function_id", functionId);
      if (model) params.set("model", model);
      if (asrServer) params.set("asr_server", asrServer);
      if (asrModel) params.set("asr_model", asrModel);
      if (asrFunctionId) params.set("asr_function_id", asrFunctionId);
      const url = params.size > 0 ? `/api/tts-config?${params.toString()}` : "/api/tts-config";
      return fetchJson<TTSConfig>(url);
    },
  });
}

export async function createSessionConfig(config: Record<string, string>): Promise<string> {
  const res = await fetch("/api/session-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }
  const data = await res.json();
  return data.session_id;
}

export async function createWebRTCSession(config: Record<string, string>): Promise<string> {
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }

  const data = await res.json() as { webrtcUrl?: string };
  if (!data.webrtcUrl) {
    throw new Error("WebRTC start did not return a connection URL");
  }
  return data.webrtcUrl;
}

export async function uploadAttachment(sessionId: string, file: File, kind: string) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/attachments?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }
  return res.json();
}

export async function getWebcamConfig(): Promise<WebcamConfig> {
  return fetchJson<WebcamConfig>("/api/webcam-config");
}

export async function uploadWebcamFrame(sessionId: string, frame: Blob) {
  const form = new FormData();
  form.append("file", frame, "webcam-frame.jpg");
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/webcam/frames`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }
  return res.json();
}

export async function uploadWebcamCapture(sessionId: string, frame: Blob, requestId: string) {
  const form = new FormData();
  form.append("file", frame, "focused-capture.jpg");
  form.append("request_id", requestId);
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/webcam/capture`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    const details = body ? `: ${body.slice(0, 200)}` : "";
    throw new Error(`HTTP ${res.status}${details}`);
  }
  return res.json();
}
