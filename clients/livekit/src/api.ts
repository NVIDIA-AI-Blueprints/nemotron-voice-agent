// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient, useQuery } from "@tanstack/react-query";
import type {
  AuthStatus,
  ClientConfig,
  Deployment,
  PhoneNumber,
  PromptEntry,
  ServiceCatalog,
  TokenRequest,
  TokenResponse,
  Tool,
  VoiceList,
} from "./lib/types";

export type * from "./lib/types";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, retry: 1 } },
});

/** How often to ask again while the agent worker has not published yet: it can still be loading models. */
const STARTING_POLL_MS = 2000;
/** How often the service status refreshes once known. The worker checks each self-hosted service every 10 s. */
const STATUS_POLL_MS = 10_000;

export class HttpError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as { error?: unknown };
    return typeof body.error === "string" && body.error ? body.error : fallback;
  } catch {
    return fallback;
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new HttpError(await errorMessage(response, `${path} request failed (${response.status})`), response.status);
  }
  return (await response.json()) as T;
}

export const isUnauthorized = (error: unknown) => error instanceof HttpError && error.status === 401;

export function useAuthStatus() {
  return useQuery({ queryKey: ["auth-status"], queryFn: () => fetchJson<AuthStatus>("/auth/status"), retry: false });
}

export function useClientConfig() {
  return useQuery({ queryKey: ["client-config"], queryFn: () => fetchJson<Partial<ClientConfig>>("/client-config") });
}

export function useDeployment() {
  return useQuery({
    queryKey: ["deployment"],
    queryFn: () => fetchJson<Partial<Deployment>>("/deployment"),
    refetchInterval: (query) => (query.state.data?.models ? false : STARTING_POLL_MS),
  });
}

export function useServices() {
  return useQuery({
    queryKey: ["services"],
    queryFn: () => fetchJson<ServiceCatalog>("/services"),
    refetchInterval: (query) => (Object.keys(query.state.data ?? {}).length > 0 ? STATUS_POLL_MS : STARTING_POLL_MS),
  });
}

/** The voices of an offered TTS entry. Empty until the worker could ask that TTS server. */
export function useVoices(ttsKey: string) {
  return useQuery({
    queryKey: ["voices", ttsKey],
    queryFn: () => fetchJson<VoiceList>(`/voices?${new URLSearchParams({ tts: ttsKey })}`),
    enabled: Boolean(ttsKey),
    refetchInterval: (query) => ((query.state.data?.voices.length ?? 0) > 0 ? false : STARTING_POLL_MS),
  });
}

export function usePromptCatalog() {
  return useQuery({ queryKey: ["prompts"], queryFn: () => fetchJson<PromptEntry[]>("/prompts") });
}

export function useTools() {
  return useQuery({ queryKey: ["tools"], queryFn: () => fetchJson<Tool[]>("/tools") });
}

export function usePhoneNumbers() {
  return useQuery({
    queryKey: ["phone"],
    queryFn: async () => (await fetchJson<{ numbers?: PhoneNumber[] }>("/phone")).numbers ?? [],
  });
}

/** Ask the UI server for a join token with the session choices. A refused choice throws with the server's reason. */
export function requestToken(body: TokenRequest): Promise<TokenResponse> {
  return fetchJson<TokenResponse>("/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Post the typed credentials. The server checks them and sets the session cookie. Returns the HTTP status. */
export async function login(username: string, password: string): Promise<number> {
  const response = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return response.status;
}
