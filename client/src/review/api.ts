// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

// ---- Types (mirror src/monitoring/api.py) ----

export interface SessionModels {
  asr?: string | null;
  llm?: string | null;
  tts?: string | null;
  voice?: string | null;
}

export interface SessionSummary {
  id: string;
  example: string;
  started_at: number;
  ended_at: number | null;
  end_reason: string | null;
  language: string | null;
  person?: { id: string; name: string } | null;
  models: SessionModels;
  user_turns: number;
  audio_turns: number;
  reviewed_turns: number;
  wer: Record<string, number | null>;
  jobs: Record<string, string>;
}

export interface MediaRef {
  id: number;
  key: string;
  mime: string | null;
  duration_secs: number | null;
  source: string | null;
  ts: number;
  width?: number | null;
  height?: number | null;
}

export interface LlmCallBrief {
  id: number;
  started_at: number;
  model: string | null;
  ttfb: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  n_images: number | null;
  interrupted: boolean;
  output_text: string | null;
  function_calls: { name: string; tool_call_id: string; arguments: unknown }[] | null;
}

export interface WerValue {
  wer: number;
  cer: number;
  word_errors: number;
  ref_words: number;
  reference_source: string;
}

export interface HumanReference {
  source: string;
  text: string;
  at: number;
}

export type TurnKind = "plain" | "tool" | "intent" | "vision";

/** One stretch of the turn timeline owned by a stage (epoch seconds). */
export interface StageSegment {
  stage: string;
  start: number;
  end: number;
}

/** Row of ``turn_metrics`` (src/monitoring/turn_metrics.py); stage columns are ``<stage>_secs``. */
export interface TurnMetrics {
  kind: TurnKind;
  barge_in: boolean | null;
  voice_latency: number | null;
  response_latency: number | null;
  response_via: "audio" | "tool" | null;
  total_secs: number | null;
  unexplained_secs: number | null;
  segments: StageSegment[] | null;
  n_llm_calls: number | null;
  n_tool_calls: number | null;
  n_images: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  gpu_load_mean: number | null;
  gpu_load_peak: number | null;
  [stageSecs: `${string}_secs`]: number | null;
}

export interface ToolCall {
  call_id: string;
  turn_idx: number | null;
  name: string;
  trigger: "llm" | "intent";
  target: "client" | "home_assistant";
  sent_at: number;
  duration_secs: number | null;
  outcome: "ok" | "error" | "timeout" | "cancelled";
  perceivable: boolean;
}

/** Host-wide sample taken once per second while a session is live. */
export interface SystemSample {
  ts: number;
  live_sessions: number;
  gpu_load: number | null;
  gpu_temp_c: number | null;
  power_w: number | null;
  /** What power_w measures: whole-board input power, or GPU power draw only. */
  power_source: "board" | "gpu" | null;
  ram_used_mb: number | null;
  cpu_load: number | null;
}

export interface TurnDetail {
  idx: number;
  user_text?: string | null;
  bot_text?: string | null;
  user_started_at?: number | null;
  /** Turn released to the LLM (after turn detection and ASR), not the end of speech. */
  user_stopped_at?: number | null;
  /** The user actually stopped speaking. */
  user_speech_stopped_at?: number | null;
  /** LLM response start, not the first audio. */
  bot_started_at?: number | null;
  /** First bot audio. */
  bot_speech_started_at?: number | null;
  bot_stopped_at?: number | null;
  /** The bot's response was cut (interruption or session end while it was open). */
  interrupted?: boolean;
  /** The user started this turn while the bot was still speaking. */
  barge_in?: boolean | null;
  metrics: TurnMetrics | null;
  tool_calls: ToolCall[];
  language?: string | null;
  audio: { user: MediaRef[]; bot: MediaRef[] };
  images: MediaRef[];
  llm_calls: LlmCallBrief[];
  transcripts: Record<string, string>;
  human_reference: HumanReference | null;
  wer: Record<string, WerValue>;
  user_bot_latency: number | null;
  /** Explicit per-turn speaker override (the session speaker applies otherwise). */
  speaker?: string | null;
}

export interface JobBrief {
  kind: string;
  status: string;
  error: string | null;
  attempts: number;
  finished_at?: number | null;
}

export interface SessionDetail {
  session: SessionSummary & { config: Record<string, unknown> };
  live_source: string;
  reference_source: string | null;
  conversation_audio: MediaRef | null;
  turns: TurnDetail[];
  system_samples: SystemSample[];
  wer_summary: Record<string, { wer: number | null; turns: number; reference_sources: string[] }>;
  jobs: JobBrief[];
  speakers: { session: string | null };
  memories_used: { used: MemoryRow[]; extracted: MemoryRow[] };
}

// ---- People & memories ----

export interface Person {
  id: string;
  name: string;
  created_at: number;
  archived: boolean;
  sessions?: number;
  memories?: Partial<Record<MemoryStatus, number>>;
  /** Voice/face samples bound at enrollment, by modality (``other`` when the modality is unknown). */
  identity_samples?: Partial<Record<IdentityModality | "other", number>>;
  /** Turns attributed live by voice ID, and how many of them a face confirmed. */
  voice_id_turns?: { turns: number; verified: number };
}

export type IdentityModality = "voice" | "face";

export interface IdentityModel {
  model: string;
  modality: IdentityModality | null;
  count: number;
  first_at: number;
  last_at: number;
  sessions: string[];
}

export interface DuplicatePair {
  people: Person[];
  scores: { model: string; modality: IdentityModality | null; score: number }[];
  same_name: boolean;
}

export interface PersonIdentity {
  models: IdentityModel[];
  duplicates: DuplicatePair[];
}

export interface MergeResult {
  person: Person;
  merged: Person;
  moved: { turns: number; embeddings: number; memories: number };
}

export type MemoryStatus = "proposed" | "active" | "forgotten" | "superseded";
export type MemoryFilter = "open" | "usable" | "all" | MemoryStatus;

export interface MemoryRow {
  id: number;
  person_id: string;
  text: string;
  category: string | null;
  language: string | null;
  confidence: number | null;
  status: MemoryStatus;
  source: string;
  supersedes: number | null;
  superseded_by: number | null;
  reviewed_by: string | null;
  reviewed_at: number | null;
  created_at: number;
}

export interface MemoryEvidence {
  session_id: string;
  turn_idx: number | null;
  quote: string | null;
  user_text?: string | null;
  bot_text?: string | null;
  audio: { key: string; duration_secs: number | null }[];
}

export interface MemoryDetail extends MemoryRow {
  person_name: string | null;
  supersedes_text: string | null;
  evidence: MemoryEvidence[];
  used_in_sessions: number;
  used_in_replies: number;
}

export interface AsrQueueItem {
  session_id: string;
  turn_idx: number;
  started_at: number;
  language: string | null;
  audio: { key: string; duration_secs: number | null }[];
  live: { source: string; text: string };
  models: Record<string, string>;
  reference_source: string | null;
  human_reference: HumanReference | null;
  disagreement: number | null;
  agree: boolean;
}

export interface AsrQueue {
  total: number;
  open: number;
  agreeing: number;
  matching: number;
  items: AsrQueueItem[];
}

export type AsrQueueSort = "recent" | "disagreement";

export interface Activity {
  id: string;
  title: string;
  job: string;
  description: string;
  open: number;
}

export interface DreamerJob {
  id: number;
  kind: string;
  target: string;
  status: string;
  attempts: number;
  error: string | null;
  progress: Record<string, unknown> | null;
}

export interface DreamerWorker {
  state: "starting" | "idle" | "waiting" | "running" | "paused" | string;
  alive: boolean;
  heartbeat_at: number;
  host?: string;
  kind?: string;
  target?: string;
  job_id?: number;
  reason?: string;
  started_services?: string[];
}

export interface DreamerStatus {
  live_sessions: number;
  idle: boolean;
  last_activity: number | null;
  paused: boolean;
  paused_at: number | null;
  worker: DreamerWorker | null;
  services: { name: string; state: string }[];
  job_kinds: string[];
  counts: Record<string, Record<string, number>>;
  running: DreamerJob[];
  failed: DreamerJob[];
  pending: number;
}

export type JobStatus = "pending" | "running" | "done" | "failed" | "cancelled";

export interface JobRow extends DreamerJob {
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface Distribution {
  n: number;
  p10: number | null;
  p50: number | null;
  p90: number | null;
}

/** Latency and stage distributions of a set of turns; ``turns`` is the sample count. */
export interface KindStats {
  turns: number;
  response: Distribution;
  voice: Distribution;
  stages: Record<string, Distribution>;
  barge_in_rate: number | null;
  llm_calls_mean: number | null;
  prompt_tokens_p50: number | null;
  completion_tokens_p50: number | null;
  gpu_load_mean: number | null;
  gpu_load_peak: number | null;
  /** Trend buckets only: set when the median response is clearly worse than the previous bucket. */
  regression?: { previous: string; previous_p50: number } | null;
}

/** Keyed by "all" and by each turn kind present. */
export type ByKind = Partial<Record<TurnKind | "all", KindStats>>;

export interface TrendBucket {
  key: string;
  label: string;
  started_at: number;
  sessions: number;
  by_kind: ByKind;
}

export interface ToolStats {
  name: string;
  trigger: "llm" | "intent";
  target: "client" | "home_assistant";
  perceivable: boolean;
  calls: number;
  duration: Distribution;
  failure_rate: number;
  error_rate: number;
  timeout_rate: number;
  cancelled_rate: number;
}

export interface MetricsVariant {
  key: Record<string, string>;
  label: string;
  sessions: number;
  turns: number;
  by_kind: ByKind;
  interrupt_rate: number | null;
  latency: Distribution;
  first_speech: Distribution;
  ttfb: Record<string, Distribution>;
  ttft: { text: Distribution; vision: Distribution };
  prompt_tokens_p50: number | null;
  wer: Record<string, { wer: number | null; ref_words: number }>;
}

export interface MetricsResponse {
  group_by: string[];
  kinds: TurnKind[];
  stages: string[];
  totals: { sessions: number; turns: number; latency_p50: number | null; live_wer: number | null; by_kind: ByKind };
  variants: MetricsVariant[];
  trend: { by_day: TrendBucket[]; by_revision: TrendBucket[] };
  tools: ToolStats[];
  sessions: {
    id: string;
    started_at: number;
    variant: number;
    latency_p50: number | null;
    response_p50: number | null;
    live_wer: number | null;
  }[];
}

export interface ReferenceItem {
  session_id: string;
  turn_idx: number;
  text: string;
}

// ---- Fetch helpers ----

const BASE = "/api/review";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}${body ? `: ${body.slice(0, 200)}` : ""}`);
  }
  return res.json();
}

function postJson<T>(path: string, body: unknown, method = "POST"): Promise<T> {
  return request<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function artifactUrl(key: string): string {
  return `${BASE}/artifacts/${key.split("/").map(encodeURIComponent).join("/")}`;
}

// ---- Queries ----

const live = { staleTime: 0, refetchOnWindowFocus: true } as const;

export function useActivities() {
  return useQuery({
    queryKey: ["review", "activities"],
    queryFn: () => request<{ activities: Activity[] }>("/activities"),
    ...live,
  });
}

export function useSessions(limit = 100) {
  return useQuery({
    queryKey: ["review", "sessions", limit],
    queryFn: () => request<{ total: number; sessions: SessionSummary[] }>(`/sessions?limit=${limit}`),
    ...live,
  });
}

export function useSessionDetail(sessionId: string) {
  return useQuery({
    queryKey: ["review", "session", sessionId],
    queryFn: () => request<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`),
    enabled: Boolean(sessionId),
    ...live,
  });
}

export function useLlmCall(sessionId: string, callId: number | null) {
  return useQuery({
    queryKey: ["review", "llm-call", sessionId, callId],
    queryFn: () =>
      request<{ messages: unknown; output_text: string; tools: string[] }>(
        `/sessions/${encodeURIComponent(sessionId)}/llm-calls/${callId}`
      ),
    enabled: callId !== null,
    staleTime: Infinity,
  });
}

export function useAsrQueue(status: "open" | "reviewed" | "all", sort: AsrQueueSort, limit = 200) {
  return useQuery({
    queryKey: ["review", "asr-queue", status, sort, limit],
    queryFn: () => request<AsrQueue>(`/asr/queue?status=${status}&sort=${sort}&limit=${limit}`),
    ...live,
  });
}

export function useDreamerStatus() {
  return useQuery({
    queryKey: ["review", "dreamer"],
    queryFn: () => request<DreamerStatus>("/dreamer"),
    refetchInterval: 5000,
    staleTime: 0,
  });
}

export function useJobs(status: JobStatus | "", page: number, pageSize = 25) {
  return useQuery({
    queryKey: ["review", "jobs", status, page, pageSize],
    queryFn: () =>
      request<{ total: number; jobs: JobRow[] }>(
        `/jobs?limit=${pageSize}&offset=${page * pageSize}${status ? `&status=${status}` : ""}`
      ),
    refetchInterval: 5000,
    staleTime: 0,
    placeholderData: (previous) => previous,
  });
}

export function useMetrics(sinceDays: number | null, groupBy: string[]) {
  const params = new URLSearchParams({ group_by: groupBy.join(",") });
  if (sinceDays) params.set("since_days", String(sinceDays));
  return useQuery({
    queryKey: ["review", "metrics", sinceDays, groupBy.join(",")],
    queryFn: () => request<MetricsResponse>(`/metrics?${params}`),
    staleTime: 0,
    // Keep the previous render while a new filter loads (no skeleton flash).
    placeholderData: (previous) => previous,
  });
}

export function usePeople(includeArchived = false) {
  return useQuery({
    queryKey: ["review", "people", includeArchived],
    queryFn: () => request<{ people: Person[] }>(`/people${includeArchived ? "?include_archived=true" : ""}`),
    ...live,
  });
}

export function usePersonIdentity(personId: string) {
  return useQuery({
    queryKey: ["review", "people", "identity", personId],
    queryFn: () => request<PersonIdentity>(`/people/${encodeURIComponent(personId)}/identity`),
    ...live,
  });
}

export function useDuplicatePeople() {
  return useQuery({
    queryKey: ["review", "people", "duplicates"],
    queryFn: () => request<{ duplicates: DuplicatePair[] }>("/people/duplicates"),
    ...live,
  });
}

export function useMemories(status: MemoryFilter, personId: string, page: number, pageSize = 20) {
  const params = new URLSearchParams({ status, limit: String(pageSize), offset: String(page * pageSize) });
  if (personId) params.set("person_id", personId);
  return useQuery({
    queryKey: ["review", "memories", status, personId, page, pageSize],
    queryFn: () => request<{ total: number; memories: MemoryDetail[] }>(`/memories?${params}`),
    ...live,
    placeholderData: (previous) => previous,
  });
}

// ---- Mutations ----

export function useSaveReferences() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: { annotator: string; items: ReferenceItem[] }) =>
      postJson<{ saved: number; wer_summary: Record<string, Record<string, unknown>> }>(
        "/asr/references",
        payload
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }),
  });
}

export function useRequeueJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: { kind: string; session_ids: string[] }) =>
      postJson<{ queued: number }>("/jobs", payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }),
  });
}

function useReviewMutation<TIn, TOut>(fn: (input: TIn) => Promise<TOut>) {
  const qc = useQueryClient();
  return useMutation({ mutationFn: fn, onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }) });
}

export function usePauseDreamer() {
  return useReviewMutation((paused: boolean) => postJson<{ paused: boolean }>("/dreamer/pause", { paused }));
}

export function useCancelJob() {
  return useReviewMutation((jobId: number) => postJson<{ cancelled: number }>(`/jobs/${jobId}/cancel`, {}));
}

export function useEnqueueAll() {
  return useReviewMutation((kind: string) => postJson<{ queued: number }>("/jobs/enqueue-all", { kind }));
}

export function useCreatePerson() {
  return useReviewMutation((name: string) => postJson<Person>("/people", { name }));
}

export function useUpdatePerson() {
  return useReviewMutation((input: { id: string; name?: string; archived?: boolean }) =>
    postJson<Person>(`/people/${encodeURIComponent(input.id)}`, { name: input.name, archived: input.archived }, "PATCH")
  );
}

export function useMergePeople() {
  return useReviewMutation((input: { targetId: string; sourceId: string; annotator: string }) =>
    postJson<MergeResult>(`/people/${encodeURIComponent(input.targetId)}/merge`, {
      annotator: input.annotator,
      source_id: input.sourceId,
    })
  );
}

export function useForgetIdentity() {
  return useReviewMutation((input: { personId: string; annotator: string; model?: string }) =>
    postJson<{ deleted: number }>(`/people/${encodeURIComponent(input.personId)}/forget-identity`, {
      annotator: input.annotator,
      model: input.model ?? null,
    })
  );
}

export function useAssignSpeaker() {
  return useReviewMutation(
    (input: { sessionId: string; annotator: string; personId: string | null; turnIdx?: number }) =>
      postJson<{ session: string | null; turns: Record<string, string> }>(
        `/sessions/${encodeURIComponent(input.sessionId)}/speaker`,
        { annotator: input.annotator, person_id: input.personId, turn_idx: input.turnIdx ?? null }
      )
  );
}

export function useReviewMemory() {
  return useReviewMutation(
    (input: {
      id: number;
      annotator: string;
      action: "approve" | "correct" | "forget";
      text?: string;
      personId?: string;
    }) =>
      postJson<MemoryDetail>(`/memories/${input.id}/review`, {
        annotator: input.annotator,
        action: input.action,
        text: input.text,
        person_id: input.personId,
      })
  );
}
