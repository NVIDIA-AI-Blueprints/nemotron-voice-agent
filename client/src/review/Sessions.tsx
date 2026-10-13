// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  artifactUrl,
  useAssignSpeaker,
  useLlmCall,
  usePeople,
  useRequeueJob,
  useSaveReferences,
  useSessionDetail,
  useSessions,
  type LlmCallBrief,
  type MemoryRow,
  type Person,
  type SystemSample,
  type TurnDetail,
} from "./api";
import { powerLabel } from "./chartTheme";
import { TurnTimeline } from "./TurnTimeline";
import { MemoryStatusBadge } from "./Memories";
import { AudioClips, ImageStrip, StatusLine, TranscriptTable, WerBadge } from "./components";
import { formatDate, formatDuration, formatSecs, reviewHref, useAnnotator } from "./utils";

export function SessionsList() {
  const sessions = useSessions();
  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Sessions</h2>
          <p className="text-sm text-muted">Recorded conversations, newest first.</p>
        </div>
      </div>
      <StatusLine loading={sessions.isLoading} error={sessions.error} />
      {sessions.data && sessions.data.sessions.length === 0 && (
        <p className="text-sm text-muted">No recorded sessions yet. Set MONITORING_ENABLED=true and have a conversation.</p>
      )}
      {sessions.data && sessions.data.sessions.length > 0 && (
        <table className="rv-table">
          <thead>
            <tr>
              <th>Started</th>
              <th>Session</th>
              <th>Lang</th>
              <th>Person</th>
              <th>Models (ASR · LLM · TTS)</th>
              <th>Turns</th>
              <th>Reviewed</th>
              <th>Live WER</th>
              <th>Jobs</th>
            </tr>
          </thead>
          <tbody>
            {sessions.data.sessions.map((s) => {
              const liveWer = Object.entries(s.wer).find(([source]) => source.startsWith("live:"))?.[1];
              const duration = s.ended_at ? s.ended_at - s.started_at : null;
              return (
                <tr key={s.id} onClick={() => (globalThis.location.hash = reviewHref("session", s.id))}>
                  <td>
                    {formatDate(s.started_at)}
                    <div className="text-xs text-muted">{s.ended_at ? formatDuration(duration) : "live"}</div>
                  </td>
                  <td className="rv-mono">{s.id}</td>
                  <td>{s.language ?? "–"}</td>
                  <td>{s.person?.name ?? "–"}</td>
                  <td className="text-xs">
                    {s.models.asr ?? "?"} · {s.models.llm ?? "?"} · {s.models.tts ?? "?"}
                  </td>
                  <td>{s.user_turns}</td>
                  <td>
                    {s.reviewed_turns}/{s.audio_turns}
                  </td>
                  <td>{liveWer === undefined ? "–" : <WerBadge value={liveWer} />}</td>
                  <td className="text-xs">
                    {Object.entries(s.jobs)
                      .map(([kind, status]) => `${kind}: ${status}`)
                      .join(", ") || "–"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

function LlmCallRow({ sessionId, call }: Readonly<{ sessionId: string; call: LlmCallBrief }>) {
  const [open, setOpen] = useState(false);
  const detail = useLlmCall(sessionId, open ? call.id : null);
  return (
    <div className="rv-llm-call">
      <button className="btn-ghost text-xs" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} LLM {call.model ?? ""} · TTFT {formatSecs(call.ttfb)} · {call.prompt_tokens ?? "?"}→
        {call.completion_tokens ?? "?"} tok
        {call.n_images ? ` · ${call.n_images} image(s)` : ""}
        {call.function_calls?.length ? ` · tools: ${call.function_calls.map((f) => f.name).join(", ")}` : ""}
        {call.interrupted ? " · interrupted" : ""}
      </button>
      {open && (
        <pre className="rv-pre">
          {detail.isLoading ? "Loading…" : JSON.stringify(detail.data?.messages ?? detail.error?.message, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ReferenceEditor({ sessionId, turn }: Readonly<{ sessionId: string; turn: TurnDetail }>) {
  const [annotator] = useAnnotator();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(turn.human_reference?.text ?? turn.user_text ?? "");
  const save = useSaveReferences();
  if (!editing) {
    return (
      <button className="btn-ghost text-xs" onClick={() => setEditing(true)}>
        {turn.human_reference ? "Edit reference" : "Set reference"}
      </button>
    );
  }
  return (
    <div className="rv-inline-edit">
      <input className="input" value={draft} onChange={(e) => setDraft(e.target.value)} />
      <button
        className="btn-primary"
        disabled={!annotator || !draft.trim() || save.isPending}
        onClick={() =>
          save.mutate(
            { annotator, items: [{ session_id: sessionId, turn_idx: turn.idx, text: draft }] },
            { onSuccess: () => setEditing(false) }
          )
        }
      >
        Save
      </button>
      <button className="btn-ghost" onClick={() => setEditing(false)}>
        Cancel
      </button>
      {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
    </div>
  );
}

function SpeakerSelect({
  sessionId,
  people,
  value,
  turnIdx,
  inherited,
}: Readonly<{ sessionId: string; people: Person[]; value: string | null; turnIdx?: number; inherited?: string }>) {
  const [annotator] = useAnnotator();
  const assign = useAssignSpeaker();
  const isTurn = turnIdx !== undefined;
  return (
    <label className="text-xs text-muted rv-speaker" title={annotator ? undefined : "Set your annotator name first"}>
      {isTurn ? "Speaker" : "Who's talking"}
      <select
        className="input rv-select"
        value={value ?? ""}
        disabled={!annotator || assign.isPending}
        onChange={(e) =>
          assign.mutate({ sessionId, annotator, personId: e.target.value || null, turnIdx })
        }
      >
        <option value="">{isTurn ? `same as session${inherited ? ` (${inherited})` : ""}` : "unknown"}</option>
        {people
          .filter((p) => !p.archived || p.id === value)
          .map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
      </select>
      {assign.error && <span className="rv-error"> {assign.error.message}</span>}
    </label>
  );
}

function MemoryList({ title, memories }: Readonly<{ title: string; memories: MemoryRow[] }>) {
  if (!memories.length) return null;
  return (
    <div className="rv-side">
      <span className="rv-label">{title}</span>
      {memories.map((m) => (
        <p key={m.id} className="text-sm">
          <MemoryStatusBadge memory={m} /> {m.text}
        </p>
      ))}
    </div>
  );
}

/** Host-wide load while the session was live (samples of src/monitoring/system_metrics.py). */
function SystemSummary({ samples }: Readonly<{ samples: SystemSample[] }>) {
  if (!samples.length) return null;
  const stat = (key: "gpu_load" | "cpu_load" | "gpu_temp_c" | "ram_used_mb", power?: string | null) => {
    const values = samples
      .filter((sample) => power === undefined || sample.power_source === power)
      .map((sample) => (power === undefined ? sample[key] : sample.power_w))
      .filter((value): value is number => value !== null);
    if (!values.length) return null;
    return { mean: values.reduce((sum, value) => sum + value, 0) / values.length, max: Math.max(...values) };
  };
  const parts: string[] = [];
  const gpu = stat("gpu_load");
  if (gpu) parts.push(`GPU load ${gpu.mean.toFixed(0)}% mean, ${gpu.max.toFixed(0)}% peak`);
  const cpu = stat("cpu_load");
  if (cpu) parts.push(`CPU load ${cpu.mean.toFixed(0)}% mean, ${cpu.max.toFixed(0)}% peak`);
  const temp = stat("gpu_temp_c");
  if (temp) parts.push(`GPU temperature ${temp.max.toFixed(0)} °C max`);
  // Board input power and GPU power are different quantities: never merged under one label.
  for (const source of [...new Set(samples.map((sample) => sample.power_source))]) {
    const power = stat("gpu_load", source);
    if (power) parts.push(`${powerLabel(source)} ${power.mean.toFixed(1)} W mean, ${power.max.toFixed(1)} W peak`);
  }
  const ram = stat("ram_used_mb");
  if (ram) parts.push(`RAM used ${(ram.max / 1024).toFixed(1)} GB peak`);
  const sessions = Math.max(...samples.map((sample) => sample.live_sessions));
  if (!parts.length) return null;
  return (
    <p className="text-xs text-muted">
      Host during the session ({samples.length} samples, up to {sessions} live session{sessions === 1 ? "" : "s"}):{" "}
      {parts.join(" · ")}
    </p>
  );
}

function TurnCard({
  sessionId,
  liveSource,
  turn,
  people,
  sessionSpeaker,
  samples,
}: Readonly<{
  sessionId: string;
  liveSource: string;
  turn: TurnDetail;
  samples: SystemSample[];
  people: Person[];
  sessionSpeaker: string | null;
}>) {
  const reference = turn.human_reference?.text;
  const inherited = people.find((p) => p.id === sessionSpeaker)?.name;
  const liveWer = turn.wer[liveSource];
  return (
    <div className="card rv-turn">
      <div className="rv-card-head">
        <span className="rv-badge rv-badge-muted">turn {turn.idx}</span>
        {turn.idx === 0 && <span className="text-xs text-muted">greeting</span>}
        {liveWer && <WerBadge value={liveWer.wer} title={`vs ${liveWer.reference_source}`} />}
        {!turn.metrics && turn.user_bot_latency !== null && (
          <span className="rv-badge rv-badge-muted">user→bot {formatSecs(turn.user_bot_latency)}</span>
        )}
        {turn.interrupted && (
          <span
            className="rv-badge rv-badge-warn"
            title="The assistant's response was cut short (interruption or end of session)"
          >
            interrupted
          </span>
        )}
      </div>
      <TurnTimeline turn={turn} samples={samples} />
      {(turn.user_text || turn.audio.user.length > 0) && (
        <div className="rv-side rv-side-user">
          <span className="rv-label">User</span>
          {turn.user_text && (
            <SpeakerSelect
              sessionId={sessionId}
              people={people}
              value={turn.speaker ?? null}
              turnIdx={turn.idx}
              inherited={inherited}
            />
          )}
          <AudioClips clips={turn.audio.user} />
          <TranscriptTable transcripts={turn.transcripts} referenceText={reference} />
          {reference && (
            <p className="text-sm">
              <span className="rv-badge rv-badge-good">{turn.human_reference?.source}</span> {reference}
            </p>
          )}
          {turn.audio.user.length > 0 && <ReferenceEditor key={reference ?? ""} sessionId={sessionId} turn={turn} />}
        </div>
      )}
      <ImageStrip images={turn.images} />
      {turn.llm_calls.map((call) => (
        <LlmCallRow key={call.id} sessionId={sessionId} call={call} />
      ))}
      {(turn.bot_text || turn.audio.bot.length > 0) && (
        <div className="rv-side rv-side-bot">
          <span className="rv-label">Assistant</span>
          <AudioClips clips={turn.audio.bot} />
          <p className="text-sm">{turn.bot_text}</p>
        </div>
      )}
    </div>
  );
}

export function SessionDetailView({ sessionId }: Readonly<{ sessionId: string }>) {
  const detail = useSessionDetail(sessionId);
  const people = usePeople(true);
  const requeue = useRequeueJob();
  const data = detail.data;
  const peopleList = people.data?.people ?? [];
  return (
    <section className="rv-page">
      <a className="text-sm" href={reviewHref("sessions")}>
        ← Sessions
      </a>
      <StatusLine loading={detail.isLoading} error={detail.error} />
      {data && (
        <>
          <div className="rv-page-head">
            <div>
              <h2 className="text-lg font-semibold rv-mono">{data.session.id}</h2>
              <p className="text-sm text-muted">
                {formatDate(data.session.started_at)} · {data.session.language} · ASR {data.session.models.asr} · LLM{" "}
                {data.session.models.llm} · TTS {data.session.models.tts} ({data.session.models.voice})
              </p>
            </div>
            <div className="rv-actions">
              {Object.entries(data.wer_summary).map(([source, summary]) => (
                <WerBadge key={source} value={summary.wer} title={`${source} over ${summary.turns} turns`} />
              ))}
              <button
                className="btn-secondary"
                disabled={requeue.isPending}
                onClick={() => requeue.mutate({ kind: "reasr", session_ids: [sessionId] })}
                title="Transcribes missing turns with the reference/candidate ASR when the machine is idle"
              >
                Re-run reasr
              </button>
              <button
                className="btn-secondary"
                disabled={requeue.isPending}
                onClick={() => requeue.mutate({ kind: "dream", session_ids: [sessionId] })}
                title="Extracts memories again when the machine is idle (unreviewed memories from this session are replaced)"
              >
                Re-run dream
              </button>
            </div>
          </div>
          <div className="rv-filters">
            <SpeakerSelect sessionId={sessionId} people={peopleList} value={data.speakers.session} />
            <span className="text-xs text-muted">Changing who is talking re-runs memory extraction (dream).</span>
          </div>
          <MemoryList title="Memories used in this session" memories={data.memories_used.used} />
          <MemoryList title="Memories extracted from this session" memories={data.memories_used.extracted} />
          {data.jobs.length > 0 && (
            <p className="text-xs text-muted">
              Jobs: {data.jobs.map((j) => `${j.kind} ${j.status}${j.error ? ` (${j.error})` : ""}`).join(" · ")}
            </p>
          )}
          <SystemSummary samples={data.system_samples ?? []} />
          {data.conversation_audio && (
            <div className="rv-audio">
              <span className="rv-label">Whole call (user left · assistant right)</span>
              <audio controls preload="none" src={artifactUrl(data.conversation_audio.key)} />
            </div>
          )}
          {data.turns.map((turn) => (
            <TurnCard
              key={turn.idx}
              sessionId={sessionId}
              liveSource={data.live_source}
              turn={turn}
              people={peopleList}
              sessionSpeaker={data.speakers.session}
              samples={data.system_samples ?? []}
            />
          ))}
        </>
      )}
    </section>
  );
}
