// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-turn timeline of the session view: latency stages from the end of user
// speech to the first response, tool calls, and the host load during the turn.

import type { SystemSample, ToolCall, TurnDetail } from "./api";
import { INK, KIND_LABELS, SERIES, stageInfo } from "./chartTheme";
import { formatSecs } from "./utils";

const OUTCOME_TONE: Record<ToolCall["outcome"], string> = {
  ok: "rv-badge-muted",
  error: "rv-badge-bad",
  timeout: "rv-badge-bad",
  cancelled: "rv-badge-warn",
};

/** GPU and CPU load (both percent, one axis) over the turn; the band is the wait for the first response. */
function LoadSparkline({
  samples,
  start,
  end,
  band,
}: Readonly<{ samples: SystemSample[]; start: number; end: number; band: [number, number] | null }>) {
  const width = 220;
  const height = 36;
  const span = Math.max(end - start, 1e-6);
  const x = (ts: number) => ((ts - start) / span) * width;
  const y = (value: number) => height - 1 - (Math.min(Math.max(value, 0), 100) / 100) * (height - 2);
  const line = (key: "gpu_load" | "cpu_load") =>
    samples
      .filter((sample) => sample[key] !== null)
      .map((sample) => `${x(sample.ts).toFixed(1)},${y(sample[key] as number).toFixed(1)}`)
      .join(" ");
  const gpu = line("gpu_load");
  const cpu = line("cpu_load");
  if (!gpu && !cpu) return null;
  return (
    <svg
      className="rv-load"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="GPU and CPU load during the turn"
    >
      <title>Host load during the turn, 0–100%. Shaded: waiting for the first response.</title>
      {band && (
        <rect x={x(band[0])} y={0} width={Math.max(x(band[1]) - x(band[0]), 1)} height={height} fill="rgba(255,255,255,0.08)" />
      )}
      <line x1={0} y1={height - 0.5} x2={width} y2={height - 0.5} stroke={INK.axis} />
      {cpu && <polyline points={cpu} fill="none" stroke={SERIES[1]} strokeWidth={2} strokeLinejoin="round" />}
      {gpu && <polyline points={gpu} fill="none" stroke={SERIES[0]} strokeWidth={2} strokeLinejoin="round" />}
    </svg>
  );
}

export function TurnTimeline({ turn, samples }: Readonly<{ turn: TurnDetail; samples: SystemSample[] }>) {
  const metrics = turn.metrics;
  if (!metrics) return null;
  const segments = metrics.segments ?? [];
  const total = metrics.total_secs;
  const start = segments.length ? segments[0].start : null;
  // Fill the gaps between stage stretches with the unexplained remainder.
  const parts: { stage: string; secs: number }[] = [];
  if (start !== null && total) {
    let cursor = start;
    for (const segment of segments) {
      if (segment.start - cursor > 0.0005) parts.push({ stage: "unexplained", secs: segment.start - cursor });
      parts.push({ stage: segment.stage, secs: segment.end - segment.start });
      cursor = segment.end;
    }
    if (start + total - cursor > 0.0005) parts.push({ stage: "unexplained", secs: start + total - cursor });
  }
  const turnStart = turn.user_started_at ?? start;
  const turnEnd = Math.max(turn.bot_stopped_at ?? 0, start !== null && total ? start + total : 0) || null;
  const turnSamples =
    turnStart && turnEnd ? samples.filter((sample) => sample.ts >= turnStart - 1 && sample.ts <= turnEnd + 1) : [];
  const stageKeys = [...new Set(parts.map((part) => part.stage))];
  return (
    <div className="rv-timeline">
      <div className="rv-timeline-head">
        <span className="rv-badge rv-badge-muted">{KIND_LABELS[metrics.kind] ?? metrics.kind}</span>
        {metrics.response_latency !== null && (
          <span
            className="rv-badge rv-badge-good"
            title={
              metrics.response_via === "tool"
                ? "user stops → a perceivable tool call is sent"
                : "user stops → first perceivable response (the first bot audio)"
            }
          >
            response {formatSecs(metrics.response_latency)}
            {metrics.response_via === "tool" ? " (action)" : ""}
          </span>
        )}
        {metrics.voice_latency !== null && (
          <span className="rv-badge rv-badge-muted" title="user stops → first bot audio">
            voice {formatSecs(metrics.voice_latency)}
          </span>
        )}
        {metrics.barge_in && (
          <span className="rv-badge rv-badge-warn" title="The user started speaking while the bot was speaking">
            barge-in
          </span>
        )}
        <span className="text-xs text-muted">
          {metrics.n_llm_calls ?? 0} LLM call{metrics.n_llm_calls === 1 ? "" : "s"}
          {metrics.prompt_tokens !== null && ` · ${metrics.prompt_tokens}→${metrics.completion_tokens ?? "?"} tok`}
          {metrics.gpu_load_mean !== null &&
            ` · GPU ${Math.round(metrics.gpu_load_mean)}% mean, ${Math.round(metrics.gpu_load_peak ?? 0)}% peak`}
        </span>
      </div>
      {parts.length > 0 && total && (
        <div className="rv-timeline-row">
          <div className="rv-stage-track" role="img" aria-label="Latency stages up to the first response">
            {parts.map((part, i) => {
              const info = stageInfo(part.stage);
              return (
                <span
                  key={`${part.stage}-${i}`}
                  className="rv-stage-seg"
                  style={{ flexGrow: part.secs, background: info.color }}
                  title={`${info.label}: ${formatSecs(part.secs)}`}
                />
              );
            })}
          </div>
          {start !== null && turnStart && turnEnd && (
            <LoadSparkline samples={turnSamples} start={turnStart} end={turnEnd} band={[start, start + total]} />
          )}
        </div>
      )}
      {parts.length > 0 && (
        <div className="rv-stage-legend">
          {stageKeys.map((key) => {
            const info = stageInfo(key);
            const secs = parts.filter((part) => part.stage === key).reduce((sum, part) => sum + part.secs, 0);
            return (
              <span key={key} className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: info.color }} />
                {info.label} {formatSecs(secs)}
              </span>
            );
          })}
          {turnSamples.length > 0 && (
            <>
              <span className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: SERIES[0] }} />
                GPU load
              </span>
              <span className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: SERIES[1] }} />
                CPU load
              </span>
            </>
          )}
        </div>
      )}
      {turn.tool_calls.length > 0 && (
        <div className="rv-stage-legend">
          {turn.tool_calls.map((call) => (
            <span
              key={call.call_id}
              className={`rv-badge ${OUTCOME_TONE[call.outcome]}`}
              title={`${call.trigger === "intent" ? "intent engine" : "LLM"} → ${call.target}${call.perceivable ? " · perceivable" : ""}`}
            >
              {call.name} · {formatSecs(call.duration_secs)} · {call.outcome}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
