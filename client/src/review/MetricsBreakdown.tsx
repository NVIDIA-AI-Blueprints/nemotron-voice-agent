// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-turn latency breakdown views of the Metrics page: stages per variant,
// trend per day / git revision, and the tool and intent call table.

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { KindStats, MetricsResponse, MetricsVariant, ToolStats, TurnKind } from "./api";
import { INK, KIND_LABELS, SERIES, STAGES, axisTick, secondsTick, short, tooltipStyle, variantColor } from "./chartTheme";
import { ChartCard, Empty } from "./components";
import { formatPct, formatSecs } from "./utils";

type StageRow = { name: string; title: string; stats: KindStats | undefined };

/** Legend of the stages that appear in the given stats (stage color is the same everywhere). */
export function StageLegend({ stages }: Readonly<{ stages: string[] }>) {
  return (
    <div className="rv-stage-legend">
      {STAGES.filter((stage) => stages.includes(stage.key)).map((stage) => (
        <span key={stage.key} className="rv-stage-legend-item">
          <span className="rv-swatch" style={{ background: stage.color }} />
          {stage.label}
        </span>
      ))}
    </div>
  );
}

/** Horizontal stacked bars: one row per variant or bucket, one segment per stage (median). */
export function StageBars({ rows, what }: Readonly<{ rows: StageRow[]; what: string }>) {
  const data = rows
    .filter((row) => row.stats && row.stats.response.n > 0)
    .map((row) => {
      const out: Record<string, number | string> = {
        name: `${row.name} · n=${row.stats!.response.n}`,
        title: row.title,
      };
      for (const stage of STAGES) out[stage.key] = row.stats!.stages[stage.key]?.p50 ?? 0;
      return out;
    });
  if (!data.length) return <Empty what={what} />;
  const present = STAGES.filter((stage) => data.some((row) => Number(row[stage.key]) > 0));
  return (
    <>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 34 + 40)}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={INK.grid} horizontal={false} />
          <XAxis type="number" tick={axisTick} tickLine={false} axisLine={{ stroke: INK.axis }} tickFormatter={secondsTick} />
          <YAxis type="category" dataKey="name" tick={axisTick} tickLine={false} axisLine={false} width={84} />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            content={({ active, payload }) => {
              const row = payload?.[0]?.payload as Record<string, number | string> | undefined;
              if (!active || !row) return null;
              return (
                <div style={tooltipStyle} className="rv-tooltip">
                  <strong>{row.title}</strong>
                  {present.map((stage) => (
                    <div key={stage.key} className="rv-tooltip-row">
                      <span className="rv-swatch" style={{ background: stage.color }} />
                      <span>{stage.label}</span>
                      <span className="rv-tooltip-value">{formatSecs(Number(row[stage.key]))}</span>
                    </div>
                  ))}
                  <div className="text-muted">{String(row.name).split(" · ")[1]} turns · stage medians do not add up</div>
                </div>
              );
            }}
          />
          {present.map((stage) => (
            <Bar
              key={stage.key}
              dataKey={stage.key}
              stackId="stages"
              fill={stage.color}
              stroke="#1a1a1a"
              strokeWidth={1}
              barSize={16}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <StageLegend stages={present.map((stage) => stage.key)} />
    </>
  );
}

/** One row per variant and turn kind: sample count first, then response, voice, and stage medians. */
export function KindTable({
  variants,
  kinds,
  selected,
}: Readonly<{ variants: MetricsVariant[]; kinds: (TurnKind | "all")[]; selected: TurnKind | "all" }>) {
  const shown = selected === "all" ? kinds : [selected];
  const stages = STAGES.filter((stage) =>
    variants.some((v) => shown.some((k) => (v.by_kind[k]?.stages[stage.key]?.n ?? 0) > 0))
  );
  return (
    <div className="rv-table-scroll">
      <table className="rv-table rv-table-static rv-num">
        <thead>
          <tr>
            <th>Variant</th>
            <th>Turn kind</th>
            <th>Turns</th>
            <th>Timed</th>
            <th>Response p50</th>
            <th>p90</th>
            <th>Voice p50</th>
            <th>p90</th>
            {stages.map((stage) => (
              <th key={stage.key} title="median">
                <span className="rv-swatch" style={{ background: stage.color }} /> {stage.label}
              </th>
            ))}
            <th>Barge-in</th>
            <th>Prompt tok p50</th>
            <th>GPU load mean / peak</th>
          </tr>
        </thead>
        <tbody>
          {variants.flatMap((v, i) =>
            shown
              .map((k) => ({ k, stats: v.by_kind[k] }))
              .filter(({ stats }) => stats && stats.turns > 0)
              .map(({ k, stats }) => (
                <tr key={`${v.label}-${k}`}>
                  <td title={v.label}>
                    <span className="rv-swatch" style={{ background: variantColor(v.label) }} /> {short(i)}
                  </td>
                  <td>{KIND_LABELS[k] ?? k}</td>
                  <td>{stats!.turns}</td>
                  <td>{stats!.response.n}</td>
                  <td>
                    <strong>{formatSecs(stats!.response.p50)}</strong>
                  </td>
                  <td>{formatSecs(stats!.response.p90)}</td>
                  <td>{formatSecs(stats!.voice.p50)}</td>
                  <td>{formatSecs(stats!.voice.p90)}</td>
                  {stages.map((stage) => (
                    <td key={stage.key}>{formatSecs(stats!.stages[stage.key]?.p50)}</td>
                  ))}
                  <td>{formatPct(stats!.barge_in_rate)}</td>
                  <td>{stats!.prompt_tokens_p50 === null ? "–" : Math.round(stats!.prompt_tokens_p50)}</td>
                  <td>
                    {stats!.gpu_load_mean === null
                      ? "–"
                      : `${Math.round(stats!.gpu_load_mean)}% / ${Math.round(stats!.gpu_load_peak ?? 0)}%`}
                  </td>
                </tr>
              ))
          )}
        </tbody>
      </table>
    </div>
  );
}

/** Latency and stage percentiles per day or per git revision, flagging clearly slower buckets. */
export function TrendSection({ trend, kind }: Readonly<{ trend: MetricsResponse["trend"]; kind: TurnKind | "all" }>) {
  const [by, setBy] = useState<"by_day" | "by_revision">("by_day");
  const buckets = trend[by].map((bucket) => ({ bucket, stats: bucket.by_kind[kind] })).filter((b) => b.stats);
  const timed = buckets.filter((b) => b.stats!.response.p50 !== null);
  const chart = timed.map(({ bucket, stats }) => {
    const d = stats!.response;
    return {
      name: bucket.label,
      p50: d.p50,
      range: [(d.p50 ?? 0) - (d.p10 ?? d.p50 ?? 0), (d.p90 ?? d.p50 ?? 0) - (d.p50 ?? 0)],
      n: d.n,
      flag: stats!.regression ? "▲ slower" : "",
      dist: d,
    };
  });
  const stages = STAGES.filter((stage) => timed.some((b) => (b.stats!.stages[stage.key]?.n ?? 0) > 0));
  return (
    <>
      <div className="rv-filters rv-section-head">
        <h3 className="text-sm font-semibold">Trend</h3>
        <div className="rv-segmented" role="group" aria-label="Trend by">
          <button className={`tab-btn ${by === "by_day" ? "active" : ""}`} onClick={() => setBy("by_day")}>
            Per day
          </button>
          <button className={`tab-btn ${by === "by_revision" ? "active" : ""}`} onClick={() => setBy("by_revision")}>
            Per git revision
          </button>
        </div>
        <span className="text-xs text-muted">
          {KIND_LABELS[kind]} · “slower” marks a median response more than 20% and 0.1 s above the previous{" "}
          {by === "by_day" ? "day" : "revision"} (5 turns minimum on both)
        </span>
      </div>
      <div className="rv-chart-grid">
        <ChartCard title="Response latency" subtitle="median, whisker p10–p90">
          {chart.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chart} margin={{ top: 20, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="name" tick={axisTick} tickLine={false} axisLine={{ stroke: INK.axis }} />
                <YAxis tick={axisTick} tickLine={false} axisLine={false} width={56} tickFormatter={secondsTick} />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  content={({ active, payload }) => {
                    const row = payload?.[0]?.payload as (typeof chart)[number] | undefined;
                    if (!active || !row) return null;
                    return (
                      <div style={tooltipStyle} className="rv-tooltip">
                        <strong>{row.name}</strong>
                        <div>
                          median {formatSecs(row.dist.p50)} · p10 {formatSecs(row.dist.p10)} · p90{" "}
                          {formatSecs(row.dist.p90)}
                        </div>
                        <div className="text-muted">
                          {row.n} turns{row.flag && " · clearly slower than the previous one"}
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="p50" fill={SERIES[0]} barSize={22} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="range" direction="y" width={6} stroke={INK.secondary} strokeWidth={1.5} />
                  <LabelList dataKey="flag" position="top" fill={INK.primary} fontSize={11} offset={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty what="timed turns of this kind" />
          )}
        </ChartCard>
        <ChartCard title="Stage breakdown" subtitle="median of each stage">
          <StageBars
            rows={timed.map(({ bucket, stats }) => ({ name: bucket.label, title: bucket.label, stats }))}
            what="timed turns of this kind"
          />
        </ChartCard>
      </div>
      <div className="rv-table-scroll">
        <table className="rv-table rv-table-static rv-num">
          <thead>
            <tr>
              <th>{by === "by_day" ? "Day" : "Git revision"}</th>
              <th>Sessions</th>
              <th>Turns</th>
              <th>Timed</th>
              <th>Response p50</th>
              <th>p90</th>
              <th>Voice p50</th>
              <th>p90</th>
              {stages.map((stage) => (
                <th key={stage.key} title="median / p90">
                  {stage.label} p50 / p90
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {buckets.map(({ bucket, stats }) => (
              <tr key={bucket.key || "unknown"}>
                <td title={bucket.key}>
                  {bucket.label}{" "}
                  {stats!.regression && (
                    <span
                      className="rv-badge rv-badge-warn"
                      title={`previous (${stats!.regression.previous}): ${formatSecs(stats!.regression.previous_p50)}`}
                    >
                      ▲ slower
                    </span>
                  )}
                </td>
                <td>{bucket.sessions}</td>
                <td>{stats!.turns}</td>
                <td>{stats!.response.n}</td>
                <td>
                  <strong>{formatSecs(stats!.response.p50)}</strong>
                </td>
                <td>{formatSecs(stats!.response.p90)}</td>
                <td>{formatSecs(stats!.voice.p50)}</td>
                <td>{formatSecs(stats!.voice.p90)}</td>
                {stages.map((stage) => (
                  <td key={stage.key}>
                    {formatSecs(stats!.stages[stage.key]?.p50)} / {formatSecs(stats!.stages[stage.key]?.p90)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function ToolsTable({ tools }: Readonly<{ tools: ToolStats[] }>) {
  if (!tools.length) return <p className="text-sm text-muted">No tool or intent calls recorded in this range.</p>;
  return (
    <div className="rv-table-scroll">
      <table className="rv-table rv-table-static rv-num">
        <thead>
          <tr>
            <th>Tool or intent</th>
            <th>Triggered by</th>
            <th>Target</th>
            <th>Perceivable</th>
            <th>Calls</th>
            <th>Duration p50</th>
            <th>p90</th>
            <th>Failed</th>
            <th>Errors</th>
            <th>Timeouts</th>
            <th>Cancelled</th>
          </tr>
        </thead>
        <tbody>
          {tools.map((tool) => (
            <tr key={`${tool.name}-${tool.trigger}-${tool.target}`}>
              <td className="rv-mono">{tool.name}</td>
              <td>{tool.trigger === "intent" ? "intent engine" : "LLM"}</td>
              <td>{tool.target === "home_assistant" ? "Home Assistant" : "client"}</td>
              <td>{tool.perceivable ? "yes" : "–"}</td>
              <td>{tool.calls}</td>
              <td>{formatSecs(tool.duration.p50)}</td>
              <td>{formatSecs(tool.duration.p90)}</td>
              <td>
                {tool.failure_rate > 0 ? (
                  <span className="rv-badge rv-badge-warn">{formatPct(tool.failure_rate)}</span>
                ) : (
                  formatPct(0)
                )}
              </td>
              <td>{formatPct(tool.error_rate)}</td>
              <td>{formatPct(tool.timeout_rate)}</td>
              <td>{formatPct(tool.cancelled_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
