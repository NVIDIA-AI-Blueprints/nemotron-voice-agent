// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared chart constants of the review Metrics and Session views.

// Dark-mode categorical steps of the dataviz reference palette, validated on the
// card surface #1a1a1a (all checks pass). Slot order is the CVD-safety mechanism.
export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const OVERFLOW = "#898781"; // past 8 variants: muted, never a generated hue
export const INK = { primary: "#ffffff", secondary: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835" };

// Color follows the variant, not its rank: a label keeps its slot for the whole
// page lifetime even when a filter removes or reorders variants.
const slotByLabel = new Map<string, number>();
export function variantColor(label: string): string {
  if (!slotByLabel.has(label)) slotByLabel.set(label, slotByLabel.size);
  const slot = slotByLabel.get(label) ?? SERIES.length;
  return SERIES[slot] ?? OVERFLOW;
}

export const tooltipStyle = {
  background: "#252525",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 6,
  color: INK.primary,
  fontSize: 12,
};
export const axisTick = { fill: INK.muted, fontSize: 11 };

/** Seconds axis ticks: milliseconds below 1 s so sub-second ticks don't round together. */
export function secondsTick(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 1 ? `${Math.round(value * 1000)} ms` : `${Number(value.toFixed(2))} s`;
}

export function short(index: number): string {
  return `V${index + 1}`;
}


// ---- Latency stages (src/monitoring/turn_metrics.py), in the order they happen ----
// Color follows the stage everywhere (variant charts, trend, session timeline). The
// remainder is neutral gray: it is the absence of an explanation, not a stage.
export const STAGES: { key: string; label: string; color: string }[] = [
  { key: "asr", label: "ASR", color: SERIES[0] },
  { key: "turn_detection", label: "Turn detection", color: SERIES[1] },
  { key: "intent_match", label: "Intent match", color: SERIES[2] },
  { key: "llm_first", label: "First LLM call", color: SERIES[3] },
  { key: "tool", label: "Tool round trip", color: SERIES[4] },
  { key: "llm_later", label: "Later LLM calls", color: SERIES[5] },
  { key: "text_aggregation", label: "Sentence aggregation", color: SERIES[6] },
  { key: "tts", label: "TTS", color: SERIES[7] },
  { key: "unexplained", label: "Unexplained", color: OVERFLOW },
];

export function stageInfo(key: string): { key: string; label: string; color: string } {
  return STAGES.find((stage) => stage.key === key) ?? { key, label: key, color: OVERFLOW };
}

/** Label of a power sample: it follows the source the value came from. */
export function powerLabel(source: string | null | undefined): string {
  if (source === "board") return "Board input power";
  if (source === "gpu") return "GPU power";
  return "Power";
}

export const KIND_LABELS: Record<string, string> = {
  all: "All turns",
  plain: "Plain",
  tool: "Tool",
  intent: "Intent-handled",
  vision: "Vision",
};
