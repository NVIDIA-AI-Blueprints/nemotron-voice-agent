// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";
import { artifactUrl, type MediaRef } from "./api";
import { formatPct, wordDiff } from "./utils";

export function DiffText({ reference, hypothesis }: Readonly<{ reference: string; hypothesis: string }>) {
  return (
    <span className="rv-diff">
      {wordDiff(reference, hypothesis).map((token, i) => (
        <span key={`${i}-${token.text}`} className={`rv-diff-${token.kind}`}>
          {token.text}
          {token.sep}
        </span>
      ))}
    </span>
  );
}

export function AudioClips({ clips, label }: Readonly<{ clips: { key: string }[]; label?: string }>) {
  if (!clips.length) return null;
  return (
    <div className="rv-audio">
      {label && <span className="rv-label">{label}</span>}
      {clips.map((clip) => (
        <audio key={clip.key} controls preload="none" src={artifactUrl(clip.key)} />
      ))}
    </div>
  );
}

export function WerBadge({ value, title }: Readonly<{ value: number | null | undefined; title?: string }>) {
  let tone = "rv-badge-muted";
  if (value !== null && value !== undefined) {
    if (value === 0) tone = "rv-badge-good";
    else tone = value < 0.15 ? "rv-badge-warn" : "rv-badge-bad";
  }
  return (
    <span className={`rv-badge ${tone}`} title={title}>
      WER {formatPct(value)}
    </span>
  );
}

export function ImageStrip({ images }: Readonly<{ images: MediaRef[] }>) {
  if (!images.length) return null;
  return (
    <div className="rv-images">
      {images.map((image) => (
        <a key={image.id} href={artifactUrl(image.key)} target="_blank" rel="noreferrer" title={image.source ?? ""}>
          <img src={artifactUrl(image.key)} alt={image.source ?? "image"} loading="lazy" />
        </a>
      ))}
    </div>
  );
}

export function StatusLine({ loading, error }: Readonly<{ loading: boolean; error: unknown }>) {
  if (loading) return <p className="text-sm text-muted">Loading…</p>;
  if (error) {
    return (
      <p className="text-sm rv-error">
        {error instanceof Error ? error.message : String(error)}
        {String(error).includes("404") && " — is MONITORING_ENABLED=true on the server?"}
      </p>
    );
  }
  return null;
}

/** Transcripts of a turn per source, diffed against the reference text when there is one. */
export function TranscriptTable({
  transcripts,
  referenceText,
  referenceSource,
}: Readonly<{ transcripts: Record<string, string>; referenceText?: string | null; referenceSource?: string | null }>) {
  const entries = Object.entries(transcripts).filter(([source]) => !source.startsWith("human:"));
  return (
    <table className="rv-transcripts">
      <tbody>
        {entries.map(([source, text]) => (
          <tr key={source}>
            <th>{source}</th>
            <td>
              {referenceText && source !== referenceSource ? (
                <DiffText reference={referenceText} hypothesis={text} />
              ) : (
                text || <span className="text-muted">(empty)</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ChartCard({
  title,
  subtitle,
  children,
}: Readonly<{ title: string; subtitle?: string; children: ReactNode }>) {
  return (
    <figure className="card rv-chart">
      <figcaption>
        <span className="text-sm font-semibold">{title}</span>
        {subtitle && <span className="text-xs text-muted"> {subtitle}</span>}
      </figcaption>
      {children}
    </figure>
  );
}

export function Empty({ what }: Readonly<{ what: string }>) {
  return <p className="text-xs text-muted rv-chart-empty">No {what} recorded in this range.</p>;
}
