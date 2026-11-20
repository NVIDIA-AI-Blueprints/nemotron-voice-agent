// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMultibandTrackVolume, type TrackReferenceOrPlaceholder } from "@livekit/components-react";

type VoiceLevelVisualizerProps = {
  track: TrackReferenceOrPlaceholder | undefined;
  ariaLabel: string;
  barCount?: number;
  barGap?: number;
  barMaxHeight?: number;
  barWidth?: number;
};

const MIN_BAR_HEIGHT = 4;

/** Level bars for an audio track, drawn like the Pipecat client's visualizer. The theme sets the bar color. */
export function VoiceLevelVisualizer({
  track,
  ariaLabel,
  barCount = 20,
  barGap = 4,
  barMaxHeight = 44,
  barWidth = 8,
}: Readonly<VoiceLevelVisualizerProps>) {
  const levels = useMultibandTrackVolume(track, { bands: barCount });

  return (
    <div className="audio-level-visualizer" role="img" aria-label={ariaLabel} style={{ gap: barGap, minHeight: barMaxHeight }}>
      {Array.from({ length: barCount }, (_, index) => (
        <span
          key={index}
          className="audio-level-bar"
          style={{
            height: Math.max(MIN_BAR_HEIGHT, Math.round(barMaxHeight * Math.min(1, levels[index] ?? 0))),
            width: barWidth,
          }}
        />
      ))}
    </div>
  );
}
