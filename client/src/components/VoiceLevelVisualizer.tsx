// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useMemo, useState } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";

type ParticipantType = "local" | "bot";

type VoiceLevelVisualizerProps = {
  participantType: ParticipantType;
  ariaLabel: string;
  backgroundColor?: string;
  barColor?: string;
  barCount?: number;
  barGap?: number;
  barLineCap?: "round" | "square";
  barMaxHeight?: number;
  barWidth?: number;
};

const REMOTE_AUDIO_LEVEL_GAIN = 6;
const REMOTE_AUDIO_LEVEL_CURVE = 0.65;

function normalizeAudioLevel(level: number) {
  if (!Number.isFinite(level) || level <= 0) return 0;
  return Math.min(1, level > 1 ? level / 255 : level);
}

function normalizeRemoteAudioLevel(level: number) {
  const normalized = normalizeAudioLevel(level);
  if (!normalized) return 0;
  return Math.min(1, (normalized * REMOTE_AUDIO_LEVEL_GAIN) ** REMOTE_AUDIO_LEVEL_CURVE);
}

function AudioLevelBars({
  ariaLabel,
  backgroundColor = "#0a0a0a",
  barColor = "#76b900",
  barCount = 20,
  barGap = 4,
  barLineCap = "round",
  barMaxHeight = 44,
  barWidth = 8,
  level,
}: Readonly<VoiceLevelVisualizerProps & { level: number }>) {
  const bars = useMemo(
    () =>
      Array.from({ length: barCount }, (_, index) => {
        const phase = (index + 1) / barCount;
        const shapedLevel = level * (0.35 + Math.abs(Math.sin(phase * Math.PI * 2.5)) * 0.65);
        return Math.max(4, Math.round(barMaxHeight * shapedLevel));
      }),
    [barCount, barMaxHeight, level]
  );

  return (
    <div
      className="audio-level-visualizer"
      aria-label={ariaLabel}
      style={{ backgroundColor, gap: barGap, minHeight: barMaxHeight }}
    >
      {bars.map((height, index) => (
        <span
          key={index}
          className="audio-level-bar"
          style={{
            backgroundColor: barColor,
            borderRadius: barLineCap === "round" ? 999 : 0,
            height,
            width: barWidth,
          }}
        />
      ))}
    </div>
  );
}

export function VoiceLevelVisualizer({
  participantType,
  ariaLabel,
  backgroundColor = "#0a0a0a",
  barColor = "#76b900",
  barCount = 20,
  barGap = 4,
  barLineCap = "round",
  barMaxHeight = 44,
  barWidth = 8,
}: Readonly<VoiceLevelVisualizerProps>) {
  const [level, setLevel] = useState(0);

  useRTVIClientEvent(
    RTVIEvent.LocalAudioLevel,
    useCallback((rawLevel: number) => {
      if (participantType === "local") setLevel(normalizeAudioLevel(rawLevel));
    }, [participantType])
  );

  useRTVIClientEvent(
    RTVIEvent.RemoteAudioLevel,
    useCallback((rawLevel: number) => {
      if (participantType === "bot") setLevel(normalizeRemoteAudioLevel(rawLevel));
    }, [participantType])
  );

  useRTVIClientEvent(
    RTVIEvent.BotStoppedSpeaking,
    useCallback(() => {
      if (participantType === "bot") setLevel(0);
    }, [participantType])
  );

  useRTVIClientEvent(
    RTVIEvent.UserStoppedSpeaking,
    useCallback(() => {
      if (participantType === "local") setLevel(0);
    }, [participantType])
  );

  useRTVIClientEvent(
    RTVIEvent.Disconnected,
    useCallback(() => {
      setLevel(0);
    }, [])
  );

  return (
    <AudioLevelBars
      participantType={participantType}
      ariaLabel={ariaLabel}
      backgroundColor={backgroundColor}
      barColor={barColor}
      barCount={barCount}
      barGap={barGap}
      barLineCap={barLineCap}
      barMaxHeight={barMaxHeight}
      barWidth={barWidth}
      level={level}
    />
  );
}
