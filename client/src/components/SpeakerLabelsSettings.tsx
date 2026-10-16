// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useApp } from "../context/useApp";
import { useConnectionState } from "../hooks/useConnectionState";
import { PanelSection } from "./PanelSection";

export function SpeakerLabelsSettings() {
  const { isLocked } = useConnectionState();
  const {
    selectedExample,
    selectedASR,
    speakerDiarizationEnabled,
    setSpeakerDiarizationEnabled,
  } = useApp();

  const exampleSupportsSpeakerLabels = selectedExample?.capabilities?.includes("speaker_labels") ?? false;
  const asrSupportsDiarization = selectedASR?.speakerDiarizationSupported === true;
  const maxSpeakers = selectedASR?.speakerDiarizationMaxSpeakers;

  if (!exampleSupportsSpeakerLabels || !asrSupportsDiarization) return null;

  const lockedTitle = "Disconnect, change the setting, then Connect again to apply";
  const toggleTitle = isLocked
    ? lockedTitle
    : "Tags user speech with session-scoped labels from the selected ASR service";

  return (
    <PanelSection label="SPEAKER LABELS">
      <div className="settings-row">
        <span className="settings-label">Diarization</span>
        <button
          type="button"
          className={`nvidia-checkbox${speakerDiarizationEnabled ? " is-checked" : ""}`}
          role="switch"
          aria-checked={speakerDiarizationEnabled}
          onClick={() => setSpeakerDiarizationEnabled(!speakerDiarizationEnabled)}
          disabled={isLocked}
          title={toggleTitle}
        >
          <span className="nvidia-checkbox__box" aria-hidden="true">
            {speakerDiarizationEnabled ? "✓" : ""}
          </span>
          <span>{speakerDiarizationEnabled ? "On" : "Off"}</span>
        </button>
      </div>
      {speakerDiarizationEnabled && maxSpeakers !== undefined && (
        <div className="settings-row">
          <span className="text-secondary">Max speakers: {maxSpeakers}</span>
        </div>
      )}
    </PanelSection>
  );
}
