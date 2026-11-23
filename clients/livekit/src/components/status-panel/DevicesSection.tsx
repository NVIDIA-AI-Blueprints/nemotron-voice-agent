// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { useLocalParticipant, useMediaDeviceSelect } from "@livekit/components-react";
import { Track } from "livekit-client";
import { VoiceLevelVisualizer } from "../VoiceLevelVisualizer";

export function DevicesSection() {
  const { localParticipant, isMicrophoneEnabled, microphoneTrack } = useLocalParticipant();
  const { devices, activeDeviceId, setActiveMediaDevice } = useMediaDeviceSelect({ kind: "audioinput" });
  const [showDeviceMenu, setShowDeviceMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const micTrack = { participant: localParticipant, source: Track.Source.Microphone, publication: microphoneTrack };

  useEffect(() => {
    if (!showDeviceMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowDeviceMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showDeviceMenu]);

  const handleMicToggle = () => {
    localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled).catch((err: unknown) => {
      console.warn("Could not change the microphone:", err);
    });
  };

  const handleMicSelect = (deviceId: string) => {
    setActiveMediaDevice(deviceId).catch((err: unknown) => console.warn("Could not switch the microphone:", err));
    setShowDeviceMenu(false);
  };

  return (
    <div className="header-devices" ref={menuRef}>
      <div className="device-row">
        <button
          className={`device-item device-button ${!isMicrophoneEnabled ? 'device-disabled' : ''}`}
          onClick={handleMicToggle}
          aria-label={isMicrophoneEnabled ? "Mute microphone" : "Unmute microphone"}
          title={isMicrophoneEnabled ? "Click to mute" : "Click to unmute"}
        >
          <span className="device-icon" aria-hidden="true">🎤</span>
          <div className="device-visualizer">
            <VoiceLevelVisualizer
              track={micTrack}
              ariaLabel="Microphone level"
              barCount={16}
              barGap={3}
              barWidth={4}
              barMaxHeight={28}
            />
          </div>
        </button>
        <button
          className="device-dropdown-btn"
          onClick={() => setShowDeviceMenu(!showDeviceMenu)}
          aria-label="Select microphone"
          aria-expanded={showDeviceMenu}
          title="Select microphone"
        >
          <span className={`chevron ${showDeviceMenu ? 'open' : ''}`} aria-hidden="true">▼</span>
        </button>
      </div>

      {showDeviceMenu && (
        <div className="device-menu header-device-menu">
          <p className="device-menu-label"><span aria-hidden="true">🎤</span> Microphones</p>
          {devices.map((mic) => (
            <button
              key={mic.deviceId}
              className={`device-menu-item ${activeDeviceId === mic.deviceId ? 'selected' : ''}`}
              onClick={() => handleMicSelect(mic.deviceId)}
              aria-pressed={activeDeviceId === mic.deviceId}
            >
              {activeDeviceId === mic.deviceId && <span className="checkmark">✓</span>}
              {mic.label || 'Unknown Microphone'}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
