// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useState, useRef, useEffect } from "react";
import {
  usePipecatClientMicControl,
  usePipecatClientMediaDevices,
  VoiceVisualizer
} from "@pipecat-ai/client-react";

export function DevicesSection() {
  const { enableMic, isMicEnabled } = usePipecatClientMicControl();
  const { availableMics, selectedMic, updateMic } = usePipecatClientMediaDevices();
  const [showDeviceMenu, setShowDeviceMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

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
    enableMic(!isMicEnabled);
  };

  const handleMicSelect = (deviceId: string) => {
    updateMic(deviceId);
    setShowDeviceMenu(false);
  };

  return (
    <div className="header-devices" ref={menuRef}>
      <div className="device-row">
        <button
          className={`device-item device-button ${!isMicEnabled ? 'device-disabled' : ''}`}
          onClick={handleMicToggle}
          title={isMicEnabled ? "Click to mute" : "Click to unmute"}
        >
          <span className="device-icon">🎤</span>
          <div className="device-visualizer">
            <VoiceVisualizer
              participantType="local"
              backgroundColor="transparent"
              barColor={isMicEnabled ? "#76b900" : "#666666"}
              barCount={16}
              barGap={3}
              barWidth={4}
              barMaxHeight={28}
              barLineCap="round"
            />
          </div>
        </button>
        <button
          className="device-dropdown-btn"
          onClick={() => setShowDeviceMenu(!showDeviceMenu)}
          title="Select microphone"
        >
          <span className={`chevron ${showDeviceMenu ? 'open' : ''}`}>▼</span>
        </button>
      </div>

      {showDeviceMenu && (
        <div className="device-menu header-device-menu">
          <p className="device-menu-label">🎤 Microphones</p>
          {availableMics.map((mic) => (
            <button
              key={mic.deviceId}
              className={`device-menu-item ${selectedMic?.deviceId === mic.deviceId ? 'selected' : ''}`}
              onClick={() => handleMicSelect(mic.deviceId)}
            >
              {selectedMic?.deviceId === mic.deviceId && <span className="checkmark">✓</span>}
              {mic.label || 'Unknown Microphone'}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
