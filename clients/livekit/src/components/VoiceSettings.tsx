// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useVoices } from "../api";
import { useApp } from "../context/useApp";
import { effectiveVoiceId } from "../lib/session";
import { useSession } from "../session/session-context";
import { PanelSection } from "./PanelSection";

function SettingsRow({ label, children }: Readonly<{ label: string; children: React.ReactNode }>) {
  return (
    <div className="settings-row">
      <span className="settings-label">{label}</span>
      {children}
    </div>
  );
}

/**
 * The voice of the session. Before connecting, the choice is sent with the join token. During a session, a change is
 * sent to the agent, which uses the new voice from its next reply.
 */
export function VoiceSettings() {
  const { isConnected, isConnecting, voices: sessionVoices, setVoice } = useSession();
  const { services, selectedVoiceId, setSelectedVoiceId } = useApp();
  const selection = services.tts.selection;
  const builtIn = selection?.kind === "builtin" ? selection.entry : undefined;
  const { data: offered, isLoading } = useVoices(builtIn?.id ?? "");

  if (selection?.kind === "custom") {
    return (
      <PanelSection label="VOICE SETTINGS">
        <SettingsRow label="Voice">
          <span className="status-value">{selection.service.fields.voice_id || "Server default"}</span>
        </SettingsRow>
      </PanelSection>
    );
  }

  if (isLoading) {
    return <PanelSection label="VOICE SETTINGS" loading />;
  }

  const list = isConnected && sessionVoices ? sessionVoices : offered;
  const voiceId = effectiveVoiceId(selectedVoiceId, list);

  if (!list || list.voices.length === 0) {
    return (
      <PanelSection label="VOICE SETTINGS">
        <p className="settings-hint">No voices available for this TTS server</p>
      </PanelSection>
    );
  }

  const handleVoiceChange = (id: string) => {
    setSelectedVoiceId(id);
    if (isConnected) setVoice(id);
  };

  return (
    <PanelSection label="VOICE SETTINGS">
      <SettingsRow label="Voice">
        <select
          className="select-dark"
          aria-label="Voice"
          value={voiceId}
          onChange={(e) => handleVoiceChange(e.target.value)}
          disabled={isConnecting || list.voices.length < 2}
          title={list.voices.length < 2 ? "The speech server offers one voice" : ""}
        >
          {list.voices.map((v) => (
            <option key={v.id} value={v.id}>{v.name}</option>
          ))}
        </select>
      </SettingsRow>
    </PanelSection>
  );
}
