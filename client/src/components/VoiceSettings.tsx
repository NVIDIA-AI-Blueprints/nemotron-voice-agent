// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useState, useMemo, useCallback, useEffect } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useVoiceCatalog } from "../api";
import { useConnectionState } from "../hooks/useConnectionState";
import { useApp } from "../context/useApp";
import { DEFAULT_SESSION_LANGUAGE } from "../context/AppContext";
import { PanelSection } from "./PanelSection";

type LanguageSwitchedMessage = {
  type?: string;
  language?: string;
  voice_id?: string;
};

type VoiceOverrideState = {
  serviceId: string;
  language: string;
  voiceId: string;
};

export function VoiceSettings() {
  const client = usePipecatClient();
  const { isConnected, isConnecting, isLocked } = useConnectionState();
  const {
    selectedExample,
    selectedASR,
    selectedTTS,
    selectedSessionLanguage,
    setSelectedSessionLanguage,
    setSelectedVoiceId,
  } = useApp();

  const sessionLanguagesEnabled = selectedExample?.capabilities?.includes("session_languages") ?? false;
  const selectedTTSServer = selectedTTS?.server;

  const { data: ttsConfig, isLoading } = useVoiceCatalog(
    selectedTTSServer,
    selectedTTS?.voiceId,
    sessionLanguagesEnabled ? selectedASR?.server : undefined,
    sessionLanguagesEnabled ? selectedASR?.model : undefined,
    sessionLanguagesEnabled ? selectedASR?.functionId : undefined,
  );

  const [voiceOverride, setVoiceOverride] = useState<VoiceOverrideState>({
    serviceId: "",
    language: "",
    voiceId: "",
  });

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((message: LanguageSwitchedMessage) => {
      if (sessionLanguagesEnabled) return;
      if (message?.type === "language-switched") {
        const lang = message.language ?? "";
        const voiceId = message.voice_id ?? "";
        if (lang || voiceId) {
          setVoiceOverride({
            serviceId: selectedTTS?.id ?? "",
            language: lang,
            voiceId,
          });
        }
      }
    }, [sessionLanguagesEnabled, selectedTTS?.id]),
  );

  useEffect(() => {
    if (!sessionLanguagesEnabled) return;
    if (!ttsConfig) return;
    const languages = ttsConfig?.languages ?? [];
    if (languages.length === 0) return;
    if (!selectedSessionLanguage || !languages.includes(selectedSessionLanguage)) {
      setSelectedSessionLanguage(languages.includes(DEFAULT_SESSION_LANGUAGE) ? DEFAULT_SESSION_LANGUAGE : languages[0]);
    }
  }, [sessionLanguagesEnabled, selectedSessionLanguage, ttsConfig, setSelectedSessionLanguage]);

  const defaultVoice = useMemo(() => {
    if (!ttsConfig?.voices?.length) return null;
    const selectedServiceVoice = selectedTTS?.voiceId || "";
    if (selectedServiceVoice) {
      const match = ttsConfig.voices.find((voice) => voice.id === selectedServiceVoice);
      if (match) return match;
    }
    if (ttsConfig.defaultVoiceId) {
      const match = ttsConfig.voices.find((voice) => voice.id === ttsConfig.defaultVoiceId);
      if (match) return match;
    }
    const enVoice = ttsConfig.voices.find((voice) => voice.language.toUpperCase() === "EN-US");
    return enVoice || ttsConfig.voices[0];
  }, [ttsConfig, selectedTTS?.voiceId]);

  const hasActiveOverride = voiceOverride.serviceId === (selectedTTS?.id ?? "");
  const activeLang = (hasActiveOverride ? voiceOverride.language : "") || (defaultVoice?.language.replace("_", "-") ?? "");
  const activeVoice = (hasActiveOverride ? voiceOverride.voiceId : "") || defaultVoice?.id || "";

  const languages = ttsConfig?.languages ?? [];

  const filteredVoices = useMemo(() => {
    if (!ttsConfig || !activeLang) return [];
    const langUpper = activeLang.toUpperCase();
    return ttsConfig.voices.filter((v) => v.language.toUpperCase() === langUpper);
  }, [ttsConfig, activeLang]);

  const handleLangChange = (lang: string) => {
    setVoiceOverride({
      serviceId: selectedTTS?.id ?? "",
      language: lang,
      voiceId: "",
    });
    setSelectedVoiceId("");
  };

  const handleVoiceChange = (voiceId: string) => {
    setVoiceOverride((current) => ({
      serviceId: selectedTTS?.id ?? "",
      language: current.serviceId === (selectedTTS?.id ?? "") ? current.language : activeLang,
      voiceId,
    }));
    setSelectedVoiceId(voiceId);
    if (isConnected && client?.state === "ready" && voiceId) {
      try {
        client.sendClientMessage("set-voice", {
          voice_id: voiceId,
          language: activeLang,
        });
      } catch (err) {
        console.warn("Could not send voice update:", err);
      }
    }
  };

  if (isLoading) {
    return <PanelSection label="VOICE SETTINGS" loading loadingText="Loading..." />;
  }

  if (sessionLanguagesEnabled) {
    const availableLanguages = ttsConfig?.languages ?? [];
    if (availableLanguages.length === 0) {
      return (
        <PanelSection label="VOICE SETTINGS">
          <p style={{ fontSize: "11px", color: "var(--text-muted)" }}>
            No shared ASR/TTS languages found for the selected services
          </p>
        </PanelSection>
      );
    }

    const languageSelectTitle = isConnected
      ? "Disconnect, change language, then Connect again to apply"
      : "Fixes ASR, TTS voice, and LLM language for the next connection";

    return (
      <PanelSection label="VOICE SETTINGS">
        <div className="settings-row">
          <span className="settings-label">Language</span>
          <select
            className="select-dark"
            value={selectedSessionLanguage || DEFAULT_SESSION_LANGUAGE}
            onChange={(e) => setSelectedSessionLanguage(e.target.value)}
            disabled={isConnecting || isLocked}
            title={languageSelectTitle}
          >
            {availableLanguages.map((lang) => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>
        </div>
      </PanelSection>
    );
  }

  if (!ttsConfig || ttsConfig.voices.length === 0) {
    return (
      <PanelSection label="VOICE SETTINGS">
        <p style={{ fontSize: "11px", color: "var(--text-muted)" }}>No voices available for this TTS server</p>
      </PanelSection>
    );
  }

  return (
    <PanelSection label="VOICE SETTINGS">
      <div className="settings-row">
        <span className="settings-label">Language</span>
        <select
          className="select-dark"
          value={activeLang}
          onChange={(e) => handleLangChange(e.target.value)}
          disabled
          title="Language selection is not yet supported"
        >
          {languages.map((lang) => (
            <option key={lang} value={lang}>{lang}</option>
          ))}
        </select>
      </div>

      <div className="settings-row">
        <span className="settings-label">Voice</span>
        <select
          className="select-dark"
          value={activeVoice}
          onChange={(e) => handleVoiceChange(e.target.value)}
        >
          <option value="">Select voice</option>
          {filteredVoices.map((v) => (
            <option key={v.id} value={v.id}>{v.name}</option>
          ))}
        </select>
      </div>
    </PanelSection>
  );
}
