// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { Fragment, forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { LabeledSelect } from "./components/ui/labeled-select";
import type { VoicesMap } from "./types";

interface UploadedPrompt {
  id: string;
  name: string;
  file: File;
}

export interface VoiceSelectorRef {
  setVoiceFromBackend: (language: string, voice: string) => void;
}

interface VoiceSelectorProps {
  voices: VoicesMap;
  onVoiceChange: (language: string, voice: string) => void;
  isZeroshotModel?: boolean;
  initialVoiceId?: string;  // Preferred voice id to select on init (if available)
  activeCustomPromptId?: string;  // ID of active custom prompt (empty = none selected)
  customPromptName?: string;
  uploadedPrompts?: UploadedPrompt[];
  onFileUpload?: (file: File) => void;
  onSelectPrompt?: (promptId: string) => void;
  isConfigureMode?: boolean;  // True when in configure mode (before start)
}

export const VoiceSelector = forwardRef<VoiceSelectorRef, VoiceSelectorProps>(function VoiceSelector({
  voices,
  onVoiceChange,
  isZeroshotModel = false,
  initialVoiceId,
  activeCustomPromptId = "",
  customPromptName = "",
  uploadedPrompts = [],
  onFileUpload,
  onSelectPrompt,
  isConfigureMode = false
}, ref) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isInitializedRef = useRef<boolean>(false);  // Track if first initialization is done
  const onVoiceChangeRef = useRef(onVoiceChange);  // Stable ref to avoid effect re-triggers

  // Track the last value set by backend - used to prevent sending it back
  const lastBackendSetRef = useRef<{ language: string; voice: string } | null>(null);

  // Keep callback ref updated
  useEffect(() => {
    onVoiceChangeRef.current = onVoiceChange;
  }, [onVoiceChange]);

  // Helper to find matching language key (case-insensitive)
  const findLanguageKey = (langCode: string): string => {
    // First try exact match
    if (voices[langCode]) return langCode;
    // Try case-insensitive match
    const normalizedInput = langCode.toLowerCase();
    const matchingKey = Object.keys(voices).find(
      key => key.toLowerCase() === normalizedInput
    );
    return matchingKey || langCode;
  };

  // Expose function to parent for backend-triggered updates
  useImperativeHandle(ref, () => ({
    setVoiceFromBackend: (language: string, voice: string) => {
      // Normalize language code to match voices object keys
      const normalizedLang = findLanguageKey(language);
      // Store what backend set - this prevents sending it back in the effect
      lastBackendSetRef.current = { language: normalizedLang, voice };
      setSelectedLanguage(normalizedLang);
      setSelectedVoice([normalizedLang, voice]);
    }
  }));

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onFileUpload) {
      // Validate file type
      if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
      }
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
      }
      onFileUpload(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const [selectedLanguage, setSelectedLanguage] = useState<string>(Object.keys(voices)[0] || "");
  const [selectedVoice, setSelectedVoice] = useState<[string, string]>([selectedLanguage, voices[selectedLanguage]?.voices[0] || ""]);

  // Initialize when voices arrive; prefer initialVoiceId if provided and available
  useEffect(() => {
    if (isInitializedRef.current) return;

    const entries = Object.entries(voices);
    if (entries.length === 0) return;

    // Try to find the language containing initialVoiceId
    let initLang = "";
    let initVoice = "";
    if (initialVoiceId) {
      for (const [lang, data] of entries) {
        const match = data?.voices?.find((v: string) => v === initialVoiceId);
        if (match) {
          initLang = lang;
          initVoice = match;
          break;
        }
      }
    }

    // Fallback to first available
    if (!initLang) {
      initLang = entries[0][0];
      initVoice = entries[0][1]?.voices?.[0] || "";
    }

    setSelectedLanguage(initLang);
    setSelectedVoice([initLang, initVoice]);

    if (initLang && initVoice) {
      isInitializedRef.current = true;
    }
  }, [voices, initialVoiceId]);

  // When language changes, update voice to first available for that language
  // Skip if backend already set a valid voice for this language
  useEffect(() => {
    if (!isInitializedRef.current) return;

    // If backend set a voice, don't override it
    if (lastBackendSetRef.current && lastBackendSetRef.current.language === selectedLanguage) {
      return;
    }

    const voicesForLang = voices[selectedLanguage]?.voices || [];
    if (voicesForLang.length > 0 && !voicesForLang.includes(selectedVoice[1])) {
      setSelectedVoice([selectedLanguage, voicesForLang[0]]);
    }
  }, [selectedLanguage, voices]);

  // Notify parent of voice changes - skip initial load and backend syncs
  useEffect(() => {
    // Skip if not yet initialized (prevents firing on initial mount)
    if (!isInitializedRef.current) return;

    // Check if this matches what backend just set - if so, don't send it back
    if (lastBackendSetRef.current &&
        lastBackendSetRef.current.language === selectedLanguage &&
        lastBackendSetRef.current.voice === selectedVoice[1]) {
      // This change came from backend, clear the ref and skip sending
      lastBackendSetRef.current = null;
      return;
    }

    // User-initiated change - send to backend
    if (selectedLanguage && selectedVoice[1]) {
      onVoiceChangeRef.current(selectedLanguage, selectedVoice[1]);
    }
  }, [selectedLanguage, selectedVoice]);  // Using ref for callback - no onVoiceChange in deps

  // Check if we have any custom prompts available
  const hasBackendPrompt = customPromptName !== "";
  const hasAnyCustomPrompts = isZeroshotModel && (hasBackendPrompt || uploadedPrompts.length > 0);
  const hasActiveCustomPrompt = activeCustomPromptId !== "";
  const hasActiveDefaultVoice = selectedVoice[1] !== "" && !hasActiveCustomPrompt;
  const languageCodes = useMemo(() => Object.keys(voices), [voices]);

  return (
    <div className="mt-4">
      {/* Custom Voice Prompts Section */}
      {isZeroshotModel && (
        <div className="mb-3">
          <div className="text-sm text-gray-600 mb-2">
            Custom Voices:
          </div>

          {/* File Upload Button - Only in Configure Mode */}
          {onFileUpload && isConfigureMode && (
            <div className="mb-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-nvidia text-white px-4 py-2 rounded-lg text-sm hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
              >
                <span>📁</span>
                Upload Custom Voice
              </button>
            </div>
          )}

          <select
            className={`w-full border rounded p-2 text-sm transition-opacity ${hasActiveDefaultVoice ? 'opacity-50' : 'opacity-100'}`}
            value={activeCustomPromptId}
            onChange={(e) => {
              if (e.target.value && onSelectPrompt) {
                onSelectPrompt(e.target.value);
              }
            }}
            disabled={!hasAnyCustomPrompts}
          >
            <option value="">
              {hasAnyCustomPrompts ? "Select a custom voice" : "No custom voices available"}
            </option>
            {hasBackendPrompt && (
              <option value="backend">
                {customPromptName} (Backend)
              </option>
            )}
            {uploadedPrompts.map((prompt) => (
              <option key={prompt.id} value={prompt.id}>
                {prompt.name} (Uploaded)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Default Voices Section */}
      {selectedLanguage === "" ? (
        <div className="text-sm text-gray-500">No voices available</div>
      ) : (
        <Fragment>
          <LabeledSelect
            label="Language:"
            options={languageCodes}
            value={selectedLanguage}
            onChange={(value) => {
              setSelectedLanguage(value);
            }}
            placeholder="Select a language"
            dimmed={hasActiveCustomPrompt}
            disabled={true}
          />
          <LabeledSelect
            label="Default Voices:"
            options={voices[selectedLanguage]?.voices || []}
            value={selectedVoice[1]}
            onChange={(value) => {
              setSelectedVoice([selectedLanguage, value]);
            }}
            placeholder="Select a voice"
            dimmed={hasActiveCustomPrompt}
            containerClassName="mb-3"
          />
        </Fragment>
      )}
    </div>
  );
});
