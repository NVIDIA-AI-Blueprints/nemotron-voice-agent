// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { toast } from "sonner";
import { useCallback, useEffect, useRef, useState } from "react";
import { AudioStream } from "./AudioStream";
import { AudioWaveForm } from "./AudioWaveForm";
import { Toaster } from "./components/ui/sonner";
import { RTC_CONFIG, RTC_OFFER_URL } from "./config";
import usePipecatWebRTC from "./hooks/use-pipecat-webrtc";
import { Transcripts } from "./Transcripts";
import MicrophoneButton from "./MicrophoneButton";
import { PromptInput } from "./PromptInput";
import { VoiceSelector, type VoiceSelectorRef } from "./VoiceSelector";
import type { VoicesMap } from "./types";
import { Header } from "./components/ui/header";

function App() {
  // UI state
  type RolePrompt = { role: "system" | "user" | "assistant"; content: string };
  const [currentPrompts, setCurrentPrompts] = useState<RolePrompt[]>([]);
  const [started, setStarted] = useState<boolean>(false);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [pendingStart, setPendingStart] = useState<boolean>(false);
  const [hasSystemPrompt, setHasSystemPrompt] = useState<boolean>(false);  // Track if system prompt was received

  // TTS state
  const [voicesByLanguage, setVoicesByLanguage] = useState<VoicesMap>({});
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [isZeroshotModel, setIsZeroshotModel] = useState<boolean>(false);
  const [customPromptName, setCustomPromptName] = useState<string>("");  // Backend prompt filename
  const [activeCustomPromptId, setActiveCustomPromptId] = useState<string>("");  // Active custom prompt ID (from backend's zero_shot_prompt)

  // Uploaded prompts management
  interface UploadedPrompt {
    id: string;
    name: string;
    file: File;
  }
  const [uploadedPrompts, setUploadedPrompts] = useState<UploadedPrompt[]>([]);

  // Track if we've already synced for this connection
  const hasSyncedRef = useRef<boolean>(false);
  const syncInProgressRef = useRef<boolean>(false);
  // Track if begin_conversation has been sent this session (prevents duplicates)
  const conversationStartedRef = useRef<boolean>(false);

  // Ref to VoiceSelector for backend-triggered updates
  const voiceSelectorRef = useRef<VoiceSelectorRef>(null);

  // Track voice state from last session for persistence
  const lastSessionVoiceRef = useRef<{ defaultVoice: string; customPromptId: string }>({
    defaultVoice: "",
    customPromptId: ""
  });
  const currentPromptsRef = useRef<RolePrompt[]>([]);
  // Base prompts cache - stores prompts before session starts, never includes runtime turns
  const basePromptsRef = useRef<RolePrompt[]>([]);

  const sanitizePrompts = useCallback(
    (prompts: any): RolePrompt[] => {
      if (!Array.isArray(prompts)) return [];
      return prompts
        .map((p) => {
          const role = typeof p?.role === "string" ? p.role : "system";
          const content = typeof p?.content === "string" ? p.content : "";
          return { role: role as RolePrompt["role"], content };
        })
        .filter(
          (p) =>
            ["system", "user", "assistant"].includes(p.role) &&
            p.content.trim().length > 0
        );
    },
    []
  );

  const promptsPayload = useCallback(() => {
    const sanitized = sanitizePrompts(currentPrompts);
    return sanitized.length ? sanitized : [];
  }, [currentPrompts, sanitizePrompts]);

  // Ensure the prompt edit panel stays available while we still have cached prompts
  useEffect(() => {
    setHasSystemPrompt(currentPrompts.length > 0);
    currentPromptsRef.current = currentPrompts;
  }, [currentPrompts]);

  // When session stops, restore from base prompts (excludes any runtime messages like intro)
  useEffect(() => {
    if (!started && basePromptsRef.current.length > 0) {
      setCurrentPrompts(basePromptsRef.current);
    }
  }, [started]);

  const webRTC = usePipecatWebRTC({
    url: RTC_OFFER_URL,
    rtcConfig: RTC_CONFIG,
    onError: (e) => toast.error(e.message),
  });


  // When connected via Configure, show config panel until Save; hide when save is clicked
  useEffect(() => {
    if (webRTC.status === "connected" && !started) {
      setShowConfig(true);
    }
    if (webRTC.status !== "connected") {
      setShowConfig(false);
      setStarted(false);
    }
  }, [webRTC.status, started]);

  // If user clicked Start before connecting, auto-begin once dataChannel is ready
  useEffect(() => {
    if (!pendingStart || webRTC.status !== "connected" || started) return;
    const ch = webRTC.dataChannel as RTCDataChannel | null;
    if (!ch) return;
    const sendStart = () => {
      const promptData = promptsPayload();
      if (promptData.length > 0) {
        ch.send(
          JSON.stringify({ id: "prompt-start", label: "rtvi-ai", type: "client-message", data: { t: "context_reset", d: promptData } })
        );
      }
      if (selectedVoice.trim()) {
        ch.send(
          JSON.stringify({
            id: "voice-reapply",
            label: "rtvi-ai",
            type: "client-message",
            data: {
              t: "set_tts_voice",
              d: {
                voice_type: "default",
                voice_id: selectedVoice.trim()
              }
            }
          })
        );
      }

      // If no prompts to sync OR prompts were already synced during configure, begin conversation immediately
      const needsSync = uploadedPrompts.length > 0 && !hasSyncedRef.current;
      if (!needsSync) {
        beginConversation(ch);
      }

      setStarted(true);
      setShowConfig(false);
      setPendingStart(false);
    };
    if (ch.readyState === "open") sendStart();
    else ch.addEventListener("open", sendStart, { once: true });
  }, [pendingStart, webRTC.status, started, promptsPayload, selectedVoice, uploadedPrompts.length]);

  // Handle TTS messages from RTVI data channel (ignore transcripts)
  useEffect(() => {
    if (webRTC.status !== "connected") return;
    const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
    if (!ch) return;
    const onMessage = (ev: MessageEvent) => {
      try {
        const envelope = JSON.parse(ev.data);
        const payload = typeof envelope?.data === "string" ? JSON.parse(envelope.data) : envelope?.data;
        if (!payload || typeof payload !== "object") return;
        if (payload?.type === "riva_voices") {
          // Handle consolidated voice information from backend
          if (payload.available_voices) {
            setVoicesByLanguage(payload.available_voices as VoicesMap);
          }
          const backendVoiceId = payload.current_voice_id || "";
          const zeroShotPrompt = payload.zero_shot_prompt || "";
          setIsZeroshotModel(payload.is_zeroshot_model === true);

          // Store backend's custom prompt filename if available
          if (zeroShotPrompt) {
            setCustomPromptName(zeroShotPrompt);
          }

          // Determine if we need to restore last session state or sync with backend
          const hasLastSessionState = lastSessionVoiceRef.current.defaultVoice !== "" ||
                                       lastSessionVoiceRef.current.customPromptId !== "";

          if (hasLastSessionState) {
            // Restore last session state
            const lastDefaultVoice = lastSessionVoiceRef.current.defaultVoice;
            const lastCustomPromptId = lastSessionVoiceRef.current.customPromptId;

            if (lastCustomPromptId) {
              // Last session had custom prompt active
              setSelectedVoice(lastDefaultVoice);  // Keep the default voice in UI
              setActiveCustomPromptId(lastCustomPromptId);

              // Reapply custom prompt to backend
              ch.send(
                JSON.stringify({
                  id: "restore-voice-state",
                  label: "rtvi-ai",
                  type: "client-message",
                  data: {
                    t: "set_tts_voice",
                    d: {
                      voice_type: "custom",
                      prompt_id: lastCustomPromptId
                    }
                  }
                })
              );
            } else if (lastDefaultVoice) {
              // Last session had only default voice (no custom prompt)
              setSelectedVoice(lastDefaultVoice);
              setActiveCustomPromptId("");  // Explicitly no custom prompt

              // Reapply default voice to backend
              ch.send(
                JSON.stringify({
                  id: "restore-voice-state",
                  label: "rtvi-ai",
                  type: "client-message",
                  data: {
                    t: "set_tts_voice",
                    d: {
                      voice_type: "default",
                      voice_id: lastDefaultVoice
                    }
                  }
                })
              );
            }
          } else {
            // No last session state - sync with backend's current state
            if (!selectedVoice && backendVoiceId) {
              setSelectedVoice(backendVoiceId);
            // Cache backend-provided default voice for subsequent reconnects
            lastSessionVoiceRef.current = {
              defaultVoice: backendVoiceId,
              customPromptId: lastSessionVoiceRef.current.customPromptId,
            };
            }

            if (zeroShotPrompt) {
              // Backend has custom prompt active
              const matchingUploadedPrompt = uploadedPrompts.find(p => p.name === zeroShotPrompt);
              setActiveCustomPromptId(matchingUploadedPrompt ? matchingUploadedPrompt.id : "backend");
            // Cache custom prompt selection alongside the current voice
            lastSessionVoiceRef.current = {
              defaultVoice: lastSessionVoiceRef.current.defaultVoice || backendVoiceId,
              customPromptId: matchingUploadedPrompt ? matchingUploadedPrompt.id : "backend",
            };
            } else {
              // No custom prompt active
              setActiveCustomPromptId("");
            // If we learned the backend voice, keep it cached even without custom prompt
            if (backendVoiceId) {
              lastSessionVoiceRef.current = {
                defaultVoice: backendVoiceId,
                customPromptId: "",
              };
            }
            }
          }
        } else if (payload?.type === "tts_update_settings") {
          console.log("Received TTS update settings: ", payload);
          // Backend (LLM) triggered voice change - update UI only via ref
          const newLanguage = payload.language_code || "";
          const newVoiceId = payload.voice_id || "";
          if (newLanguage && newVoiceId) {
            voiceSelectorRef.current?.setVoiceFromBackend(newLanguage, newVoiceId);
          }
        }
        else if (payload?.type === "system_prompt") {
          // Only accept backend prompts if we don't have cached base prompts
          // basePromptsRef stores the pristine prompts before any session starts
          if (basePromptsRef.current.length === 0) {
            const promptsArray = Array.isArray(payload.prompts) ? payload.prompts : [];
            const fallbackPrompt = typeof payload.prompt === "string" ? payload.prompt : "";
            const parsed = sanitizePrompts(
              promptsArray.length > 0 ? promptsArray : [{ role: "system", content: fallbackPrompt }]
            );
            basePromptsRef.current = parsed;
            setCurrentPrompts(parsed);
            setHasSystemPrompt(parsed.length > 0);
          }
        }
      } catch {}
    };
    ch.addEventListener("message", onMessage);
    return () => ch.removeEventListener("message", onMessage);
  }, [webRTC.status, selectedVoice, uploadedPrompts, sanitizePrompts]);

  // Reset sync flags when disconnected
  useEffect(() => {
    if (webRTC.status !== "connected") {
      console.log("Resetting sync flags because status is:", webRTC.status);
      hasSyncedRef.current = false;
      syncInProgressRef.current = false;
      conversationStartedRef.current = false;
    }
  }, [webRTC.status]);

  // Helper to send begin_conversation exactly once per session
  const beginConversation = useCallback((ch: RTCDataChannel) => {
    if (conversationStartedRef.current) return;
    if (ch.readyState !== "open") return;
    conversationStartedRef.current = true;
    ch.send(JSON.stringify({
      id: "begin-conversation",
      label: "rtvi-ai",
      type: "client-message",
      data: { t: "begin_conversation" }
    }));
  }, []);

  const handleVoiceChange = useCallback(
    (language: string, voice: string) => {
      const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
      if (ch && ch.readyState === "open" && language && voice) {
        setSelectedVoice(voice);
        // Selecting default voice will automatically deselect custom prompts on backend
        setActiveCustomPromptId("");  // Clear UI selection immediately for responsiveness

        // Save state for next session: default voice selected, no custom prompt
        lastSessionVoiceRef.current = {
          defaultVoice: voice,
          customPromptId: ""
        };

        // Use unified voice selection action
        ch.send(
          JSON.stringify({
            id: "voice-set",
            label: "rtvi-ai",
            type: "client-message",
            data: {
              t: "set_tts_voice",
              d: {
                voice_type: "default",
                language_code: language,
                voice_id: voice
              }
            }
          })
        );
        toast.success(`Switched voice: ${voice}`);
      } else {
        console.warn("Voice change ignored; data channel not open or missing language/voice.");
      }
    },
    [webRTC.status, webRTC]
  );

  const handlePromptChange = useCallback((index: number, content: string) => {
    setCurrentPrompts((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      const next = [...prev];
      next[index] = { ...next[index], content };
      // Also update basePromptsRef so user edits persist across stop/start
      basePromptsRef.current = next;
      return next;
    });
  }, []);


  const handleFileUpload = useCallback(async (file: File, isReSync: boolean = false, existingPromptId?: string) => {
    console.log("=== START: handleFileUpload ===");
    console.log("File:", file.name, "Size:", file.size, "bytes", isReSync ? "(RE-SYNC)" : "");

    const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
    if (!ch) {
      console.error("No data channel available");
      toast.error("Not connected");
      return;
    }

    console.log("Data channel state:", ch.readyState);

    // Add listener for channel closure
    const onChannelClose = () => console.error("Data channel closed during upload!");
    ch.addEventListener('close', onChannelClose);

    try {
      // Convert file to base64
      console.log("Converting file to base64...");
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Convert to base64 in chunks to avoid stack overflow
      let binary = '';
      const chunkSize = 8192; // Process 8KB at a time
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode(...chunk);
      }
      const base64 = btoa(binary);
      console.log("Base64 length:", base64.length);

      // Generate or use existing ID
      const promptId = existingPromptId || `${Date.now()}_${file.name}`;
      console.log(isReSync ? "Using existing prompt ID:" : "Generated prompt ID:", promptId);

      // Split large messages into chunks (max 50KB per chunk to be very safe)
      const maxChunkSize = 50000;
      const totalChunks = Math.ceil(base64.length / maxChunkSize);

      console.log(`Splitting upload into ${totalChunks} chunks...`);

      if (totalChunks === 1) {
        // Single message for small files
        const message = {
          id: "upload-prompt",
          label: "rtvi-ai",
          type: "client-message",
          data: {
            t: "upload_custom_audio_prompt",
            d: {
              audio: base64,
              filename: file.name,
              prompt_id: promptId
            }
          }
        };
        ch.send(JSON.stringify(message));
        console.log("Single message sent");
      } else {
        // Send chunks for large files
        for (let i = 0; i < totalChunks; i++) {
          // Check if channel is still open
          if (ch.readyState !== "open") {
            console.error(`Data channel closed at chunk ${i + 1}/${totalChunks}, readyState: ${ch.readyState}`);
            throw new Error(`Data channel closed while uploading (chunk ${i + 1}/${totalChunks})`);
          }

          const start = i * maxChunkSize;
          const end = Math.min(start + maxChunkSize, base64.length);
          const chunk = base64.substring(start, end);

          const message = {
            id: `upload-prompt-chunk-${i}`,
            label: "rtvi-ai",
            type: "client-message",
            data: {
              t: "upload_custom_audio_prompt",
              d: {
                chunk: chunk,
                chunk_index: i,
                total_chunks: totalChunks,
                filename: file.name,
                prompt_id: promptId
              }
            }
          };

          const messageStr = JSON.stringify(message);
          console.log(`Sending chunk ${i + 1}/${totalChunks}, size: ${messageStr.length} bytes, bufferedAmount: ${ch.bufferedAmount}`);

          // Wait if buffer is too full
          while (ch.bufferedAmount > 1024 * 1024 && ch.readyState === "open") { // Wait if more than 1MB buffered
            console.log(`Buffer full (${ch.bufferedAmount} bytes), waiting...`);
            await new Promise(resolve => setTimeout(resolve, 100));
          }

          ch.send(messageStr);
          console.log(`Sent chunk ${i + 1}/${totalChunks}, new bufferedAmount: ${ch.bufferedAmount}`);

          // Delay between chunks to let the channel recover
          if (i < totalChunks - 1) {
            console.log("Waiting before next chunk...");
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }
      }
      console.log("Upload message(s) sent successfully");

      // Add to uploaded prompts list (only if it's a new upload, not a re-sync)
      if (!isReSync) {
        const newPrompt: UploadedPrompt = {
          id: promptId,
          name: file.name,
          file: file
        };

        setUploadedPrompts(prev => [...prev, newPrompt]);
        setActiveCustomPromptId(promptId);  // Set as active

        // Save state for next session: new custom prompt selected
        lastSessionVoiceRef.current = {
          defaultVoice: selectedVoice,  // Keep current default voice
          customPromptId: promptId       // Save newly uploaded prompt
        };

        toast.success(`Uploaded and activated: ${file.name}`);
      } else {
        console.log("Re-sync upload completed, state unchanged");
      }
      console.log("=== END: handleFileUpload ===");

    } catch (error) {
      console.error("Upload failed:", error);
      toast.error("Failed to upload audio file");
    } finally {
      // Clean up listener
      ch.removeEventListener('close', onChannelClose);
    }
  }, [webRTC.status, webRTC, selectedVoice]);

  // Re-upload files on reconnection (placed after handleFileUpload is defined)
  useEffect(() => {
    if (webRTC.status !== "connected" || uploadedPrompts.length === 0) return;

    // Skip if we've already synced
    if (hasSyncedRef.current) {
      console.log("Already synced for this connection, skipping");
      return;
    }

    // Skip if sync is already in progress
    if (syncInProgressRef.current) {
      console.log("Sync already in progress, skipping");
      return;
    }

    // Mark sync as in progress immediately
    syncInProgressRef.current = true;
    console.log("Starting re-sync for new connection...");

    // Wait a bit for the connection to stabilize
    const timer = setTimeout(async () => {
      // Double-check data channel is ready
      const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
      if (!ch || ch.readyState !== "open") {
        console.log("Data channel not ready for re-sync, aborting");
        syncInProgressRef.current = false;
        return;
      }

      console.log("=== Re-syncing uploaded prompts after reconnection ===");
      console.log(`Found ${uploadedPrompts.length} prompts to re-upload`);

      for (const prompt of uploadedPrompts) {
        try {
          console.log(`Re-uploading: ${prompt.name} (ID: ${prompt.id})`);
          await handleFileUpload(prompt.file, true, prompt.id);
          console.log(`Re-uploaded: ${prompt.name}`);
        } catch (error) {
          console.error(`Failed to re-upload ${prompt.name}:`, error);
          toast.error(`Failed to re-sync: ${prompt.name}`);
        }
      }

      // Restore the previously active prompt selection if one was active
      if (activeCustomPromptId && activeCustomPromptId !== "") {
        console.log(`Restoring active prompt selection: ${activeCustomPromptId}`);
        const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
        if (ch && ch.readyState === "open") {
          ch.send(
            JSON.stringify({
              id: "select-prompt-resync",
              label: "rtvi-ai",
              type: "client-message",
              data: {
                t: "set_tts_voice",
                d: {
                  voice_type: "custom",
                  prompt_id: activeCustomPromptId
                }
              }
            })
          );
        }
      }

      // Mark as successfully synced
      hasSyncedRef.current = true;
      syncInProgressRef.current = false;
      toast.success(`Re-synced ${uploadedPrompts.length} custom voice(s)`);
      console.log("=== Re-sync completed ===");

      // Begin conversation after re-sync completes
      if (started) {
        const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
        if (ch) {
          beginConversation(ch);
        }
      }
    }, 2000); // Wait 2 seconds for backend to be ready

    return () => {
      clearTimeout(timer);
      // If component unmounts during sync, reset the flag
      syncInProgressRef.current = false;
    };
  }, [webRTC.status, uploadedPrompts, handleFileUpload, activeCustomPromptId, started]);

  const handleSelectPrompt = useCallback((promptId: string) => {
    const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
    if (!ch) return;

    // Update UI state immediately for responsiveness
    setActiveCustomPromptId(promptId);

    // Save state for next session: custom prompt selected (keep current default voice)
    lastSessionVoiceRef.current = {
      defaultVoice: selectedVoice,  // Keep the default voice selection
      customPromptId: promptId      // Save active custom prompt
    };

    // Send voice selection to backend
    ch.send(
      JSON.stringify({
        id: "select-prompt",
        label: "rtvi-ai",
        type: "client-message",
        data: {
          t: "set_tts_voice",
          d: {
            voice_type: "custom",
            prompt_id: promptId
          }
        }
      })
    );

    // Get friendly name for toast
    const promptName = promptId === "backend"
      ? customPromptName
      : uploadedPrompts.find(p => p.id === promptId)?.name || promptId;

    toast.success(`Switched to custom voice: ${promptName}`);
  }, [webRTC.status, webRTC, uploadedPrompts, customPromptName, selectedVoice]);

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <section className="flex-1 flex">
        <div className="flex-1 p-5">
          <AudioStream
            streamOrTrack={webRTC.status === "connected" ? webRTC.stream : null}
          />
          <Transcripts
            dataChannel={webRTC.status === "connected" ? webRTC.dataChannel : null}
          />
        </div>
        <div className="p-5 border-l-1 border-gray-200 flex flex-col">
          <div className="flex-1 mb-4">
            <AudioWaveForm
              streamOrTrack={webRTC.status === "connected" ? webRTC.stream : null}
            />
          </div>
          {webRTC.status === "connected" && (
            <>
              {!started && showConfig && hasSystemPrompt && (
                <div className="mt-4">
                  <div className="text-sm text-gray-600 mb-2">Edit Prompts:</div>
                  <div className="space-y-4 max-h-80 overflow-y-auto">
                    {currentPrompts.map((p, idx) => (
                      <div key={`${p.role}-${idx}`} className="h-60">
                        <div className="text-xs font-semibold uppercase text-gray-500 mb-1">
                          {p.role} prompt
                        </div>
                        <PromptInput
                          defaultValue={p.content}
                          onChange={(val) => handlePromptChange(idx, val)}
                          disabled={false}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <VoiceSelector
                ref={voiceSelectorRef}
                voices={voicesByLanguage}
                onVoiceChange={handleVoiceChange}
                isZeroshotModel={isZeroshotModel}
                initialVoiceId={lastSessionVoiceRef.current.defaultVoice}
                activeCustomPromptId={activeCustomPromptId}
                customPromptName={customPromptName}
                uploadedPrompts={uploadedPrompts}
                onFileUpload={handleFileUpload}
                onSelectPrompt={handleSelectPrompt}
                isConfigureMode={!started && showConfig}
              />
            </>
          )}
        </div>
      </section>
      <footer className="border-t-1 border-gray-200 p-6 flex items-center justify-between">
        {/* Left cluster: Start / Stop + Mic */}
        <div className="flex items-center">
          {!started && (
            <button
              className={`bg-nvidia px-4 py-2 rounded-lg text-white flex items-center gap-2 transition-opacity ${
                webRTC.status === "connected" && showConfig
                  ? "opacity-40 cursor-not-allowed"
                  : pendingStart || webRTC.status === "connecting"
                  ? "opacity-80 cursor-wait"
                  : ""
              }`}
              onClick={() => {
                if (webRTC.status !== "connected") {
                  setPendingStart(true);
                  (webRTC as any).start?.();
                  return;
                }
                // If config is open, require Save first
                if (showConfig) return;
                const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
                const promptData = promptsPayload();
                if (hasSystemPrompt && ch && promptData.length > 0) {
                  ch.send(JSON.stringify({ id: "prompt-start", label: "rtvi-ai", type: "client-message", data: { t: "context_reset", d: promptData } }));
                }

                // If no prompts to sync OR prompts were already synced during configure, begin conversation immediately
                const needsSync = uploadedPrompts.length > 0 && !hasSyncedRef.current;
                if (ch && !needsSync) {
                  beginConversation(ch);
                }

                setStarted(true);
                setShowConfig(false);
              }}
              disabled={(webRTC.status === "connected" && showConfig) || pendingStart}
            >
              {(pendingStart || webRTC.status === "connecting") && (
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
              {pendingStart || webRTC.status === "connecting" ? "Starting..." : "Start"}
            </button>
          )}
          {webRTC.status === "connected" && started && (
            <>
              <button
                className="ml-3 bg-nvidia px-4 py-2 rounded-lg text-white"
                onClick={() => {
                  setStarted(false);
                  (webRTC as any).stop?.();
                }}
              >
                Stop
              </button>
              <MicrophoneButton stream={(webRTC as any).micStream} />
            </>
          )}
        </div>

        {/* Right cluster: Configure (init) and Save (connected pre-start) */}
        <div className="flex items-center">
          {webRTC.status !== "connected" && (
            <button
              className={`bg-nvidia px-4 py-2 rounded-lg text-white flex items-center gap-2 ${webRTC.status === "connecting" ? "opacity-80 cursor-wait" : ""}`}
              onClick={() => (webRTC as any).start?.()}
              disabled={webRTC.status === "connecting"}
            >
              {webRTC.status === "connecting" && (
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
              {webRTC.status === "connecting" ? "Connecting..." : "Configure"}
            </button>
          )}
          {webRTC.status === "connected" && !started && showConfig && (
            <button
              className="bg-nvidia px-4 py-2 rounded-lg text-white ml-3"
              onClick={() => {
                const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
                const promptData = promptsPayload();
                if (hasSystemPrompt && ch && promptData.length > 0) {
                  ch.send(JSON.stringify({ id: "prompt-save", label: "rtvi-ai", type: "client-message", data: { t: "context_reset", d: promptData } }));
                }
                setShowConfig(false);
                toast.success("Configuration saved");
              }}
            >
              Save
            </button>
          )}
        </div>
      </footer>
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
