// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { toast } from "sonner";
import React from "react";
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
import {
  Phone,
  Mic,
  Waves,
  ShieldCheck,
  Cpu,
  Database,
  Brain,
  Activity,
  CheckCircle2,
  Clock,
  Gauge,
  Volume2,
} from "lucide-react";

function App() {
  // UI state
  type RolePrompt = { role: "system" | "user" | "assistant"; content: string };
  const [currentPrompts, setCurrentPrompts] = useState<RolePrompt[]>([]);
  const [started, setStarted] = useState<boolean>(false);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [pendingStart, setPendingStart] = useState<boolean>(false);
  const [hasSystemPrompt, setHasSystemPrompt] = useState<boolean>(false); // Track if system prompt was received
  const [callDuration, setCallDuration] = useState<number>(0);

  // Call duration timer
  useEffect(() => {
    if (!started) {
      setCallDuration(0);
      return;
    }
    const interval = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [started]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // TTS state
  const [voicesByLanguage, setVoicesByLanguage] = useState<VoicesMap>({});
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [isZeroshotModel, setIsZeroshotModel] = useState<boolean>(false);
  const [customPromptName, setCustomPromptName] = useState<string>(""); // Backend prompt filename
  const [activeCustomPromptId, setActiveCustomPromptId] = useState<string>(""); // Active custom prompt ID (from backend's zero_shot_prompt)
  const [tokenUsage, setTokenUsage] = useState({
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  });
  const [latencies, setLatencies] = useState({
    asr: 0,
    llm: 0,
    tts: 0,
  });
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
  const lastSessionVoiceRef = useRef<{
    defaultVoice: string;
    customPromptId: string;
  }>({
    defaultVoice: "",
    customPromptId: "",
  });
  const currentPromptsRef = useRef<RolePrompt[]>([]);
  // Base prompts cache - stores prompts before session starts, never includes runtime turns
  const basePromptsRef = useRef<RolePrompt[]>([]);

  const sanitizePrompts = useCallback((prompts: any): RolePrompt[] => {
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
          p.content.trim().length > 0,
      );
  }, []);

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
          JSON.stringify({
            id: "prompt-start",
            label: "rtvi-ai",
            type: "client-message",
            data: { t: "context_reset", d: promptData },
          }),
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
                voice_id: selectedVoice.trim(),
              },
            },
          }),
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
  }, [
    pendingStart,
    webRTC.status,
    started,
    promptsPayload,
    selectedVoice,
    uploadedPrompts.length,
  ]);

  // Handle TTS messages from RTVI data channel (ignore transcripts)
  useEffect(() => {
    if (webRTC.status !== "connected") return;
    const ch = (webRTC as any).dataChannel as RTCDataChannel | null;
    if (!ch) return;
    const onMessage = (ev: MessageEvent) => {
      try {
        const envelope = JSON.parse(ev.data);
        const payload =
          typeof envelope?.data === "string"
            ? JSON.parse(envelope.data)
            : envelope?.data;
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
          const hasLastSessionState =
            lastSessionVoiceRef.current.defaultVoice !== "" ||
            lastSessionVoiceRef.current.customPromptId !== "";

          if (hasLastSessionState) {
            // Restore last session state
            const lastDefaultVoice = lastSessionVoiceRef.current.defaultVoice;
            const lastCustomPromptId =
              lastSessionVoiceRef.current.customPromptId;

            if (lastCustomPromptId) {
              // Last session had custom prompt active
              setSelectedVoice(lastDefaultVoice); // Keep the default voice in UI
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
                      prompt_id: lastCustomPromptId,
                    },
                  },
                }),
              );
            } else if (lastDefaultVoice) {
              // Last session had only default voice (no custom prompt)
              setSelectedVoice(lastDefaultVoice);
              setActiveCustomPromptId(""); // Explicitly no custom prompt

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
                      voice_id: lastDefaultVoice,
                    },
                  },
                }),
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
              const matchingUploadedPrompt = uploadedPrompts.find(
                (p) => p.name === zeroShotPrompt,
              );
              setActiveCustomPromptId(
                matchingUploadedPrompt ? matchingUploadedPrompt.id : "backend",
              );
              // Cache custom prompt selection alongside the current voice
              lastSessionVoiceRef.current = {
                defaultVoice:
                  lastSessionVoiceRef.current.defaultVoice || backendVoiceId,
                customPromptId: matchingUploadedPrompt
                  ? matchingUploadedPrompt.id
                  : "backend",
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
            voiceSelectorRef.current?.setVoiceFromBackend(
              newLanguage,
              newVoiceId,
            );
          }
        } else if (payload?.type === "system_prompt") {
          // Only accept backend prompts if we don't have cached base prompts
          // basePromptsRef stores the pristine prompts before any session starts
          if (basePromptsRef.current.length === 0) {
            const promptsArray = Array.isArray(payload.prompts)
              ? payload.prompts
              : [];
            const fallbackPrompt =
              typeof payload.prompt === "string" ? payload.prompt : "";
            const parsed = sanitizePrompts(
              promptsArray.length > 0
                ? promptsArray
                : [{ role: "system", content: fallbackPrompt }],
            );
            basePromptsRef.current = parsed;
            setCurrentPrompts(parsed);
            setHasSystemPrompt(parsed.length > 0);
          }
        } else if (
          payload?.tokens &&
          Array.isArray(payload.tokens) &&
          payload.tokens.length > 0
        ) {
          const latest = payload.tokens[0];

          setTokenUsage({
            prompt_tokens: latest.prompt_tokens ?? 0,
            completion_tokens: latest.completion_tokens ?? 0,
            total_tokens: latest.total_tokens ?? 0,
          });

          console.log("Token Usage:", latest);
        } else if (
          payload?.processing &&
          Array.isArray(payload.processing) &&
          payload.processing.length > 0
        ) {
          console.log("Received processing metrics array:", payload.processing);
          const metric = payload.processing[0];
          console.log("Received metric:", metric);

          if (metric.processor?.includes("NvidiaLLMService")) {
            setLatencies((prev) => ({
              ...prev,
              llm: Math.round(metric.value * 1000),
            }));
          }

          if (metric.processor?.includes("NemotronTTSService")) {
            setLatencies((prev) => ({
              ...prev,
              tts: Math.round(metric.value * 1000),
            }));
          }

          if (metric.processor?.includes("RivaASRService")) {
            setLatencies((prev) => ({
              ...prev,
              asr: Math.round(metric.value * 1000),
            }));
          }
        }
      } catch {}
    };
    ch.addEventListener("message", onMessage);
    return () => ch.removeEventListener("message", onMessage);
  }, [webRTC.status, selectedVoice, uploadedPrompts, sanitizePrompts]);

  const totalLatency = latencies.asr + latencies.llm + latencies.tts;

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
    ch.send(
      JSON.stringify({
        id: "begin-conversation",
        label: "rtvi-ai",
        type: "client-message",
        data: { t: "begin_conversation" },
      }),
    );
  }, []);

  const handleVoiceChange = useCallback(
    (language: string, voice: string) => {
      const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
      if (ch && ch.readyState === "open" && language && voice) {
        setSelectedVoice(voice);
        // Selecting default voice will automatically deselect custom prompts on backend
        setActiveCustomPromptId(""); // Clear UI selection immediately for responsiveness

        // Save state for next session: default voice selected, no custom prompt
        lastSessionVoiceRef.current = {
          defaultVoice: voice,
          customPromptId: "",
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
                voice_id: voice,
              },
            },
          }),
        );
        toast.success(`Switched voice: ${voice}`);
      } else {
        console.warn(
          "Voice change ignored; data channel not open or missing language/voice.",
        );
      }
    },
    [webRTC.status, webRTC],
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

  const handleFileUpload = useCallback(
    async (
      file: File,
      isReSync: boolean = false,
      existingPromptId?: string,
    ) => {
      console.log("=== START: handleFileUpload ===");
      console.log(
        "File:",
        file.name,
        "Size:",
        file.size,
        "bytes",
        isReSync ? "(RE-SYNC)" : "",
      );

      const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
      if (!ch) {
        console.error("No data channel available");
        toast.error("Not connected");
        return;
      }

      console.log("Data channel state:", ch.readyState);

      // Add listener for channel closure
      const onChannelClose = () =>
        console.error("Data channel closed during upload!");
      ch.addEventListener("close", onChannelClose);

      try {
        // Convert file to base64
        console.log("Converting file to base64...");
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        // Convert to base64 in chunks to avoid stack overflow
        let binary = "";
        const chunkSize = 8192; // Process 8KB at a time
        for (let i = 0; i < bytes.length; i += chunkSize) {
          const chunk = bytes.subarray(i, i + chunkSize);
          binary += String.fromCharCode(...chunk);
        }
        const base64 = btoa(binary);
        console.log("Base64 length:", base64.length);

        // Generate or use existing ID
        const promptId = existingPromptId || `${Date.now()}_${file.name}`;
        console.log(
          isReSync ? "Using existing prompt ID:" : "Generated prompt ID:",
          promptId,
        );

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
                prompt_id: promptId,
              },
            },
          };
          ch.send(JSON.stringify(message));
          console.log("Single message sent");
        } else {
          // Send chunks for large files
          for (let i = 0; i < totalChunks; i++) {
            // Check if channel is still open
            if (ch.readyState !== "open") {
              console.error(
                `Data channel closed at chunk ${i + 1}/${totalChunks}, readyState: ${ch.readyState}`,
              );
              throw new Error(
                `Data channel closed while uploading (chunk ${i + 1}/${totalChunks})`,
              );
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
                  prompt_id: promptId,
                },
              },
            };

            const messageStr = JSON.stringify(message);
            console.log(
              `Sending chunk ${i + 1}/${totalChunks}, size: ${messageStr.length} bytes, bufferedAmount: ${ch.bufferedAmount}`,
            );

            // Wait if buffer is too full
            while (
              ch.bufferedAmount > 1024 * 1024 &&
              ch.readyState === "open"
            ) {
              // Wait if more than 1MB buffered
              console.log(
                `Buffer full (${ch.bufferedAmount} bytes), waiting...`,
              );
              await new Promise((resolve) => setTimeout(resolve, 100));
            }

            ch.send(messageStr);
            console.log(
              `Sent chunk ${i + 1}/${totalChunks}, new bufferedAmount: ${ch.bufferedAmount}`,
            );

            // Delay between chunks to let the channel recover
            if (i < totalChunks - 1) {
              console.log("Waiting before next chunk...");
              await new Promise((resolve) => setTimeout(resolve, 100));
            }
          }
        }
        console.log("Upload message(s) sent successfully");

        // Add to uploaded prompts list (only if it's a new upload, not a re-sync)
        if (!isReSync) {
          const newPrompt: UploadedPrompt = {
            id: promptId,
            name: file.name,
            file: file,
          };

          setUploadedPrompts((prev) => [...prev, newPrompt]);
          setActiveCustomPromptId(promptId); // Set as active

          // Save state for next session: new custom prompt selected
          lastSessionVoiceRef.current = {
            defaultVoice: selectedVoice, // Keep current default voice
            customPromptId: promptId, // Save newly uploaded prompt
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
        ch.removeEventListener("close", onChannelClose);
      }
    },
    [webRTC.status, webRTC, selectedVoice],
  );

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
        console.log(
          `Restoring active prompt selection: ${activeCustomPromptId}`,
        );
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
                  prompt_id: activeCustomPromptId,
                },
              },
            }),
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
  }, [
    webRTC.status,
    uploadedPrompts,
    handleFileUpload,
    activeCustomPromptId,
    started,
  ]);

  const handleSelectPrompt = useCallback(
    (promptId: string) => {
      const ch = webRTC.status === "connected" ? webRTC.dataChannel : null;
      if (!ch) return;

      // Update UI state immediately for responsiveness
      setActiveCustomPromptId(promptId);

      // Save state for next session: custom prompt selected (keep current default voice)
      lastSessionVoiceRef.current = {
        defaultVoice: selectedVoice, // Keep the default voice selection
        customPromptId: promptId, // Save active custom prompt
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
              prompt_id: promptId,
            },
          },
        }),
      );

      // Get friendly name for toast
      const promptName =
        promptId === "backend"
          ? customPromptName
          : uploadedPrompts.find((p) => p.id === promptId)?.name || promptId;

      toast.success(`Switched to custom voice: ${promptName}`);
    },
    [webRTC.status, webRTC, uploadedPrompts, customPromptName, selectedVoice],
  );

  const outcomes = [
    { k: "68%", v: "First-call resolution" },
    { k: "24/7", v: "Availability, every language" },
    { k: "100%", v: "Calls scored for compliance" },
    { k: "$48 k", v: "Recovered in pilot quarter" },
    { k: "0", v: "Agent burnout on repetitive asks" },
    { k: "4.3/5", v: "Customer experience score" },
  ];

  const stats = [
    { k: "3.2×", v: "Contact rate lift" },
    { k: "42%", v: "Cost per resolution" },
    { k: "<800ms", v: "End-to-end latency" },
  ];

  const STAGES = [
    { id: 1, label: "Voice capture", detail: "NVIDIA Riva ASR", icon: Mic },
    {
      id: 2,
      label: "Intent & sentiment",
      detail: "NeMo Classifiers",
      icon: Brain,
    },
    {
      id: 3,
      label: "Policy & context",
      detail: "RAG over Collections Playbook",
      icon: Database,
    },
    { id: 4, label: "Reasoning", detail: "LLM / Triton", icon: Cpu },
    {
      id: 5,
      label: "Voice synthesis",
      detail: "Riva TTS — neural voice",
      icon: Waves,
    },
    {
      id: 6,
      label: "Compliance guardrails",
      detail: "PII redaction · Regulatory check",
      icon: ShieldCheck,
    },
  ];

  const customerDetails = [
    { label: "Outstanding", value: "$1150", bold: true },
    { label: "Days past due", value: "32 days" },
    { label: "Segment", value: "Insurance" },
    { label: "Language Preference", value: "English" },
  ];

  return (
    <div className="h-screen flex flex-col">
      <Header />

      {/*=============== Hero Section================== */}
      <section className="max-w-350 mx-auto px-8 pt-14 pb-10 grain">
        <div className="grid grid-cols-12 gap-8 items-end">
          {/* Left Content */}
          <div className="col-span-12 lg:col-span-8">
            <div className="flex items-center gap-3 mb-6">
              <span className="inline-block w-8 h-0.5 bg-[#FB4E0B]" />

              <span className="text-[11px] font-bold tracking-[0.22em] text-[#808080] uppercase font-yantramanav">
                A live working prototype · not a mock
              </span>
            </div>

            <h1 className="font-yantramanav font-light text-[42px] md:text-[56px] lg:text-[66px] leading-[0.95] tracking-[-0.02em] text-black">
              Collections,
              <br />
              <span className="font-bold text-[#FB4E0B]">reimagined</span> as
              <br />a conversation.
            </h1>

            <p className="mt-7 max-w-130 text-[17px] leading-[1.65] font-light text-[#414141] font-yantramanav">
              An autonomous voice agent that negotiates repayment plans with
              empathy, respects every regulator, and scales to a million
              conversations a day, built on the NVIDIA stack.
            </p>
          </div>

          {/* Right Stats */}
          <div className="col-span-12 lg:col-span-4 lg:text-right">
            <div className="inline-grid grid-cols-3 gap-6 lg:gap-8 border-l-[3px] border-[#FB4E0B] pl-6">
              {stats.map((stat, index) => (
                <div key={index}>
                  <h3 className="text-2xl lg:text-3xl font-bold text-[#FB4E0B]">
                    {stat.k}
                  </h3>
                  <p className="mt-2 text-sm text-[#808080]">{stat.v}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ============== MAIN DEMO SURFACE ============== */}
      <section className="w-full max-w-350 mx-auto px-8 pb-16">
        <div className="grid grid-cols-12 gap-5">
          {/* --- LEFT --- */}
          <aside className="col-span-12 lg:col-span-3 space-y-4">
            {/* Customer Card */}
            <div className="border border-[#E6E5E5] bg-white p-5">
              <div className="mb-3 font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                Customer in Focus
              </div>

              <div className="flex items-start gap-3">
                <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center bg-[#FB4E0B]">
                  <span className="font-yantramanav text-lg font-bold text-white">
                    N
                  </span>
                </div>

                <div className="min-w-0 flex-1">
                  <div className="font-yantramanav text-[17px] font-bold leading-tight text-black">
                    Nathan Reeves
                  </div>
                  <div className="font-yantramanav text-[11px] tracking-[0.04em] text-[#808080]">
                    R5T2Y9N7
                  </div>
                </div>
              </div>

              <div className="mt-5 space-y-3">
                {customerDetails.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between"
                  >
                    <span className="font-yantramanav text-sm text-[#808080]">
                      {item.label}
                    </span>

                    <span
                      className={`font-yantramanav text-sm text-black ${
                        item.bold ? "font-bold" : ""
                      }`}
                    >
                      {item.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            {/* Call Session */}
            <div className="relative overflow-hidden bg-black p-5">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(251,78,11,0.8),transparent_50%)] opacity-20" />

              <div className="relative">
                <div className="mb-6 flex items-center justify-between">
                  <div className="font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                    Call Session
                  </div>

                  <div className="flex items-center gap-1.5 font-yantramanav text-xs text-[#ABABAB]">
                    <Clock className="h-3 w-3" />
                    {formatDuration(callDuration)}
                  </div>
                </div>

                <div className="flex flex-col items-center py-4">
                  <div className="relative">
                    {/* Pulse Rings */}
                    {webRTC.status === "connected" && started && (
                      <>
                        <div className="pulse-ring absolute inset-0 rounded-full border border-[#FB4E0B]/50" />
                        <div
                          className="pulse-ring absolute inset-0 rounded-full border border-[#FB4E0B]/50"
                          style={{ animationDelay: "0.5s" }}
                        />
                      </>
                    )}

                    {/* Phone Button */}
                    <button
                      onClick={() => {
                        if (
                          webRTC.status === "init" ||
                          webRTC.status === "error"
                        ) {
                          setPendingStart(true);
                          webRTC.start();
                        } else if (webRTC.status === "connected") {
                          if (!started) {
                            const ch =
                              webRTC.dataChannel as RTCDataChannel | null;
                            const promptData = promptsPayload();
                            if (
                              hasSystemPrompt &&
                              ch &&
                              promptData.length > 0
                            ) {
                              ch.send(
                                JSON.stringify({
                                  id: "prompt-start",
                                  label: "rtvi-ai",
                                  type: "client-message",
                                  data: { t: "context_reset", d: promptData },
                                }),
                              );
                            }
                            const needsSync =
                              uploadedPrompts.length > 0 &&
                              !hasSyncedRef.current;
                            if (ch && !needsSync) {
                              beginConversation(ch);
                            }
                            setStarted(true);
                            setShowConfig(false);
                          } else {
                            setStarted(false);
                            webRTC.stop();
                          }
                        }
                      }}
                      className={`relative flex h-24 w-24 items-center justify-center rounded-full text-white transition-all duration-200 hover:scale-105 active:scale-95 ${
                        webRTC.status === "connected" && started
                          ? "bg-red-600 animate-pulse"
                          : webRTC.status === "connecting"
                            ? "bg-yellow-600 animate-pulse"
                            : "bg-[#FB4E0B] hover:bg-[#e03a00]"
                      }`}
                      disabled={webRTC.status === "connecting"}
                    >
                      {webRTC.status === "connected" && started ? (
                        <Phone className="h-8 w-8 rotate-135" />
                      ) : (
                        <Phone className="h-8 w-8" />
                      )}
                    </button>
                  </div>

                  <div className="mt-4 flex flex-col items-center gap-2">
                    <div className="flex items-center gap-2 font-yantramanav text-sm font-light text-[#DBD8DB]">
                      {webRTC.status === "connected" && started ? (
                        <>
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
                          Live Call
                        </>
                      ) : webRTC.status === "connected" && !started ? (
                        <>
                          <span className="h-1.5 w-1.5 rounded-full bg-yellow-500" />
                          Ready to Start
                        </>
                      ) : webRTC.status === "connecting" ? (
                        <>
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-yellow-500" />
                          Connecting...
                        </>
                      ) : (
                        <>
                          <span className="h-1.5 w-1.5 rounded-full bg-gray-500" />
                          Offline
                        </>
                      )}
                    </div>

                    <div className="mt-2 flex items-center justify-center gap-2">
                      {(webRTC.status === "init" ||
                        webRTC.status === "error") && (
                        <button
                          onClick={() => {
                            setShowConfig(true);
                            webRTC.start();
                          }}
                          className="px-3 py-1.5 text-[#ABABAB] hover:text-[#FB4E0B] text-[10px] font-bold uppercase tracking-wider transition-colors font-yantramanav"
                        >
                          Configure Agent
                        </button>
                      )}
                      {webRTC.status === "connected" && started && (
                        <MicrophoneButton stream={webRTC.micStream} />
                      )}
                    </div>
                  </div>
                </div>

                {/* Waveform */}
                <div className="mt-2 flex h-10 items-center justify-center gap-0.5">
                  {webRTC.status === "connected" && started ? (
                    <AudioWaveForm
                      streamOrTrack={webRTC.stream}
                      lineColor="#FB4E0B"
                      backgroundColor="transparent"
                      height={40}
                      width={250}
                    />
                  ) : (
                    <div className="flex h-10 items-center justify-center gap-0.5 opacity-30">
                      {Array.from({ length: 28 }).map((_, i) => (
                        <div
                          key={i}
                          className="wave-bar rounded-full bg-[#FB4E0B]"
                          style={{
                            width: "2px",
                            height: `${12 + (Math.sin(i) + 1.2) * 4}px`,
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Agent Configuration */}
            {webRTC.status === "connected" && (
              <div className="border border-[#E6E5E5] bg-white p-5 space-y-4">
                <div className="font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                  Agent Configuration
                </div>
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
                {!started && showConfig && hasSystemPrompt && (
                  <div className="mt-4 border-t border-[#E6E5E5] pt-4 space-y-3">
                    <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#808080] font-yantramanav">
                      System Prompt Context
                    </div>
                    <div className="space-y-4 max-h-60 overflow-y-auto pr-1">
                      {currentPrompts.map((p, idx) => (
                        <div key={`${p.role}-${idx}`} className="h-44">
                          <div className="text-[10px] font-semibold uppercase text-gray-500 mb-1 font-yantramanav">
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
                    <button
                      className="w-full mt-3 bg-[#FB4E0B] text-white py-2 rounded font-bold text-xs uppercase tracking-wider font-yantramanav hover:bg-[#e03a00] transition-colors"
                      onClick={() => {
                        const ch = webRTC.dataChannel as RTCDataChannel | null;
                        const promptData = promptsPayload();
                        if (hasSystemPrompt && ch && promptData.length > 0) {
                          ch.send(
                            JSON.stringify({
                              id: "prompt-save",
                              label: "rtvi-ai",
                              type: "client-message",
                              data: { t: "context_reset", d: promptData },
                            }),
                          );
                        }
                        setShowConfig(false);
                        toast.success("Configuration saved");
                      }}
                    >
                      Save Configuration
                    </button>
                  </div>
                )}
              </div>
            )}
          </aside>
          {/* --- MAIN --- */}
          <main className="col-span-12 lg:col-span-6">
            <div className="bg-white border border-[#E6E5E5] min-h-[640px] flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between px-7 py-5 border-b border-[#E6E5E5]">
                <div>
                  <div className="text-[10px] font-bold tracking-[0.18em] text-[#808080] uppercase font-['Yantramanav']">
                    Live Transcript
                  </div>

                  <div className="mt-[2px] text-[20px] font-bold text-black font-['Yantramanav']">
                    Aria → Nathan Reeves
                  </div>
                </div>

                <div className="flex items-center gap-2 text-[11px] text-[#808080] font-['Yantramanav']">
                  <Volume2 className="w-[13px] h-[13px]" />
                  {selectedVoice ? `Riva TTS · ${selectedVoice}` : ""}
                </div>
              </div>

              {/* Dynamic Transcript Area */}
              <div className="flex-1 p-7 overflow-y-auto max-h-[520px]">
                <AudioStream
                  streamOrTrack={
                    webRTC.status === "connected" ? webRTC.stream : null
                  }
                />
                {webRTC.status === "connected" && started ? (
                  <Transcripts
                    dataChannel={
                      webRTC.status === "connected" ? webRTC.dataChannel : null
                    }
                  />
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center text-[#808080] font-yantramanav py-20">
                    <Phone className="w-12 h-12 text-[#E6E5E5] mb-4 animate-pulse" />
                    <p className="text-lg font-medium text-black font-yantramanav">
                      Awaiting Connection
                    </p>
                    <p className="text-sm mt-1 max-w-xs text-gray-500 font-yantramanav">
                      Connect the agent and start a call session on the left to
                      begin the conversation.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </main>
          {/* --- RIGHT --- */}
          <aside className="col-span-12 lg:col-span-3 space-y-4">
            <div className="border border-[#E6E5E5] bg-white p-5">
              <div className="mb-4 text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                Live Signals
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Brain size={16} className="text-[#FB4E0B]" />
                    <span className="text-sm text-[#666666]">Intent</span>
                  </div>
                  <span className="text-sm font-medium text-[#111111]">
                    ---
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity size={16} className="text-green-600" />
                    <span className="text-sm text-[#666666]">Sentiment</span>
                  </div>
                  <span className="text-sm font-medium text-[#111111]">
                    Neutral
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-green-600" />
                    <span className="text-sm text-[#666666]">
                      Promise to Pay
                    </span>
                  </div>
                  <span className="text-sm font-medium text-green-600">
                    15 June 2026
                  </span>
                </div>
              </div>
            </div>

            {/* Stack Latency */}
            <div className="border border-[#E6E5E5] bg-white p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                  Stack Latency
                </div>
                <Gauge className="h-[13px] w-[13px] text-[#808080]" />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-yantramanav text-[12px] text-[#808080]">
                    Nemotron ASR
                  </span>
                  <span className="font-yantramanav text-[12px] font-semibold text-[#000000]">
                    {latencies.asr} ms
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="font-yantramanav text-[12px] text-[#808080]">
                    Nemotron LLM
                  </span>
                  <span className="font-yantramanav text-[12px] font-semibold text-[#000000]">
                    {latencies.llm} ms
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="font-yantramanav text-[12px] text-[#808080]">
                    Nemotron TTS
                  </span>
                  <span className="font-yantramanav text-[12px] font-semibold text-[#000000]">
                    {latencies.tts} ms
                  </span>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-[#E6E5E5] pt-3">
                <div className="font-yantramanav text-[11px] text-[#808080]">
                  Total turn
                </div>
                <div className="font-yantramanav text-[14px] font-bold text-[#000000]">
                  {totalLatency} ms
                </div>
              </div>
            </div>

            {/* Token Usage */}
            <div className="border border-[#E6E5E5] bg-white p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                  Token Usage
                </div>
                <Cpu className="h-[13px] w-[13px] text-[#808080]" />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-yantramanav text-[12px] text-[#808080]">
                    Prompt
                  </span>
                  <span className="font-yantramanav text-[12px] font-semibold text-[#000000]">
                    {tokenUsage.prompt_tokens.toLocaleString()}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="font-yantramanav text-[12px] text-[#808080]">
                    Completion
                  </span>
                  <span className="font-yantramanav text-[12px] font-semibold text-[#000000]">
                    {tokenUsage.completion_tokens.toLocaleString()}
                  </span>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-[#E6E5E5] pt-3">
                <div className="font-yantramanav text-[11px] text-[#808080]">
                  Total
                </div>
                <div className="font-yantramanav text-[14px] font-bold text-[#000000]">
                  {tokenUsage.total_tokens.toLocaleString()}
                </div>
              </div>
            </div>

            {/* Compliance */}
            <div className="border border-[#E6E5E5] bg-white p-5">
              <div className="mb-3 flex items-center gap-1.5 font-yantramanav text-[10px] font-bold uppercase tracking-[0.18em] text-[#808080]">
                <ShieldCheck className="h-3 w-3" />
                Compliance
              </div>

              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] text-[#333333]">
                    PII redaction on logs
                  </span>
                  <span className="text-[12px] font-medium text-green-600">
                    ✓
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[12px] text-[#333333]">
                    Consent to record captured
                  </span>
                  <span className="text-[12px] font-medium text-green-600">
                    ✓
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[12px] text-[#333333]">
                    Call window: 08:00–19:00
                  </span>
                  <span className="text-[12px] font-medium text-green-600">
                    ✓
                  </span>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </section>

      {/* ============== PIPELINE VIZ ============== */}
      <section className="max-w-350 mx-auto px-8 pb-20">
        <div className="grid grid-cols-12 gap-8 items-start mb-8">
          <div className="col-span-12 lg:col-span-5">
            <div className="flex items-center gap-3 mb-4">
              <span className="inline-block w-8 h-[2px] bg-[#FB4E0B]" />

              <span className="text-[11px] font-bold tracking-[0.22em] text-[#808080] uppercase font-yantramanav">
                The pipeline, in motion
              </span>
            </div>

            <h2 className="font-yantramanav font-light text-[38px] leading-[1.15] tracking-[-0.01em] text-black">
              Every spoken turn flows through{" "}
              <span className="font-bold text-[#FB4E0B]">six</span>{" "}
              NVIDIA-accelerated stages.
            </h2>
          </div>

          <div className="col-span-12 lg:col-span-7 lg:pt-10">
            <p className="text-[15px] leading-[1.7] font-light text-[#414141] font-yantramanav">
              When the customer speaks, audio streams into Riva for
              transcription. NeMo classifiers infer intent and sentiment in
              parallel. The retrieval layer pulls live policy context. An LLM
              hosted on NVIDIA NIM reasons over the full state and drafts a
              response, which Riva TTS renders in a natural neural voice with
              compliance guardrails at every boundary.
            </p>
          </div>
        </div>
        <div className="bg-white border border-[#E6E5E5] p-8">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
            {STAGES.map((stage, index) => {
              const Icon = stage.icon;

              return (
                <div
                  key={stage.id}
                  className="relative flex flex-col items-center text-center px-2 py-2"
                >
                  {/* Connector Line */}
                  {index < STAGES.length - 1 && (
                    <div className="hidden lg:block absolute top-7 left-[60%] w-full h-[2px] bg-gradient-to-r from-[#FB4E0B] via-[#FF9B6B] to-[#FFE1D3] z-0" />
                  )}

                  {/* Icon */}
                  <div className="relative z-10 w-14 h-14 flex items-center justify-center bg-[#FB4E0B] text-white">
                    <Icon className="w-5 h-5" />
                  </div>

                  <div className="mt-2 text-[10px] text-[#ABABAB] tracking-[0.08em] font-yantramanav">
                    0{stage.id}
                  </div>

                  <div className="mt-1 text-[13px] font-medium leading-[1.3] text-black font-yantramanav">
                    {stage.label}
                  </div>

                  <div className="mt-1 text-[11px] leading-[1.4] text-[#808080] font-yantramanav">
                    {stage.detail}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <footer className="border-t-[3px] border-[#FB4E0B] bg-black text-white">
        <div className="max-w-350 mx-auto px-8 py-16">
          <div className="grid grid-cols-12 gap-8">
            {/* Left Content */}
            <div className="col-span-12 lg:col-span-5">
              <div className="flex items-center gap-3 mb-4">
                <span className="inline-block w-8 h-0.5 bg-[#FB4E0B]" />

                <span className="text-[11px] font-bold tracking-[0.22em] text-[#808080] uppercase font-yantramanav">
                  What this unlocks
                </span>
              </div>

              <h3 className="font-yantramanav font-light text-[38px] leading-[1.2] text-white">
                A contact centre that thinks, remembers, and{" "}
                <span className="font-bold text-[#FB4E0B]">complies</span>.
              </h3>

              <p className="mt-5 max-w-100 text-[14px] leading-[1.7] font-light text-[#808080] font-yantramanav">
                Deployed on your infrastructure or ours, the agent learns your
                policies, speaks your customers' language, and hands off
                gracefully to human agents when nuance is needed.
              </p>
            </div>

            {/* Outcomes Grid */}

            <div className="col-span-12 lg:col-span-7 grid grid-cols-2 md:grid-cols-3 gap-6">
              {outcomes.map((item, index) => (
                <div key={index} className="p-4 rounded-lg">
                  <h4 className="text-[#FB4E0B] text-3xl font-bold">
                    {item.k}
                  </h4>
                  <p className="text-sm text-[#808080] mt-2">{item.v}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="mt-14 pt-6 border-t border-[#414141] flex flex-wrap items-center justify-between gap-4">
            <div className="text-[11px] tracking-[0.04em] text-[#414141] font-yantramanav">
              EXL Service · Bengaluru
            </div>

            <div className="flex items-center gap-6 text-[12px] text-[#808080] font-yantramanav">
              {["Riva", "NIM", "NeMo", "Triton", "cuVS"].map((t, i, arr) => (
                <React.Fragment key={t}>
                  <span>{t}</span>
                  {i < arr.length - 1 && (
                    <span className="text-[#414141]">·</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </footer>

      <Toaster position="top-right" />
    </div>
  );
}

export default App;
