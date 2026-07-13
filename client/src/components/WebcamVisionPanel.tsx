// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { getWebcamConfig, uploadWebcamCapture, uploadWebcamFrame, type WebcamConfig } from "../api";
import { isRecord, numberField, stringField } from "../utils";

type WebcamStatus = "idle" | "starting" | "live" | "uploading" | "error";
type NormalizedWebcamConfig = Required<WebcamConfig>;
type WebcamUploadState = {
  mode: string;
  label: string;
};
type VisualControlIntent = "none" | "greet" | "stop" | "continue" | "down";
type VisualControl = {
  intent: VisualControlIntent;
  confidence: number;
  reason: string;
};
type WebcamAgentUpdate = {
  observation: string;
  eventReason: string;
  focus: string;
  visualControl: VisualControl;
  propagated: boolean;
  createdAt: string;
};
type WebcamControlUpdate = {
  action: string;
  state: string;
  visualControl: VisualControl;
  createdAt: string;
};

const DEFAULT_WEBCAM_CONFIG: Required<WebcamConfig> = {
  sample_interval_seconds: 1.5,
  frame_max_width: 640,
  jpeg_quality: 0.7,
  initial_upload_enabled: true,
  initial_upload_delay_ms: 700,
};
const IDLE_UPLOAD_STATE: WebcamUploadState = { mode: "idle", label: "" };
const HIGHRES_JPEG_QUALITY = 0.92;
const DEFAULT_CHUNK_SECONDS = 8;

function normalizeWebcamConfig(config: WebcamConfig): NormalizedWebcamConfig {
  return {
    sample_interval_seconds: Math.max(
      0.5,
      Number(config.sample_interval_seconds || DEFAULT_WEBCAM_CONFIG.sample_interval_seconds)
    ),
    frame_max_width: Math.max(160, Number(config.frame_max_width || DEFAULT_WEBCAM_CONFIG.frame_max_width)),
    jpeg_quality: Math.min(0.95, Math.max(0.1, Number(config.jpeg_quality || DEFAULT_WEBCAM_CONFIG.jpeg_quality))),
    initial_upload_enabled: config.initial_upload_enabled ?? DEFAULT_WEBCAM_CONFIG.initial_upload_enabled,
    initial_upload_delay_ms: Math.max(
      0,
      Number(config.initial_upload_delay_ms ?? DEFAULT_WEBCAM_CONFIG.initial_upload_delay_ms)
    ),
  };
}

function canvasToJpegBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Could not encode webcam frame"));
      },
      "image/jpeg",
      quality
    );
  });
}

function visualControlFromMessage(message: Record<string, unknown>): VisualControl {
  const control = isRecord(message.visual_control) ? message.visual_control : {};
  const intent = stringField(control, "intent");
  const known = intent === "greet" || intent === "stop" || intent === "continue" || intent === "down";
  return {
    intent: known ? (intent as VisualControlIntent) : "none",
    confidence: Math.min(1, Math.max(0, numberField(control, "confidence"))),
    reason: stringField(control, "reason"),
  };
}

function proactiveActionLabel(update: WebcamControlUpdate): string {
  switch (update.state) {
    case "greeted":
      return "Greeted you back";
    case "barged_in":
      return "Stopped on your signal";
    case "resumed":
      return "Resumed where I left off";
    case "acknowledged":
      return "Saw your thumbs-up";
    case "noted":
      return "Noted your thumbs-down";
    default:
      return update.action ? `Reacted to your ${update.action} gesture` : "";
  }
}

export function WebcamVisionPanel({ sessionId }: Readonly<{ sessionId: string }>) {
  const client = usePipecatClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const initialUploadTimeoutRef = useRef<number | null>(null);
  const inFlightRef = useRef(0);
  const captureGenerationRef = useRef(0);
  const uploadModeRef = useRef("idle");
  const configRef = useRef<NormalizedWebcamConfig | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [chunkSeconds, setChunkSeconds] = useState(DEFAULT_CHUNK_SECONDS);
  const [uploadState, setUploadState] = useState<WebcamUploadState>(IDLE_UPLOAD_STATE);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState("");
  const [agentUpdate, setAgentUpdate] = useState<WebcamAgentUpdate | null>(null);
  const [controlUpdate, setControlUpdate] = useState<WebcamControlUpdate | null>(null);
  const [captureState, setCaptureState] = useState<"idle" | "capturing" | "analyzing">("idle");
  const captureClearTimerRef = useRef<number | null>(null);
  const captureAttachmentIdRef = useRef("");
  const captureInProgressRef = useRef(false);

  const clearCaptureProgress = useCallback(() => {
    if (captureClearTimerRef.current !== null) {
      window.clearTimeout(captureClearTimerRef.current);
      captureClearTimerRef.current = null;
    }
    captureAttachmentIdRef.current = "";
    captureInProgressRef.current = false;
    setCaptureState("idle");
  }, []);

  const cleanupStream = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (initialUploadTimeoutRef.current !== null) {
      window.clearTimeout(initialUploadTimeoutRef.current);
      initialUploadTimeoutRef.current = null;
    }
    captureGenerationRef.current += 1;
    clearCaptureProgress();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    inFlightRef.current = 0;
    uploadModeRef.current = "idle";
    setUploadState(IDLE_UPLOAD_STATE);
    configRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, [clearCaptureProgress]);

  const sendWebcamState = useCallback((isEnabled: boolean) => {
    if (!client || client.state !== "ready") return;
    try {
      client.sendClientMessage("webcam-state", { enabled: isEnabled });
    } catch (err) {
      console.warn("Could not send webcam state update:", err);
    }
  }, [client]);

  const sendWebcamChunk = useCallback((seconds: number) => {
    if (!client || client.state !== "ready") return false;
    try {
      client.sendClientMessage("webcam-chunk", { seconds });
      return true;
    } catch (err) {
      console.warn("Could not send webcam chunk update:", err);
      return false;
    }
  }, [client]);

  const handleStreamEnded = useCallback(() => {
    cleanupStream();
    sendWebcamState(false);
    setEnabled(false);
    setControlUpdate(null);
    setStatus("idle");
  }, [cleanupStream, sendWebcamState]);

  const stop = useCallback(() => {
    cleanupStream();
    sendWebcamState(false);
    setEnabled(false);
    setControlUpdate(null);
    setStatus("idle");
  }, [cleanupStream, sendWebcamState]);

  const captureFrame = useCallback(
    async (
      config: NormalizedWebcamConfig,
      { highRes = false, captureRequestId = "" }: { highRes?: boolean; captureRequestId?: string } = {},
    ): Promise<string | null> => {
      if (!sessionId || (!highRes && inFlightRef.current > 0)) return null;
      const generation = captureGenerationRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !video.videoWidth || !video.videoHeight) return null;

      const scale = highRes ? 1 : Math.min(1, config.frame_max_width / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const context = canvas.getContext("2d");
      if (!context) return null;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      inFlightRef.current += 1;
      setStatus("uploading");
      try {
        const blob = await canvasToJpegBlob(canvas, highRes ? HIGHRES_JPEG_QUALITY : config.jpeg_quality);
        const uploaded = await (highRes
          ? uploadWebcamCapture(sessionId, blob, captureRequestId)
          : uploadWebcamFrame(sessionId, blob));
        if (generation !== captureGenerationRef.current) return null;
        setError("");
        return (isRecord(uploaded) ? stringField(uploaded, "id") : "") || null;
      } catch (err) {
        if (generation !== captureGenerationRef.current) return null;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Webcam upload failed");
        return null;
      } finally {
        if (generation === captureGenerationRef.current) {
          inFlightRef.current -= 1;
          if (inFlightRef.current === 0) setStatus((prev) => (prev === "error" ? prev : "live"));
        }
      }
    },
    [sessionId]
  );

  const start = useCallback(async () => {
    if (enabled) {
      stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setError("Browser webcam capture is not available.");
      return;
    }

    setStatus("starting");
    setError("");
    try {
      const [config, stream] = await Promise.all([
        getWebcamConfig().then(normalizeWebcamConfig).catch(() => DEFAULT_WEBCAM_CONFIG),
        navigator.mediaDevices.getUserMedia({ video: true, audio: false }),
      ]);
      streamRef.current = stream;
      configRef.current = config;
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", handleStreamEnded, { once: true });
      });
      setEnabled(true);
      setStatus("live");
      sendWebcamState(true);
      sendWebcamChunk(chunkSeconds);

      if (config.initial_upload_enabled) {
        initialUploadTimeoutRef.current = window.setTimeout(() => {
          initialUploadTimeoutRef.current = null;
          void captureFrame(config);
        }, config.initial_upload_delay_ms);
      }
    } catch (err) {
      cleanupStream();
      setEnabled(false);
      setStatus("error");
      setError(err instanceof Error ? err.message : "Could not start webcam");
    }
  }, [captureFrame, chunkSeconds, cleanupStream, enabled, handleStreamEnded, sendWebcamChunk, sendWebcamState, stop]);

  const stopFrameUploads = useCallback((mode = "") => {
    if (mode && uploadModeRef.current !== mode) return;
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    uploadModeRef.current = "idle";
    setUploadState(IDLE_UPLOAD_STATE);
  }, []);

  const startFrameUploads = useCallback((mode: string, label: string, intervalMs: number) => {
    stopFrameUploads();
    const config = configRef.current;
    if (!config) return;
    const normalizedMode = mode || "server";
    const safeIntervalMs = Math.max(250, intervalMs);
    uploadModeRef.current = normalizedMode;
    setUploadState({
      mode: normalizedMode,
      label: label || `capturing (${normalizedMode.replace(/_/g, " ")})`,
    });
    void captureFrame(config);
    intervalRef.current = window.setInterval(() => void captureFrame(config), safeIntervalMs);
  }, [captureFrame, stopFrameUploads]);

  const captureFrameOnce = useCallback(() => {
    const config = configRef.current;
    if (config) void captureFrame(config);
  }, [captureFrame]);

  const captureHighRes = useCallback(async (requestId: string) => {
    if (!requestId || captureInProgressRef.current) return;
    const generation = captureGenerationRef.current;
    const config = configRef.current;
    if (!config) return;
    captureInProgressRef.current = true;
    if (captureClearTimerRef.current !== null) window.clearTimeout(captureClearTimerRef.current);
    setCaptureState("capturing");
    const capturedId = await captureFrame(config, { highRes: true, captureRequestId: requestId });
    if (generation !== captureGenerationRef.current) return;
    if (!capturedId) {
      captureInProgressRef.current = false;
      setCaptureState("idle");
      return;
    }
    captureAttachmentIdRef.current = capturedId;
    setCaptureState("analyzing");
    captureClearTimerRef.current = window.setTimeout(() => {
      if (generation !== captureGenerationRef.current) return;
      captureAttachmentIdRef.current = "";
      captureInProgressRef.current = false;
      setCaptureState("idle");
    }, 60000);
  }, [captureFrame]);

  useEffect(() => {
    if (!enabled || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
    void videoRef.current.play().catch((err) => {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Could not play webcam preview");
    });
  }, [enabled]);

  useEffect(() => stop, [stop]);

  useRTVIClientEvent(
    RTVIEvent.Disconnected,
    useCallback(() => {
      cleanupStream();
      setEnabled(false);
      setStatus("idle");
      setControlUpdate(null);
    }, [cleanupStream])
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((message: unknown) => {
      if (!isRecord(message)) return;
      const messageType = stringField(message, "type");
      if (messageType === "webcam-upload-control") {
        const action = stringField(message, "action");
        const mode = stringField(message, "mode");
        const intervalMs = numberField(message, "interval_ms");
        if (action === "repeat" || message.active === true) {
          const config = configRef.current;
          const fallbackIntervalMs = config ? config.sample_interval_seconds * 1000 : 1500;
          startFrameUploads(mode, stringField(message, "label"), intervalMs || fallbackIntervalMs);
        } else if (action === "once") {
          captureFrameOnce();
        } else {
          stopFrameUploads(mode === "idle" ? "" : mode);
        }
        return;
      }
      if (messageType === "webcam-capture-request") {
        void captureHighRes(stringField(message, "request_id"));
        return;
      }
      if (messageType === "agent-task-update") {
        const status = stringField(message, "status");
        if (status !== "done" && status !== "error") return;
        const attachment = message.attachment;
        const attachmentId = isRecord(attachment) ? stringField(attachment, "id") : "";
        if (attachmentId && attachmentId === captureAttachmentIdRef.current) {
          clearCaptureProgress();
        }
        return;
      }
      if (messageType === "webcam-control-update") {
        setControlUpdate({
          action: stringField(message, "action"),
          state: stringField(message, "state"),
          visualControl: visualControlFromMessage(message),
          createdAt: new Date().toISOString(),
        });
        return;
      }
      if (messageType !== "webcam-agent-update") return;
      const observation = stringField(message, "observation");
      if (!observation) return;
      setAgentUpdate({
        observation,
        eventReason: stringField(message, "event_reason"),
        focus: stringField(message, "focus"),
        visualControl: visualControlFromMessage(message),
        propagated: message.propagated === true,
        createdAt: new Date().toISOString(),
      });
    }, [captureFrameOnce, captureHighRes, clearCaptureProgress, startFrameUploads, stopFrameUploads])
  );

  const statusLabel = !enabled
    ? "Camera off — not capturing"
    : uploadState.mode !== "idle"
      ? uploadState.label
      : "Watching the live view";
  const agentUpdateStale = Boolean(agentUpdate && !enabled);
  const proactiveLabel = controlUpdate ? proactiveActionLabel(controlUpdate) : "";

  return (
    <div className={`webcam-control webcam-control-${status} ${enabled ? "webcam-control-enabled" : "webcam-control-off"}`}>
      <div className="webcam-control-main">
        <button
          className="btn-icon webcam-icon-button"
          type="button"
          onClick={start}
          title={enabled ? "Stop webcam vision" : "Enable webcam vision"}
          aria-label={enabled ? "Stop webcam vision" : "Enable webcam vision"}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M4 8.5A2.5 2.5 0 0 1 6.5 6h7A2.5 2.5 0 0 1 16 8.5v7a2.5 2.5 0 0 1-2.5 2.5h-7A2.5 2.5 0 0 1 4 15.5v-7Z" />
            <path d="m16 10 4-2.5v9L16 14" />
            <path d="M8 10h4" />
          </svg>
        </button>
        <div className="webcam-control-text">
          <strong>
            <span
              className={`webcam-status-dot${enabled ? " webcam-status-dot-live" : ""}`}
              aria-hidden="true"
            />
            {enabled ? "Vision enabled" : "Webcam off"}
          </strong>
          <small className="webcam-status-label">{statusLabel}</small>
          {error && <small className="webcam-error">{error}</small>}
        </div>
        <label className="webcam-chunk-control" title="Recent seconds sent to the model as one continuous video">
          <span>Chunk</span>
          <select
            value={chunkSeconds}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              const seconds = Number(e.target.value);
              if (sendWebcamChunk(seconds)) setChunkSeconds(seconds);
            }}
          >
            <option value={2}>2s</option>
            <option value={4}>4s</option>
            <option value={8}>8s</option>
            <option value={15}>15s</option>
            <option value={30}>30s</option>
          </select>
        </label>
      </div>
      {enabled && (
        <div className="webcam-preview">
          <video ref={videoRef} muted playsInline />
          {captureState !== "idle" && (
            <div className={`webcam-capture-overlay webcam-capture-overlay-${captureState}`}>
              <span className="webcam-capture-spinner" aria-hidden="true" />
              <span>
                {captureState === "capturing"
                  ? "Capturing a high-resolution snapshot — hold on..."
                  : "Reading the high-resolution snapshot..."}
              </span>
            </div>
          )}
        </div>
      )}
      {!enabled && !agentUpdate && (
        <div className="webcam-off-state">
          <strong>Camera is off</strong>
          <small>The agent is not receiving live webcam frames.</small>
        </div>
      )}
      {agentUpdate && (
        <div className={`webcam-agent-update ${agentUpdateStale ? "webcam-agent-update-stale" : ""}`}>
          <div className="webcam-agent-update-header">
            <strong>{agentUpdateStale ? "Last webcam summary" : "Regular webcam summary"}</strong>
            <small>{new Date(agentUpdate.createdAt).toLocaleTimeString()}</small>
          </div>
          <small>
            {agentUpdateStale
              ? "Past context only; webcam is off now"
              : agentUpdate.propagated
                ? "Shared with agent bus"
                : "UI only, no meaningful scene change"}
          </small>
          <div className={`webcam-focus-note webcam-focus-note-${agentUpdate.focus ? "steered" : "generic"}`}>
            <span className="webcam-focus-tag">Steering</span>
            <small>{agentUpdate.focus || "Generic — no conversational focus set"}</small>
          </div>
          <p>{agentUpdate.observation}</p>
          {agentUpdate.eventReason && (
            <small>{agentUpdate.propagated ? "Why it propagated" : "Summary note"}: {agentUpdate.eventReason}</small>
          )}
          <div className={`webcam-visual-control webcam-visual-control-${agentUpdate.visualControl.intent}`}>
            <span>{agentUpdate.visualControl.intent}</span>
            <strong>{Math.round(agentUpdate.visualControl.confidence * 100)}%</strong>
            {agentUpdate.visualControl.reason && <small>{agentUpdate.visualControl.reason}</small>}
          </div>
          {controlUpdate && proactiveLabel && (
            <div className={`webcam-visual-control webcam-visual-control-${controlUpdate.visualControl.intent}`}>
              <span>{proactiveLabel}</span>
              <small>{new Date(controlUpdate.createdAt).toLocaleTimeString()}</small>
            </div>
          )}
        </div>
      )}
      <canvas ref={canvasRef} hidden />
    </div>
  );
}
