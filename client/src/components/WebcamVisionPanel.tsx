// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useEffect, useRef, useState } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { getWebcamConfig, uploadWebcamFrame, type WebcamConfig } from "../api";
import { isRecord, numberField, stringField } from "../utils";

type WebcamStatus = "idle" | "starting" | "live" | "uploading" | "error";
type NormalizedWebcamConfig = Required<WebcamConfig>;
type WebcamUploadState = {
  mode: string;
  label: string;
};
type VisualControlIntent = "none" | "stop" | "continue";
type VisualControl = {
  intent: VisualControlIntent;
  confidence: number;
  reason: string;
};
type WebcamAgentUpdate = {
  observation: string;
  eventReason: string;
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
  return {
    intent: intent === "stop" || intent === "continue" ? intent : "none",
    confidence: Math.min(1, Math.max(0, numberField(control, "confidence"))),
    reason: stringField(control, "reason"),
  };
}

export function WebcamVisionPanel({ sessionId }: Readonly<{ sessionId: string }>) {
  const client = usePipecatClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const initialUploadTimeoutRef = useRef<number | null>(null);
  const uploadingRef = useRef(false);
  const uploadModeRef = useRef("idle");
  const configRef = useRef<NormalizedWebcamConfig | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [uploadState, setUploadState] = useState<WebcamUploadState>(IDLE_UPLOAD_STATE);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState("");
  const [agentUpdate, setAgentUpdate] = useState<WebcamAgentUpdate | null>(null);
  const [controlUpdate, setControlUpdate] = useState<WebcamControlUpdate | null>(null);

  const cleanupStream = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (initialUploadTimeoutRef.current !== null) {
      window.clearTimeout(initialUploadTimeoutRef.current);
      initialUploadTimeoutRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    uploadingRef.current = false;
    uploadModeRef.current = "idle";
    setUploadState(IDLE_UPLOAD_STATE);
    configRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const sendWebcamState = useCallback((isEnabled: boolean) => {
    if (!client || client.state !== "ready") return;
    try {
      client.sendClientMessage("webcam-state", { enabled: isEnabled });
    } catch (err) {
      console.warn("Could not send webcam state update:", err);
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

  const captureFrame = useCallback(async (config: NormalizedWebcamConfig) => {
    if (!sessionId || uploadingRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

    const scale = Math.min(1, config.frame_max_width / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    uploadingRef.current = true;
    setStatus("uploading");
    try {
      const blob = await canvasToJpegBlob(canvas, config.jpeg_quality);
      await uploadWebcamFrame(sessionId, blob);
      setStatus("live");
      setError("");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Webcam upload failed");
    } finally {
      uploadingRef.current = false;
    }
  }, [sessionId]);

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
  }, [captureFrame, cleanupStream, enabled, handleStreamEnded, sendWebcamState, stop]);

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
        visualControl: visualControlFromMessage(message),
        propagated: message.propagated === true,
        createdAt: new Date().toISOString(),
      });
    }, [captureFrameOnce, startFrameUploads, stopFrameUploads])
  );

  const statusLabel = enabled
    ? status === "uploading"
      ? "Updating latest frame..."
      : uploadState.mode !== "idle"
        ? uploadState.label
        : "ready for server capture"
    : "camera is off; not capturing";
  const agentUpdateStale = Boolean(agentUpdate && !enabled);

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
          <strong>{enabled ? "Vision enabled" : "Webcam off"}</strong>
          <small>{statusLabel}</small>
          {error && <small className="webcam-error">{error}</small>}
        </div>
      </div>
      {enabled && (
        <div className="webcam-preview">
          <video ref={videoRef} muted playsInline />
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
          <p>{agentUpdate.observation}</p>
          {agentUpdate.eventReason && (
            <small>{agentUpdate.propagated ? "Why it propagated" : "Summary note"}: {agentUpdate.eventReason}</small>
          )}
          <div className={`webcam-visual-control webcam-visual-control-${agentUpdate.visualControl.intent}`}>
            <span>{agentUpdate.visualControl.intent}</span>
            <strong>{Math.round(agentUpdate.visualControl.confidence * 100)}%</strong>
            {agentUpdate.visualControl.reason && <small>{agentUpdate.visualControl.reason}</small>}
          </div>
          {controlUpdate && (
            <small>
              Last control action: {controlUpdate.action || "none"} | state {controlUpdate.state || "listening"} |{" "}
              {new Date(controlUpdate.createdAt).toLocaleTimeString()}
            </small>
          )}
        </div>
      )}
      <canvas ref={canvasRef} hidden />
    </div>
  );
}
