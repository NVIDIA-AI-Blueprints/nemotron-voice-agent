// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Chunked media uploads carried over the session's own RTVI connection (WebRTC data channel or
// WebSocket) instead of HTTP routes, so the bytes reach the worker that runs the bot at any worker
// count. Mirrors src/nemotron_voice_agent/pipecat/runtime/session_uploads.py on the server.

export type UploadTarget = "attachment" | "webcam-frame" | "webcam-capture";

export type UploadMeta = {
  target: UploadTarget;
  name: string;
  /** Media kind for attachments: image, audio or video. */
  kind?: string;
  /** Required for `webcam-capture`: the id from the server's capture request. */
  requestId?: string;
};

/** Stored-upload metadata the server returns, e.g. `{ id, bytes, ... }`. */
export type UploadResult = Record<string, unknown>;

export type SendClientMessage = (type: string, data: unknown) => void;

// Raw bytes per chunk; base64 inflates this by a third, which stays well below data-channel message limits.
export const UPLOAD_CHUNK_BYTES = 16 * 1024;
const ACK_TIMEOUT_MS = 30_000;
// Yield to the event loop after this many chunks so one large upload cannot starve audio and UI work.
const CHUNKS_PER_YIELD = 16;

type Waiter = {
  resolve: (result: UploadResult) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/**
 * Sends uploads as `upload-start`, `upload-chunk` and `upload-end` client messages and resolves each
 * upload from the server's `upload-ack` (stored) or `upload-error` server message. Feed every RTVI
 * server message to `handleServerMessage`.
 */
export class UploadSession {
  private readonly waiters = new Map<string, Waiter>();

  private readonly send: SendClientMessage;
  private readonly ackTimeoutMs: number;

  constructor(send: SendClientMessage, ackTimeoutMs = ACK_TIMEOUT_MS) {
    this.send = send;
    this.ackTimeoutMs = ackTimeoutMs;
  }

  /** Returns true when the message belonged to the upload protocol. */
  handleServerMessage(message: unknown): boolean {
    if (typeof message !== "object" || message === null) return false;
    const record = message as Record<string, unknown>;
    const uploadId = typeof record.upload_id === "string" ? record.upload_id : "";
    if (record.type === "upload-error") {
      this.settle(uploadId, (waiter) => waiter.reject(new Error(String(record.error ?? "Upload failed"))));
      return true;
    }
    if (record.type !== "upload-ack") return false;
    if (record.status === "stored") this.settle(uploadId, (waiter) => waiter.resolve(record));
    return true;
  }

  /** Fails every in-flight upload, e.g. when the connection drops. */
  abortAll(reason = "Connection closed"): void {
    for (const uploadId of [...this.waiters.keys()]) {
      this.settle(uploadId, (waiter) => waiter.reject(new Error(reason)));
    }
  }

  async upload(blob: Blob, meta: UploadMeta): Promise<UploadResult> {
    const uploadId = crypto.randomUUID();
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const stored = this.awaitStored(uploadId);
    // The server rejects bad uploads asynchronously; a rejection is surfaced through `stored`.
    stored.catch(() => undefined);

    this.send("upload-start", {
      upload_id: uploadId,
      target: meta.target,
      name: meta.name,
      content_type: blob.type,
      total_bytes: bytes.length,
      ...(meta.kind ? { kind: meta.kind } : {}),
      ...(meta.requestId ? { request_id: meta.requestId } : {}),
    });
    let sent = 0;
    for (let offset = 0, seq = 0; offset < bytes.length; offset += UPLOAD_CHUNK_BYTES, seq += 1) {
      if (!this.waiters.has(uploadId)) break; // already rejected by the server
      this.send("upload-chunk", {
        upload_id: uploadId,
        seq,
        data: toBase64(bytes.subarray(offset, offset + UPLOAD_CHUNK_BYTES)),
      });
      sent += 1;
      if (sent % CHUNKS_PER_YIELD === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (this.waiters.has(uploadId)) this.send("upload-end", { upload_id: uploadId });
    return stored;
  }

  private awaitStored(uploadId: string): Promise<UploadResult> {
    return new Promise<UploadResult>((resolve, reject) => {
      const timer = setTimeout(
        () => this.settle(uploadId, (waiter) => waiter.reject(new Error("Upload timed out"))),
        this.ackTimeoutMs,
      );
      this.waiters.set(uploadId, { resolve, reject, timer });
    });
  }

  private settle(uploadId: string, finish: (waiter: Waiter) => void): void {
    const waiter = this.waiters.get(uploadId);
    if (!waiter) return;
    this.waiters.delete(uploadId);
    clearTimeout(waiter.timer);
    finish(waiter);
  }
}
