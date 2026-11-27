// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient, useRTVIClientEvent } from "@pipecat-ai/client-react";
import { UploadSession, type UploadMeta, type UploadResult } from "../lib/uploads";

/** Returns `upload(blob, meta)`, which sends media over the session's RTVI connection. */
export function useUploader(): (blob: Blob, meta: UploadMeta) => Promise<UploadResult> {
  const client = usePipecatClient();
  const session = useMemo(
    () => new UploadSession((type, data) => {
      if (!client) throw new Error("Not connected");
      client.sendClientMessage(type, data);
    }),
    [client],
  );

  useRTVIClientEvent(RTVIEvent.ServerMessage, useCallback((message: unknown) => {
    session.handleServerMessage(message);
  }, [session]));
  useRTVIClientEvent(RTVIEvent.Disconnected, useCallback(() => session.abortAll(), [session]));
  useEffect(() => () => session.abortAll("Upload cancelled"), [session]);

  return useCallback((blob, meta) => session.upload(blob, meta), [session]);
}
