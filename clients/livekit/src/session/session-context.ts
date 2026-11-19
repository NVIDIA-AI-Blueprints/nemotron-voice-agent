// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from "react";
import type { Model, VoiceList } from "../api";
import type { EventItem } from "../lib/timeline";
import type { TranscriptSegment } from "./useTranscript";

export type SessionPhase = "idle" | "requesting" | "connecting" | "connected";

export interface SessionState {
  phase: SessionPhase;
  isConnected: boolean;
  isConnecting: boolean;
  /** Whether the session choices are fixed: from the token request until the session ends. */
  isLocked: boolean;
  /** Whether Connect can start a session: idle, with the client config, services, and prompts loaded. */
  canConnect: boolean;
  roomName: string;
  serverUrl: string;
  error: string;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  /** The agent events of this session, for the conversation. */
  events: EventItem[];
  /** The transcription segments of the current room, kept while the conversation panel remounts. */
  segments: TranscriptSegment[];
  /** The models in use, as the agent reports them when the session starts. Null when no session is connected. */
  models: Model[] | null;
  /** The voices that the agent offers in this session. */
  voices: VoiceList | null;
  /** Ask the agent to use another voice from its next reply. */
  setVoice: (voiceId: string) => void;
}

export const SessionContext = createContext<SessionState | null>(null);

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be inside SessionProvider");
  return ctx;
}
