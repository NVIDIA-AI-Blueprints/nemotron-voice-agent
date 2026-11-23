// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { RoomAudioRenderer, RoomContext, useConnectionState, useDataChannel } from "@livekit/components-react";
import { ConnectionState, DisconnectReason, MediaDeviceFailure, Room, RoomEvent } from "livekit-client";
import { isUnauthorized, queryClient, requestToken, useClientConfig } from "../api";
import { useApp } from "../context/useApp";
import { CONTROL_TOPIC } from "../lib/agentEvents";
import { resolveServerUrl } from "../lib/serverUrl";
import { buildTokenRequest } from "../lib/session";
import { SessionContext, type SessionPhase, type SessionState } from "./session-context";
import { useAgentEvents } from "./useAgentEvents";
import { useTranscriptionSegments } from "./useTranscript";

const newRoom = () => new Room({ adaptiveStream: true, dynacast: true });
const encoder = new TextEncoder();

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function microphoneError(error: unknown): string {
  return MediaDeviceFailure.getFailure(error) === MediaDeviceFailure.PermissionDenied
    ? "Microphone access was denied. Allow it in the browser, then Connect again."
    : "The microphone could not be opened. Check the input device, then Connect again.";
}

interface RoomSession {
  room: Room;
  roomName: string;
  error: string;
  setError: (message: string) => void;
  /** Join a new room. `isCurrent` turns false once a newer attempt or a disconnect supersedes this one. */
  join: (token: string, roomName: string, isCurrent: () => boolean) => Promise<void>;
  leave: () => Promise<void>;
}

/**
 * The LiveKit room of the current session.
 *
 * Each session joins a new room with a new token, so nothing from an earlier session leaks into the next one. Any
 * failure (the connection, or the microphone) leaves the room, so the agent does not wait for a user who never
 * arrived, and leaves the page ready for another try.
 */
function useRoomSession(serverUrl: string): RoomSession {
  const [room, setRoom] = useState(newRoom);
  const [roomName, setRoomName] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const ended = () => setRoomName("");
    room.on(RoomEvent.Disconnected, ended);
    return () => {
      room.off(RoomEvent.Disconnected, ended);
      room.disconnect().catch(() => undefined);
    };
  }, [room]);

  const join = useCallback(async (token: string, name: string, isCurrent: () => boolean) => {
    const next = newRoom();
    setRoom(next);
    setRoomName(name);
    const fail = async (message: string) => {
      await next.disconnect();
      if (!isCurrent()) return;
      setError(message);
      setRoomName("");
    };
    try {
      await next.connect(serverUrl, token);
    } catch (err) {
      return fail(`Connection failed: ${messageOf(err)}`);
    }
    if (!isCurrent()) return next.disconnect();
    try {
      await next.localParticipant.setMicrophoneEnabled(true);
    } catch (err) {
      return fail(microphoneError(err));
    }
  }, [serverUrl]);

  const leave = useCallback(() => {
    setRoomName("");
    return room.disconnect();
  }, [room]);

  return { room, roomName, error, setError, join, leave };
}

function SessionStateProvider({ session, serverUrl, children }: Readonly<{ session: RoomSession; serverUrl: string; children: ReactNode }>) {
  const { room, roomName, error, setError, join, leave } = session;
  const connectionState = useConnectionState(room);
  const { choices, ready, setLocked } = useApp();
  const agentEvents = useAgentEvents();
  const segments = useTranscriptionSegments(room);
  const { send } = useDataChannel(CONTROL_TOPIC);
  const [requesting, setRequesting] = useState(false);
  const connecting = useRef(false);
  const attempt = useRef(0);
  const lastFailure = useRef("");
  const { clear: clearEvents } = agentEvents;

  useEffect(() => {
    lastFailure.current = agentEvents.events.findLast((item) => item.entry.tone === "error")?.entry.text ?? "";
  }, [agentEvents.events]);

  useEffect(() => {
    // The agent deletes the room when the session cannot go on, and the idle page hides the conversation that says why.
    const ended = (reason?: DisconnectReason) => {
      if (reason === DisconnectReason.ROOM_DELETED) setError(lastFailure.current || "The agent ended the session.");
    };
    room.on(RoomEvent.Disconnected, ended);
    return () => {
      room.off(RoomEvent.Disconnected, ended);
    };
  }, [room, setError]);

  const connect = useCallback(async () => {
    if (connecting.current || !ready) return;
    connecting.current = true;
    const id = ++attempt.current;
    const isCurrent = () => attempt.current === id;
    setError("");
    setRequesting(true);
    try {
      let grant;
      try {
        grant = await requestToken(buildTokenRequest(choices));
      } catch (err) {
        if (!isCurrent()) return;
        if (isUnauthorized(err)) await queryClient.invalidateQueries({ queryKey: ["auth-status"] });
        setError(messageOf(err));
        return;
      } finally {
        if (isCurrent()) setRequesting(false);
      }
      if (!isCurrent()) return;
      clearEvents();
      await join(grant.token, grant.room, isCurrent);
    } finally {
      if (isCurrent()) connecting.current = false;
    }
  }, [choices, clearEvents, join, ready, setError]);

  const disconnect = useCallback(() => {
    attempt.current += 1;
    connecting.current = false;
    setRequesting(false);
    return leave();
  }, [leave]);

  const setVoice = useCallback((voiceId: string) => {
    send(encoder.encode(JSON.stringify({ type: "set_voice", voice: voiceId })), { reliable: true }).catch(
      (err: unknown) => console.warn("Could not send the voice change:", err),
    );
  }, [send]);

  let phase: SessionPhase = "idle";
  if (requesting) phase = "requesting";
  else if (roomName && connectionState === ConnectionState.Connected) phase = "connected";
  else if (roomName) phase = "connecting";
  const isLocked = phase !== "idle";

  useEffect(() => {
    setLocked(isLocked);
  }, [isLocked, setLocked]);

  const value = useMemo<SessionState>(() => ({
    phase,
    isConnected: phase === "connected",
    isConnecting: phase === "requesting" || phase === "connecting",
    isLocked,
    canConnect: ready && phase === "idle",
    roomName,
    serverUrl,
    error,
    connect,
    disconnect,
    events: agentEvents.events,
    segments,
    models: phase === "connected" ? agentEvents.models : null,
    voices: agentEvents.voices,
    setVoice,
  }), [phase, isLocked, ready, roomName, serverUrl, error, connect, disconnect, agentEvents.events, segments,
    agentEvents.models, agentEvents.voices, setVoice]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

/** The LiveKit room of the current session, and the session state for the page. */
export function SessionProvider({ children }: Readonly<{ children: ReactNode }>) {
  const { data: clientConfig } = useClientConfig();
  const serverUrl = resolveServerUrl(globalThis.location, clientConfig?.ws_url ?? "");
  const session = useRoomSession(serverUrl);

  return (
    <RoomContext.Provider value={session.room}>
      <SessionStateProvider session={session} serverUrl={serverUrl}>
        {children}
      </SessionStateProvider>
      <RoomAudioRenderer />
    </RoomContext.Provider>
  );
}
