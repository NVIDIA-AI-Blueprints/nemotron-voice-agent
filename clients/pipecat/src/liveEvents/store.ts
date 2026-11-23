// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Event log of the current call, and whether the call is on. It is one small store outside React, so it needs no
// provider in the shared app and it outlives the call's client, which the app replaces after a hang up: a call that has
// just ended stays on screen until the next one starts.
//
// A call begins at its connection, or at its first event when that comes first; it ends at the disconnect. The reducer is
// pure and the tests drive it directly; `liveEventsStore` wraps it for React (`useSyncExternalStore`).

import { parseLiveEvent } from "./liveEvent.ts";
import { EMPTY_LOG, logReducer, type LogState } from "./log.ts";

export interface CallLog {
  /** Which call this is: 0 before any call, then 1, 2, ... */
  call: number;
  /** "live" from the call's start until it ends; "ended" after; "idle" before any call. */
  phase: "idle" | "live" | "ended";
  log: LogState;
}

export const EMPTY_CALL_LOG: CallLog = { call: 0, phase: "idle", log: EMPTY_LOG };

export type CallLogAction =
  | { type: "connected" }
  | { type: "disconnected" }
  | { type: "message"; message: unknown }
  | { type: "clear" };

function begin(state: CallLog): CallLog {
  const call = state.call + 1;
  return { call, phase: "live", log: { ...EMPTY_LOG, session: String(call), phase: "live" } };
}

export function callLogReducer(state: CallLog, action: CallLogAction): CallLog {
  switch (action.type) {
    case "connected":
      return state.phase === "live" ? state : begin(state);
    case "disconnected":
      return state.phase === "live" ? { ...state, phase: "ended", log: { ...state.log, phase: "ended" } } : state;
    case "clear": {
      const log = logReducer(state.log, { type: "clear" });
      return log === state.log ? state : { ...state, log };
    }
    case "message": {
      const row = parseLiveEvent(action.message);
      if (!row) return state;
      // Each bot numbers its events from 1: a first event after later ones is a new call's, even when the hang up and
      // the next connection were not reported.
      const last = state.log.rows.at(-1);
      const restarted = row.seq === 1 && last !== undefined && last.seq > 1;
      const live = state.phase === "live" && !restarted ? state : begin(state);
      const log = logReducer(live.log, { type: "event", session: live.log.session, event: row });
      // The same event twice, or one out of order, is ignored by the log.
      return log === live.log ? live : { ...live, log };
    }
  }
}

// ---- The store for React ----------------------------------------------------------------------------------------------

type Listener = () => void;

/** A store that holds a CallLog: `getSnapshot` and `subscribe` are what `useSyncExternalStore` takes. */
export class CallLogStore {
  private state: CallLog = EMPTY_CALL_LOG;
  private readonly listeners = new Set<Listener>();

  getSnapshot = (): CallLog => this.state;

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  dispatch = (action: CallLogAction): void => {
    const next = callLogReducer(this.state, action);
    if (next === this.state) return;
    this.state = next;
    for (const listener of [...this.listeners]) listener();
  };
}

/** The one store of the page: the bridge feeds it and the Event log reads it. */
export const liveEventsStore = new CallLogStore();
