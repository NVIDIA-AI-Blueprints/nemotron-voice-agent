// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Event log of the current call, as the store holds it. The log remembers which call (session) it belongs to: a
// different call never sees it, a hang up leaves the finished call's log on screen until the next call starts, and the
// newest MAX_ROWS events are kept. Pure reducer and selectors, no React.

import { familyOf, type Family, type LiveEvent } from "./liveEvent.ts";

/** The events kept; the oldest go first, and `dropped` counts them. A call of a few minutes sends far fewer. */
export const MAX_ROWS = 2000;

export interface LogState {
  /** The session the rows belong to. */
  session: string;
  rows: LiveEvent[];
  /** How many of the oldest events were dropped to stay within MAX_ROWS. */
  dropped: number;
  /** "live" from the call's connection until it ends; "ended" after; "idle" before any call. */
  phase: "idle" | "live" | "ended";
}

export type LogAction =
  | { type: "started"; session: string }
  | { type: "event"; session: string; event: LiveEvent }
  | { type: "ended" }
  | { type: "clear" };

export const EMPTY_LOG: LogState = { session: "", rows: [], dropped: 0, phase: "idle" };

export function logReducer(state: LogState, action: LogAction): LogState {
  switch (action.type) {
    case "started":
      // Events that came just before the connection was reported belong to this call and stay.
      return state.session === action.session ? { ...state, phase: "live" } : { ...EMPTY_LOG, session: action.session, phase: "live" };
    case "ended":
      return state.phase === "live" ? { ...state, phase: "ended" } : state;
    case "clear":
      return state.rows.length === 0 && state.dropped === 0 ? state : { ...state, rows: [], dropped: 0 };
    case "event": {
      const base: LogState = state.session === action.session ? state : { ...EMPTY_LOG, session: action.session, phase: "live" };
      const last = base.rows.at(-1);
      // The same event twice, or one that arrives out of order, is ignored: the log is in the server's order.
      if (last && action.event.seq <= last.seq) return base === state ? state : base;
      const rows = [...base.rows, action.event];
      const overflow = Math.max(0, rows.length - MAX_ROWS);
      return { ...base, rows: overflow ? rows.slice(overflow) : rows, dropped: base.dropped + overflow };
    }
  }
}

/** The rows to show while `currentSession` is the app's current session ("" between calls). */
export function visibleRows(state: LogState, currentSession: string): LiveEvent[] {
  return currentSession === "" || state.session === currentSession ? state.rows : [];
}

/** The rows whose family is not hidden. */
export function filterRows(rows: readonly LiveEvent[], hidden: ReadonlySet<Family>): LiveEvent[] {
  return hidden.size === 0 ? [...rows] : rows.filter((row) => !hidden.has(familyOf(row)));
}

/** One JSON object per line, for copying the log into a bug report or a script. */
export function toJsonLines(rows: readonly LiveEvent[]): string {
  return rows.map((row) => JSON.stringify(row)).join("\n");
}

/** How many events the server dropped between the first and last row shown: the gaps in `seq`. */
export function missingCount(rows: readonly LiveEvent[]): number {
  if (rows.length < 2) return 0;
  return rows.at(-1)!.seq - rows[0].seq + 1 - rows.length;
}
