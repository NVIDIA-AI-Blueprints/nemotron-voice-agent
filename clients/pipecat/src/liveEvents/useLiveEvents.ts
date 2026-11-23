// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useSyncExternalStore } from "react";
import type { LiveEvent } from "./liveEvent";
import { liveEventsStore } from "./store";

export interface LiveEventsValue {
  /** The current call's events. */
  rows: LiveEvent[];
  /** How many of the oldest events were dropped to keep the log short. */
  dropped: number;
  /** True after a call has ended; its log stays on screen until the next call starts. */
  callEnded: boolean;
  /** Empties the log. */
  clear: () => void;
}

// While the tab is hidden its panel does not follow the store, so a streamed answer does not re-render a list nobody sees;
// it reads the newest state when it is shown again.
const unsubscribed = () => () => {};

export function useLiveEvents(active = true): LiveEventsValue {
  const state = useSyncExternalStore(active ? liveEventsStore.subscribe : unsubscribed, liveEventsStore.getSnapshot);
  const clear = useCallback(() => liveEventsStore.dispatch({ type: "clear" }), []);
  return { rows: state.log.rows, dropped: state.log.dropped, callEnded: state.phase === "ended", clear };
}
