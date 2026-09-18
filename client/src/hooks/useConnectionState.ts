// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePipecatClientTransportState } from "@pipecat-ai/client-react";

const CONNECTING_STATES = new Set(["authenticating", "authenticated", "connecting"]);
const CONNECTED_STATES = new Set(["connected", "ready"]);

export function useConnectionState() {
  const state = usePipecatClientTransportState();
  return {
    state,
    isConnected: CONNECTED_STATES.has(state),
    isConnecting: CONNECTING_STATES.has(state),
    isLocked: CONNECTED_STATES.has(state) || CONNECTING_STATES.has(state),
  };
}
