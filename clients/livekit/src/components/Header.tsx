// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useSession } from "../session/session-context";
import { DevicesSection } from "./status-panel/DevicesSection";

const BUTTON_TEXT = { idle: "Connect", requesting: "Connecting...", connecting: "Connecting...", connected: "Disconnect" };

export function Header() {
  const { phase, isConnected, isConnecting, canConnect, error, connect, disconnect } = useSession();
  const waiting = !isConnected && !isConnecting && !canConnect;

  return (
    <header className="px-4 py-3 border-b">
      <div className="d-flex justify-between items-center">
        <h1 className="text-lg font-semibold">
          <span className="brand-accent">Nemotron</span> Voice Agent
        </h1>
        <div className="d-flex items-center gap-3">
          {isConnected && <DevicesSection />}
          <button
            className={isConnected ? "btn-secondary" : "btn-primary"}
            onClick={() => (isConnected ? disconnect() : connect())}
            disabled={isConnecting || waiting}
            title={waiting ? "Loading the services and prompts" : undefined}
          >
            {BUTTON_TEXT[phase]}
          </button>
        </div>
      </div>
      {error && <p className="mt-2 error-text">{error}</p>}
    </header>
  );
}
