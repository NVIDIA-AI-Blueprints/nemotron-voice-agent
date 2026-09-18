// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useConnectionState } from "../hooks/useConnectionState";
import { useApp } from "../context/useApp";
import { PanelSection } from "./PanelSection";

export function TransportSelector() {
  const { isLocked } = useConnectionState();
  const { availableTransports, selectedTransport, setTransport } = useApp();

  return (
    <PanelSection label="TRANSPORT">
      <div className="transport-options">
        {availableTransports.map((transport) => (
          <button
            key={transport.id}
            className={`transport-btn ${selectedTransport === transport.id ? "transport-btn--active" : ""}`}
            onClick={() => setTransport(transport.id)}
            disabled={isLocked || availableTransports.length === 1}
            aria-pressed={selectedTransport === transport.id}
          >
            {transport.label}
          </button>
        ))}
      </div>
    </PanelSection>
  );
}
