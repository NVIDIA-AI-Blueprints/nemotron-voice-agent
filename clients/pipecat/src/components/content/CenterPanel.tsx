// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { lazy, Suspense, useState, type CSSProperties } from "react";
import { useApp } from "../../context/useApp";
import { logEnabled } from "../../liveEvents/gate";
import { useConnectionState } from "../../hooks/useConnectionState";
import { IdleHero } from "./IdleHero";
import { ConversationPanel } from "./ConversationPanel";
import { MetricsPanel } from "./MetricsPanel";
import { ServicesPanel } from "./ServicesPanel";
import { PromptsPanel } from "./PromptsPanel";
import { ToolsPanel } from "./ToolsPanel";

// The Event log loads only for an example whose registry entry lists `event_log`.
const EventLogScreen = lazy(() => import("../events/EventLogScreen"));

// With the extra tab, the labels stay on one line and the buttons are a little narrower; a narrow panel wraps the strip.
const STRIP_STYLE: CSSProperties = { flexWrap: "wrap" };
const TAB_STYLE: CSSProperties = { whiteSpace: "nowrap", paddingInline: "var(--space-4)" };

type Tab = "conversation" | "events" | "metrics" | "services" | "prompts" | "tools";

const ALL_TABS: { id: Tab; label: string }[] = [
  { id: "conversation", label: "CONVERSATION" },
  { id: "metrics", label: "METRICS" },
  { id: "services", label: "SERVICES" },
  { id: "prompts", label: "PROMPTS" },
  { id: "tools", label: "TOOLS" },
];

function ConversationContent() {
  const { isConnected, isConnecting } = useConnectionState();

  if (!isConnected) {
    return <IdleHero connecting={isConnecting} fadingOut={false} />;
  }

  return (
    <ConversationPanel />
  );
}

export function CenterPanel() {
  const logOn = logEnabled(useApp().selectedExample);
  const [chosenTab, setActiveTab] = useState<Tab>("conversation");
  const activeTab = logOn || chosenTab !== "events" ? chosenTab : "conversation";
  const tabs = logOn ? [ALL_TABS[0], { id: "events" as const, label: "EVENT LOG" }, ...ALL_TABS.slice(1)] : ALL_TABS;

  return (
    <main className="flex-1 d-flex flex-col overflow-hidden">
      <div className="tab-header" style={logOn ? STRIP_STYLE : undefined}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            id={`center-tab-${tab.id}`}
            style={logOn ? TAB_STYLE : undefined}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className={`flex-1 min-h-0 relative ${activeTab !== "conversation" ? "hidden" : ""}`}>
        <div className="conversation-overlay overflow-y-auto">
          <ConversationContent />
        </div>
      </div>
      <div className={`flex-1 min-h-0 overflow-y-auto ${activeTab !== "metrics" ? "hidden" : ""}`}>
        <MetricsPanel />
      </div>
      <div className={`flex-1 min-h-0 overflow-y-auto ${activeTab !== "services" ? "hidden" : ""}`}>
        <ServicesPanel />
      </div>
      <div className={`flex-1 min-h-0 overflow-y-auto ${activeTab !== "prompts" ? "hidden" : ""}`}>
        <PromptsPanel />
      </div>
      <div className={`flex-1 min-h-0 overflow-y-auto ${activeTab !== "tools" ? "hidden" : ""}`}>
        <ToolsPanel />
      </div>
      {logOn && (
        <Suspense fallback={null}>
          <EventLogScreen shown={activeTab === "events"} />
        </Suspense>
      )}
    </main>
  );
}
