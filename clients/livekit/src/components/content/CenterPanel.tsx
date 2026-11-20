// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useApp } from "../../context/useApp";
import { useSession } from "../../session/session-context";
import { ConversationPanel } from "./ConversationPanel";
import { IdleHero } from "./IdleHero";
import { PromptsPanel } from "./PromptsPanel";
import { ServicesPanel } from "./ServicesPanel";
import { ToolsPanel } from "./ToolsPanel";

type Tab = "conversation" | "services" | "prompts" | "tools";

const ALL_TABS: { id: Tab; label: string }[] = [
  { id: "conversation", label: "CONVERSATION" },
  { id: "services", label: "SERVICES" },
  { id: "prompts", label: "PROMPTS" },
  { id: "tools", label: "TOOLS" },
];

const TAB_STEPS: Record<string, (index: number) => number> = {
  ArrowRight: (index) => (index + 1) % ALL_TABS.length,
  ArrowLeft: (index) => (index - 1 + ALL_TABS.length) % ALL_TABS.length,
  Home: () => 0,
  End: () => ALL_TABS.length - 1,
};

function ConversationContent() {
  const { isConnected, isConnecting } = useSession();
  const { selectedPrompt } = useApp();

  if (!isConnected) {
    return <IdleHero connecting={isConnecting} examples={selectedPrompt?.examples ?? []} />;
  }

  return <ConversationPanel />;
}

function TabPanel({ tab, active, className, children }: Readonly<{ tab: Tab; active: Tab; className: string; children: ReactNode }>) {
  return (
    <div
      id={`panel-${tab}`}
      role="tabpanel"
      aria-labelledby={`tab-${tab}`}
      className={`${className} ${active === tab ? "" : "hidden"}`}
    >
      {children}
    </div>
  );
}

export function CenterPanel() {
  const [activeTab, setActiveTab] = useState<Tab>("conversation");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = TAB_STEPS[event.key];
    if (!step) return;
    event.preventDefault();
    const next = step(ALL_TABS.findIndex((tab) => tab.id === activeTab));
    setActiveTab(ALL_TABS[next].id);
    tabRefs.current[next]?.focus();
  };

  return (
    <main className="flex-1 d-flex flex-col overflow-hidden">
      <div className="tab-header" role="tablist" aria-label="Agent views" onKeyDown={handleKeyDown}>
        {ALL_TABS.map((tab, index) => (
          <button
            key={tab.id}
            ref={(element) => { tabRefs.current[index] = element; }}
            type="button"
            id={`tab-${tab.id}`}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <TabPanel tab="conversation" active={activeTab} className="flex-1 min-h-0 relative">
        <div className="conversation-overlay overflow-y-auto">
          <ConversationContent />
        </div>
      </TabPanel>
      <TabPanel tab="services" active={activeTab} className="flex-1 min-h-0 overflow-y-auto">
        <ServicesPanel />
      </TabPanel>
      <TabPanel tab="prompts" active={activeTab} className="flex-1 min-h-0 overflow-y-auto">
        <PromptsPanel />
      </TabPanel>
      <TabPanel tab="tools" active={activeTab} className="flex-1 min-h-0 overflow-y-auto">
        <ToolsPanel />
      </TabPanel>
    </main>
  );
}
