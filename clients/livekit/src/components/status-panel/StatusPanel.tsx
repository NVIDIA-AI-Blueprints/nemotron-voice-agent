// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PhoneSection } from "../PhoneSection";
import { PromptSelector } from "../PromptSelector";
import { ToolsSection } from "../ToolsSection";
import { VoiceSettings } from "../VoiceSettings";

export function StatusPanel() {
  return (
    <aside className="status-panel">
      <PromptSelector />
      <ToolsSection />
      <VoiceSettings />
      <PhoneSection />
    </aside>
  );
}
