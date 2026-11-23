// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePhoneNumbers } from "../api";
import { PanelSection } from "./PanelSection";

/** The phone numbers that reach the agent when telephony (SIP) is on. Hidden when no number is mapped. */
export function PhoneSection() {
  const { data: numbers = [] } = usePhoneNumbers();
  if (numbers.length === 0) return null;

  return (
    <PanelSection label="PHONE">
      <p className="settings-hint">Dial a number from any phone to reach the same agent.</p>
      <ul className="tools-list">
        {numbers.map((item) => (
          <li key={item.number} className="tools-row is-active">
            <a className="tools-name phone-number" href={`tel:${item.number}`}>{item.number}</a>
            {item.name && <span className="settings-hint">answers as {item.name}</span>}
          </li>
        ))}
      </ul>
      <p className="settings-hint">
        Each number answers with one use case. The SIP guide in the example README covers number setup.
      </p>
    </PanelSection>
  );
}
