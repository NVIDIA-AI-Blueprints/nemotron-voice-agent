// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";
import { useCollapsible } from "../hooks/useCollapsible";
import { CollapsibleHeader } from "./CollapsibleHeader";

interface PanelSectionProps {
  label: string;
  children?: ReactNode;
  loading?: boolean;
  storageKey?: string;
}

export function PanelSection({ label, children, loading, storageKey }: Readonly<PanelSectionProps>) {
  const { collapsed, toggle } = useCollapsible(storageKey ?? label);

  return (
    <div className={`panel-section${collapsed ? " is-collapsed" : ""}`}>
      <CollapsibleHeader
        label={label}
        collapsed={collapsed}
        onToggle={toggle}
        className="panel-label panel-label-toggle"
      />
      {!collapsed && (loading ? <p className="settings-hint">Loading...</p> : children)}
    </div>
  );
}
