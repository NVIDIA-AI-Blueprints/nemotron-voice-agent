// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import type { ReactNode } from "react";
import { useCollapsible } from "../hooks/useCollapsible";
import { CollapsibleHeader } from "./CollapsibleHeader";

interface PanelSectionProps {
  label: string;
  children?: ReactNode;
  loading?: boolean;
  loadingText?: string;
  collapsible?: boolean;
  storageKey?: string;
}

export function PanelSection({
  label,
  children,
  loading,
  loadingText = "Loading...",
  collapsible = true,
  storageKey,
}: Readonly<PanelSectionProps>) {
  const { collapsed, toggle } = useCollapsible(storageKey ?? label);
  const isCollapsed = collapsible && collapsed;

  return (
    <div className={`panel-section${isCollapsed ? " is-collapsed" : ""}`}>
      {collapsible ? (
        <CollapsibleHeader
          label={label}
          collapsed={collapsed}
          onToggle={toggle}
          className="panel-label panel-label-toggle"
        />
      ) : (
        <p className="panel-label">{label}</p>
      )}
      {!isCollapsed &&
        (loading ? <p style={{ fontSize: "11px", color: "var(--text-muted)" }}>{loadingText}</p> : children)}
    </div>
  );
}
