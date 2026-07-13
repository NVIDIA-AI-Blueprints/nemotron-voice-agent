// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { type ReactNode } from "react";
import { useCollapsible } from "../hooks/useCollapsible";
import { CollapsibleHeader } from "./CollapsibleHeader";

interface CollapsibleCardProps {
  label: string;
  children: ReactNode;
  className?: string;
  storageKey?: string;
  defaultCollapsed?: boolean;
  keepMounted?: boolean;
}

export function CollapsibleCard({
  label,
  children,
  className = "",
  storageKey,
  defaultCollapsed = false,
  keepMounted = false,
}: Readonly<CollapsibleCardProps>) {
  const { collapsed, toggle } = useCollapsible(storageKey ?? label, defaultCollapsed);
  const stateClass = collapsed ? "is-collapsed" : "is-expanded";

  return (
    <div className={`card sidebar-card collapsible-card ${stateClass} ${className}`.trim()}>
      <CollapsibleHeader
        label={label}
        collapsed={collapsed}
        onToggle={toggle}
        className="collapsible-header"
        labelClassName="text-xs text-secondary"
      />
      {keepMounted ? (
        <div className="collapsible-body" hidden={collapsed}>
          {children}
        </div>
      ) : (
        !collapsed && <div className="collapsible-body">{children}</div>
      )}
    </div>
  );
}
