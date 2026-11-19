// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from "react";
import { useCollapsible } from "../hooks/useCollapsible";
import { CollapsibleHeader } from "./CollapsibleHeader";

interface CollapsibleCardProps {
  label: string;
  children: ReactNode;
  className: string;
  storageKey: string;
}

export function CollapsibleCard({ label, children, className, storageKey }: Readonly<CollapsibleCardProps>) {
  const { collapsed, toggle } = useCollapsible(storageKey);

  return (
    <div className={`card sidebar-card collapsible-card ${className}`}>
      <CollapsibleHeader
        label={label}
        collapsed={collapsed}
        onToggle={toggle}
        className="collapsible-header"
        labelClassName="text-xs text-secondary"
      />
      {!collapsed && <div className="collapsible-body">{children}</div>}
    </div>
  );
}
