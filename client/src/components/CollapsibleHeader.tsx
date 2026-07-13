// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { CollapseChevron } from "./CollapseChevron";

interface CollapsibleHeaderProps {
  label: string;
  collapsed: boolean;
  onToggle: () => void;
  className: string;
  labelClassName?: string;
}

export function CollapsibleHeader({
  label,
  collapsed,
  onToggle,
  className,
  labelClassName,
}: Readonly<CollapsibleHeaderProps>) {
  return (
    <button type="button" className={className} onClick={onToggle} aria-expanded={!collapsed}>
      <span className={labelClassName}>{label}</span>
      <CollapseChevron collapsed={collapsed} />
    </button>
  );
}
