// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function CollapseChevron({ collapsed }: Readonly<{ collapsed: boolean }>) {
  return (
    <svg
      className={`collapsible-chevron${collapsed ? " is-collapsed" : ""}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}
