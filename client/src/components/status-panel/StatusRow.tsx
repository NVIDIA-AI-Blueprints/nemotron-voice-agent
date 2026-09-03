// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

interface StatusRowProps {
  label: string;
  value: string;
  title?: string;
  valueClassName?: string;
  children?: React.ReactNode;
}

export function StatusRow({ label, value, title, valueClassName, children }: Readonly<StatusRowProps>) {
  return (
    <div className="status-row">
      <span>{label}</span>
      <span className={valueClassName ? `status-value ${valueClassName}` : "status-value"} title={title}>
        {value}
        {children}
      </span>
    </div>
  );
}
