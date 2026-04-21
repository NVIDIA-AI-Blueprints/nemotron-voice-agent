// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import type { SelectHTMLAttributes } from "react";

interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface LabeledSelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  label: string;
  options: SelectOption[] | string[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  dimmed?: boolean;
  containerClassName?: string;
}

export function LabeledSelect({
  label,
  options,
  value,
  onChange,
  placeholder = "Select an option",
  dimmed = false,
  containerClassName = "",
  className = "",
  disabled,
  ...props
}: LabeledSelectProps) {
  // Normalize options to always be SelectOption[]
  const normalizedOptions: SelectOption[] = options.map((opt) =>
    typeof opt === "string" ? { value: opt, label: opt } : opt
  );

  return (
    <div className={containerClassName}>
      <div className={`text-sm mb-2 ${disabled ? "text-gray-400" : "text-gray-600"}`}>{label}</div>
      <select
        className={`w-full border rounded p-2 text-sm transition-all ${
          dimmed ? "opacity-50" : "opacity-100"
        } ${
          disabled
            ? "bg-gray-100 text-gray-500 cursor-not-allowed border-gray-200"
            : "bg-white cursor-pointer border-gray-300 hover:border-gray-400"
        } ${className}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        {...props}
      >
        <option value="" disabled>
          {placeholder}
        </option>
        {normalizedOptions.map((opt) => (
          <option key={opt.value} value={opt.value} disabled={opt.disabled}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
