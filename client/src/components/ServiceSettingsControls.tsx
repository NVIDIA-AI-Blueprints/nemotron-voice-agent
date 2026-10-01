// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties, ReactNode } from "react";
import type { ServiceSettingSpec, ServiceSettingsSchema, ServiceSettingValues } from "../api";
import { Toggle } from "./Toggle";

type Props = Readonly<{
  schema: ServiceSettingsSchema;
  values: ServiceSettingValues;
  disabled: boolean;
  onChange: (name: string, value: unknown) => void;
  children?: ReactNode;
}>;

export function SettingRow({ label, inactive = false, children }: Readonly<{ label: string; inactive?: boolean; children: ReactNode }>) {
  return (
    <div className={`param-row${inactive ? " param-row--inactive" : ""}`}>
      <span>{label}</span>
      <div className="param-row__control">{children}</div>
    </div>
  );
}

type Entry = [string, ServiceSettingSpec];

function orderSettings(schema: ServiceSettingsSchema): Entry[] {
  const entries = Object.entries(schema);
  const dependents = (name: string) => entries.filter(([, spec]) => spec.requires === name);
  const toggles = entries.filter(([, spec]) => spec.type === "bool");
  const rest = entries.filter(([, spec]) => spec.type !== "bool" && !spec.requires);
  return [...toggles.flatMap((entry) => [entry, ...dependents(entry[0])]), ...rest];
}

function parseNumber(raw: string, integer: boolean): number | undefined {
  if (raw === "") return undefined;
  const value = Number(raw);
  return integer ? Math.round(value) : value;
}

export function ServiceSettingsControls({ schema, values, disabled, onChange, children }: Props) {
  const resolve = (name: string) => values[name] ?? schema[name]?.default;

  const renderControl = (name: string, spec: ServiceSettingSpec, rowDisabled: boolean) => {
    const value = resolve(name);
    if (spec.type === "bool") {
      return (
        <Toggle
          checked={Boolean(value)}
          onChange={(checked) => onChange(name, checked)}
          label={spec.label ?? name}
          disabled={rowDisabled}
        />
      );
    }
    if (spec.type === "enum") {
      return (
        <select className="select-dark" value={String(value ?? "")} disabled={rowDisabled} onChange={(e) => onChange(name, e.target.value)}>
          {(spec.options ?? []).map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      );
    }
    const numeric = typeof value === "number" ? value : undefined;
    const step = spec.type === "int" ? 1 : 0.01;
    const min = spec.min ?? 0;
    const max = spec.max ?? 0;
    const ranged = spec.min !== undefined && spec.max !== undefined && max > min;
    const fill = ranged && numeric !== undefined ? ((numeric - min) / (max - min)) * 100 : undefined;
    return (
      <>
        {ranged && (
          <input
            className={`param-row__slider${numeric === undefined ? " param-row__slider--unset" : ""}`}
            type="range"
            min={min}
            max={max}
            step={step}
            value={numeric ?? min}
            disabled={rowDisabled}
            aria-label={spec.label ?? name}
            style={fill === undefined ? undefined : ({ "--slider-fill": `${fill}%` } as CSSProperties)}
            onChange={(e) => onChange(name, parseNumber(e.target.value, spec.type === "int"))}
          />
        )}
        <input
          className="param-row__value"
          type="number"
          min={spec.min}
          max={spec.max}
          step={step}
          placeholder="Auto"
          value={numeric ?? ""}
          disabled={rowDisabled}
          aria-label={spec.label ?? name}
          onChange={(e) => onChange(name, parseNumber(e.target.value, spec.type === "int"))}
        />
      </>
    );
  };

  return (
    <div className="param-grid">
      {children}
      {orderSettings(schema).map(([name, spec]) => {
        const inactive = Boolean(spec.requires) && !resolve(spec.requires ?? "");
        return (
          <SettingRow key={name} label={spec.label ?? name} inactive={inactive}>
            {renderControl(name, spec, disabled || inactive)}
          </SettingRow>
        );
      })}
    </div>
  );
}
