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

function clamp(value: number, spec: ServiceSettingSpec): number {
  return Math.min(spec.max ?? value, Math.max(spec.min ?? value, value));
}

export function ServiceSettingsControls({ schema, values, disabled, onChange, children }: Props) {
  const resolve = (name: string) => values[name] ?? schema[name]?.default;

  const bounded = (spec: ServiceSettingSpec): { spec: ServiceSettingSpec; belowMin: boolean } => {
    const limit = spec.below ? resolve(spec.below) : undefined;
    if (typeof limit !== "number") return { spec, belowMin: false };
    const cap = limit - (spec.reserve ?? 0);
    if (cap < (spec.min ?? 0)) return { spec, belowMin: true };
    return { spec: { ...spec, max: Math.min(spec.max ?? cap, cap) }, belowMin: false };
  };

  const bounds = Object.fromEntries(Object.entries(schema).map(([name, spec]) => [name, bounded(spec)]));
  const forcedOff = new Set(
    Object.entries(schema).flatMap(([name, spec]) => (spec.requires && bounds[name].belowMin ? [spec.requires] : [])),
  );

  const current = (name: string) => (forcedOff.has(name) ? false : resolve(name));

  const renderControl = (name: string, spec: ServiceSettingSpec, rowDisabled: boolean) => {
    const raw = current(name);
    const value = typeof raw === "number" && spec.max !== undefined ? Math.min(raw, spec.max) : raw;
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
          onBlur={() => {
            const typed = values[name];
            if (typeof typed === "number" && clamp(typed, spec) !== typed) onChange(name, clamp(typed, spec));
          }}
        />
      </>
    );
  };

  return (
    <div className="param-grid">
      {children}
      {orderSettings(schema).map(([name]) => {
        const { spec, belowMin } = bounds[name];
        const inactive = (Boolean(spec.requires) && !current(spec.requires ?? "")) || belowMin || forcedOff.has(name);
        return (
          <SettingRow key={name} label={spec.label ?? name} inactive={inactive}>
            {renderControl(name, spec, disabled || inactive)}
          </SettingRow>
        );
      })}
    </div>
  );
}
