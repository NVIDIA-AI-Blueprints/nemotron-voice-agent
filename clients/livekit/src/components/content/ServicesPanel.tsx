// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, type KeyboardEvent, type ReactNode } from "react";
import { useApp } from "../../context/useApp";
import { useSession } from "../../session/session-context";
import type { CustomFieldSpec, CustomService, ServiceEntry, ServiceStatus, Slot } from "../../api";
import { SLOTS } from "../../lib/types";
import { SLOT_TITLES, customServiceError, endpointFields, fieldLabel, serviceDetails, settingsKey } from "../../lib/services";
import { ServiceSettingsControls } from "../ServiceSettingsControls";

const SOURCE_LABELS: Record<string, string> = { "self-hosted": "Self-hosted", "cloud-nim": "NVIDIA Cloud" };
const STATUS_DISPLAY: Record<ServiceStatus, { label: string; indicator: string }> = {
  ready: { label: "Ready", indicator: "done" },
  starting: { label: "Starting…", indicator: "running" },
  error: { label: "Error", indicator: "failed" },
  offline: { label: "Not running", indicator: "failed" },
};

function onActivate(action: () => void) {
  return (event: KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    action();
  };
}

function RowActions({ onEdit, onRemove }: Readonly<{ onEdit: () => void; onRemove: () => void }>) {
  const { isLocked } = useSession();
  const title = (action: string) => (isLocked ? "Locked during session" : action);
  return (
    <div className="svc-row__actions">
      <button type="button" className="svc-icon-btn" onClick={(event) => { event.stopPropagation(); onEdit(); }} disabled={isLocked} title={title("Edit")}>✎</button>
      <button type="button" className="svc-icon-btn svc-icon-btn--remove" onClick={(event) => { event.stopPropagation(); onRemove(); }} disabled={isLocked} title={title("Remove")}>−</button>
    </div>
  );
}

function CustomServiceForm({
  slot, spec, initial, submitLabel, onSubmit, onCancel,
}: Readonly<{
  slot: Slot;
  spec: CustomFieldSpec;
  initial?: CustomService;
  submitLabel: string;
  onSubmit: (name: string, fields: Record<string, string>) => void;
  onCancel: () => void;
}>) {
  const { isLocked } = useSession();
  const [name, setName] = useState(initial?.name ?? "");
  const [fields, setFields] = useState<Record<string, string>>(initial?.fields ?? {});
  const [error, setError] = useState("");
  const canSubmit = Boolean(name.trim()) && spec.required.every((field) => fields[field]?.trim());

  const handleSubmit = () => {
    if (!canSubmit || isLocked) return;
    const reason = customServiceError(slot, spec, fields);
    if (reason) {
      setError(reason);
      return;
    }
    const trimmed = Object.fromEntries(Object.entries(fields).map(([key, value]) => [key, value.trim()]));
    onSubmit(name.trim(), trimmed);
  };

  return (
    <form
      className={initial ? "svc-row svc-row--editing" : "svc-add-form"}
      onSubmit={(event) => { event.preventDefault(); handleSubmit(); }}
      onKeyDown={(event) => { if (event.key === "Escape") onCancel(); }}
    >
      <fieldset className={initial ? "svc-edit-fields" : "d-flex flex-col gap-2"} disabled={isLocked}>
        <input className="svc-input" placeholder="Display name" aria-label="Display name" value={name} onChange={(e) => { setName(e.target.value); setError(""); }} autoFocus />
        {endpointFields(spec).map((field) => {
          const label = spec.required.includes(field) ? fieldLabel(field) : `${fieldLabel(field)} (optional)`;
          return (
            <input
              key={field}
              className="svc-input"
              placeholder={label}
              aria-label={label}
              value={fields[field] ?? ""}
              onChange={(e) => {
                setFields((previous) => ({ ...previous, [field]: e.target.value }));
                setError("");
              }}
            />
          );
        })}
        {error && <p className="prompts-status prompts-status--error">{error}</p>}
        <div className="svc-edit-actions">
          <button type="submit" className="btn-primary svc-add-btn" disabled={!canSubmit}>{submitLabel}</button>
          {initial && <button type="button" className="svc-icon-btn" onClick={onCancel} title="Cancel">✕</button>}
        </div>
      </fieldset>
    </form>
  );
}

function CustomServiceRow({
  service, spec, isActive, onSelect,
}: Readonly<{ service: CustomService; spec: CustomFieldSpec; isActive: boolean; onSelect?: (id: string) => void }>) {
  const { services } = useApp();
  const [editing, setEditing] = useState(false);
  const slotState = services[service.slot];

  if (editing) {
    return (
      <CustomServiceForm
        slot={service.slot}
        spec={spec}
        initial={service}
        submitLabel="Save"
        onSubmit={(name, fields) => { slotState.updateCustom(service.id, name, fields); setEditing(false); }}
        onCancel={() => setEditing(false)}
      />
    );
  }

  return (
    <div
      className={`svc-row${onSelect ? " svc-row--clickable" : ""}${isActive ? " svc-row--active" : ""}`}
      onClick={onSelect ? () => onSelect(service.id) : undefined}
      onKeyDown={onSelect ? onActivate(() => onSelect(service.id)) : undefined}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      style={onSelect ? undefined : { opacity: 0.6, cursor: "default" }}
    >
      <div className="svc-row__info">
        <span className="svc-row__name">{service.name}</span>
        {serviceDetails(service.slot, service.fields).map((detail, index) => (
          <span key={detail} className={`svc-row__detail${index === 0 ? " svc-row__url" : ""}`}>{detail}</span>
        ))}
        {service.fields.extra_params && <span className="svc-row__detail svc-row__sys">extra: {service.fields.extra_params}</span>}
      </div>
      <RowActions onEdit={() => setEditing(true)} onRemove={() => slotState.removeCustom(service.id)} />
    </div>
  );
}

function ServiceCard({
  slot, entry, isActive, expanded, isLocked, onSelect, onToggle,
}: Readonly<{
  slot: Slot;
  entry: ServiceEntry;
  isActive: boolean;
  expanded: boolean;
  isLocked: boolean;
  onSelect: (id: string) => void;
  onToggle: () => void;
}>) {
  const { serviceSettings, setServiceSetting } = useApp();
  const hasSettings = Object.keys(entry.settings).length > 0;
  const key = settingsKey(slot, entry.id);
  const handleClick = () => {
    if (!isActive && !isLocked) onSelect(entry.id);
    if (hasSettings) onToggle();
  };

  return (
    <div className={`svc-row svc-row--clickable svc-row--card${isActive ? " svc-row--active" : ""}`}>
      <div
        className="svc-row__header"
        onClick={handleClick}
        onKeyDown={onActivate(handleClick)}
        role="button"
        tabIndex={0}
        aria-expanded={hasSettings ? expanded : undefined}
      >
        <div className="svc-row__info">
          <span className="svc-row__name">{entry.name}</span>
          {serviceDetails(slot, entry).map((detail, index) => (
            <span key={detail} className={`svc-row__detail${index === 0 ? " svc-row__url" : ""}`}>{detail}</span>
          ))}
        </div>
        <div className="svc-row__controls">
          {entry.status && (
            <>
              <span className="svc-row__detail">{STATUS_DISPLAY[entry.status].label}</span>
              <span className={`agent-task-indicator agent-task-indicator-${STATUS_DISPLAY[entry.status].indicator}`} aria-hidden="true" />
            </>
          )}
          <div className="svc-source" role="group" aria-label="Endpoint">
            <button type="button" className="svc-source__option svc-source__option--selected" aria-pressed disabled>
              {SOURCE_LABELS[entry.source] ?? entry.source}
            </button>
          </div>
          {hasSettings && (
            <button
              type="button"
              className={`svc-icon-btn svc-chevron${expanded ? " svc-chevron--open" : ""}`}
              onClick={(event) => { event.stopPropagation(); onToggle(); }}
              title={expanded ? "Hide parameters" : "Show parameters"}
              aria-expanded={expanded}
            >
              ▾
            </button>
          )}
        </div>
      </div>
      {expanded && hasSettings && (
        <div className="svc-row__section">
          <p className="prompts-section-label">Parameters</p>
          <ServiceSettingsControls
            schema={entry.settings}
            values={serviceSettings[key] ?? {}}
            disabled={isLocked}
            onChange={(name, value) => setServiceSetting(key, name, value)}
          />
        </div>
      )}
    </div>
  );
}

function ServiceSection({
  title, children, onAdd, addDisabled,
}: Readonly<{ title: string; children: ReactNode; onAdd?: () => void; addDisabled: boolean }>) {
  return (
    <div style={{ marginBottom: "var(--space-6)" }}>
      <div className="services-header">
        <h3 className="metrics-title">{title}</h3>
        {onAdd && (
          <button
            type="button"
            className="svc-icon-btn svc-icon-btn--add"
            onClick={onAdd}
            disabled={addDisabled}
            title={addDisabled ? "Locked during session" : `Add ${title}`}
          >
            +
          </button>
        )}
      </div>
      <div className="svc-list">{children}</div>
    </div>
  );
}

function CustomServicesGroup({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <div className="svc-group">
      <div className="svc-group__label">Custom</div>
      <div className="svc-list">{children}</div>
    </div>
  );
}

function SlotServices({ slot }: Readonly<{ slot: Slot }>) {
  const { services, servicesLoading, selectService } = useApp();
  const { isLocked } = useSession();
  const [adding, setAdding] = useState(false);
  const [expanded, setExpanded] = useState("");
  const state = services[slot];
  const spec = state.customSpec;
  const select = (id: string) => selectService(slot, id);

  return (
    <ServiceSection title={SLOT_TITLES[slot]} onAdd={spec ? () => setAdding(!adding) : undefined} addDisabled={isLocked}>
      {servicesLoading && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>Loading...</p>}
      {!servicesLoading && state.entries.length === 0 && (
        <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>Waiting for the agent worker to list its services...</p>
      )}
      {adding && spec && (
        <CustomServiceForm
          slot={slot}
          spec={spec}
          submitLabel="Add"
          onSubmit={(name, fields) => {
            const service = state.addCustom(name, fields);
            if (!service) return;
            select(service.id);
            setAdding(false);
          }}
          onCancel={() => setAdding(false)}
        />
      )}
      {state.entries.map((entry) => (
        <ServiceCard
          key={entry.id}
          slot={slot}
          entry={entry}
          isActive={state.selectedId === entry.id}
          expanded={expanded === entry.id}
          isLocked={isLocked}
          onSelect={select}
          onToggle={() => setExpanded(expanded === entry.id ? "" : entry.id)}
        />
      ))}
      {spec && state.custom.length > 0 && (
        <CustomServicesGroup>
          {state.custom.map((service) => (
            <CustomServiceRow
              key={service.id}
              service={service}
              spec={spec}
              isActive={state.selectedId === service.id}
              onSelect={isLocked ? undefined : select}
            />
          ))}
        </CustomServicesGroup>
      )}
    </ServiceSection>
  );
}

export function ServicesPanel() {
  return (
    <div className="services-panel p-4">
      {SLOTS.map((slot) => <SlotServices key={slot} slot={slot} />)}
    </div>
  );
}
