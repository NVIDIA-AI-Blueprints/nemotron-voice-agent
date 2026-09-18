// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useApp } from "../../context/useApp";
import { useConnectionState } from "../../hooks/useConnectionState";
import { useServiceCatalog, type LLMService, type SimpleService, type ServiceEntry } from "../../api";

type SourceGroupedService = { builtIn: boolean; source?: LLMService["source"] };

/* ── Generic simple service row (ASR / TTS) ── */

function SimpleServiceRow({
  svc, isActive, canRemove, fields, onSelect, onUpdate, onRemove,
}: Readonly<{
  svc: SimpleService; isActive: boolean; canRemove: boolean;
  fields: { label: string; key: keyof SimpleService }[];
  onSelect?: (id: string) => void;
  onUpdate: (id: string, updates: Partial<SimpleService>) => void;
  onRemove: (id: string) => void;
}>) {
  const buildForm = () => {
    const next: Record<string, string> = { name: svc.name };
    fields.forEach(({ key }) => { next[key as string] = String(svc[key] ?? ""); });
    return next;
  };
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<Record<string, string>>(() => buildForm());

  const set = (k: string, v: string) => setForm((p) => ({ ...p, [k]: v }));

  const handleSave = () => {
    if (!form.name?.trim()) return;
    onUpdate(svc.id, form as Partial<SimpleService>);
    setEditing(false);
  };

  if (!svc.builtIn && editing) {
    return (
      <div
        className="svc-row svc-row--editing"
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            handleSave();
          }
          if (e.key === "Escape") {
            setEditing(false);
          }
        }}
      >
        <div className="svc-edit-fields">
          <input className="svc-input" value={form.name ?? ""} onChange={(e) => set("name", e.target.value)} placeholder="Name" autoFocus />
          {fields.map(({ label, key }) => (
            <input key={key as string} className="svc-input" value={form[key as string] ?? ""} onChange={(e) => set(key as string, e.target.value)} placeholder={label} />
          ))}
          <div className="svc-edit-actions">
            <button className="btn-primary svc-add-btn" onClick={handleSave} disabled={!form.name?.trim()}>Save</button>
            <button className="svc-icon-btn" onClick={() => setEditing(false)} title="Cancel">✕</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`svc-row${onSelect ? " svc-row--clickable" : ""}${isActive ? " svc-row--active" : ""}`}
      onClick={onSelect ? () => onSelect(svc.id) : undefined}
      onKeyDown={onSelect ? (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(svc.id);
        }
      } : undefined}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      style={onSelect ? undefined : { opacity: 0.6, cursor: "default" }}
    >
      <div className="svc-row__info">
        <span className="svc-row__name">{svc.name}</span>
        {svc.server && <span className="svc-row__detail svc-row__url">{svc.server}</span>}
        {svc.model && <span className="svc-row__detail">{svc.model}</span>}
        {svc.voiceId && <span className="svc-row__detail">voice: {svc.voiceId}</span>}
        {svc.functionId && <span className="svc-row__detail svc-row__sys">function_id: {svc.functionId}</span>}
      </div>
      <div className="svc-row__actions">
        {isActive && <span className="prompt-card__badge">Active</span>}
        {!svc.builtIn && (
          <>
            <button
              className="svc-icon-btn"
              onClick={(event) => {
                event.stopPropagation();
                setForm(buildForm());
                setEditing(true);
              }}
              title="Edit"
            >
              ✎
            </button>
            {canRemove && <button className="svc-icon-btn svc-icon-btn--remove" onClick={(event) => { event.stopPropagation(); onRemove(svc.id); }} title="Remove">−</button>}
          </>
        )}
      </div>
    </div>
  );
}

/* ── Add form for simple services ── */

function SimpleAddForm({
  fields, onAdd, onCancel,
}: Readonly<{
  fields: { label: string; key: string; required?: boolean }[];
  onAdd: (values: Record<string, string>) => void;
  onCancel: () => void;
}>) {
  const [form, setForm] = useState<Record<string, string>>({});
  const set = (k: string, v: string) => setForm((p) => ({ ...p, [k]: v }));

  const allFields = [{ label: "Display name", key: "name", required: true }, ...fields];
  const canSubmit = allFields.filter((f) => f.required).every((f) => form[f.key]?.trim());

  const handleAdd = () => { if (canSubmit) { onAdd(form); } };

  return (
    <div
      className="svc-add-form"
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          handleAdd();
        }
        if (e.key === "Escape") {
          onCancel();
        }
      }}
    >
      {allFields.map((f) => (
        <input key={f.key} className="svc-input" placeholder={f.label} value={form[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)} autoFocus={f.key === "name"} />
      ))}
      <button className="btn-primary svc-add-btn" onClick={handleAdd} disabled={!canSubmit}>Add</button>
    </div>
  );
}

/* ── LLM service row (unchanged, richer fields) ── */

function LLMServiceRow({ svc, isActive, canRemove, onSelect }: Readonly<{ svc: LLMService; isActive: boolean; canRemove: boolean; onSelect?: (id: string) => void }>) {
  const { updateLLM, removeLLM } = useApp();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(svc.name);
  const [modelId, setModelId] = useState(svc.modelId);
  const [baseUrl, setBaseUrl] = useState(svc.baseUrl);
  const [systemPrompt, setSystemPrompt] = useState(svc.systemPrompt);
  const [extraParams, setExtraParams] = useState(svc.extraParams);

  const resetForm = () => {
    setName(svc.name); setModelId(svc.modelId); setBaseUrl(svc.baseUrl);
    setSystemPrompt(svc.systemPrompt); setExtraParams(svc.extraParams);
  };

  const handleSave = () => {
    if (!name.trim() || !modelId.trim() || !baseUrl.trim()) return;
    updateLLM(svc.id, { name: name.trim(), modelId: modelId.trim(), baseUrl: baseUrl.trim(), systemPrompt: systemPrompt.trim(), extraParams: extraParams.trim() });
    setEditing(false);
  };

  const handleCancel = () => {
    resetForm();
    setEditing(false);
  };

  if (!svc.builtIn && editing) {
    return (
      <div
        className="svc-row svc-row--editing"
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            handleSave();
          }
          if (e.key === "Escape") {
            handleCancel();
          }
        }}
      >
        <div className="svc-edit-fields">
          <input className="svc-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" autoFocus />
          <input className="svc-input" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Server URL" />
          <input className="svc-input" value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder="Model ID" />
          <input className="svc-input" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} placeholder="System prompt (optional)" />
          <input className="svc-input" value={extraParams} onChange={(e) => setExtraParams(e.target.value)} placeholder="Extra params JSON (optional)" />
          <div className="svc-edit-actions">
            <button className="btn-primary svc-add-btn" onClick={handleSave} disabled={!name.trim() || !modelId.trim() || !baseUrl.trim()}>Save</button>
            <button className="svc-icon-btn" onClick={handleCancel} title="Cancel">✕</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`svc-row${onSelect ? " svc-row--clickable" : ""}${isActive ? " svc-row--active" : ""}`}
      onClick={onSelect ? () => onSelect(svc.id) : undefined}
      onKeyDown={onSelect ? (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(svc.id);
        }
      } : undefined}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      style={onSelect ? undefined : { opacity: 0.6, cursor: "default" }}
    >
      <div className="svc-row__info">
        <span className="svc-row__name">{svc.name}</span>
        <span className="svc-row__detail svc-row__url">{svc.baseUrl}</span>
        <span className="svc-row__detail">{svc.modelId}</span>
        {svc.systemPrompt && <span className="svc-row__detail svc-row__sys">sys: {svc.systemPrompt}</span>}
        {svc.extraParams && <span className="svc-row__detail svc-row__sys">extra: {svc.extraParams}</span>}
      </div>
      <div className="svc-row__actions">
        {isActive && <span className="prompt-card__badge">Active</span>}
        {!svc.builtIn && (
          <>
            <button className="svc-icon-btn" onClick={() => { resetForm(); setEditing(true); }} title="Edit">✎</button>
            {canRemove && <button className="svc-icon-btn svc-icon-btn--remove" onClick={() => removeLLM(svc.id)} title="Remove">−</button>}
          </>
        )}
      </div>
    </div>
  );
}

/* ── Read-only row for catalog-owned services ── */

function ReadOnlyServiceRow({ entry, isLocked = false }: Readonly<{ entry: ServiceEntry; isLocked?: boolean }>) {
  const server = entry.server ? String(entry.server) : "";
  const baseUrl = entry.base_url ? String(entry.base_url) : "";
  const modelId = entry.model_id ? String(entry.model_id) : "";
  const extraParams = entry.extra_params ? String(entry.extra_params) : "";
  const timeoutSecs = entry.timeout_secs ? String(entry.timeout_secs) : "";
  const isSelected = entry.selected === true;
  return (
    <div className={`svc-row${isSelected ? " svc-row--active" : ""}${isLocked ? " svc-row--disabled" : ""}`}>
      <div className="svc-row__info">
        <span className="svc-row__name">{entry.name}</span>
        {server && <span className="svc-row__detail svc-row__url">{server}</span>}
        {baseUrl && <span className="svc-row__detail svc-row__url">{baseUrl}</span>}
        {modelId && <span className="svc-row__detail">Model: {modelId}</span>}
        {extraParams && <span className="svc-row__detail">Extra params: {extraParams}</span>}
        {timeoutSecs && <span className="svc-row__detail">Timeout: {timeoutSecs}s</span>}
      </div>
      {isSelected && (
        <div className="svc-row__actions">
          <span className="prompt-card__badge">Active</span>
        </div>
      )}
    </div>
  );
}

/* ── Section wrapper ── */

function ServiceSection({ title, children, onAdd }: Readonly<{ title: string; children: React.ReactNode; onAdd?: () => void }>) {
  return (
    <div style={{ marginBottom: "var(--space-6)" }}>
      <div className="services-header">
        <h3 className="metrics-title">{title}</h3>
        {onAdd && (
          <button className="svc-icon-btn svc-icon-btn--add" onClick={onAdd} title={`Add ${title}`}>+</button>
        )}
      </div>
      <div className="svc-list">{children}</div>
    </div>
  );
}

function ServiceSourceGroup({ title, children }: Readonly<{ title: string; children: React.ReactNode }>) {
  return (
    <div className="svc-group">
      <div className="svc-group__label">{title}</div>
      <div className="svc-list">{children}</div>
    </div>
  );
}

function groupServicesBySource<T extends SourceGroupedService>(items: T[]) {
  const groups: Array<{ key: string; title: string; items: T[] }> = [];
  const selfHosted = items.filter((item) => item.builtIn && item.source === "self-hosted");
  const cloud = items.filter((item) => item.builtIn && item.source === "cloud-nim");
  const custom = items.filter((item) => !item.builtIn);

  if (selfHosted.length > 0) groups.push({ key: "self-hosted", title: "Self-hosted", items: selfHosted });
  if (cloud.length > 0) groups.push({ key: "cloud-nim", title: "NVIDIA Cloud", items: cloud });
  if (custom.length > 0) groups.push({ key: "custom", title: "Custom", items: custom });

  return groups;
}

/* ── Main panel ── */

export function ServicesPanel() {
  const {
    selectedExample,
    llms, llmsLoading, selectedLLMId, selectLLM, addLLM,
    asrServices, asrLoading, selectedASRId, selectASR, addASR, updateASR, removeASR,
    ttsServices, ttsLoading, selectedTTSId, selectTTS, addTTS, updateTTS, removeTTS,
  } = useApp();
  const { data: catalog } = useServiceCatalog(selectedExample?.key ?? "");
  const { isLocked } = useConnectionState();

  const [addingLLM, setAddingLLM] = useState(false);
  const [addingASR, setAddingASR] = useState(false);
  const [addingTTS, setAddingTTS] = useState(false);

  const llmCustomCount = llms.filter((s) => !s.builtIn).length;
  const asrCustomCount = asrServices.filter((s) => !s.builtIn).length;
  const ttsCustomCount = ttsServices.filter((s) => !s.builtIn).length;
  const slotList = selectedExample?.slots ?? [];
  const llmGroups = groupServicesBySource(llms);
  const asrGroups = groupServicesBySource(asrServices);
  const ttsGroups = groupServicesBySource(ttsServices);

  const renderCatalogSection = (slot: string) => {
    const groups = groupServicesBySource(catalog?.[slot] ?? []);
    const title = `${slot.split("-").map((part) => (part.toUpperCase() === "LLM" ? "LLM" : part[0]?.toUpperCase() + part.slice(1))).join(" ")} Services`;
    return (
      <ServiceSection key={slot} title={title}>
        {groups.length === 0 && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>No services configured</p>}
        {groups.map((group) => (
          <ServiceSourceGroup key={group.key} title={group.title}>
            {group.items.map((entry) => <ReadOnlyServiceRow key={entry.id} entry={entry} isLocked={isLocked} />)}
          </ServiceSourceGroup>
        ))}
      </ServiceSection>
    );
  };

  const renderSlot = (slot: string) => {
    if (slot === "llm") {
      return (
        <ServiceSection key={slot} title="LLM Services" onAdd={() => setAddingLLM(!addingLLM)}>
          {llmsLoading && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>Loading...</p>}
          {addingLLM && (
            <SimpleAddForm
              fields={[
                { label: "Server URL", key: "baseUrl", required: true },
                { label: "Model ID", key: "modelId", required: true },
                { label: "System prompt (optional)", key: "systemPrompt" },
                { label: "Extra params JSON (optional)", key: "extraParams" },
              ]}
              onAdd={(v) => { addLLM(v.name, v.modelId, v.baseUrl, v.systemPrompt ?? "", v.extraParams ?? ""); setAddingLLM(false); }}
              onCancel={() => setAddingLLM(false)}
            />
          )}
          {llmGroups.map((group) => (
            <ServiceSourceGroup key={group.key} title={group.title}>
              {group.items.map((svc) => (
                <LLMServiceRow key={svc.id} svc={svc} isActive={selectedLLMId === svc.id} canRemove={!svc.builtIn && llmCustomCount > 0} onSelect={isLocked ? undefined : selectLLM} />
              ))}
            </ServiceSourceGroup>
          ))}
        </ServiceSection>
      );
    }

    if (slot === "asr") {
      return (
        <ServiceSection key={slot} title="ASR Services" onAdd={() => setAddingASR(!addingASR)}>
          {asrLoading && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>Loading...</p>}
          {addingASR && (
            <SimpleAddForm
              fields={[
                { label: "Server (gRPC endpoint)", key: "server", required: true },
                { label: "Model (optional)", key: "model" },
                { label: "Function ID (optional)", key: "functionId" },
              ]}
              onAdd={(v) => { const svc = addASR(v.name, v.server, v.model || undefined); if (v.functionId) updateASR(svc.id, { functionId: v.functionId }); setAddingASR(false); }}
              onCancel={() => setAddingASR(false)}
            />
          )}
          {asrGroups.map((group) => (
            <ServiceSourceGroup key={group.key} title={group.title}>
              {group.items.map((svc) => (
                <SimpleServiceRow
                  key={svc.id} svc={svc} isActive={selectedASRId === svc.id}
                  canRemove={!svc.builtIn && asrCustomCount > 0}
                  fields={[{ label: "Server", key: "server" }, { label: "Model", key: "model" }, { label: "Function ID", key: "functionId" }]}
                  onSelect={isLocked ? undefined : selectASR} onUpdate={updateASR} onRemove={removeASR}
                />
              ))}
            </ServiceSourceGroup>
          ))}
        </ServiceSection>
      );
    }

    if (slot === "tts") {
      return (
        <ServiceSection key={slot} title="TTS Services" onAdd={() => setAddingTTS(!addingTTS)}>
          {ttsLoading && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>Loading...</p>}
          {addingTTS && (
            <SimpleAddForm
              fields={[
                { label: "Server (gRPC endpoint)", key: "server", required: true },
                { label: "Voice ID (optional)", key: "voiceId" },
                { label: "Function ID (optional)", key: "functionId" },
              ]}
              onAdd={(v) => { const svc = addTTS(v.name, v.server, v.voiceId || undefined); if (v.functionId) updateTTS(svc.id, { functionId: v.functionId }); setAddingTTS(false); }}
              onCancel={() => setAddingTTS(false)}
            />
          )}
          {ttsGroups.map((group) => (
            <ServiceSourceGroup key={group.key} title={group.title}>
              {group.items.map((svc) => (
                <SimpleServiceRow
                  key={svc.id} svc={svc} isActive={selectedTTSId === svc.id}
                  canRemove={!svc.builtIn && ttsCustomCount > 0}
                  fields={[{ label: "Server", key: "server" }, { label: "Voice ID", key: "voiceId" }, { label: "Function ID", key: "functionId" }]}
                  onSelect={isLocked ? undefined : selectTTS} onUpdate={updateTTS} onRemove={removeTTS}
                />
              ))}
            </ServiceSourceGroup>
          ))}
        </ServiceSection>
      );
    }

    return renderCatalogSection(slot);
  };

  return (
    <div className="services-panel p-4">
      {slotList.map(renderSlot)}
    </div>
  );
}
