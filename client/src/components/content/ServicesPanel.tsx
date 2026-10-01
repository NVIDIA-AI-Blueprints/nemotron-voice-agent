// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useApp } from "../../context/useApp";
import { useConnectionState } from "../../hooks/useConnectionState";
import {
  catalogKey,
  serviceSettingsKey,
  serviceSettingsSchema,
  useServiceCatalog,
  type LLMService,
  type ServiceEntry,
  type ServiceSettingsSchema,
  type SimpleService,
} from "../../api";
import { ServiceSettingsControls, SettingRow } from "../ServiceSettingsControls";
import { Toggle } from "../Toggle";

/* ── Custom ASR / TTS service row ── */

function SimpleServiceRow({
  svc, isActive, fields, onSelect, onUpdate, onRemove,
}: Readonly<{
  svc: SimpleService; isActive: boolean;
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

  if (editing) {
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
        <button className="svc-icon-btn svc-icon-btn--remove" onClick={(event) => { event.stopPropagation(); onRemove(svc.id); }} title="Remove">−</button>
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

/* ── Custom LLM service row ── */

function LLMServiceRow({ svc, isActive, onSelect }: Readonly<{ svc: LLMService; isActive: boolean; onSelect?: (id: string) => void }>) {
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

  if (editing) {
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
        <button className="svc-icon-btn" onClick={() => { resetForm(); setEditing(true); }} title="Edit">✎</button>
        <button className="svc-icon-btn svc-icon-btn--remove" onClick={() => removeLLM(svc.id)} title="Remove">−</button>
      </div>
    </div>
  );
}

/* ── Built-in service card ── */

type ServiceVariant = {
  id: string;
  source?: LLMService["source"];
  details: string[];
  schema?: ServiceSettingsSchema;
  streamingUrl?: string;
};

const SOURCE_LABELS: Record<string, string> = { "self-hosted": "Self-hosted", "cloud-nim": "NVIDIA Cloud" };

function BuiltInServiceCard({
  slot, name, variants, activeId, expanded, streaming, isLocked, onSelect, onToggle, renderSettings,
}: Readonly<{
  slot: string;
  name: string;
  variants: ServiceVariant[];
  activeId: string;
  expanded: boolean;
  streaming: boolean;
  isLocked: boolean;
  onSelect?: (id: string) => void;
  onToggle: () => void;
  renderSettings: (variant: ServiceVariant, settingsKey: string) => React.ReactNode;
}>) {
  const active = variants.find((variant) => variant.id === activeId);
  const shown = active ?? variants[0];
  const canSelect = Boolean(onSelect) && !isLocked;
  const handleClick = () => {
    if (!active && canSelect) onSelect?.(shown.id);
    if (shown.schema) onToggle();
  };

  const showSettings = expanded && Boolean(shown.schema);
  const [endpoint, ...details] = shown.details;
  const shownDetails = [streaming && shown.streamingUrl ? shown.streamingUrl : endpoint, ...details];
  return (
    <div className={`svc-row svc-row--clickable svc-row--card${active ? " svc-row--active" : ""}`}>
      <div
        className="svc-row__header"
        onClick={handleClick}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            handleClick();
          }
        }}
        role="button"
        tabIndex={0}
        aria-expanded={shown.schema ? expanded : undefined}
      >
        <div className="svc-row__info">
          <span className="svc-row__name">{name}</span>
          {shownDetails.map((detail, index) => (
            <span key={detail} className={`svc-row__detail${index === 0 ? " svc-row__url" : ""}`}>{detail}</span>
          ))}
        </div>
        <div className="svc-row__controls">
          <div className="svc-source" role="group" aria-label="Endpoint">
            {variants.map((variant) => (
              <button
                key={variant.id}
                className={`svc-source__option${variant.id === shown.id ? " svc-source__option--selected" : ""}`}
                aria-pressed={variant.id === shown.id}
                disabled={!canSelect || variants.length === 1}
                onClick={(event) => { event.stopPropagation(); onSelect?.(variant.id); }}
              >
                {SOURCE_LABELS[variant.source ?? ""] ?? variant.source}
              </button>
            ))}
          </div>
          {shown.schema && (
            <button
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
      {showSettings && (
        <div className="svc-row__section">
          <p className="prompts-section-label">Parameters</p>
          {renderSettings(shown, serviceSettingsKey(slot, shown.id))}
        </div>
      )}
    </div>
  );
}

function groupByCatalogKey<T extends { id: string; name: string }>(items: T[], toVariant: (item: T) => ServiceVariant) {
  const cards = new Map<string, { name: string; variants: ServiceVariant[] }>();
  for (const item of items) {
    const key = catalogKey(item.id);
    const card = cards.get(key) ?? { name: item.name, variants: [] };
    card.variants.push(toVariant(item));
    cards.set(key, card);
  }
  return [...cards.entries()].map(([key, card]) => ({ key, ...card }));
}

function compact(values: Array<string | undefined>): string[] {
  return values.filter((value): value is string => Boolean(value));
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

function CustomServicesGroup({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="svc-group">
      <div className="svc-group__label">Custom</div>
      <div className="svc-list">{children}</div>
    </div>
  );
}

/* ── Main panel ── */

export function ServicesPanel() {
  const {
    selectedExample,
    llms, llmsLoading, selectedLLMId, selectLLM, addLLM,
    asrServices, asrLoading, selectedASRId, selectASR, addASR, updateASR, removeASR,
    ttsServices, ttsLoading, selectedTTSId, selectTTS, addTTS, updateTTS, removeTTS,
    serviceSettings, setServiceSetting, streamingInput, setStreamingInput,
  } = useApp();

  const { data: catalog } = useServiceCatalog(selectedExample?.key ?? "");
  const { isLocked } = useConnectionState();

  const [addingLLM, setAddingLLM] = useState(false);
  const [addingASR, setAddingASR] = useState(false);
  const [addingTTS, setAddingTTS] = useState(false);
  const [expandedCard, setExpandedCard] = useState("");

  const slotList = selectedExample?.slots ?? [];
  const canStream = selectedExample?.capabilities?.includes("streaming_input") ?? false;

  const renderSettings = (variant: ServiceVariant, settingsKey: string) => variant.schema && (
    <ServiceSettingsControls
      schema={variant.schema}
      values={serviceSettings[settingsKey] ?? {}}
      disabled={isLocked}
      onChange={(name, value) => setServiceSetting(settingsKey, name, value)}
    >
      {variant.streamingUrl && (
        <SettingRow label="Streaming Input">
          <Toggle
            checked={Boolean(streamingInput[settingsKey])}
            onChange={(enabled) => setStreamingInput(settingsKey, enabled)}
            label="Streaming Input"
            disabled={isLocked}
          />
        </SettingRow>
      )}
    </ServiceSettingsControls>
  );

  const renderCards = (
    slot: string,
    cards: ReturnType<typeof groupByCatalogKey>,
    activeId: string,
    onSelect?: (id: string) => void,
  ) => cards.map((card) => {
    const cardId = `${slot}:${card.key}`;
    return (
      <BuiltInServiceCard
        key={cardId}
        slot={slot}
        name={card.name}
        variants={card.variants}
        activeId={activeId}
        expanded={expandedCard === cardId}
        streaming={Boolean(streamingInput[serviceSettingsKey(slot, card.variants[0].id)])}
        isLocked={isLocked}
        onSelect={onSelect}
        onToggle={() => setExpandedCard(expandedCard === cardId ? "" : cardId)}
        renderSettings={renderSettings}
      />
    );
  });

  const llmCards = groupByCatalogKey(llms.filter((s) => s.builtIn), (s) => ({
    id: s.id,
    source: s.source,
    details: compact([s.baseUrl, s.modelId]),
    schema: s.settings,
    streamingUrl: canStream ? s.streamingUrl : undefined,
  }));
  const asrCards = groupByCatalogKey(asrServices.filter((s) => s.builtIn), (s) => ({
    id: s.id, source: s.source, details: compact([s.server, s.model]), schema: s.settings,
  }));
  const ttsCards = groupByCatalogKey(ttsServices.filter((s) => s.builtIn), (s) => ({
    id: s.id, source: s.source, details: compact([s.server, s.voiceId && `voice: ${s.voiceId}`]), schema: s.settings,
  }));

  const renderCatalogSection = (slot: string) => {
    const entries: ServiceEntry[] = catalog?.[slot] ?? [];
    const cards = groupByCatalogKey(entries, (entry) => ({
      id: entry.id,
      source: entry.source,
      details: compact([String(entry.server ?? entry.base_url ?? ""), String(entry.model ?? entry.model_id ?? "")]),
      schema: serviceSettingsSchema(entry),
    }));
    const activeId = entries.find((entry) => entry.selected === true)?.id ?? "";
    const title = `${slot.split("-").map((part) => (part.toUpperCase() === "LLM" ? "LLM" : part[0]?.toUpperCase() + part.slice(1))).join(" ")} Services`;
    return (
      <ServiceSection key={slot} title={title}>
        {cards.length === 0 && <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)" }}>No services configured</p>}
        {renderCards(slot, cards, activeId)}
      </ServiceSection>
    );
  };

  const renderSlot = (slot: string) => {
    if (slot === "llm") {
      const custom = llms.filter((s) => !s.builtIn);
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
          {renderCards(slot, llmCards, selectedLLMId, selectLLM)}
          {custom.length > 0 && (
            <CustomServicesGroup>
              {custom.map((svc) => (
                <LLMServiceRow key={svc.id} svc={svc} isActive={selectedLLMId === svc.id} onSelect={isLocked ? undefined : selectLLM} />
              ))}
            </CustomServicesGroup>
          )}
        </ServiceSection>
      );
    }

    if (slot === "asr") {
      const custom = asrServices.filter((s) => !s.builtIn);
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
          {renderCards(slot, asrCards, selectedASRId, selectASR)}
          {custom.length > 0 && (
            <CustomServicesGroup>
              {custom.map((svc) => (
                <SimpleServiceRow
                  key={svc.id} svc={svc} isActive={selectedASRId === svc.id}
                  fields={[{ label: "Server", key: "server" }, { label: "Model", key: "model" }, { label: "Function ID", key: "functionId" }]}
                  onSelect={isLocked ? undefined : selectASR} onUpdate={updateASR} onRemove={removeASR}
                />
              ))}
            </CustomServicesGroup>
          )}
        </ServiceSection>
      );
    }

    if (slot === "tts") {
      const custom = ttsServices.filter((s) => !s.builtIn);
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
          {renderCards(slot, ttsCards, selectedTTSId, selectTTS)}
          {custom.length > 0 && (
            <CustomServicesGroup>
              {custom.map((svc) => (
                <SimpleServiceRow
                  key={svc.id} svc={svc} isActive={selectedTTSId === svc.id}
                  fields={[{ label: "Server", key: "server" }, { label: "Voice ID", key: "voiceId" }, { label: "Function ID", key: "functionId" }]}
                  onSelect={isLocked ? undefined : selectTTS} onUpdate={updateTTS} onRemove={removeTTS}
                />
              ))}
            </CustomServicesGroup>
          )}
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
