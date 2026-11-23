// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";
import { useApp } from "../../context/useApp";
import { useConnectionState } from "../../hooks/useConnectionState";
import type { Prompt } from "../../api";

function PromptRow({ prompt, isActive, canRemove, disabled }: Readonly<{ prompt: Prompt; isActive: boolean; canRemove: boolean; disabled?: boolean }>) {
  const { updatePrompt, removePrompt } = useApp();
  const [editing, setEditing] = useState(false);
  const [description, setDescription] = useState(prompt.description);
  const [content, setContent] = useState(prompt.content);
  const displayName = prompt.scope === "agent" && prompt.promptName ? prompt.promptName : prompt.key;

  const resetForm = () => {
    setDescription(prompt.description);
    setContent(prompt.content);
  };

  const handleSave = () => {
    if (!content.trim()) return;
    updatePrompt(prompt.key, description.trim(), content.trim());
    setEditing(false);
  };

  const handleCancel = () => {
    resetForm();
    setEditing(false);
  };

  if (!prompt.builtIn && editing) {
    return (
      <div
        className="prompt-card prompt-card--editing"
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            handleCancel();
          }
          if (e.key === "Enter" && !e.shiftKey && e.target === e.currentTarget) {
            handleSave();
          }
        }}
      >
        <div className="svc-edit-fields">
          <input className="svc-input" value={prompt.key} disabled title="Key cannot be changed" />
          <input className="svc-input" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description (optional)" autoFocus />
          <textarea className="svc-input prompt-content-input" value={content} onChange={(e) => setContent(e.target.value)} placeholder="Prompt content" rows={6} />
          <div className="svc-edit-actions">
            <button className="btn-primary svc-add-btn" onClick={handleSave} disabled={!content.trim()}>Save</button>
            <button className="svc-icon-btn" onClick={handleCancel} title="Cancel">✕</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`prompt-card${isActive ? " prompt-card--selected" : ""}${disabled ? " prompt-card--disabled" : ""}`}>
      <div className="prompt-card__header">
        <span className="prompt-card__name">
          {displayName}
        </span>
        <div className="prompt-card__actions">
          {prompt.scope === "agent" && prompt.agent && <span className="prompt-card__badge">{prompt.agent}</span>}
          {isActive && <span className="prompt-card__badge">Active</span>}
          {!prompt.builtIn && (
            <>
              <button className="svc-icon-btn" onClick={() => { if (!disabled) { resetForm(); setEditing(true); } }} disabled={disabled} aria-disabled={disabled} title={disabled ? "Locked during session" : "Edit"}>✎</button>
              {canRemove && <button className="svc-icon-btn svc-icon-btn--remove" onClick={() => { if (!disabled) removePrompt(prompt.key); }} disabled={disabled || !canRemove} aria-disabled={disabled || !canRemove} title={disabled ? "Locked during session" : "Remove"}>−</button>}
            </>
          )}
        </div>
      </div>
      {prompt.description && <p className="prompt-card__desc">{prompt.description}</p>}
      <pre className="prompt-card__content">{prompt.content}</pre>
    </div>
  );
}

function PromptPart({ prompt, name, active, editable, locked }: Readonly<{
  prompt: Prompt;
  name: string;
  active: boolean;
  editable: boolean;
  locked: boolean;
}>) {
  const { promptOverrides, setPromptOverride } = useApp();
  const override = promptOverrides.get(prompt.key);
  const content = override ?? prompt.content;
  const [draft, setDraft] = useState<string | null>(null);
  const canEdit = editable && prompt.builtIn && !locked;

  const save = () => {
    if (!canEdit || !draft?.trim()) return;
    setPromptOverride(prompt.key, draft === prompt.content ? null : draft);
    setDraft(null);
  };

  return (
    <details className="prompt-card">
      <summary className="prompt-part__summary">
        <span className="prompt-card__name">{name}</span>
        {active && <span className="prompt-card__badge">Active</span>}
        {override !== undefined && <span className="prompt-card__badge">Edited</span>}
      </summary>
      <div className="prompt-part__body">
        {prompt.description && <p className="prompt-card__desc">{prompt.description}</p>}
        {draft !== null && canEdit ? (
          <>
            <textarea
              className="svc-input prompt-content-input"
              aria-label={name}
              value={draft}
              rows={8}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Escape") setDraft(null); }}
              autoFocus
            />
            <div className="svc-edit-actions">
              <button className="btn-primary svc-add-btn" onClick={save} disabled={!draft.trim()}>Save</button>
              <button className="svc-icon-btn prompt-part__action" onClick={() => setDraft(null)}>Cancel</button>
            </div>
          </>
        ) : (
          <>
            <pre className="prompt-card__content">{content}</pre>
            {canEdit && (
              <div className="svc-edit-actions">
                <button className="svc-icon-btn prompt-part__action" onClick={() => setDraft(content)}>Edit</button>
                {override !== undefined && (
                  <button className="svc-icon-btn prompt-part__action" onClick={() => setPromptOverride(prompt.key, null)}>Reset</button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </details>
  );
}

function OrderedPrompts({ locked }: Readonly<{ locked: boolean }>) {
  const { promptLayout, prompts, selectedPrompt } = useApp();
  const catalog = new Map(prompts.filter((prompt) => prompt.builtIn).map((prompt) => [prompt.key, prompt]));
  const arrangedKeys = new Set(promptLayout.flatMap(({ parts }) => parts.map((part) =>
    part.active ? selectedPrompt?.key : part.key,
  )));
  const otherPrompts = Array.from(catalog.values()).filter((prompt) => !arrangedKeys.has(prompt.key));

  return (
    <>
      {promptLayout.map(({ id, label, parts }) => (
        <section key={id} className="prompt-part__group" aria-label={label}>
          <h4 className="prompts-section-label">{label}</h4>
          <ol className="prompt-part__list">
            {parts.map((part, index) => {
              const prompt = part.active ? selectedPrompt : part.key ? catalog.get(part.key) : undefined;
              if (!prompt && !part.label) return null;
              return (
                <li key={`${index}:${prompt?.key ?? part.label}`}>
                  {prompt ? (
                    <PromptPart
                      prompt={prompt}
                      name={part.label ?? prompt.promptName ?? prompt.key}
                      active={Boolean(part.active)}
                      editable={Boolean(part.editable)}
                      locked={locked}
                    />
                  ) : (
                    <div className="prompt-part__runtime">
                      <span className="prompt-card__name">{part.label}</span>
                      <span className="prompt-card__badge">Runtime</span>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        </section>
      ))}
      {otherPrompts.length > 0 && (
        <details className="prompt-part__catalog">
          <summary className="prompts-section-label">Other templates</summary>
          <div className="prompts-list">
            {otherPrompts.map((prompt) => (
              <PromptRow key={prompt.key} prompt={prompt} isActive={false} canRemove={false} />
            ))}
          </div>
        </details>
      )}
    </>
  );
}

export function PromptsPanel() {
  const { prompts, promptsLoading, promptLayout, selectedPromptKey, addPrompt } = useApp();
  const { isLocked } = useConnectionState();
  const [adding, setAdding] = useState(false);
  const [key, setKey] = useState("");
  const [description, setDescription] = useState("");
  const [content, setContent] = useState("");
  const [addError, setAddError] = useState("");

  const resetAddForm = () => {
    setKey("");
    setDescription("");
    setContent("");
    setAddError("");
  };

  const customPrompts = prompts.filter((p) => !p.builtIn);
  const defaultPrompts = prompts.filter((p) => p.builtIn && p.scope !== "agent");
  const agentPromptsByAgent = useMemo(
    () =>
      prompts
        .filter((p) => p.builtIn && p.scope === "agent")
        .reduce<Record<string, Prompt[]>>((groups, prompt) => {
          const agent = prompt.agent || "Agent";
          groups[agent] = [...(groups[agent] ?? []), prompt];
          return groups;
        }, {}),
    [prompts],
  );

  const handleAdd = () => {
    if (!key.trim() || !content.trim()) return;
    const err = addPrompt(key.trim(), description.trim(), content.trim());
    if (err) setAddError(err);
    else {
      resetAddForm();
      setAdding(false);
    }
  };

  if (promptsLoading) {
    return (
      <div className="prompts-panel p-4">
        <div className="services-header"><h3 className="metrics-title">Prompts</h3></div>
        <p className="prompts-status">Loading prompts...</p>
      </div>
    );
  }

  return (
    <div className="prompts-panel p-4">
      <div className="services-header">
        <h3 className="metrics-title">Prompts</h3>
        <button
          className="svc-icon-btn svc-icon-btn--add"
          onClick={() => {
            if (adding) {
              resetAddForm();
              setAdding(false);
            } else {
              setAdding(true);
            }
          }}
          title={adding ? "Cancel" : "Add prompt"}
        >
          {adding ? "✕" : "+"}
        </button>
      </div>

      {adding && (
        <div
          className="svc-add-form"
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              resetAddForm();
              setAdding(false);
            }
          }}
        >
          <input className="svc-input" placeholder="Prompt key (e.g. my_assistant)" value={key} onChange={(e) => setKey(e.target.value)} autoFocus />
          <input className="svc-input" placeholder="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />
          <textarea className="svc-input prompt-content-input" placeholder="Prompt content" value={content} onChange={(e) => setContent(e.target.value)} rows={6} />
          {addError && <p className="prompts-status prompts-status--error">{addError}</p>}
          <button className="btn-primary svc-add-btn" onClick={handleAdd} disabled={!key.trim() || !content.trim()}>Add</button>
        </div>
      )}

      {promptLayout.length > 0 && <OrderedPrompts locked={isLocked} />}

      {customPrompts.length > 0 && (
        <>
          <p className="prompts-section-label">Custom Prompts</p>
          <div className="prompts-list">
            {customPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={selectedPromptKey === p.key} canRemove={customPrompts.length > 0} disabled={isLocked} />
            ))}
          </div>
        </>
      )}

      {promptLayout.length === 0 && defaultPrompts.length > 0 && (
        <>
          <p className="prompts-section-label">Default Prompts</p>
          <div className="prompts-list">
            {defaultPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={selectedPromptKey === p.key} canRemove={false} disabled={isLocked} />
            ))}
          </div>
        </>
      )}

      {promptLayout.length === 0 && Object.entries(agentPromptsByAgent).map(([agent, agentPrompts]) => (
        <div key={agent}>
          <p className="prompts-section-label">Agent Prompts: {agent}</p>
          <div className="prompts-list">
            {agentPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={false} canRemove={false} disabled={isLocked} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
