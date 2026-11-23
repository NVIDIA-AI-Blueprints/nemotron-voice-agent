// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useApp } from "../../context/useApp";
import { useSession } from "../../session/session-context";
import type { Prompt } from "../../api";

function PromptRow({ prompt, isActive, disabled }: Readonly<{ prompt: Prompt; isActive: boolean; disabled: boolean }>) {
  const { updatePrompt, removePrompt } = useApp();
  const [editing, setEditing] = useState(false);
  const [description, setDescription] = useState(prompt.description);
  const [content, setContent] = useState(prompt.content);
  const [saveError, setSaveError] = useState("");

  const resetForm = () => {
    setDescription(prompt.description);
    setContent(prompt.content);
    setSaveError("");
  };

  const handleSave = () => {
    if (!content.trim() || disabled) return;
    const err = updatePrompt(prompt.key, description.trim(), content.trim());
    if (err) setSaveError(err);
    else setEditing(false);
  };

  const handleCancel = () => {
    resetForm();
    setEditing(false);
  };

  if (!prompt.builtIn && editing) {
    return (
      <form
        className="prompt-card prompt-card--editing"
        onSubmit={(e) => { e.preventDefault(); handleSave(); }}
        onKeyDown={(e) => { if (e.key === "Escape") handleCancel(); }}
      >
        <fieldset className="svc-edit-fields" disabled={disabled}>
          <input className="svc-input" value={prompt.key} aria-label="Prompt key" disabled title="Key cannot be changed" />
          <input className="svc-input" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description (optional)" aria-label="Description" autoFocus />
          <textarea className="svc-input prompt-content-input" value={content} onChange={(e) => setContent(e.target.value)} placeholder="Prompt content" aria-label="Prompt content" rows={6} />
          {saveError && <p className="prompts-status prompts-status--error">{saveError}</p>}
          <div className="svc-edit-actions">
            <button type="submit" className="btn-primary svc-add-btn" disabled={!content.trim()}>Save</button>
            <button type="button" className="svc-icon-btn" onClick={handleCancel} title="Cancel">✕</button>
          </div>
        </fieldset>
      </form>
    );
  }

  const lockedTitle = (action: string) => (disabled ? "Locked during session" : action);
  return (
    <div className={`prompt-card${isActive ? " prompt-card--selected" : ""}${disabled ? " prompt-card--disabled" : ""}`}>
      <div className="prompt-card__header">
        <span className="prompt-card__name">{prompt.label}</span>
        <div className="prompt-card__actions">
          {isActive && <span className="prompt-card__badge">Active</span>}
          {!prompt.builtIn && (
            <>
              <button type="button" className="svc-icon-btn" onClick={() => { resetForm(); setEditing(true); }} disabled={disabled} title={lockedTitle("Edit")}>✎</button>
              <button type="button" className="svc-icon-btn svc-icon-btn--remove" onClick={() => removePrompt(prompt.key)} disabled={disabled} title={lockedTitle("Remove")}>−</button>
            </>
          )}
        </div>
      </div>
      {prompt.description && <p className="prompt-card__desc">{prompt.description}</p>}
      <pre className="prompt-card__content">{prompt.content}</pre>
    </div>
  );
}

export function PromptsPanel() {
  const { prompts, promptsLoading, selectedPromptKey, addPrompt } = useApp();
  const { isLocked } = useSession();
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

  const closeAddForm = () => {
    resetAddForm();
    setAdding(false);
  };

  const customPrompts = prompts.filter((p) => !p.builtIn);
  const defaultPrompts = prompts.filter((p) => p.builtIn);

  const handleAdd = () => {
    if (isLocked) return;
    const err = addPrompt(key, description, content);
    if (err) setAddError(err);
    else closeAddForm();
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
          type="button"
          className="svc-icon-btn svc-icon-btn--add"
          onClick={() => (adding ? closeAddForm() : setAdding(true))}
          disabled={isLocked}
          title={isLocked ? "Locked during session" : adding ? "Cancel" : "Add prompt"}
        >
          {adding ? "✕" : "+"}
        </button>
      </div>

      {adding && (
        <form
          className="svc-add-form"
          onSubmit={(e) => { e.preventDefault(); handleAdd(); }}
          onKeyDown={(e) => { if (e.key === "Escape") closeAddForm(); }}
        >
          <fieldset className="d-flex flex-col gap-2" disabled={isLocked}>
            <input className="svc-input" placeholder="Prompt key (e.g. my_assistant)" aria-label="Prompt key" value={key} onChange={(e) => setKey(e.target.value)} autoFocus />
            <input className="svc-input" placeholder="Description (optional)" aria-label="Description" value={description} onChange={(e) => setDescription(e.target.value)} />
            <textarea className="svc-input prompt-content-input" placeholder="Prompt content" aria-label="Prompt content" value={content} onChange={(e) => setContent(e.target.value)} rows={6} />
            <p className="prompts-section-label">A custom prompt runs without tools, on the default use case's greeting.</p>
            {addError && <p className="prompts-status prompts-status--error">{addError}</p>}
            <button type="submit" className="btn-primary svc-add-btn" disabled={!key.trim() || !content.trim()}>Add</button>
          </fieldset>
        </form>
      )}

      {customPrompts.length > 0 && (
        <>
          <p className="prompts-section-label">Custom Prompts</p>
          <div className="prompts-list">
            {customPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={selectedPromptKey === p.key} disabled={isLocked} />
            ))}
          </div>
        </>
      )}

      {defaultPrompts.length > 0 && (
        <>
          <p className="prompts-section-label">Default Prompts</p>
          <div className="prompts-list">
            {defaultPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={selectedPromptKey === p.key} disabled={isLocked} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
