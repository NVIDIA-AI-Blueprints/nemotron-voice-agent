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

export function PromptsPanel() {
  const { prompts, promptsLoading, selectedPromptKey, addPrompt } = useApp();
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

      {defaultPrompts.length > 0 && (
        <>
          <p className="prompts-section-label">Default Prompts</p>
          <div className="prompts-list">
            {defaultPrompts.map((p) => (
              <PromptRow key={p.key} prompt={p} isActive={selectedPromptKey === p.key} canRemove={false} disabled={isLocked} />
            ))}
          </div>
        </>
      )}

      {Object.entries(agentPromptsByAgent).map(([agent, agentPrompts]) => (
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
