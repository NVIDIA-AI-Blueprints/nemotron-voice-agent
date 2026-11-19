// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Prompt, PromptEntry } from "./types.ts";

/** The longest custom prompt that the UI server accepts. */
export const MAX_PROMPT_CHARS = 8000;

export function fromCatalog(entry: PromptEntry): Prompt {
  return {
    key: entry.id,
    label: entry.name,
    description: entry.description,
    content: entry.content,
    examples: entry.examples,
    tools: entry.tools,
    default: entry.default,
    builtIn: true,
  };
}

export function customPrompt(key: string, description: string, content: string): Prompt {
  return { key, label: key, description, content, examples: [], tools: [], default: false, builtIn: false };
}

/** Return the well-formed custom prompts from stored JSON. Malformed entries are dropped. */
export function parseStoredPrompts(value: unknown): Prompt[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item: unknown) => {
    if (typeof item !== "object" || item === null) return [];
    const { key, description, content } = item as Record<string, unknown>;
    if (typeof key !== "string" || !key || typeof description !== "string" || typeof content !== "string") return [];
    return [customPrompt(key, description, content)];
  });
}

/** Turn a typed name into a prompt key: lower case, with underscores for spaces. */
export function promptKey(name: string): string {
  return name.trim().toLowerCase().replaceAll(/\s+/g, "_");
}

/** Return why a custom prompt cannot be added, or "" when it can. */
export function customPromptError(key: string, content: string, existing: readonly Prompt[]): string {
  if (!key) return "A prompt key is required";
  if (existing.some((prompt) => prompt.key === key)) return `Prompt '${key}' already exists`;
  return promptContentError(content);
}

/** Return why prompt content cannot be saved, or "" when it can. */
export function promptContentError(content: string): string {
  if (!content.trim()) return "Prompt content is required";
  if (content.trim().length > MAX_PROMPT_CHARS) return `Prompt content is limited to ${MAX_PROMPT_CHARS} characters`;
  return "";
}

/** Return the key of the prompt to use: the chosen one while it exists, else the server default, else the first. */
export function effectivePromptKey(chosen: string, prompts: readonly Prompt[]): string {
  if (chosen && prompts.some((prompt) => prompt.key === chosen)) return chosen;
  return prompts.find((prompt) => prompt.default)?.key ?? prompts[0]?.key ?? "";
}
