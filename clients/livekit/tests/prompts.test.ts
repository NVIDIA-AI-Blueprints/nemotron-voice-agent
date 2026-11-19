// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_PROMPT_CHARS,
  customPrompt,
  customPromptError,
  effectivePromptKey,
  fromCatalog,
  parseStoredPrompts,
  promptContentError,
  promptKey,
} from "../src/lib/prompts.ts";

const generic = fromCatalog({
  id: "generic",
  name: "Generic assistant",
  description: "Answers anything",
  examples: ["Hi"],
  content: "You are helpful.",
  tools: [],
  default: true,
});

test("a catalog entry becomes a built-in prompt labeled with its name", () => {
  assert.equal(generic.key, "generic");
  assert.equal(generic.label, "Generic assistant");
  assert.ok(generic.builtIn && generic.default);
});

test("a typed name becomes a key in lower case with underscores", () => {
  assert.equal(promptKey("  My Bank  Agent "), "my_bank_agent");
});

test("a custom prompt needs a new key and some content within the server limit", () => {
  const existing = [generic, customPrompt("mine", "", "x")];
  assert.equal(customPromptError("brief", "Answer briefly.", existing), "");
  assert.match(customPromptError("", "x", existing), /key/);
  assert.match(customPromptError("mine", "x", existing), /already exists/);
  assert.match(customPromptError("generic", "x", existing), /already exists/);
  assert.match(customPromptError("brief", "  ", existing), /content/);
  assert.match(customPromptError("brief", "x".repeat(MAX_PROMPT_CHARS + 1), existing), /limited/);
});

test("edited prompt content keeps the same limit and errors as a new prompt", () => {
  assert.equal(promptContentError("x".repeat(MAX_PROMPT_CHARS)), "");
  assert.equal(promptContentError(" "), customPromptError("brief", " ", []));
  assert.equal(
    promptContentError("x".repeat(MAX_PROMPT_CHARS + 1)),
    customPromptError("brief", "x".repeat(MAX_PROMPT_CHARS + 1), []),
  );
  assert.match(promptContentError("x".repeat(9000)), /limited to 8000 characters/);
});

test("the chosen prompt is kept while it exists, else the default is used", () => {
  const prompts = [customPrompt("mine", "", "x"), generic];
  assert.equal(effectivePromptKey("mine", prompts), "mine");
  assert.equal(effectivePromptKey("gone", prompts), "generic");
  assert.equal(effectivePromptKey("", [customPrompt("mine", "", "x")]), "mine");
  assert.equal(effectivePromptKey("", []), "");
});

test("stored custom prompts keep only well-formed entries, as custom prompts", () => {
  const stored = [
    { key: "brief", description: "Short", content: "Answer briefly.", builtIn: true, tools: ["x"] },
    null,
    "brief",
    { key: "", description: "", content: "x" },
    { key: 3, description: "", content: "x" },
    { key: "nodesc", content: "x" },
    { key: "badcontent", description: "", content: { text: "x" } },
  ];
  assert.deepEqual(parseStoredPrompts(stored), [customPrompt("brief", "Short", "Answer briefly.")]);
  assert.deepEqual(parseStoredPrompts({ key: "brief" }), []);
  assert.deepEqual(parseStoredPrompts(undefined), []);
});
