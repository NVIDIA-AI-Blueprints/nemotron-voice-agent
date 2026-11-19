// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { readLSJson } from "../src/utils.ts";

function withStorage(items: Record<string, string>, run: () => void) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: (key: string) => items[key] ?? null },
  });
  try {
    run();
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else Reflect.deleteProperty(globalThis, "localStorage");
  }
}

test("stored JSON reads back, and missing or corrupt values read as undefined", () => {
  withStorage({ good: '[{"key":"a"}]', corrupt: "{not json", empty: "" }, () => {
    assert.deepEqual(readLSJson("good"), [{ key: "a" }]);
    assert.equal(readLSJson("corrupt"), undefined);
    assert.equal(readLSJson("empty"), undefined);
    assert.equal(readLSJson("missing"), undefined);
  });
});

test("a blocked storage reads as undefined", () => {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new Error("SecurityError");
    },
  });
  try {
    assert.equal(readLSJson("any"), undefined);
  } finally {
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});
