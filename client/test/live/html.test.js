// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The HTML the console builds from data (client/public/live/html.js). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { test } from "node:test";

import { cartHtml, escapeHtml, menuHtml, optionsHtml } from "../../public/live/html.js";

const ATTACK = `"><img src=x onerror="alert(1)">`;
const SCRIPT = "<script>steal(token)</script>";

/** Whether `html` holds markup that came from data: a tag or an attribute break that the page did not write. */
const injected = (html) => /<img|<script|onerror="/.test(html);

test("escapeHtml escapes the five characters that end text or an attribute", () => {
  assert.equal(escapeHtml(`<a href="x" title='y'>&</a>`), "&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;");
  assert.equal(escapeHtml(null), "");
  assert.equal(escapeHtml(undefined), "");
  assert.equal(escapeHtml(42), "42");
});

test("a customer name, an item and a line id from a tool are text in the cart", () => {
  const cart = {
    lines: [{ quantity: 1, size: ATTACK, name: SCRIPT, modifiers: [ATTACK], line_id: SCRIPT, line_total: 4.5 }],
    subtotal: 4.5, tax: 0.38, total: 4.88,
  };
  const html = cartHtml(cart, { order_id: ATTACK, customer_name: SCRIPT, ready_in_minutes: ATTACK });
  assert.equal(injected(html), false, html);
  assert.match(html, /&lt;script&gt;steal\(token\)&lt;\/script&gt;/);
  assert.match(html, /\$4\.50/);
  assert.match(html, /\$4\.88/);
});

test("an empty cart and an order with no name", () => {
  const html = cartHtml({ lines: [], subtotal: 0, tax: 0, total: 0 }, { order_id: "BB-42", customer_name: null, ready_in_minutes: 4 });
  assert.match(html, /cart is empty/);
  assert.match(html, /order BB-42 for guest · ready in 4 min/);
});

test("voices and models from the server are text in the select's options", () => {
  const html = optionsHtml(["marin", ATTACK], "marin");
  assert.equal(injected(html), false, html);
  assert.match(html, /<option value="marin" selected>marin<\/option>/);
  // A datalist: values only.
  assert.equal(optionsHtml(["m1"], null, []), `<option value="m1"></option>`);
  // Labels of their own.
  assert.equal(optionsHtml(["0"], "0", ["Greet first"]), `<option value="0" selected>Greet first</option>`);
});

test("the menu is escaped too", () => {
  const html = menuHtml(
    [{ id: "x", name: SCRIPT, category: ATTACK, price: 3 }], { small: 0 }, { [SCRIPT]: { price: 0.75 } },
  );
  assert.equal(injected(html), false, html);
  assert.match(html, /\$3\.00/);
  assert.match(html, /small \+\$0\.00/);
});
