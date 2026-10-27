// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The browser cafe (client/public/live/cafe.js) gives the server's tools.py results, which cafe-parity.json records
// (tests/unit/cross_box/test_console_cafe_parity.py). Run: node --test client/test/live/*.test.js

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { CafeTools, money, pyRepr, wholeNumber } from "../../public/live/cafe.js";

const fixture = JSON.parse(await readFile(new URL("./cafe-parity.json", import.meta.url), "utf8"));

/** `value` with every `placed_at` replaced by "<placed_at>", as the fixture records it. */
function withoutClock(value) {
  if (Array.isArray(value)) return value.map(withoutClock);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, key === "placed_at" ? "<placed_at>" : withoutClock(item)]));
  }
  return value;
}

for (const fixtureCase of fixture.cases) {
  test(`as tools.py: ${fixtureCase.name}`, async () => {
    const cafe = new CafeTools({ latency: {} });
    const results = [];
    for (const call of fixtureCase.calls) {
      results.push(withoutClock(JSON.parse(await cafe.execute(call.call_id, call.name, call.arguments))));
    }
    assert.deepEqual(results, fixtureCase.results);
  });
}

test("the string \"false\", or any value but true, never places an order", async () => {
  const cafe = new CafeTools({ latency: {} });
  await cafe.execute("a", "manage_cart", JSON.stringify({ action: "add", item_id: "lat" }));
  for (const [index, confirmed] of ["false", "yes", 1, {}, [true]].entries()) {
    const result = JSON.parse(await cafe.execute(`p${index}`, "place_order", JSON.stringify({ confirmed })));
    assert.equal(result.status, "needs_confirmation", JSON.stringify(confirmed));
  }
  assert.equal(cafe.lastOrder, null);
  assert.equal(JSON.parse(await cafe.execute("go", "place_order", JSON.stringify({ confirmed: true }))).status, "confirmed");
  assert.equal(cafe.lastOrder.order_id, "BB-42");
});

test("a repeated call id, even while the first still runs, adds one line", async () => {
  const cafe = new CafeTools({ latency: { manage_cart: 20 } });
  const args = JSON.stringify({ action: "add", item_id: "esp" });
  const [first, second] = await Promise.all([cafe.execute("same", "manage_cart", args), cafe.execute("same", "manage_cart", args)]);
  assert.equal(first, second);
  assert.equal(cafe.snapshot().lines.length, 1);
});

test("arguments that are not a JSON object are an error result", async () => {
  const cafe = new CafeTools({ latency: {} });
  assert.equal(JSON.parse(await cafe.execute("a", "get_menu", "[1]")).error, "'list' object has no attribute 'get'");
  assert.equal(JSON.parse(await cafe.execute("b", "get_menu", "{")).status, "error");
});

test("a service delay is waited, and place_order waits only to place the order", async () => {
  const cafe = new CafeTools({ latency: { place_order: 60 } });
  await cafe.execute("a", "manage_cart", JSON.stringify({ action: "add", item_id: "lat" }));
  let started = performance.now();
  await cafe.execute("b", "place_order", "{}");
  assert.ok(performance.now() - started < 50);
  started = performance.now();
  await cafe.execute("c", "place_order", JSON.stringify({ confirmed: true }));
  assert.ok(performance.now() - started >= 55);
});

test("an order's placed_at is tools.py's isoformat", async () => {
  const cafe = new CafeTools({ latency: {}, now: () => new Date(Date.UTC(2026, 9, 27, 1, 2, 3, 45)) });
  await cafe.execute("a", "manage_cart", JSON.stringify({ action: "add", item_id: "lat" }));
  const result = JSON.parse(await cafe.execute("b", "place_order", JSON.stringify({ confirmed: true })));
  assert.equal(result.order.placed_at, "2026-10-27T01:02:03.045000+00:00");
});

test("money rounds the shortest decimal half up, as Decimal does", () => {
  assert.equal(money(1.005), 1.01);   // binary rounding gives 1.00
  assert.equal(money(2.675), 2.68);   // binary rounding gives 2.67
  assert.equal(money(0.255), 0.26);
  assert.equal(money(12.345), 12.35);
  assert.equal(money(-1.005), -1.01); // half up is away from zero
  assert.equal(money(1.5e-7), 0);
  assert.equal(money(3), 3);
  assert.equal(money(0.1 + 0.2), 0.3);
});

test("whole numbers as Python's int() reads them, else the default", () => {
  assert.equal(wholeNumber("3", 1), 3);
  assert.equal(wholeNumber(" 1_0 ", 1), 10);
  assert.equal(wholeNumber("+4", 1), 4);
  assert.equal(wholeNumber("2.5", 1), 1);
  assert.equal(wholeNumber("abc", 1), 1);
  assert.equal(wholeNumber(2.9, 1), 2);
  assert.equal(wholeNumber(-2.9, 1), -2);
  assert.equal(wholeNumber(true, 7), 1);
  assert.equal(wholeNumber(null, 7), 7);
  assert.equal(wholeNumber(Number.NaN, 7), 7);
  assert.equal(wholeNumber([3], 7), 7);
});

test("repr as Python writes it in tools.py's messages", () => {
  assert.equal(pyRepr("Latte"), "'Latte'");
  assert.equal(pyRepr("it's"), `"it's"`);
  assert.equal(pyRepr(`it's "x"`), `'it\\'s "x"'`);
  assert.equal(pyRepr("a\nb\\"), "'a\\nb\\\\'");
  assert.equal(pyRepr(null), "None");
  assert.equal(pyRepr(true), "True");
  assert.equal(pyRepr(5), "5");
  assert.equal(pyRepr([1, "a"]), "[1, 'a']");
  assert.equal(pyRepr({ k: null }), "{'k': None}");
});
