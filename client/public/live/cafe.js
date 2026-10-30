// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mock Bluebird Cafe backend — runs entirely in the browser.
//
// These are the *client tools*. The backend model decides to call them; this file is what actually executes.
// The schemas the backend sees come from GET /v1/live/info (the example's own cafe tool schemas).
//
// They follow src/delegation_core/backend/tools/cafe/tools.py rule for rule, and the two change together:
// client/test/live/cafe-parity.json holds tools.py's results for a set of calls, which cafe.test.js checks here and
// tests/unit/cross_box/test_console_cafe_parity.py checks there. Python's types make a few inputs differ: arguments
// that are not a JSON object, a list or object as a size, modifier, action or query, the line id a failed add uses
// up in tools.py, a float such as 1.0 (1 here), and digits other than 0-9 in a quantity.

export const MENU = [
  { id: "esp", name: "Espresso",         category: "coffee", price: 3.0,  sizable: true,  notes: "Double shot, pulled to order." },
  { id: "lat", name: "Latte",            category: "coffee", price: 4.5,  sizable: true,  notes: "Espresso with steamed milk." },
  { id: "cap", name: "Cappuccino",       category: "coffee", price: 4.25, sizable: true,  notes: "Equal parts espresso, milk, foam." },
  { id: "cbr", name: "Cold Brew",        category: "coffee", price: 4.75, sizable: true,  notes: "18-hour steep, served over ice." },
  { id: "mch", name: "Matcha Latte",     category: "tea",    price: 5.0,  sizable: true,  notes: "Ceremonial grade matcha." },
  { id: "erl", name: "Earl Grey",        category: "tea",    price: 3.25, sizable: true,  notes: "Bergamot black tea." },
  { id: "crs", name: "Butter Croissant", category: "pastry", price: 3.75, sizable: false, notes: "Baked this morning." },
  { id: "bmf", name: "Blueberry Muffin", category: "pastry", price: 3.5,  sizable: false, notes: "Made with Maine blueberries." },
  { id: "avt", name: "Avocado Toast",    category: "food",   price: 8.5,  sizable: false, notes: "Sourdough, chili flake, lemon." },
];

export const SIZES = { small: 0, medium: 0.5, large: 1 };
export const MODIFIERS = {
  oat_milk: { label: "Oat milk", price: 0.75 },
  extra_shot: { label: "Extra shot", price: 1 },
  decaf: { label: "Decaf", price: 0 },
  iced: { label: "Iced", price: 0 },
};
export const TAX_RATE = 0.085;

/** The functions this client executes, as the schemas name them; a hosted tool such as web_search is not one. */
export const CLIENT_TOOL_NAMES = ["get_menu", "manage_cart", "place_order"];

/** Simulated service delays in ms, which leave time for progress commentary; `place_order` only when it places. */
export const LATENCY_MS = { get_menu: 120, manage_cart: 180, place_order: 2600 };

/** A tool's refusal: the call's result is `{status: "error", error: message}`, as tools.py's caught exceptions. */
class CafeError extends Error {}

// --------------------------------------------------------------------------------- Python's rules, in JavaScript
const own = (table, key) => typeof key === "string" && Object.hasOwn(table, key);
/** `args.get(key)`: the value, or null when it is missing. */
const get = (args, key) => (Object.hasOwn(args, key) ? args[key] : null);
/** Python's truth value: null, false, 0, "" and an empty list or object are false. */
const truthy = (value) =>
  Array.isArray(value) ? value.length > 0
    : value !== null && typeof value === "object" ? Object.keys(value).length > 0
      : Boolean(value);

/** Python's type name of a JSON value. */
function pyType(value) {
  if (value === null || value === undefined) return "NoneType";
  if (Array.isArray(value)) return "list";
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
  if (typeof value === "string") return "str";
  return "dict";
}

/** Python's repr() of a JSON value, as tools.py's messages write one with `!r`. */
export function pyRepr(value) {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  if (typeof value === "object") return `{${Object.entries(value).map(([k, v]) => `${pyRepr(k)}: ${pyRepr(v)}`).join(", ")}}`;
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = [...value].map((c) => {
    if (c === "\\") return "\\\\";
    if (c === quote) return `\\${c}`;
    if (c === "\n") return "\\n";
    if (c === "\r") return "\\r";
    if (c === "\t") return "\\t";
    const code = c.codePointAt(0);
    return code < 0x20 || code === 0x7f ? `\\x${code.toString(16).padStart(2, "0")}` : c;
  }).join("");
  return `${quote}${escaped}${quote}`;
}

/** tools.py's `_whole_number`: `int(value)`, or `fallback` when it is missing or not a number. */
export function wholeNumber(value, fallback) {
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") return Number.isFinite(value) ? Math.trunc(value) : fallback;
  if (typeof value === "string" && /^\s*[+-]?\d+(_\d+)*\s*$/.test(value)) return parseInt(value.replace(/_/g, ""), 10);
  return fallback;
}

/**
 * tools.py's `money`: the price to cents, half up, from its shortest decimal (`Decimal(str(value))`), exact even where
 * binary rounding is not (1.005 is 1.01).
 */
export function money(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number === 0) return number;
  const [mantissa, exponent = "0"] = String(Math.abs(number)).split("e");
  const [whole, fraction = ""] = mantissa.split(".");
  const scale = fraction.length - Number(exponent);  // the decimal digits after the point
  const digits = BigInt(whole + fraction);
  let cents;
  if (scale <= 2) {
    cents = digits * 10n ** BigInt(2 - scale);
  } else {
    const unit = 10n ** BigInt(scale - 2);
    cents = digits / unit + (2n * (digits % unit) >= unit ? 1n : 0n);
  }
  return (number < 0 ? -1 : 1) * Number(cents) / 100;
}

/** `datetime.now(UTC).isoformat()`: microseconds, and an offset of +00:00. */
function isoUtc(date) {
  const [stamp, millis] = date.toISOString().slice(0, -1).split(".");
  return `${stamp}${millis === "000" ? "" : `.${millis}000`}+00:00`;
}

/** A tool call's arguments: a JSON object. */
function parseArguments(text) {
  let args;
  try {
    args = JSON.parse(text);
  } catch (error) {
    throw new CafeError(error.message);
  }
  if (pyType(args) !== "dict") throw new CafeError(`'${pyType(args)}' object has no attribute 'get'`);
  return args;
}

/** The modifiers a call names: Python iterates a list, a string's characters, or an object's keys. */
function names(value) {
  if (!truthy(value)) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return [...value];
  if (typeof value === "object") return Object.keys(value);
  throw new CafeError(`'${pyType(value)}' object is not iterable`);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * One cart and its orders, in this page (tools.py's `CafeTools`). No real order is ever placed. `latency`: the delay of
 * each tool's call, in ms, by tool name (a tool it leaves out has none); `now()`: the clock of an order's
 * `placed_at`.
 */
export class CafeTools {
  constructor({ latency = LATENCY_MS, now = () => new Date() } = {}) {
    this.lines = [];
    this.lineSeq = 0;
    this.orderSeq = 41;
    this.lastOrder = null;
    this.byCall = new Map();  // each call_id's output, so a repeated call never adds or orders twice
    this.latency = latency;
    this.now = now;
  }

  /** The cart with line totals, subtotal, tax and total. */
  snapshot() {
    const lines = this.lines.map((line) => {
      const item = MENU.find((x) => x.id === line.item_id);
      let price = item.price + (item.sizable ? SIZES[line.size] : 0);
      price += line.modifiers.reduce((sum, m) => sum + MODIFIERS[m].price, 0);
      return { ...line, name: item.name, line_total: money(price * line.quantity) };
    });
    const subtotal = money(lines.reduce((sum, x) => sum + x.line_total, 0));
    const tax = money(subtotal * TAX_RATE);
    return {
      lines,
      item_count: lines.reduce((sum, x) => sum + x.quantity, 0),
      subtotal,
      tax,
      total: money(subtotal + tax),
      currency: "USD",
    };
  }

  /** Run one call; returns its result as JSON text. A repeated `call_id` returns the first call's result. */
  execute(callId, name, argumentsJson) {
    if (!this.byCall.has(callId)) this.byCall.set(callId, this.run(name, argumentsJson));
    return this.byCall.get(callId);
  }

  async run(name, argumentsJson) {
    let result;
    if (!CLIENT_TOOL_NAMES.includes(name)) {
      result = { status: "error", error: `Unknown tool: ${name}` };
    } else {
      try {
        const args = parseArguments(argumentsJson);
        const delay = name === "place_order" && get(args, "confirmed") !== true ? 0 : this.latency[name] || 0;
        if (delay) await sleep(delay);
        result = { get_menu: () => this.getMenu(args), manage_cart: () => this.manageCart(args), place_order: () => this.placeOrder(args) }[name]();
      } catch (error) {
        result = { status: "error", error: error instanceof CafeError ? error.message : String(error.message || error) };
      }
    }
    return JSON.stringify(result);
  }

  /** Menu items matching `category` and `query`, with the modifiers and the tax rate. */
  getMenu(args) {
    const query = String(truthy(get(args, "query")) ? get(args, "query") : "").toLowerCase().trim();
    const category = truthy(get(args, "category")) ? get(args, "category") : "all";
    const items = MENU
      .filter((x) => (category === "all" || category === x.category) &&
        (!query || x.name.toLowerCase().includes(query) || x.notes.toLowerCase().includes(query)))
      .map((x) => ({ id: x.id, name: x.name, category: x.category, price: x.price, sizes: x.sizable ? SIZES : null, notes: x.notes }));
    return { items, modifiers: MODIFIERS, tax_rate: TAX_RATE };
  }

  /** View or change the cart; returns the full cart afterwards. */
  manageCart(args) {
    if (!Object.hasOwn(args, "action")) throw new CafeError("'action'");
    const action = args.action;
    let status = null;
    if (action === "add") {
      const item = MENU.find((x) => x.id === get(args, "item_id"));
      if (!item) throw new CafeError(`Unknown item_id ${pyRepr(get(args, "item_id"))}. Call get_menu for valid ids.`);
      const size = get(args, "size");
      const modifiers = names(get(args, "modifiers")).filter((m) => own(MODIFIERS, m));
      this.lineSeq += 1;
      this.lines.push({
        line_id: `line_${this.lineSeq}`,
        item_id: item.id,
        quantity: Math.max(1, Math.min(20, wholeNumber(get(args, "quantity"), 1))),
        size: item.sizable ? (own(SIZES, size) ? size : "medium") : null,
        modifiers,
      });
      status = "added";
    } else if (action === "clear") {
      this.lines = [];
      status = "cleared";
    } else if (action === "remove" || action === "update_quantity") {
      const line = this.lines.find((x) => x.line_id === get(args, "line_id"));
      if (!line) throw new CafeError(`Unknown line_id ${pyRepr(get(args, "line_id"))}. View the cart for its lines.`);
      const quantity = Math.max(0, Math.min(20, wholeNumber(get(args, "quantity"), 1)));
      if (action === "remove" || quantity === 0) {
        this.lines.splice(this.lines.indexOf(line), 1);
        status = "removed";
      } else {
        line.quantity = quantity;
        status = "updated";
      }
    } else if (action !== "view") {
      throw new CafeError("Unsupported cart action");
    }
    const result = { action, cart: this.snapshot() };
    if (status) result.status = status;
    return result;
  }

  /** A read-back summary, or the order placed when `confirmed` is exactly true. */
  placeOrder(args) {
    const cart = this.snapshot();
    if (!cart.lines.length) return { status: "error", error: "The cart is empty.", cart };
    if (get(args, "confirmed") !== true) {
      return {
        status: "needs_confirmation",
        cart,
        total: cart.total,
        confirmation_summary: cart.lines.map((x) => `${x.quantity} ${x.name}`).join(", "),
      };
    }
    this.orderSeq += 1;
    this.lastOrder = {
      order_id: `BB-${this.orderSeq}`,
      customer_name: Object.hasOwn(args, "customer_name") ? args.customer_name : "guest",
      ready_in_minutes: 4,
      total: cart.total,
      lines: cart.lines,
      placed_at: isoUtc(this.now()),
    };
    this.lines = [];
    return { status: "confirmed", order: this.lastOrder, cart: this.snapshot() };
  }
}
