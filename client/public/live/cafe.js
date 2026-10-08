// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mock Bluebird Cafe backend — runs entirely in the browser.
//
// These are the *client tools*. The backend model decides to call them; this file is what actually executes.
// The schemas the backend sees come from GET /v1/live/info (the example's own cafe tool schemas).

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

export const SIZES = { small: 0.0, medium: 0.5, large: 1.0 };
export const MODIFIERS = {
  oat_milk: { label: "Oat milk", price: 0.75 },
  extra_shot: { label: "Extra shot", price: 1.0 },
  decaf: { label: "Decaf", price: 0.0 },
  iced: { label: "Iced", price: 0.0 },
};

const TAX_RATE = 0.085;

/** Mutable demo state. The cart is the source of truth the delegate is told to trust. */
export const state = {
  cart: [],
  lineSeq: 0,
  orderSeq: 41,
  lastOrder: null,
};

const round2 = (n) => Math.round(n * 100) / 100;

function findItem(id) {
  if (!id) return null;
  const needle = String(id).toLowerCase().trim();
  return (
    MENU.find((m) => m.id === needle) ||
    MENU.find((m) => m.name.toLowerCase() === needle) ||
    MENU.find((m) => m.name.toLowerCase().includes(needle)) ||
    null
  );
}

function linePrice(line) {
  const item = findItem(line.item_id);
  const size = item.sizable ? SIZES[line.size] ?? 0 : 0;
  const mods = (line.modifiers || []).reduce((sum, m) => sum + (MODIFIERS[m]?.price ?? 0), 0);
  return round2((item.price + size + mods) * line.quantity);
}

export function cartSnapshot() {
  const lines = state.cart.map((line) => {
    const item = findItem(line.item_id);
    return {
      line_id: line.line_id,
      item_id: item.id,
      name: item.name,
      quantity: line.quantity,
      size: item.sizable ? line.size : null,
      modifiers: line.modifiers,
      line_total: linePrice(line),
    };
  });
  const subtotal = round2(lines.reduce((sum, l) => sum + l.line_total, 0));
  const tax = round2(subtotal * TAX_RATE);
  return {
    lines,
    item_count: lines.reduce((sum, l) => sum + l.quantity, 0),
    subtotal,
    tax,
    total: round2(subtotal + tax),
    currency: "USD",
  };
}

// ---------------------------------------------------------------------------------
// Tool implementations. Each returns { result, latencyMs } and may be slow on purpose.
// ---------------------------------------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function get_menu(args) {
  await sleep(120); // pretend this is a menu service
  const category = args.category || "all";
  const query = (args.query || "").toLowerCase().trim();
  const items = MENU.filter(
    (m) =>
      (category === "all" || m.category === category) &&
      (!query || m.name.toLowerCase().includes(query) || m.notes.toLowerCase().includes(query))
  ).map((m) => ({
    id: m.id,
    name: m.name,
    category: m.category,
    price: m.price,
    sizes: m.sizable ? SIZES : null,
    notes: m.notes,
  }));
  return {
    items,
    modifiers: MODIFIERS,
    tax_rate: TAX_RATE,
    note: items.length ? undefined : "No menu item matched that filter.",
  };
}

async function manage_cart(args) {
  await sleep(180); // pretend this is an order service
  const action = args.action;

  if (action === "view") return { action, cart: cartSnapshot() };

  if (action === "clear") {
    state.cart = [];
    return { action, status: "cleared", cart: cartSnapshot() };
  }

  if (action === "add") {
    const item = findItem(args.item_id);
    if (!item) {
      return { action, status: "error", error: `Unknown item_id '${args.item_id}'. Call get_menu for valid ids.`, cart: cartSnapshot() };
    }
    const quantity = Math.max(1, Math.min(20, parseInt(args.quantity ?? 1, 10) || 1));
    const size = item.sizable ? (SIZES[args.size] !== undefined ? args.size : "medium") : null;
    const modifiers = (args.modifiers || []).filter((m) => MODIFIERS[m]);
    const line = {
      line_id: `line_${++state.lineSeq}`,
      item_id: item.id,
      quantity,
      size,
      modifiers,
    };
    state.cart.push(line);
    return {
      action,
      status: "added",
      line_id: line.line_id,
      applied_size_default: item.sizable && args.size === undefined ? "medium" : undefined,
      cart: cartSnapshot(),
    };
  }

  if (action === "update_quantity" || action === "remove") {
    const index = state.cart.findIndex((l) => l.line_id === args.line_id);
    if (index === -1) {
      return { action, status: "error", error: `Unknown line_id '${args.line_id}'. Call manage_cart with action 'view' to list lines.`, cart: cartSnapshot() };
    }
    if (action === "remove") {
      state.cart.splice(index, 1);
      return { action, status: "removed", cart: cartSnapshot() };
    }
    const quantity = Math.max(0, Math.min(20, parseInt(args.quantity ?? 1, 10) || 0));
    if (quantity === 0) state.cart.splice(index, 1);
    else state.cart[index].quantity = quantity;
    return { action, status: quantity === 0 ? "removed" : "updated", cart: cartSnapshot() };
  }

  return { action, status: "error", error: `Unsupported action '${action}'.`, cart: cartSnapshot() };
}

async function place_order(args) {
  const cart = cartSnapshot();
  if (!cart.lines.length) {
    return { status: "error", error: "The cart is empty. Nothing to order.", cart };
  }
  if (!args.confirmed) {
    return {
      status: "needs_confirmation",
      confirmation_summary: cart.lines
        .map((l) => `${l.quantity} ${l.size ? l.size + " " : ""}${l.name}`)
        .join(", "),
      total: cart.total,
      cart,
      next_step: "Read this back to the user and call place_order again with confirmed=true once they say yes.",
    };
  }
  // Deliberately slow: this is the window where progress commentary is useful.
  await sleep(2600);
  const order = {
    order_id: `BB-${++state.orderSeq}`,
    customer_name: args.customer_name || "guest",
    ready_in_minutes: 4 + Math.floor(Math.random() * 4),
    total: cart.total,
    lines: cart.lines,
    placed_at: new Date().toISOString(),
  };
  state.lastOrder = order;
  state.cart = [];
  return { status: "confirmed", order, cart: cartSnapshot() };
}

const IMPLEMENTATIONS = { get_menu, manage_cart, place_order };

/** Names of the functions this client executes. web_search is NOT here: it is hosted. */
export const CLIENT_TOOL_NAMES = Object.keys(IMPLEMENTATIONS);

/**
 * Execute one function call requested by the delegate.
 * Returns the JSON string that goes into function_call_output.output.
 */
export async function executeTool(name, argsJson) {
  const started = performance.now();
  let args = {};
  let parseError = null;
  try {
    args = argsJson ? JSON.parse(argsJson) : {};
  } catch (error) {
    parseError = String(error);
  }

  let result;
  if (parseError) {
    result = { status: "error", error: `Could not parse arguments: ${parseError}` };
  } else if (!IMPLEMENTATIONS[name]) {
    result = { status: "error", error: `No client implementation for tool '${name}'.` };
  } else {
    try {
      result = await IMPLEMENTATIONS[name](args);
    } catch (error) {
      result = { status: "error", error: String(error) };
    }
  }
  return { args, result, latencyMs: Math.round(performance.now() - started) };
}
