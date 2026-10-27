// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HTML the console builds from data. Every value from the server, a model or a tool goes through escapeHtml (or
// textContent in app.js): markup run in this page could read the bearer token.

const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

/** `value` as text that is safe inside an element and inside a quoted attribute; null and undefined are empty. */
export function escapeHtml(value) {
  return value == null ? "" : String(value).replace(/[&<>"']/g, (c) => ENTITIES[c]);
}

const money = (value) => `$${Number(value).toFixed(2)}`;

/** `<option>`s for `values`, the one equal to `selected` chosen; `labels` gives each one's text (default: itself). */
export function optionsHtml(values, selected, labels = values) {
  return values.map((value, index) =>
    `<option value="${escapeHtml(value)}"${value === selected ? " selected" : ""}>${escapeHtml(labels[index])}</option>`
  ).join("");
}

/** The cart view: one row per line, the totals, then the last order. `cart` is the cafe's snapshot. */
export function cartHtml(cart, lastOrder) {
  const lines = cart.lines.map((line) =>
    `<div class="line"><span>${escapeHtml(line.quantity)}× ${line.size ? `${escapeHtml(line.size)} ` : ""}${escapeHtml(line.name)}` +
    `${line.modifiers.length ? ` (${line.modifiers.map(escapeHtml).join(", ")})` : ""} ` +
    `<span style="color:var(--faint)">${escapeHtml(line.line_id)}</span></span>` +
    `<span>${money(line.line_total)}</span></div>`).join("");
  return (lines || `<div class="line" style="color:var(--faint)">cart is empty</div>`) +
    `<div class="line tot"><span>subtotal</span><span>${money(cart.subtotal)}</span></div>` +
    `<div class="line"><span>tax</span><span>${money(cart.tax)}</span></div>` +
    `<div class="line"><span><b>total</b></span><span><b>${money(cart.total)}</b></span></div>` +
    (lastOrder
      ? `<div class="order">order ${escapeHtml(lastOrder.order_id)} for ${escapeHtml(lastOrder.customer_name ?? "guest")}` +
        ` · ready in ${escapeHtml(lastOrder.ready_in_minutes)} min</div>`
      : "");
}

/** The menu: one row per item, then the sizes and the modifiers with their prices. */
export function menuHtml(menu, sizes, modifiers) {
  return menu.map((item) =>
    `<div class="mitem"><span>${escapeHtml(item.id)} · ${escapeHtml(item.name)} ` +
    `<span style="color:var(--faint)">${escapeHtml(item.category)}</span></span><span>${money(item.price)}</span></div>`
  ).join("") +
    `<div class="mitem" style="margin-top:6px"><span>sizes</span><span>${Object.entries(sizes)
      .map(([name, price]) => `${escapeHtml(name)} +${money(price)}`).join(" · ")}</span></div>` +
    `<div class="mitem"><span>modifiers</span><span>${Object.entries(modifiers)
      .map(([name, modifier]) => `${escapeHtml(name)} +${money(modifier.price)}`).join(" · ")}</span></div>`;
}
