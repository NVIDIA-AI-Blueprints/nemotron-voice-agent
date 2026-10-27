// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The event log's bounded entries; app.js draws and removes their rows. A payload becomes text only when a filter
// searches it or its row opens.

/** The most entries (and rows) the log keeps; the oldest go first. */
export const LOG_LIMIT = 2000;

/** The newest entries, at most `limit`. */
export class EventLog {
  constructor(limit = LOG_LIMIT) {
    this.limit = limit;
    this.entries = [];
  }

  /** Add `entry`; returns the entries dropped to stay within the limit, oldest first. */
  add(entry) {
    this.entries.push(entry);
    return this.entries.length > this.limit ? this.entries.splice(0, this.entries.length - this.limit) : [];
  }

  clear() {
    this.entries = [];
  }

  /** The entries as the **export** button saves them: their own fields, not the page's. */
  exported() {
    return this.entries.map(({ at, cat, dir, type, text, payload, actor }) => ({ at, cat, dir, type, text, payload, actor }));
  }
}

/** The lowercase text a filter searches in an entry: made the first time a filter needs it, then kept. */
export function searchText(entry) {
  if (entry.search === undefined) {
    entry.search = `${entry.type} ${entry.text || ""} ${JSON.stringify(entry.payload ?? "")}`.toLowerCase();
  }
  return entry.search;
}

/** A payload as its row shows it when opened. */
export const payloadText = (payload) => JSON.stringify(payload, null, 2);
