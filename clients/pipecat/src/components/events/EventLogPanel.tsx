// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FAMILIES, familyOf, formatTime, summarize, type Family, type LiveEvent } from "../../liveEvents/liveEvent";
import { filterRows, missingCount, toJsonLines } from "../../liveEvents/log";
import { useLiveEvents } from "../../liveEvents/useLiveEvents";
import "../../styles/event-log.scss";

// How close to the bottom the list must be, in pixels, to count as following the newest event.
const FOLLOW_SLACK = 24;

interface RowProps {
  row: LiveEvent;
  open: boolean;
  onToggle: (seq: number) => void;
}

const EventRow = memo(function EventRow({ row, open, onToggle }: Readonly<RowProps>) {
  const family = familyOf(row);
  const line = summarize(row);
  return (
    <li className="evlog__item" data-family={family} data-open={open || undefined}>
      <button
        type="button"
        className="evlog__row"
        aria-expanded={open}
        onClick={() => onToggle(row.seq)}
        title={`${row.kind}${line ? ` — ${line}` : ""}`}
      >
        <span className="evlog__t">{formatTime(row.t)}</span>
        <span className="evlog__dir" aria-label={row.direction === "out" ? "sent to the backend" : "from the backend"}>
          {row.direction === "out" ? "→" : "←"}
        </span>
        <span className="evlog__stream">{row.stream ?? ""}</span>
        <span className="evlog__kind">{row.kind}</span>
        <span className="evlog__sum">{line}</span>
      </button>
      {open && <pre className="evlog__json">{JSON.stringify(row.event, null, 2)}</pre>}
    </li>
  );
});

/** The Event log: the requests the voice agent sends to its backend and the OpenAI-format events that stream back. */
export function EventLogPanel({ active = true }: Readonly<{ active?: boolean }>) {
  const { rows, dropped, callEnded, clear } = useLiveEvents(active);
  const [hidden, setHidden] = useState<ReadonlySet<Family>>(() => new Set());
  const [open, setOpen] = useState<ReadonlySet<number>>(() => new Set());
  const [follow, setFollow] = useState(true);
  const [note, setNote] = useState("");
  const scroller = useRef<HTMLDivElement>(null);

  const shown = useMemo(() => filterRows(rows, hidden), [rows, hidden]);
  const missing = useMemo(() => missingCount(rows), [rows]);

  // Follow the newest event, as `tail -f` does, until the reader scrolls up; scrolling back to the bottom follows again.
  useEffect(() => {
    const node = scroller.current;
    if (follow && node) node.scrollTop = node.scrollHeight;
  }, [follow, shown.length]);

  // Events that arrive while the tab is hidden cannot be scrolled to (a hidden list has no height), so a list that is following
  // goes to the bottom when the tab is shown, and when the window is resized.
  const following = useRef(true);
  useEffect(() => {
    following.current = follow;
  }, [follow]);
  useEffect(() => {
    const node = scroller.current;
    if (!node || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      if (following.current) node.scrollTop = node.scrollHeight;
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const onScroll = useCallback(() => {
    const node = scroller.current;
    if (!node) return;
    setFollow(node.scrollHeight - node.scrollTop - node.clientHeight <= FOLLOW_SLACK);
  }, []);

  const jumpToLatest = useCallback(() => {
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
    setFollow(true);
  }, []);

  const toggleFamily = useCallback((family: Family) => {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(family)) next.delete(family);
      else next.add(family);
      return next;
    });
  }, []);

  const toggleRow = useCallback((seq: number) => {
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!note) return undefined;
    const timer = setTimeout(() => setNote(""), 2500);
    return () => clearTimeout(timer);
  }, [note]);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(toJsonLines(shown));
      setNote(`Copied ${shown.length} ${shown.length === 1 ? "event" : "events"}`);
    } catch {
      setNote("Copy is blocked in this browser");
    }
  }, [shown]);

  const onClear = useCallback(() => {
    clear();
    setOpen(new Set());
    setFollow(true);
  }, [clear]);

  const countText =
    shown.length === rows.length
      ? `${rows.length} ${rows.length === 1 ? "event" : "events"}`
      : `${shown.length} of ${rows.length} events`;

  return (
    <section className="evlog" aria-label="Event log">
      <header className="evlog__bar">
        <div className="evlog__title">
          <h2>Event log</h2>
          <p>Requests to the backend and the OpenAI-format events it streams back, as they happen.</p>
        </div>
        <div className="evlog__tools">
          <div className="evlog__filters" role="group" aria-label="Show event types">
            {FAMILIES.map((family) => (
              <button
                key={family.id}
                type="button"
                className="evlog__chip"
                data-family={family.id}
                aria-pressed={!hidden.has(family.id)}
                onClick={() => toggleFamily(family.id)}
              >
                {family.label}
              </button>
            ))}
          </div>
          <button type="button" className="evlog__btn" onClick={copy} disabled={shown.length === 0}>
            Copy JSON
          </button>
          <button type="button" className="evlog__btn" onClick={onClear} disabled={rows.length === 0}>
            Clear
          </button>
        </div>
      </header>

      <p className="evlog__meta" role="status">
        <span>{countText}</span>
        {callEnded && <span className="evlog__badge">Call ended</span>}
        {dropped > 0 && <span className="evlog__badge evlog__badge--warn">{dropped} oldest dropped</span>}
        {missing > 0 && <span className="evlog__badge evlog__badge--warn">{missing} not received</span>}
        {note && <span className="evlog__note">{note}</span>}
      </p>

      <div className="evlog__frame">
        <div ref={scroller} className="evlog__scroll" onScroll={onScroll} tabIndex={0} aria-label="Events, oldest first">
          {rows.length === 0 ? (
            <p className="evlog__empty">
              No events yet. Start a call: each request the voice agent sends to its backend, and every event that streams
              back, shows up here.
            </p>
          ) : shown.length === 0 ? (
            <p className="evlog__empty">Every type is hidden. Turn one back on above.</p>
          ) : (
            <ol className="evlog__list">
              {shown.map((row) => (
                <EventRow key={row.seq} row={row} open={open.has(row.seq)} onToggle={toggleRow} />
              ))}
            </ol>
          )}
        </div>
        {!follow && shown.length > 0 && (
          <button type="button" className="evlog__jump" onClick={jumpToLatest}>
            Jump to latest ↓
          </button>
        )}
      </div>
    </section>
  );
}
