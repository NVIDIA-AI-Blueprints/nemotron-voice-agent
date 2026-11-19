// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from "react";

const PINNED_DISTANCE_PX = 96;

/** Keep the nearest scrolling ancestor at the bottom while the user has not scrolled away from it. */
export function useStickToBottom<T>(dependency: T) {
  const anchorRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLElement | null>(null);
  const pinnedRef = useRef(true);

  const resolveScroller = useCallback((): HTMLElement | null => {
    if (scrollerRef.current) return scrollerRef.current;
    let element = anchorRef.current?.parentElement ?? null;
    while (element) {
      const overflowY = window.getComputedStyle(element).overflowY;
      if (overflowY === "auto" || overflowY === "scroll") {
        scrollerRef.current = element;
        return element;
      }
      element = element.parentElement;
    }
    return null;
  }, []);

  const scrollToBottom = useCallback(() => {
    const scroller = resolveScroller();
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }, [resolveScroller]);

  useEffect(() => {
    const scroller = resolveScroller();
    if (!scroller) return;
    const onScroll = () => {
      const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      pinnedRef.current = distanceFromBottom <= PINNED_DISTANCE_PX;
    };
    onScroll();
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => scroller.removeEventListener("scroll", onScroll);
  }, [resolveScroller]);

  useEffect(() => {
    const scroller = resolveScroller();
    const content = anchorRef.current?.parentElement;
    if (!scroller || !content) return;
    const observer = new ResizeObserver(() => {
      if (pinnedRef.current) scroller.scrollTop = scroller.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [resolveScroller]);

  useEffect(() => {
    if (!pinnedRef.current) return;
    const frame = requestAnimationFrame(scrollToBottom);
    return () => cancelAnimationFrame(frame);
  }, [dependency, scrollToBottom]);

  return anchorRef;
}
