# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background writer: the only place where recording touches disk or the DB.

Producers (observer, event handlers) call ``put_row`` / ``put_task`` which never
block and never raise; a single asyncio task drains the queue in batches and runs
each batch in a worker thread. If the queue overflows (storage far slower than
the conversation) items are dropped and counted rather than slowing the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from monitoring.store import ArtifactStore, SessionStore

# A deferred unit of work run in the writer thread; returns rows to insert.
WriterTask = Callable[[ArtifactStore], list[tuple[str, dict[str, Any]]]]
# Work that needs everything queued before it to be committed (derived metrics).
PostCommitTask = Callable[[SessionStore], None]

_STOP = object()


class RecordWriter:
    """Batching, non-blocking writer bound to one session."""

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        artifacts: ArtifactStore,
        *,
        max_queue: int = 20_000,
        batch_size: int = 200,
        flush_interval_secs: float = 0.25,
        heartbeat_secs: float = 15.0,
    ):
        """Bind the writer to a session and its stores."""
        self.session_id = session_id
        self._store = store
        self._artifacts = artifacts
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval = flush_interval_secs
        self._heartbeat_secs = heartbeat_secs
        self._last_heartbeat = 0.0
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.max_depth = 0

    def start(self) -> None:
        """Start the background writer task."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"record-writer-{self.session_id}")

    def put_row(self, table: str, row: dict[str, Any]) -> None:
        """Queue a row insert (never blocks)."""
        self._put(("row", table, row))

    def put_task(self, fn: WriterTask) -> None:
        """Queue a deferred artifact task (never blocks)."""
        self._put(("task", fn))

    def put_post_commit(self, fn: PostCommitTask) -> None:
        """Queue work run in the writer thread once everything queued before it is stored (never blocks)."""
        self._put(("post", fn))

    def _put(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                logger.warning(f"Recorder queue full; dropped {self.dropped} item(s)")
            return
        depth = self._queue.qsize()
        if depth > self.max_depth:
            self.max_depth = depth

    async def close(self, timeout_secs: float = 30.0) -> None:
        """Flush everything queued so far and stop the writer task."""
        if self._task is None:
            return
        await self._queue.put(_STOP)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_secs)
        except TimeoutError:
            logger.warning(f"Recorder writer did not drain within {timeout_secs:.0f}s; cancelling")
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        stopping = False
        while not stopping:
            batch = []
            try:
                # Wake up at least every heartbeat so a silent session still looks alive.
                item = await asyncio.wait_for(self._queue.get(), timeout=self._heartbeat_secs)
            except TimeoutError:
                item = None
            if item is None:
                pass
            elif item is _STOP:
                stopping = True
            else:
                batch.append(item)
                deadline = time.monotonic() + self._flush_interval
                while len(batch) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                    if item is _STOP:
                        stopping = True
                        break
                    batch.append(item)
            if stopping:
                # Drain whatever is left without waiting.
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is not _STOP:
                        batch.append(item)
            try:
                await asyncio.to_thread(self._process, batch)
            except Exception as exc:  # never let storage errors kill the session
                logger.opt(exception=exc).error(f"Recorder failed to persist {len(batch)} item(s)")

    def _process(self, batch: list[Any]) -> None:
        rows: list[tuple[str, dict[str, Any]]] = []

        def flush() -> None:
            if rows:
                self._store.write_batch(rows)
                rows.clear()

        for item in batch:
            if item[0] == "row":
                rows.append((item[1], item[2]))
            elif item[0] == "post":
                flush()
                try:
                    item[1](self._store)
                except Exception as exc:
                    logger.opt(exception=exc).error("Recorder post-commit task failed")
            else:
                try:
                    rows.extend(item[1](self._artifacts))
                except Exception as exc:
                    logger.opt(exception=exc).error("Recorder artifact task failed")
        flush()
        now = time.time()
        if now - self._last_heartbeat >= self._heartbeat_secs:
            self._store.touch_session(self.session_id, now)
            self._last_heartbeat = now
