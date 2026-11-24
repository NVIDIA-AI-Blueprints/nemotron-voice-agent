# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unfinished native owners of exact task continuations, shared across adapter renewal."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable, Iterable

from .protocol import Runtime


class Retirement:
    """A native owner's positive retirement proof, including any still-owned ancestors."""

    def __init__(self, registry: Retirements, keys: tuple[tuple[Hashable, str], ...], ancestors: set[Retirement]):
        """Capture exact unfinished predecessors before this run can suspend or begin native work."""
        self.registry, self.keys, self.ancestors = registry, keys, ancestors
        self._ready: asyncio.Future[None] | None = None
        self.qualified = False
        self.dependents: set[Retirement] = set()
        self.receipt: asyncio.Future[bool] | None = None
        self.runtime: Runtime | None = None
        self.native_closed = False
        self.local_ready = False
        self.finalize: Callable[[], None] | None = None
        self.error: Exception | None = None
        for ancestor in ancestors:
            ancestor.dependents.add(self)

    @property
    def ready(self) -> asyncio.Future[None]:
        """Create a loop-bound waiter only when a run actually starts awaiting ownership."""
        if self._ready is None:
            self._ready = asyncio.get_running_loop().create_future()
            if self.qualified:
                self._ready.set_result(None)
            elif self.registry.closed:
                self._ready.cancel()
        return self._ready

    async def wait(self) -> None:
        """Wait for predecessors without transferring cancellation to their cleanup owners."""
        for ancestor in self.ancestors:
            await asyncio.shield(ancestor.ready)

    def finish(
        self,
        receipt: asyncio.Future[bool] | None = None,
        *,
        runtime: Runtime | None = None,
        finalize: Callable[[], None] | None = None,
    ) -> None:
        """Record local cleanup once; a completed unqualified receipt never grants continuation."""
        if self.local_ready or self.receipt is not None:
            return
        self.receipt, self.runtime, self.finalize = receipt, runtime, finalize
        if runtime is not None:
            cleanup = runtime.cleanup_receipt
            self.native_closed = cleanup.confirmed and cleanup.scope == "runtime"
        if receipt is None:
            self.local_ready = True
            self._finish()
        else:
            receipt.add_done_callback(lambda _: self._read_receipt())
            self._read_receipt()

    def _read_receipt(self) -> None:
        receipt = self.receipt
        if self.local_ready or receipt is None or not receipt.done() or receipt.cancelled():
            return
        if receipt.exception() is None and (receipt.result() or self.native_closed):
            if self.finalize is not None:
                try:
                    self.finalize()
                except Exception as error:
                    self.error = error
                    return
            self.local_ready = True
            self._finish()

    def _finish(self) -> None:
        if self.qualified or not self.local_ready:
            return
        if any(not ancestor.qualified for ancestor in self.ancestors):
            return
        self.qualified = True
        if self._ready is not None and not self._ready.done():
            self._ready.set_result(None)
        self.registry.forget(self)
        for dependent in tuple(self.dependents):
            dependent._finish()
        self.dependents.clear()
        for ancestor in self.ancestors:
            ancestor.dependents.discard(self)


class Retirements:
    """Retain only unfinished owners; conversation references isolate names and confer no attachment authority."""

    def __init__(self):
        """Hold unresolved owners only; no event loop or runtime starts."""
        self.owners: dict[tuple[Hashable, str], set[Retirement]] = {}
        self.closed = False

    def register(
        self,
        namespace: Hashable,
        task: str,
        work: str,
        predecessors: Iterable[str],
    ) -> Retirement:
        """Register one task and its Work alias without overwriting any still-owned sibling."""
        if self.closed:
            raise RuntimeError("The harness source is closing")
        ancestors = {owner for reference in predecessors for owner in self.owners.get((namespace, reference), ())}
        keys = tuple(dict.fromkeys(((namespace, task), (namespace, work))))
        owner = Retirement(self, keys, ancestors)
        for key in keys:
            self.owners.setdefault(key, set()).add(owner)
        return owner

    def forget(self, owner: Retirement) -> None:
        """Remove aliases only after this owner and every inherited predecessor qualified."""
        for key in owner.keys:
            owners = self.owners.get(key)
            if owners is not None:
                owners.discard(owner)
                if not owners:
                    self.owners.pop(key)

    def native_retired(self, runtime: Runtime) -> None:
        """Apply independent confirmed physical shutdown only after each owner's tracked bridge calls settle."""
        for owner in {owner for owners in self.owners.values() for owner in owners}:
            if owner.runtime is runtime:
                owner.native_closed = True
                owner._read_receipt()

    def close(self) -> None:
        """Stop registration and release registry references without cancelling physical cleanup receipts."""
        self.closed = True
        for owner in {owner for owners in self.owners.values() for owner in owners}:
            if owner._ready is not None and not owner._ready.done():
                owner._ready.cancel()
        self.owners.clear()
