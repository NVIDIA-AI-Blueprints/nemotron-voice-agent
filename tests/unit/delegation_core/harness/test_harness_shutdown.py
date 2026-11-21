# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""Final source shutdown joins only its own physical cleanup, including failed opens."""

import asyncio
import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples.frontend_backend_live.delegation_core.app import DelegationCore
from examples.frontend_backend_live.delegation_core.backend.harness import adapter as adapter_module
from examples.frontend_backend_live.delegation_core.backend.harness import runtime as runtime_module
from examples.frontend_backend_live.delegation_core.backend.harness.bridge import Door
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import NativeCleanup, Role
from examples.frontend_backend_live.delegation_core.backend.harness.runtime import RuntimeOpener, RuntimeProcess
from examples.frontend_backend_live.delegation_core.backend.harness.source import HarnessSource
from examples.frontend_backend_live.delegation_core.backend.harness.workdir import NATIVE_OWNER, WorkFolder
from tests.unit.delegation_core.harness.doubles import settings, vocabulary
from tests.unit.delegation_core.helpers import CORE_SETTINGS, DEFAULTS, ENDPOINT, turns, until


class DeferredClient:
    """A client with explicit logical and physical close gates; no SDK or subprocess is used."""

    def __init__(self, *, fail_open=False, close_gate=None, cleanup_error=None, close_error=None):
        """Control a fake client's close without starting any runtime."""
        self.fail_open, self.close_gate, self.cleanup_error = fail_open, close_gate, cleanup_error
        self.close_error = close_error
        self.release, self.waiting = asyncio.Event(), asyncio.Event()
        self.physical_closed = False
        self.close_count = 0
        self.alive = True
        self.folder = None

    async def open(self):
        if self.fail_open:
            raise RuntimeError("scripted open failure")

    async def close(self):
        self.close_count += 1
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.alive = False
        if self.close_error is not None:
            raise self.close_error

    async def wait_closed(self):
        self.waiting.set()
        await self.release.wait()
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.physical_closed = True
        return NativeCleanup("runtime", True, True)


class UnconfirmedClient:
    """A failed ACP-style close has no independent physical-shutdown observation."""

    def __init__(self, error):
        """Hold an original failure without a wait_closed capability or a real native process."""
        self.close_error = error
        self.release, self.close_gate = asyncio.Event(), None
        self.close_count = 0
        self.physical_closed = False
        self.alive = True

    async def open(self):
        raise RuntimeError("scripted open failure")

    async def close(self):
        self.close_count += 1
        raise self.close_error

    async def wait_closed(self):
        raise self.close_error


class FailedRelease:
    """A conductor or worker release can fail before the remaining processes are joined."""

    def __init__(self, error):
        """Preserve the exact exception from a failed release."""
        self.error = error

    async def close(self):
        raise self.error

    async def release(self):
        raise self.error


class SourceShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="harness-final-close-")))
        self.cleanups = set(runtime_module._CLEANUPS)
        self.drains = set(adapter_module._DRAINS)
        self.clients, self.sources, self.folders = [], [], []

    async def asyncTearDown(self):
        for client in self.clients:
            client.release.set()
            if client.close_gate is not None:
                client.close_gate.set()
        owned = (set(runtime_module._CLEANUPS) - self.cleanups) | (set(adapter_module._DRAINS) - self.drains)
        await asyncio.gather(*owned, return_exceptions=True)
        await asyncio.gather(*(source.aclose() for source in self.sources), return_exceptions=True)
        # These folders contain only this fixture's fake clients, never a native process or uncertain effect.
        for folder in self.folders:
            folder.remove()

    def source(self):
        source = HarnessSource(
            settings(work_root=self.root), endpoints=lambda _: ENDPOINT, tools=lambda: [], vocabulary=vocabulary()
        )
        self.sources.append(source)
        return source

    def runtime(self, source, *, client=None, **changes):
        client = client or DeferredClient(**changes)
        self.clients.append(client)
        folder = WorkFolder.make(self.root)
        folder.protect_native_owner()
        self.folders.append(folder)
        client.folder = folder.path
        owner = source.new()
        opening = asyncio.get_running_loop().create_future()
        opening.set_result(RuntimeProcess(client, folder))
        owner._shared = {"workers": [opening]}
        return owner, client

    async def failed_open(self, source, *, client=None, **changes):
        client = client or DeferredClient(fail_open=True, **changes)
        self.clients.append(client)
        owner = source.new()

        def make_client(launch, **_):
            client.folder = launch.folder
            return client

        block = SimpleNamespace(
            work_root=self.root,
            command=("never-started",),
            pass_env=(),
            env={},
            agent=SimpleNamespace(environment=lambda: {}),
            protocol=SimpleNamespace(client=make_client),
            url=None,
            turn_timeout_seconds=300,
            permission="refuse",
            start_timeout_seconds=60,
            stop_grace_seconds=5,
        )
        owner.opener = RuntimeOpener(block)
        make = WorkFolder.make

        def make_folder(root):
            folder = make(root)
            self.folders.append(folder)
            return folder

        with mock.patch.object(WorkFolder, "make", side_effect=make_folder):
            opening = asyncio.create_task(owner.opener.open(Role("workers", "w")))
            owner._shared = {"workers": [opening]}
            message = str(client.close_error) if client.close_error is not None else "scripted open failure"
            with self.assertRaisesRegex(RuntimeError, message):
                await opening
        return owner, client

    async def test_final_source_close_waits_for_native_owner_even_when_open_failed(self):
        for opened in (True, False):
            with self.subTest(opened=opened):
                source = self.source()
                if opened:
                    owner, client = self.runtime(source)
                else:
                    owner, client = await self.failed_open(source)
                await owner.close()
                await client.waiting.wait()
                await self.check_owned_drain(source, client, require_marker=not opened)

    async def check_owned_drain(self, source, client, *, require_marker=False):
        closing = asyncio.create_task(source.aclose())
        await turns()
        try:
            self.assertFalse(closing.done())
            self.assertTrue(client.folder.exists())
            self.assertFalse(client.physical_closed)
            if require_marker:
                self.assertTrue((client.folder / NATIVE_OWNER).exists())
        finally:
            client.release.set()
            await closing
        self.assertTrue(client.physical_closed)
        self.assertFalse(client.folder.exists())

    async def test_source_join_never_waits_for_another_source(self):
        left, right = self.source(), self.source()
        first, a = self.runtime(left)
        second, b = self.runtime(right)
        await asyncio.gather(first.close(), second.close())
        a.release.set()
        await left.aclose()
        self.assertTrue(a.physical_closed)
        self.assertFalse(b.physical_closed)
        self.assertTrue(b.folder.exists())

    async def test_repeated_cancelled_close_keeps_one_owned_logical_and_physical_close(self):
        source, gate = self.source(), asyncio.Event()
        owner, client = self.runtime(source, close_gate=gate)
        first, second = asyncio.create_task(owner.close()), asyncio.create_task(owner.close())
        await until(lambda: client.close_count, "the logical close to start")
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        gate.set()
        await second
        await owner.close()
        closing = asyncio.create_task(source.aclose())
        await turns()
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        with self.assertRaisesRegex(RuntimeError, "clos"):
            source.new()
        client.release.set()
        await source.aclose()
        self.assertEqual(client.close_count, 1)
        self.assertTrue(client.physical_closed)
        self.assertFalse(client.folder.exists())

    async def test_completed_cleanup_failure_remains_owned_after_native_open_or_start_failure(self):
        for opened in (True, False):
            with self.subTest(opened=opened):
                source, error = self.source(), RuntimeError("physical shutdown is unconfirmed")
                owner, client = (
                    self.runtime(source, cleanup_error=error)
                    if opened
                    else await self.failed_open(source, cleanup_error=error)
                )
                await owner.close()
                client.release.set()
                await turns()
                if not opened:
                    self.assertFalse(set(runtime_module._CLEANUPS) - self.cleanups)
                    self.assertFalse(set(adapter_module._DRAINS) - self.drains)
                with self.assertRaises(RuntimeError) as caught:
                    await source.aclose()
                self.assertIs(caught.exception, error)
                self.assertTrue((client.folder / NATIVE_OWNER).exists())
                self.assertFalse(client.physical_closed)

    async def test_closing_one_conversation_does_not_close_another_on_the_same_source(self):
        source = self.source()
        first, a = self.runtime(source)
        second, b = self.runtime(source)
        retiring = first.own_door(Door("first"))
        continuing = second.own_door(Door("second"))
        for door in (retiring, continuing):
            source.bridge.open(door)
            source.bridge._replay.retention[door] = object()
        settle = asyncio.Event()
        tool = asyncio.create_task(settle.wait())
        retiring.pending_calls.add(tool)
        try:
            await first.close()
            self.assertTrue(retiring.retired)
            self.assertIn(retiring.token, source.bridge.doors)
        finally:
            settle.set()
        await until(lambda: retiring.token not in source.bridge.doors, "the closed adapter's credential is revoked")
        self.assertFalse(a.physical_closed)
        self.assertTrue(a.folder.exists())
        self.assertNotIn(retiring, source.bridge._replay.retention)
        self.assertIs(source.bridge.doors[continuing.token], continuing)
        self.assertIn(continuing, source.bridge._replay.retention)
        a.release.set()
        await turns()
        self.assertTrue(a.physical_closed)
        self.assertTrue(b.alive)
        self.assertEqual(b.close_count, 0)
        self.assertFalse(second.closed)

    async def test_confirmed_drained_adapter_is_not_retained_until_server_shutdown(self):
        source = self.source()
        owner, client = self.runtime(source)
        reference = weakref.ref(owner)
        client.release.set()
        await owner.close()
        await turns()
        del owner
        gc.collect()
        self.assertIsNone(reference())

    async def test_logical_close_errors_survive_successful_physical_cleanup(self):
        for stage in ("client", "conductor", "worker", "conductor cancellation"):
            with self.subTest(stage=stage):
                failure = asyncio.CancelledError if stage == "conductor cancellation" else RuntimeError
                source, error = self.source(), failure(f"original {stage} close error")
                owner, client = self.runtime(source, close_error=error if stage == "client" else None)
                if stage.startswith("conductor"):
                    owner._conductor = FailedRelease(error)
                elif stage == "worker":
                    owner._runs.add(FailedRelease(error))
                await owner.close()
                client.release.set()
                with self.assertRaises(failure) as caught:
                    await source.aclose()
                if failure is RuntimeError:
                    self.assertIs(caught.exception, error)
                self.assertTrue(client.physical_closed)
                self.assertFalse(client.folder.exists())

    async def test_close_error_without_physical_proof_preserves_workspace(self):
        for opened in (True, False):
            with self.subTest(opened=opened):
                source, error = self.source(), RuntimeError("original unconfirmed client close")
                client = UnconfirmedClient(error)
                owner, _ = (
                    self.runtime(source, client=client) if opened else await self.failed_open(source, client=client)
                )
                marker = client.folder / NATIVE_OWNER
                self.assertTrue(marker.exists())
                await owner.close()
                with self.assertRaises(RuntimeError) as caught:
                    await source.aclose()
                self.assertIs(caught.exception, error)
                self.assertTrue(marker.exists())
                self.assertFalse(client.physical_closed)


class ClosingSource:
    """A core source close with controlled completion and a preserved original failure."""

    def __init__(self, gate=None, error=None):
        """Record one source's close while its sibling remains independently runnable."""
        self.gate, self.error = gate, error
        self.entered = asyncio.Event()
        self.closed = False
        self.calls = 0

    async def aclose(self):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        self.closed = True
        if self.error is not None:
            raise self.error


class CoreShutdownTests(unittest.IsolatedAsyncioTestCase):
    def core(self, left, right):
        return DelegationCore(CORE_SETTINGS, left, right, defaults=DEFAULTS, result_format="text")

    async def test_cancelled_core_close_does_not_cancel_or_duplicate_its_distinct_sources(self):
        gate = asyncio.Event()
        source = ClosingSource(gate)
        core = self.core(source, source)
        closing = asyncio.create_task(core.aclose())
        await source.entered.wait()
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        gate.set()
        await core.aclose()
        self.assertTrue(source.closed)
        self.assertEqual(source.calls, 1)

    async def test_core_waits_for_both_sources_and_propagates_the_first_error(self):
        gate = asyncio.Event()
        error = RuntimeError("original source cleanup failure")
        failed, other = ClosingSource(error=error), ClosingSource(gate)
        closing = asyncio.create_task(self.core(failed, other).aclose())
        try:
            await failed.entered.wait()
            await other.entered.wait()
            self.assertFalse(closing.done())
        finally:
            gate.set()
            with self.assertRaises(RuntimeError) as caught:
                await closing
        self.assertIs(caught.exception, error)
        self.assertTrue(failed.closed and other.closed)
