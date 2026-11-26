# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""Session balancing across the workers of one process group."""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nemotron_voice_agent.pipecat.shared.worker_balance as balance


class ShouldAcceptTests(unittest.TestCase):
    def test_least_loaded_workers_accept(self) -> None:
        self.assertTrue(balance.should_accept(0, [0, 1, 2]))
        self.assertTrue(balance.should_accept(1, [1, 1, 2]))
        self.assertFalse(balance.should_accept(2, [1, 1, 2]))

    def test_first_sessions_go_one_per_worker_then_even(self) -> None:
        counts = [0, 0, 0, 0]
        for _ in range(10):
            # Of the workers that may accept, the first one wins in this model.
            winner = next(i for i, own in enumerate(counts) if balance.should_accept(own, counts))
            counts[winner] += 1
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sorted(counts), [2, 2, 3, 3])

    def test_no_peers_means_accept(self) -> None:
        self.assertTrue(balance.should_accept(5, []))


class SlotTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def table(self, slots: int = 4) -> balance.SlotTable:
        return balance.SlotTable(Path(self.dir.name) / "group", slots)

    def test_claims_distinct_slots_and_reports_counts(self) -> None:
        first, second = self.table(), self.table()
        # Make the second table look like another live process.
        with patch.object(balance.os, "getpid", return_value=os.getppid()):
            second.claim()
        first.claim()
        first.publish(3)
        second.publish(1)

        self.assertNotEqual(first.index, second.index)
        self.assertCountEqual(first.counts(), [3, 1])

    def test_dead_worker_is_ignored_and_its_slot_reused(self) -> None:
        table = self.table(2)
        dead_pid = 2**22 + 12345  # beyond the usual pid range
        with patch.object(balance.os, "getpid", return_value=dead_pid):
            table.claim()
            table.publish(9)
        live = self.table(2)
        live.claim()
        live.publish(2)

        self.assertEqual(live.counts(), [2])

    def test_release_frees_the_slot(self) -> None:
        table = self.table(1)
        table.claim()
        table.publish(4)
        table.release()

        self.assertEqual(table.counts(), [])
        other = self.table(1)
        self.assertEqual(other.claim(), 0)

    def test_no_free_slot_is_an_error(self) -> None:
        table = self.table(1)
        table.claim()
        with self.assertRaises(RuntimeError):
            self.table(1).claim()


class WorkerBalancerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        group = Path(self.dir.name) / "group"
        self.table = balance.SlotTable(group, 4)
        self.table.claim()
        self.peer = balance.SlotTable(group, 4)
        with patch.object(balance.os, "getpid", return_value=os.getppid()):
            self.peer.claim()
        self.accepted = 0
        loop = asyncio.get_running_loop()

        class Protocol(asyncio.Protocol):
            def connection_made(proto, transport):
                self.accepted += 1

        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.setblocking(False)
        self.server = await loop.create_server(Protocol, sock=self.listener)
        self.balancer = balance.WorkerBalancer(self.table, loop, 0.002)
        self.clients: list[socket.socket] = []
        self.addAsyncCleanup(self.cleanup_all)

    async def cleanup_all(self) -> None:
        await self.balancer.stop()
        for client in self.clients:
            client.close()
        self.server.close()

    def connect(self) -> None:
        client = socket.socket()
        client.setblocking(False)
        client.connect_ex(self.listener.getsockname())
        self.clients.append(client)

    async def test_busy_worker_stops_accepting_until_it_is_least_loaded(self) -> None:
        self.peer.publish(0)
        self.balancer.sessions = 3
        self.balancer.start()
        await asyncio.sleep(0.05)

        self.connect()
        await asyncio.sleep(0.1)
        self.assertEqual(self.accepted, 0)

        self.balancer.sessions = 0
        await asyncio.sleep(0.1)
        self.assertEqual(self.accepted, 1)

    async def test_accepted_connection_counts_until_it_reaches_the_app(self) -> None:
        self.peer.publish(1)
        self.balancer.start()
        await asyncio.sleep(0.05)

        self.connect()
        await asyncio.sleep(0.1)
        self.assertEqual(self.accepted, 1)
        self.assertEqual(self.balancer.count, 1)

        self.balancer.scope_started(websocket=True)
        self.assertEqual(self.balancer.count, 1)
        self.balancer.scope_ended(websocket=True)
        self.assertEqual(self.balancer.count, 0)


class SettingsTests(unittest.TestCase):
    def test_off_by_default_and_with_one_worker(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(balance.BALANCE_ENV, None)
            self.assertFalse(balance.balancing_enabled(8))
        with patch.dict(os.environ, {balance.BALANCE_ENV: "1"}):
            self.assertTrue(balance.balancing_enabled(8))
            self.assertFalse(balance.balancing_enabled(1))


if __name__ == "__main__":
    unittest.main()
