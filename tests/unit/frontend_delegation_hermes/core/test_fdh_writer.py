# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Single outbound writer: no interleaving, per-producer order, deterministic close."""

from __future__ import annotations

import asyncio
import threading
import unittest

from examples.frontend_delegation_hermes.core.delegation.writer import QueueWriter, WriterClosedError


class WriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_threads_and_tasks_keep_order_per_producer(self) -> None:
        received: list[str] = []

        async def send(frame: str) -> None:
            await asyncio.sleep(0)  # a slow transport must not let frames interleave
            received.append(frame)

        writer = QueueWriter(send)
        writer.start()

        def thread_producer(name: str) -> None:
            for i in range(50):
                writer.put_threadsafe(f"{name}:{i}")

        threads = [threading.Thread(target=thread_producer, args=(f"t{n}",)) for n in range(3)]
        for thread in threads:
            thread.start()
        for i in range(50):
            writer.put(f"loop:{i}")
            await asyncio.sleep(0)
        for thread in threads:
            thread.join()
        await asyncio.sleep(0.05)
        await writer.close(drain=True)
        self.assertEqual(len(received), 200)
        for producer in ("t0", "t1", "t2", "loop"):
            seq = [int(f.split(":")[1]) for f in received if f.startswith(producer + ":")]
            self.assertEqual(seq, list(range(50)), producer)

    async def test_closed_writer_rejects_frames(self) -> None:
        writer = QueueWriter(lambda frame: asyncio.sleep(0))
        writer.start()
        await writer.close()
        with self.assertRaises(WriterClosedError):
            writer.put("x")
        with self.assertRaises(WriterClosedError):
            writer.put_threadsafe("x")

    async def test_close_without_drain_drops_queued_frames(self) -> None:
        gate = asyncio.Event()
        received: list[str] = []

        async def send(frame: str) -> None:
            await gate.wait()
            received.append(frame)

        writer = QueueWriter(send)
        writer.start()
        for i in range(5):
            writer.put(str(i))
        await asyncio.sleep(0)
        closing = asyncio.create_task(writer.close(drain=False, timeout=1.0))
        await asyncio.sleep(0)
        gate.set()
        await closing
        self.assertLessEqual(len(received), 1)

    async def test_send_failure_closes_and_reports(self) -> None:
        errors: list[BaseException] = []

        async def send(frame: str) -> None:
            raise ConnectionError("gone")

        writer = QueueWriter(send, on_error=errors.append)
        writer.start()
        writer.put("x")
        await asyncio.sleep(0.01)
        self.assertTrue(writer.closed)
        self.assertEqual(len(errors), 1)
        await writer.close()


if __name__ == "__main__":
    unittest.main()
