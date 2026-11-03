# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate open-upload source ownership without any model/provider services."""

import asyncio
import unittest
from unittest.mock import patch

from examples.generic.offline.models import AudioSource, SourceAbortedError, StreamingAudioSource


class StreamingSourceTests(unittest.IsolatedAsyncioTestCase):
    """An upload gap is neither silence nor source EOF."""

    async def test_sync_and_async_readers_share_canonical_windows_before_validated_eof(self):
        """Packet fragments cannot change reader chunk identity or final extent."""
        pcm = bytes(range(256)) * 8 + b"\x01\x02" * 17
        source = StreamingAudioSource(len(pcm) // 2)
        iterator = source.async_chunks()
        first = asyncio.create_task(anext(iterator))
        source.append(pcm[:1000])
        await asyncio.sleep(0)
        self.assertFalse(first.done())
        source.append(pcm[1000:1024])
        self.assertEqual((await asyncio.wait_for(first, 1)).pcm, pcm[:1024])
        source.append(pcm[1024:])
        second = await asyncio.wait_for(anext(iterator), 1)
        self.assertEqual((second.sequence, second.start_sample, second.end_sample), (1, 512, 1024))
        partial = asyncio.create_task(anext(iterator))
        threaded = asyncio.create_task(asyncio.to_thread(lambda: tuple(source.chunks())))
        sealed = asyncio.create_task(source.wait_sealed())
        await asyncio.sleep(0)
        self.assertFalse(partial.done())
        self.assertFalse(sealed.done())
        self.assertFalse(threaded.done())
        with self.assertRaisesRegex(RuntimeError, "not sealed"):
            _ = source.pcm
        recording = source.seal()
        self.assertIs(await asyncio.wait_for(sealed, 1), recording)
        self.assertEqual((await asyncio.wait_for(partial, 1)).pcm, pcm[2048:])
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)
        chunks = await asyncio.wait_for(threaded, 1)
        self.assertEqual(chunks, tuple(recording.chunks()))
        self.assertEqual(source.pcm, pcm)
        self.assertEqual(source.received_samples, source.sample_count)
        self.assertIs(source.seal(), recording)

    async def test_full_declared_prefix_does_not_finish_iterator_until_explicit_seal(self):
        """Even an exact full-window extent still requires the EOF control."""
        source = StreamingAudioSource(512)
        source.append(bytes(1024))
        iterator = source.async_chunks()
        await anext(iterator)
        ended = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        self.assertFalse(ended.done())
        source.seal()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(ended, 1)

    async def test_abort_wakes_thread_async_and_seal_waiters_with_distinct_failure(self):
        """Cancellation must not half-close a provider as a normal source EOF."""
        source = StreamingAudioSource(512)
        iterator = source.async_chunks()
        pending = [
            asyncio.create_task(anext(iterator)),
            asyncio.create_task(asyncio.to_thread(lambda: tuple(source.chunks()))),
            asyncio.create_task(source.wait_sealed()),
        ]
        await asyncio.sleep(0)
        source.abort()
        for task in pending:
            with self.assertRaises(SourceAbortedError):
                await asyncio.wait_for(task, 1)
        with self.assertRaises(SourceAbortedError):
            source.seal()
        with self.assertRaises(SourceAbortedError):
            source.append(bytes(1024))
        self.assertFalse(source.sealed)

    async def test_cancelled_reader_does_not_cancel_another_reader_or_upload(self):
        """A consumer's cancellation cannot corrupt the shared source state."""
        source = StreamingAudioSource(512)
        cancelled = asyncio.create_task(anext(source.async_chunks()))
        remaining = asyncio.create_task(anext(source.async_chunks()))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        source.append(bytes(1024))
        self.assertEqual((await asyncio.wait_for(remaining, 1)).end_sample, 512)
        self.assertEqual(source.seal().sample_count, 512)

    async def test_burst_appends_schedule_only_one_wakeup_per_blocked_reader(self):
        """Buffered packets cannot amplify callbacks for an already-notified wait."""
        source = StreamingAudioSource(1024)
        iterator = source.async_chunks()
        first = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        with patch.object(loop, "call_soon_threadsafe", wraps=loop.call_soon_threadsafe) as notify:
            for _ in range(4):
                source.append(bytes(512))
            self.assertEqual(notify.call_count, 1)
        self.assertEqual((await asyncio.wait_for(first, 1)).end_sample, 512)
        self.assertEqual((await anext(iterator)).end_sample, 1024)
        ended = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        self.assertFalse(ended.done())
        source.seal()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(ended, 1)

    async def test_empty_source_and_finite_async_interface_preserve_actual_pcm(self):
        """No synthetic tail or final inference padding becomes source PCM."""
        source = StreamingAudioSource(0)
        source.seal()
        self.assertEqual([chunk async for chunk in source.async_chunks()], [])
        finite = AudioSource(b"\x11\x22" * 513)
        self.assertEqual([chunk async for chunk in finite.async_chunks()], list(finite.chunks()))
        self.assertIs(await finite.wait_sealed(), finite)
        source.abort()
        self.assertTrue(source.sealed)
        self.assertEqual(source.pcm, b"")

    async def test_invalid_or_excess_extent_does_not_mutate_received_prefix(self):
        """Source allocation remains bounded by a validated declaration."""
        for value in (-1, True, 1.0, 16000 * 3600 + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StreamingAudioSource(value)
        source = StreamingAudioSource(2)
        for pcm in (b"", b"a", bytearray(b"ab"), bytes(6)):
            with self.subTest(size=len(pcm)), self.assertRaises(ValueError):
                source.append(pcm)
            self.assertEqual(source.received_samples, 0)
        with self.assertRaises(ValueError):
            source.seal()
        source.append(b"aabb")
        source.seal()
        with self.assertRaises(RuntimeError):
            source.append(b"cc")
