# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""A stalled Talker stream must not hold a Realtime response open."""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from examples.frontend_backend_agent.src import reliable_talker


def _chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=0,
        model="test",
        choices=[Choice(index=0, delta=ChoiceDelta(role="assistant", content=content), finish_reason=None)],
    )


class _Stream:
    """Yield some chunks, then stall forever."""

    def __init__(self, prefix: list[ChatCompletionChunk], *, stall: bool) -> None:
        self._prefix = list(prefix)
        self._stall = stall
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._prefix:
            return self._prefix.pop(0)
        if self._stall:
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class TalkerStreamBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stalled_stream_yields_no_chunks_and_is_closed(self):
        stream = _Stream([_chunk("Reservation ABC")], stall=True)
        with patch.object(reliable_talker, "_TALKER_STREAM_TIMEOUT_SECONDS", 0.05):
            chunks = await reliable_talker._collect_stream(stream)
        self.assertEqual(chunks, [], "a truncated completion must not be emitted")
        self.assertTrue(stream.closed)

    async def test_a_stalled_stream_reports_as_an_empty_turn(self):
        """The empty result is what drives the bounded retry and spoken fallback."""
        stream = _Stream([_chunk("Reservation ABC")], stall=True)
        with patch.object(reliable_talker, "_TALKER_STREAM_TIMEOUT_SECONDS", 0.05):
            chunks = await reliable_talker._collect_stream(stream)
        self.assertEqual(reliable_talker._base_invalid_reason(chunks), "empty")

    async def test_a_stream_that_completes_in_time_is_unaffected(self):
        stream = _Stream([_chunk("Reservation ABC123 is confirmed.")], stall=False)
        with patch.object(reliable_talker, "_TALKER_STREAM_TIMEOUT_SECONDS", 5.0):
            chunks = await reliable_talker._collect_stream(stream)
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(reliable_talker._base_invalid_reason(chunks))


if __name__ == "__main__":
    unittest.main()
