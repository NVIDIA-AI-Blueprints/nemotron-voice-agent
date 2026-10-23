# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""CR149: fixes from the PR #149 review to copied prototype code (recorded as diverged)."""

from __future__ import annotations

import unittest

from _fba_pipecat_fakes import PipecatHarness

from examples.frontend_backend_agent.core.agent.messages import RoleTotals, Usage, UsageTotals
from examples.frontend_backend_agent.core.agent.session import SessionState
from examples.frontend_backend_agent.core.audio.formats import PCM, AudioFormat
from examples.frontend_backend_agent.core.engine.scripted import ScriptedAgentPort


class SessionUsageTests(unittest.TestCase):
    def test_usage_survives_a_checkpoint(self) -> None:
        usage = UsageTotals(
            frontend=RoleTotals(calls=2, usage=Usage(10, 5, 15, 3), latency_ms=12.5, cost_known_subtotal=0.25),
            backend=RoleTotals(calls=1, usage=Usage(7, 2, 9, 0), latency_ms=4.0, cost_unknown_calls=1),
        )
        state = SessionState(usage=usage)
        self.assertEqual(SessionState.from_dict(state.to_dict()).usage, usage)

    def test_checkpoint_without_usage_starts_empty(self) -> None:
        data = SessionState().to_dict()
        del data["usage"]
        self.assertEqual(SessionState.from_dict(data).usage, UsageTotals())


class InputAlignmentTests(unittest.IsolatedAsyncioTestCase):
    """The Pipecat session's input decode keeps PCM16 sample alignment (v1's ``InputPath`` fix, B6)."""

    def _session(self):  # noqa: ANN202
        harness = PipecatHarness(agent_factory=lambda _: ScriptedAgentPort([]))
        session = harness.session
        session._configure_input(AudioFormat(PCM, 16000))
        return session

    @staticmethod
    def _feed(session, chunk: bytes) -> tuple[bytes, float]:  # noqa: ANN001
        session._on_append(chunk)
        audio, session._pending_audio = session._pending_audio, None
        return audio or (b"", 0.0)

    async def test_odd_pcm16_chunks_keep_sample_alignment(self) -> None:
        audio = bytes(range(1, 41))
        whole, split = self._session(), self._session()
        expected = self._feed(whole, audio)
        parts = [self._feed(split, chunk) for chunk in (audio[:3], audio[3:20], audio[20:21], audio[21:])]
        self.assertEqual(b"".join(pcm for pcm, _ in parts), expected[0])
        self.assertEqual(sum(step for _, step in parts), expected[1])

    async def test_clear_drops_a_held_byte(self) -> None:
        session = self._session()
        self.assertEqual(self._feed(session, b"\x01"), (b"", 0.0))
        session._clear_input()
        self.assertEqual(self._feed(session, b"\x02\x03"), (b"\x02\x03", 1000.0 / 16000))
