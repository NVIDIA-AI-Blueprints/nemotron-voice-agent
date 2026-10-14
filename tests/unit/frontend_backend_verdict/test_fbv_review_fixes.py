# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""CR149: fixes from the PR #149 review to copied prototype code (recorded as diverged)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from examples.frontend_backend_verdict.text.messages import RoleTotals, Usage, UsageTotals
from examples.frontend_backend_verdict.text.session import SessionState
from examples.frontend_backend_verdict.voice.audio.formats import PCM, AudioFormat
from examples.frontend_backend_verdict.voice.engine.input_path import InputPath
from examples.frontend_backend_verdict.voice.server import _bearer_ok
from examples.frontend_backend_verdict.voice.speech.vad_energy import EnergyVad
from examples.frontend_backend_verdict.voice.wire.session_view import TurnDetectionSettings

MANUAL = TurnDetectionSettings(mode="manual", requested_type=None)


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


class InputAlignmentTests(unittest.TestCase):
    def _path(self) -> InputPath:
        fmt = AudioFormat(PCM, 16000)
        return InputPath(
            engine_rate=16000, vad=EnergyVad(16000), input_format=fmt, turn_detection=MANUAL, min_speech_ms=120
        )

    def test_odd_pcm16_chunks_keep_sample_alignment(self) -> None:
        audio = bytes(range(1, 41))
        whole, split = self._path(), self._path()
        whole.feed(audio)
        for chunk in (audio[:3], audio[3:20], audio[20:21], audio[21:]):
            split.feed(chunk)
        self.assertEqual(split.commit(), whole.commit())

    def test_clear_drops_a_held_byte(self) -> None:
        path = self._path()
        path.feed(b"\x01")
        path.clear()
        path.feed(b"\x02\x03")
        self.assertEqual(path.buffered_ms, 1000.0 / 16000)


class BearerTests(unittest.TestCase):
    def _ok(self, header: str, token: str = "secret") -> bool:
        config = SimpleNamespace(server=SimpleNamespace(require_bearer=True, bearer_token=token))
        return _bearer_ok(SimpleNamespace(headers={"authorization": header}), config)

    def test_bearer_token_checks(self) -> None:
        self.assertTrue(self._ok("Bearer secret"))
        self.assertFalse(self._ok("Bearer wrong"))
        self.assertFalse(self._ok(""))
        self.assertFalse(self._ok("Bearer ", token=""))
