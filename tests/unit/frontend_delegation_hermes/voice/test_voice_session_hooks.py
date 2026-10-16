# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107

"""The session seams: view preview and the transactional ``session.update`` (on the test head)."""

from __future__ import annotations

import json
import unittest
from typing import Any

from _fdh_test_head.ports import IdentityNormalizer, SpeechServices
from _fdh_test_head.session import RealtimeSession
from _fdh_test_head.stubs import StubRecognizer, ToneSynthesizer
from _fdh_test_head.vad_energy import EnergyVad
from _voice_fakes import MemoryTransport, base_config, tau2_session_update

from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.head_api import SessionChange, TurnContext
from examples.frontend_delegation_hermes.core.wire.session_view import RealtimeSessionView, SessionDefaults


def _services() -> SpeechServices:
    return SpeechServices(
        recognizer=StubRecognizer(),
        synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=5.0),
        vad_factory=EnergyVad,
        normalizer=IdentityNormalizer(),
    )


class RecordingTurns:
    """A minimal TurnManagerLike that records calls and can reject updates."""

    def __init__(self, context: TurnContext, *, reject: bool = False) -> None:
        self.context = context
        self.reject = reject
        self.updates: list[SessionChange] = []
        self.started = False
        self.closed = False
        self.state = "IDLE"

    async def start(self) -> None:
        self.started = True

    async def on_session_update(self, settings: Any, change: SessionChange) -> None:
        if self.reject and not change.first:
            raise RuntimeError("no tool changes after start")
        self.updates.append(change)

    async def close(self) -> None:
        self.closed = True

    def __getattr__(self, name: str) -> Any:  # every other hook is a no-op
        async def noop_async(*args: Any, **kwargs: Any) -> None:
            return None

        def noop(*args: Any, **kwargs: Any) -> Any:
            return False

        return noop_async if name in ("on_speech_started", "on_response_cancel", "on_output_audio_clear") else noop


class PreviewTests(unittest.TestCase):
    def test_preview_does_not_mutate_until_commit(self) -> None:
        config = base_config()
        view = RealtimeSessionView(
            session_id="s",
            model="m",
            defaults=SessionDefaults(
                input_format=config.audio.default_input_format,
                output_format=config.audio.default_output_format,
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                honor_client_values=True,
            ),
        )
        before = view.public()
        preview = view.preview(tau2_session_update("mock")["session"])
        self.assertEqual(view.public(), before)
        self.assertTrue(preview.result.tools_changed)
        preview.commit()
        self.assertNotEqual(view.public(), before)
        stale = view.preview({"instructions": "x"})
        view.apply({"instructions": "y"})
        with self.assertRaises(RuntimeError):
            stale.commit()


class TurnManagerFactoryTests(unittest.IsolatedAsyncioTestCase):
    def session(self, *, reject: bool = False) -> tuple[RealtimeSession, MemoryTransport, list[RecordingTurns]]:
        built: list[RecordingTurns] = []
        transport = MemoryTransport()

        def factory(context: TurnContext) -> RecordingTurns:
            built.append(RecordingTurns(context, reject=reject))
            return built[-1]

        session = RealtimeSession(
            config=base_config(),
            transport=transport,
            services=_services(),
            routing_sink=SessionRoutingSink(EventLog()),
            filler_log=FillerLog(),
            turn_manager_factory=factory,
        )
        return session, transport, built

    async def test_transactional_updates(self) -> None:
        session, transport, built = self.session(reject=True)
        await session.handle_message(json.dumps(tau2_session_update("mock")))
        self.assertEqual(built[0].updates[0].first, True)
        public = session.view.public()
        changed = tau2_session_update("mock")
        changed["session"]["instructions"] = "something else"
        await session.handle_message(json.dumps(changed))
        self.assertEqual(session.view.public(), public)  # rejected -> nothing committed
        self.assertEqual(len(built[0].updates), 1)


if __name__ == "__main__":
    unittest.main()
