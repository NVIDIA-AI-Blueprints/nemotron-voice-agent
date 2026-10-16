# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""v1's session rules on the Pipecat adapter (plan §4, S1–S22, and B5, B7).

The rules exercised end to end elsewhere are not repeated here: S1–S4 and S21 by
``test_fbv_pipecat_wire_parity`` (G2), S5 and S6's turn-manager side by
``test_fbv_voice_barge_in_verdict``, S9 by ``test_fbv_golden`` (N4), S12 by
``test_fbv_pipecat_host`` and ``test_fbv_runtime``, S13 by ``test_fbv_voice_filler``, S14 by
``test_fbv_tool_resume``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from _fbv_core_fakes import append_event, pcmu_silence, pcmu_speech, tau2_session_update, voice_config
from _fbv_pipecat_fakes import PipecatHarness, PipecatTestCase

from examples.frontend_backend_verdict.core.engine.event_log import EventLog
from examples.frontend_backend_verdict.core.engine.scripted import ScriptedAgentPort

LONG = "This is a long answer. It has several sentences. Each one takes a while to say. The caller may interrupt."


class _ConfigureSpy(ScriptedAgentPort):
    def __init__(self, script: list[dict[str, Any]]) -> None:
        super().__init__(script)
        self.configured: list[tuple[int, str]] = []

    def configure(self, *, tools: Any, instructions: str) -> None:
        self.configured.append((len(tools), instructions))
        super().configure(tools=tools, instructions=instructions)


def _harness(script: list[dict[str, Any]] | None = None, **kwargs: Any) -> PipecatHarness:
    return PipecatHarness(agent_factory=lambda _: ScriptedAgentPort(script or [{"say": "Okay."}]), **kwargs)


async def _feed_real_time(harness: PipecatHarness, audio: bytes, check: Any = None) -> int:
    """Append ``audio`` at the client's pace (20 ms frames, 20 ms apart); ``check`` runs after each."""
    count = 0
    for offset in range(0, len(audio), 160):
        harness.websocket.put_text(append_event(audio[offset : offset + 160]))
        await asyncio.sleep(0.02)
        if check is not None:
            check()
            count += 1
    return count


class ProtocolErrorTests(PipecatTestCase):
    async def test_protocol_errors_are_non_fatal_and_name_the_client_event(self) -> None:  # S18
        harness = _harness()
        await harness.start(tau2_session_update())
        await harness.send({"type": "no.such.event", "event_id": "evt_client_1"})
        error = await harness.wait_for("error")
        self.assertEqual(error["error"]["event_id"], "evt_client_1")
        await harness.send("{not json")
        await harness.wait_for("error", 2)
        await harness.speak()  # the session goes on
        await harness.wait_for("response.done")
        await harness.close()

    async def test_per_response_overrides_are_refused(self) -> None:  # S7
        harness = _harness()
        await harness.start(tau2_session_update())
        await harness.send({"type": "response.create", "response": {"instructions": "Be brief."}})
        error = await harness.wait_for("error")
        self.assertEqual(error["error"]["code"], "unsupported_response_override")
        self.assertFalse(harness.of_type("response.created"))
        await harness.close()

    async def test_manual_commit_is_refused(self) -> None:  # S15, B7
        harness = _harness()
        await harness.start(tau2_session_update())
        await harness.feed(pcmu_speech(200))
        await harness.send({"type": "input_audio_buffer.commit"})
        error = await harness.wait_for("error")
        self.assertEqual(error["error"]["code"], "unsupported_event")
        await harness.close()

    async def test_only_unconsumed_user_text_items_can_be_deleted(self) -> None:  # S16
        harness = _harness()
        await harness.start(tau2_session_update())
        await harness.send({"type": "conversation.item.delete", "item_id": "item_unknown"})
        error = await harness.wait_for("error")
        self.assertEqual(error["error"]["code"], "invalid_value")
        self.assertFalse(harness.of_type("conversation.item.deleted"))
        await harness.close()


class SessionUpdateTests(PipecatTestCase):
    async def test_configure_on_the_first_update_and_on_changes_only(self) -> None:  # S9
        agent = _ConfigureSpy([{"say": "Okay."}])
        harness = PipecatHarness(agent_factory=lambda _: agent)
        update = tau2_session_update()
        await harness.start(update)
        self.assertEqual(len(agent.configured), 1)
        await harness.send({"type": "session.update", "session": {"audio": update["session"]["audio"]}})
        await harness.wait_for("session.updated", 2)
        self.assertEqual(len(agent.configured), 1)  # neither tools nor instructions changed
        await harness.send({"type": "session.update", "session": {"instructions": "New policy."}})
        await harness.wait_for("session.updated", 3)
        self.assertEqual(agent.configured[-1][1], "New policy.")
        self.assertEqual(len(agent.configured), 2)
        await harness.close()

    async def test_client_tools_are_ignored_with_config_tools(self) -> None:  # S8
        harness = _harness(config=voice_config(tools_source="config"))
        await harness.start()
        await harness.send(tau2_session_update())
        error = await harness.wait_for("error")
        self.assertEqual(error["error"]["code"], "client_tools_ignored")
        await harness.wait_for("session.updated")
        await harness.close()

    async def test_no_update_in_time_continues_with_defaults(self) -> None:  # S10
        harness = _harness(config=voice_config(server_session_update_timeout_s=0.2))
        await harness.start()
        await asyncio.sleep(0.1)
        harness.websocket.put_text(append_event(pcmu_silence(20)))  # any message restarts the wait (v1)
        await asyncio.sleep(0.15)
        self.assertFalse(harness.session._configured)
        await asyncio.sleep(0.2)
        self.assertTrue(harness.session._configured)
        self.assertEqual(harness.websocket.close_code, None)  # the session stays open
        await harness.close()


class InputBufferTests(PipecatTestCase):
    async def test_clear_drops_the_open_turn(self) -> None:  # S6
        harness = _harness()
        await harness.start(tau2_session_update())
        await harness.feed(pcmu_silence(200) + pcmu_speech(400))
        await harness.wait_for("input_audio_buffer.speech_started")
        await harness.send({"type": "input_audio_buffer.clear"})
        await harness.wait_for("input_audio_buffer.cleared")
        await harness.feed(pcmu_silence(1500))
        self.assertFalse(harness.of_type("input_audio_buffer.committed"))
        self.assertFalse(harness.of_type("response.created"))
        await harness.close()

    async def test_transcription_events_follow_the_flags(self) -> None:  # S20
        quiet = voice_config(asr_interim_results=False, protocol_emit_input_transcription=False)
        harness = _harness(config=quiet)
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        types = harness.types()
        self.assertNotIn("conversation.item.input_audio_transcription.delta", types)
        self.assertNotIn("conversation.item.input_audio_transcription.completed", types)
        self.assertIn("input_audio_buffer.committed", types)
        await harness.close()


class OutputTests(PipecatTestCase):
    async def test_output_audio_buffer_clear_stops_the_answer(self) -> None:  # S17
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            harness = _harness([{"say": LONG}], pace_output=True, event_log=EventLog(str(path)))
            await harness.start(tau2_session_update())
            await harness.speak()
            await harness.wait_for("response.output_audio.delta")
            await harness.send({"type": "output_audio_buffer.clear"})
            await harness.settle(0.1)
            sent = len(harness.of_type("response.output_audio.delta"))
            await _feed_real_time(harness, pcmu_silence(400))
            self.assertEqual(len(harness.of_type("response.output_audio.delta")), sent)  # the rest is dropped
            await harness.close()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        barge_in = [record for record in records if record["kind"] == "barge_in"]
        self.assertEqual([record["reason"] for record in barge_in], ["client_cancelled"])
        repaired = next(record for record in records if record["kind"] == "history_repaired")
        self.assertTrue(LONG.startswith(repaired["heard"]))
        self.assertNotEqual(repaired["heard"], LONG)

    async def test_heard_never_exceeds_written_at_real_time_input(self) -> None:  # B5
        """Pipecat writes audio at twice real time; at the client's pace, playout stays behind it."""
        harness = _harness([{"say": LONG}], pace_output=True)
        await harness.start(tau2_session_update())
        progress = harness.session.turns.progress
        serializer = harness.session.pipeline.serializer
        checks = await _feed_real_time(
            harness,
            pcmu_speech(600) + pcmu_silence(2500),
            check=lambda: self.assertLessEqual(progress.heard_ms, serializer.written_ms + 1.0),
        )
        self.assertGreater(checks, 100)
        self.assertGreater(progress.heard_ms, 1000)  # the answer was playing while the checks ran
        await harness.close()


class CloseTests(PipecatTestCase):
    async def test_close_unregisters_and_logs_the_end_last(self) -> None:  # S22
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            harness = _harness(event_log=EventLog(str(path)))
            await harness.start(tau2_session_update())
            await harness.speak()
            await harness.wait_for("response.done")
            session_id = harness.session.session_id
            await harness.close()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[-1]["kind"], "session_end")
        self.assertNotIn(session_id, harness.runtime.state.routing_sink._listeners)
