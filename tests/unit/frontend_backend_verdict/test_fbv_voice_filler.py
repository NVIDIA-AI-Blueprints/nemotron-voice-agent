# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 9, 10, 10b: silenced filler is invisible on the wire; audible filler runs concurrently; timing is logged."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    normalize_trace,
    pcmu_silence,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)

from examples.frontend_backend_verdict.voice.agent.filler import FillerLog
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients


async def _delegated_turn(
    filler: str,
    *,
    config=None,
    gate: asyncio.Event | None = None,
    log: FillerLog | None = None,
    show_silent_filler: bool = False,
):
    frontend = FakeChatClient([delegate_response("Check the thing.", filler=filler)])
    backend = FakeChatClient([text_response("It is all set.")])
    if gate is not None:
        backend.gates[0] = gate
    harness = SessionHarness(
        config=config,
        clients=AgentClients(backend=backend, frontend=frontend),
        filler_log=log or FillerLog(),
        show_silent_filler=show_silent_filler,
    )
    await harness.start(tau2_session_update())
    await harness.speak()
    return harness


class LogOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_only_is_invisible(self) -> None:
        with_filler = await _delegated_turn("Let me check.")
        await with_filler.wait_for("response.done")
        await with_filler.idle(500)
        without_filler = await _delegated_turn("")
        await without_filler.wait_for("response.done")
        await without_filler.idle(500)
        self.assertEqual(normalize_trace(with_filler.events), normalize_trace(without_filler.events))
        self.assertEqual(len(with_filler.filler_log.records), 1)
        self.assertEqual(with_filler.filler_log.records[0]["text"], "Let me check.")
        self.assertFalse(with_filler.filler_log.records[0]["spoken"])
        # A delegation without filler text still gets its timing record.
        self.assertEqual(len(without_filler.filler_log.records), 1)
        await with_filler.close()
        await without_filler.close()

    async def test_opted_in_client_is_told_the_silent_filler_once(self) -> None:
        gate = asyncio.Event()
        harness = await _delegated_turn("Let me check.", gate=gate, show_silent_filler=True)
        shown = await harness.wait_for("x_nvidia.filler")
        self.assertFalse(gate.is_set(), "the filler is reported while the backend is still working")
        self.assertEqual((shown["text"], shown["mode"], shown["spoken"]), ("Let me check.", "log_only", False))
        gate.set()
        await harness.wait_for("response.done")
        await harness.idle(500)
        self.assertEqual(len(harness.of_type("x_nvidia.filler")), 1)
        # Still silent: one response, one audio item (the answer), no filler audio or transcript.
        self.assertEqual(len(harness.of_type("response.output_item.added")), 1)
        transcript = "".join(e["delta"] for e in harness.of_type("response.output_audio_transcript.delta"))
        self.assertEqual(transcript, "It is all set.")
        await harness.close()

    async def test_no_filler_event_without_text_or_opt_in(self) -> None:
        empty = await _delegated_turn("", show_silent_filler=True)
        await empty.wait_for("response.done")
        not_asked = await _delegated_turn("Let me check.")
        await not_asked.wait_for("response.done")
        await empty.idle(500)
        await not_asked.idle(500)
        self.assertEqual(empty.of_type("x_nvidia.filler"), [])
        self.assertEqual(not_asked.of_type("x_nvidia.filler"), [])
        await empty.close()
        await not_asked.close()

    async def test_timing_record_from_first_turn_on_both_clocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filler.jsonl"
            config = voice_config(filler_log_path=str(path))
            frontend = FakeChatClient(
                [delegate_response("First.", "One moment."), delegate_response("Second.", "Let me look.")]
            )
            backend = FakeChatClient([text_response("First answer."), tool_response(("get_users", {}), ids=["c1"])])
            log = FillerLog(str(path))
            harness = SessionHarness(
                config=config, clients=AgentClients(backend=backend, frontend=frontend), filler_log=log
            )
            await harness.start(tau2_session_update())
            await harness.feed(pcmu_silence(1000))
            await harness.speak()
            await harness.wait_for("response.done")
            await harness.feed(pcmu_silence(1500))
            await harness.speak()
            await harness.wait_for("response.done", 2)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            await harness.close()
        self.assertEqual([r["turn_id"] for r in records], [1, 2])
        first, second = records
        self.assertEqual(first["outcome"], "answer")
        self.assertEqual(second["outcome"], "tool_calls")
        self.assertIsNone(second["backend_done"])
        for record in records:
            self.assertTrue(record["session_start_wall"].endswith("Z"))
            self.assertIsInstance(record["turn_start"]["audio_ms"], int)
            self.assertGreater(record["turn_end"]["audio_ms"], record["turn_start"]["audio_ms"])
            self.assertIsNotNone(record["filler_ready"]["since_session_start_ms"])
            self.assertIsNotNone(record["filler_ready"]["since_turn_end_ms"])
        self.assertAlmostEqual(first["turn_start"]["audio_ms"], 1000, delta=40)
        self.assertIsNotNone(first["first_answer_audio"])
        self.assertIn("would_have_spoken", first)


class SpeakTests(unittest.IsolatedAsyncioTestCase):
    async def test_filler_audio_goes_out_while_the_backend_is_still_pending(self) -> None:
        gate = asyncio.Event()
        config = voice_config(filler_mode="speak", filler_speak_after_ms=0)
        harness = await _delegated_turn("Let me check.", config=config, gate=gate)
        await harness.wait_for("response.output_audio.delta")
        self.assertFalse(gate.is_set(), "filler audio must precede the backend's completion")
        filler_item = harness.of_type("response.output_audio.delta")[0]["item_id"]
        transcript = "".join(
            e["delta"] for e in harness.of_type("response.output_audio_transcript.delta") if e["item_id"] == filler_item
        )
        self.assertEqual(transcript, "Let me check.")
        gate.set()
        await harness.wait_for("response.done")
        items = [e["item"]["id"] for e in harness.of_type("response.output_item.added")]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], filler_item)
        types_and_items = [(e["type"], e.get("item", {}).get("id")) for e in harness.events]
        filler_done = types_and_items.index(("response.output_item.done", filler_item))
        answer_added = types_and_items.index(("response.output_item.added", items[1]))
        self.assertLess(filler_done, answer_added)
        self.assertEqual(len(harness.of_type("response.created")), 1)
        record = harness.filler_log.records[0]
        self.assertTrue(record["spoken"])
        self.assertIsNotNone(record["filler_audio_start"])
        # The filler is not part of the conversation history.
        history = harness.runners[0].state.frontend_history.messages
        self.assertFalse(any(m.content == "Let me check." for m in history))
        await harness.close()

    async def test_spoken_filler_is_not_also_reported_as_silent(self) -> None:
        config = voice_config(filler_mode="speak", filler_speak_after_ms=0)
        harness = await _delegated_turn("Let me check.", config=config, show_silent_filler=True)
        await harness.wait_for("response.done")
        self.assertEqual(harness.of_type("x_nvidia.filler"), [])
        await harness.close()

    async def test_filler_skipped_when_backend_is_fast(self) -> None:
        config = voice_config(filler_mode="speak", filler_speak_after_ms=5000)
        harness = await _delegated_turn("Let me check.", config=config)
        await harness.wait_for("response.done")
        self.assertEqual(len(harness.of_type("response.output_item.added")), 1)
        self.assertFalse(harness.filler_log.records[0]["spoken"])
        self.assertFalse(harness.filler_log.records[0]["would_have_spoken"])
        await harness.close()
