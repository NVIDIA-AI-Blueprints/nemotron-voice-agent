# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Playout facts: what each line really played, read after the output that played it."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.adapter.bridge import LiveWebSocketAudioOutput
from examples.frontend_backend_live.adapter.playout import Playout, PlayoutTracker
from examples.frontend_backend_live.adapter.services import (
    SPEECH_MODES,
    LineTextHold,
    PerSentence,
    Stitched,
    build_tts,
    speech_mode,
)
from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.shared.pipeline_utils import PIPELINE_AUDIO_OUT_SAMPLE_RATE, build_pipeline_params
from live.ports import EXCHANGE_RATE
from live.protocol import SessionConfig
from tests.unit.cascade_helpers import LIVE_CONFIG as CASCADE_TEST_CONFIG
from tests.unit.cascade_helpers import FakeTTS
from tests.unit.frontend_backend_live.adapter.adapter_helpers import Probe, RecordingPort, run_frames, tone


def audio(ms: int) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=tone(ms), sample_rate=EXCHANGE_RATE, num_channels=1)


def text(words: str) -> TTSTextFrame:
    return TTSTextFrame(text=words, aggregated_by="sentence")


class Facts:
    """A playout listener that notes each fact, in order."""

    def __init__(self):
        self.seen: list[tuple] = []

    def started(self):
        self.seen.append(("started",))

    def played(self, words):
        self.seen.append(("played", words))

    def finished(self):
        self.seen.append(("finished",))

    def silent(self):
        self.seen.append(("silent",))

    def cut(self):
        self.seen.append(("cut",))


def members(protocol) -> set[str]:
    return {name for name in vars(protocol) if not name.startswith("_")}


class TrackerTests(unittest.IsolatedAsyncioTestCase):
    """The tracker alone, fed the frames an output pushes after playing them."""

    def tracker(self):
        port, facts = RecordingPort(), Facts()
        tracker = PlayoutTracker(port)
        tracker.listen(facts)
        passed = []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            passed.append(frame)

        tracker.push_frame = record
        return tracker, port, facts, passed

    async def feed(self, tracker, port, *steps):
        """Feed frames; an int moves the session's input clock to that many milliseconds first."""
        for step in steps:
            if isinstance(step, int):
                port.input_ms = step
            else:
                await tracker.process_frame(step, FrameDirection.DOWNSTREAM)

    async def test_a_line_that_plays_to_its_end(self):
        tracker, port, facts, passed = self.tracker()
        frames = [TTSStartedFrame(), audio(300), text("Your latte is ready."), audio(200), text("Anything else?")]
        await self.feed(tracker, port, *frames, TTSStoppedFrame())
        self.assertEqual(
            facts.seen,
            [("started",), ("played", "Your latte is ready."), ("played", "Anything else?"), ("finished",)],
        )
        self.assertEqual([delta["delta"] for delta in port.transcript()], ["Your latte is ready. ", "Anything else? "])
        self.assertEqual(len(passed), 6)  # every frame passes on

    async def run_tracker(self, frames):
        """Run the tracker in a pipeline (a cut needs one), with a listener; return the port and the facts."""
        port, facts = RecordingPort(), Facts()
        tracker = PlayoutTracker(port)
        tracker.listen(facts)
        await run_frames([tracker], frames, gap=0.0, settle=0.1)
        return port, facts

    async def test_a_cut_line_reports_what_played_and_no_transcript_for_the_rest(self):
        frames = [TTSStartedFrame(), audio(300), text("Your latte is ready."), audio(100), InterruptionFrame()]
        port, facts = await self.run_tracker(frames)
        self.assertEqual(facts.seen, [("started",), ("played", "Your latte is ready."), ("cut",)])
        self.assertEqual([delta["delta"] for delta in port.transcript()], ["Your latte is ready. "])

    async def test_a_line_with_no_audio_is_silent_and_its_text_is_not_heard(self):
        tracker, port, facts, _ = self.tracker()
        await self.feed(tracker, port, TTSStartedFrame(), text("Never heard."), TTSStoppedFrame())
        self.assertEqual(facts.seen, [("silent",)])  # never started: a silent reply, which is said again
        self.assertEqual(port.transcript(), [])

    async def test_a_cut_with_no_line_playing_still_tells_the_listener(self):
        _, facts = await self.run_tracker([InterruptionFrame()])
        self.assertEqual(facts.seen, [("cut",)])

    async def test_each_sentence_is_timed_on_the_session_clock_from_its_first_audio_to_its_end(self):
        tracker, port, _, _ = self.tracker()
        frames = [TTSStartedFrame(), 1000, audio(300), 1300, text("One."), 1350, audio(200), 1550, text("Two.")]
        await self.feed(tracker, port, *frames)
        self.assertEqual(
            [(delta["start_ms"], delta["end_ms"]) for delta in port.transcript()], [(1000, 1300), (1350, 1550)]
        )

    async def test_facts_wait_for_a_listener_and_the_transcript_does_not(self):
        port = RecordingPort()
        tracker = PlayoutTracker(port)

        async def ignore(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        tracker.push_frame = ignore
        for frame in (TTSStartedFrame(), audio(100), text("Hello."), TTSStoppedFrame()):
            await tracker.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual([delta["delta"] for delta in port.transcript()], ["Hello. "])

    def test_the_listener_hears_the_plans_four_facts_and_a_silent_line(self):
        self.assertEqual(members(Playout), {"started", "played", "finished", "silent", "cut"})


class WebSocketPlayoutTests(unittest.IsolatedAsyncioTestCase):
    """The primary WebSocket's output followed by the tracker, as a live session runs them."""

    async def test_each_sentence_is_sent_to_the_client_only_after_its_audio(self):
        port, facts = RecordingPort(), Facts()
        tracker = PlayoutTracker(port)
        tracker.listen(facts)
        frames = [TTSStartedFrame(), audio(300), text("Your latte is ready."), audio(200), text("Anything else?")]
        await run_frames([LiveWebSocketAudioOutput(port), tracker], [*frames, TTSStoppedFrame()], gap=0.0, settle=0.8)
        order = [entry[2]["delta"].strip() if entry[0] == "emit" else bool(any(entry[1])) for entry in port.log]
        first = order.index(True)
        # Three packets of the first sentence, its text, two packets of the second, its text, then silence.
        expected = [True, True, True, "Your latte is ready.", True, True, "Anything else?", False]
        self.assertEqual(order[first : first + 8], expected)
        self.assertEqual(facts.seen[-1], ("finished",))

    async def test_a_barge_in_drops_the_text_with_the_audio_it_follows(self):
        port, facts, probe = RecordingPort(), Facts(), Probe()
        tracker = PlayoutTracker(port)
        tracker.listen(facts)
        frames = [TTSStartedFrame(), audio(1000), text("A long answer that never played."), InterruptionFrame()]
        await run_frames([LiveWebSocketAudioOutput(port), tracker, probe], frames, gap=0.03, settle=0.4)
        self.assertEqual(port.transcript(), [])
        self.assertEqual(probe.of(TTSTextFrame), [])
        self.assertEqual(facts.seen[-1], ("cut",))
        self.assertLess(len(port.voiced()), 4)  # at most a packet or two of the line had gone out

    async def test_a_frame_with_no_audio_ahead_of_it_passes_at_once(self):
        port, probe = RecordingPort(), Probe()
        started = TTSStartedFrame()
        task = asyncio.create_task(run_frames([LiveWebSocketAudioOutput(port), probe], [started], settle=0.3))
        await asyncio.sleep(0.1)
        self.assertIn(started, probe.frames)
        await task


class SpeechModeTests(unittest.IsolatedAsyncioTestCase):
    """Each TTS mode, chosen by name, keeps a line's text behind its audio."""

    def test_the_two_modes_are_named_in_one_table(self):
        self.assertEqual(SPEECH_MODES, {"per_sentence": PerSentence, "stitched": Stitched})
        self.assertEqual(
            {name: mode.synthesis_mode for name, mode in SPEECH_MODES.items()},
            {"per_sentence": "per_sentence", "stitched": "stitched"},
        )

    def test_per_sentence_text_already_follows_its_audio_so_nothing_is_held(self):
        self.assertEqual(PerSentence().text_hold(), [])

    def test_stitched_text_is_held_by_a_fresh_processor_each_time(self):
        first, second = Stitched().text_hold(), Stitched().text_hold()
        self.assertIsInstance(first[0], LineTextHold)
        self.assertIsNot(first[0], second[0])

    async def test_a_stitched_line_keeps_its_text_until_its_audio_has_all_come(self):
        hold, sent = LineTextHold(), []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append(frame)

        hold.push_frame = record
        start, early, later, first, second, stop = (
            TTSStartedFrame(),
            text("Your latte is ready."),  # stitched TTS: the text comes before its audio
            text("Anything else?"),
            audio(300),
            audio(200),
            TTSStoppedFrame(),
        )
        for frame in (start, early, first, later, second, stop):
            await hold.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(sent, [start, first, second, early, later, stop])

    async def test_a_cut_stitched_line_drops_its_text(self):
        probe = Probe()
        frames = [TTSStartedFrame(), text("Never heard."), audio(100), InterruptionFrame(), TTSStoppedFrame()]
        await run_frames([LineTextHold(), probe], frames, gap=0.0, settle=0.1)
        self.assertEqual(probe.of(TTSTextFrame), [])
        self.assertEqual(len(probe.of(TTSStoppedFrame)), 1)

    def test_a_host_built_text_to_speech_names_the_mode_it_runs_in(self):
        for mode, expected in (("stitched", Stitched), ("per_sentence", PerSentence), ("", PerSentence)):
            with self.subTest(mode=mode):
                tts = build_tts({}, {"server": "magpie-tts:50051", "synthesis_mode": mode})
                self.assertIsInstance(speech_mode(tts), expected)  # no mode: Pipecat's default, per sentence
        for stand_in in (FakeTTS(), MagicMock(), None):  # a stand-in yields audio, then its text
            with self.subTest(stand_in=type(stand_in).__name__):
                self.assertIsInstance(speech_mode(stand_in), PerSentence)

    async def test_a_stitched_magpie_line_from_todays_root_reaches_the_transcript_after_its_audio(self):
        # The root builds a stitched Magpie TTS from the catalog and the adapter with no mode: a stitched line's text
        # comes before its audio, and the hold keeps each sentence for the transcript.
        tts = build_tts({}, {"server": "magpie-tts:50051", "synthesis_mode": "stitched"})
        tts._service = SimpleNamespace(stub=FakeMagpie(), auth=SimpleNamespace(get_auth_metadata=lambda: []))
        adapter = CascadeAdapter(
            SessionConfig(model="live-1"),
            "websocket",
            None,
            live_config=CASCADE_TEST_CONFIG,
            frontend_endpoint=ModelEndpoint(model="talker", base_url="http://talker"),
            instructions="",
            stt=None,
            tts=tts,
        )
        self.assertIsInstance(adapter.speech, Stitched)
        port, facts = RecordingPort(), Facts()
        tracker = PlayoutTracker(port)
        tracker.listen(facts)
        line = "Your latte is ready. It costs five dollars. Anything else?"
        frames = [LLMFullResponseStartFrame(), LLMTextFrame(line), LLMFullResponseEndFrame()]
        # The speech output start() builds after the talker: the TTS, its mode's text hold, the output, the tracker.
        processors = [tts, *adapter.speech.text_hold(), LiveWebSocketAudioOutput(port), tracker]
        await run_frames(processors, frames, gap=0.0, settle=1.5, params={"params": build_pipeline_params()})
        sentences = ["Your latte is ready.", "It costs five dollars.", "Anything else?"]
        self.assertEqual([delta["delta"].strip() for delta in port.transcript()], sentences)
        self.assertEqual(facts.seen, [("started",), *[("played", s) for s in sentences], ("finished",)])
        first_delta = next(i for i, entry in enumerate(port.log) if entry[0] == "emit")
        self.assertTrue(any(entry[0] == "audio" and any(entry[1]) for entry in port.log[:first_delta]))


class FakeMagpie:
    """Magpie's gRPC synthesis service, as the text-to-speech calls it: audio for each text, after a pause."""

    def GetRivaSynthesisConfig(self, request):
        return object()

    def SynthesizeOnline(self, requests, metadata=None):
        for request in requests:  # stitched: one call per line, its sentences fed in as they come
            time.sleep(0.05)
            pcm = tone(60 * len(request.text.split()), rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE)
            for start in range(0, len(pcm), 4410):  # 100 ms responses
                yield SimpleNamespace(audio=pcm[start : start + 4410])


if __name__ == "__main__":
    unittest.main()
