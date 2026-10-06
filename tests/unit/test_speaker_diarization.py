# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy

from examples.shared.pipeline_utils import build_user_aggregator_params
from examples.shared.speaker import (
    SpeakerDiarizationProcessor,
    SpeakerRun,
    TranscriptFragment,
    finalized_turn_payloads,
    forward_speaker_events,
    group_fragments_by_speaker,
    progress_payloads,
    send_rtvi_payloads,
    split_transcript_by_speaker,
)
from utils import PROJECT_ROOT, build_services_api_response, filter_session_config, load_yaml_file


def _word(text: str, speaker_tag: int | None):
    return SimpleNamespace(word=text, speaker_tag=speaker_tag, start_time=0, end_time=100)


def _result(*words):
    return SimpleNamespace(alternatives=[SimpleNamespace(words=list(words))])


def _frame(text: str, result) -> TranscriptionFrame:
    frame = TranscriptionFrame(text=text, user_id="", timestamp="")
    frame.result = result
    return frame


def _interim_frame(text: str, result) -> InterimTranscriptionFrame:
    frame = InterimTranscriptionFrame(text=text, user_id="", timestamp="")
    frame.result = result
    return frame


class SpeakerTagParsingTest(unittest.TestCase):
    def test_splits_result_at_word_speaker_changes(self):
        result = _result(_word("hello", 0), _word("there", 0), _word("hi", 1))
        self.assertEqual(split_transcript_by_speaker(result), [(0, "hello there"), (1, "hi")])

    def test_untagged_word_extends_current_speaker(self):
        result = _result(_word("hello", 0), _word("there", None))
        self.assertEqual(split_transcript_by_speaker(result), [(0, "hello there")])

    def test_fragments_join_with_spaces_into_one_contiguous_speaker_run(self):
        runs = group_fragments_by_speaker(
            [
                TranscriptFragment(0, "this is"),
                TranscriptFragment(0, "what I expected"),
                TranscriptFragment(1, "I disagree"),
            ]
        )

        self.assertEqual(
            [(run.speaker_id, run.text) for run in runs],
            [(0, "this is what I expected"), (1, "I disagree")],
        )

    def test_untagged_fragment_does_not_inherit_the_previous_speaker(self):
        runs = group_fragments_by_speaker(
            [
                TranscriptFragment(0, "good afternoon"),
                TranscriptFragment(None, "so you're on an extended mission"),
                TranscriptFragment(None, "you've done these before"),
            ]
        )

        self.assertEqual(
            [(run.speaker_id, run.text) for run in runs],
            [(0, "good afternoon"), (None, "so you're on an extended mission you've done these before")],
        )


class SpeakerDiarizationProcessorTest(unittest.IsolatedAsyncioTestCase):
    async def _processor(self, *, multi_speaker_support: bool) -> SpeakerDiarizationProcessor:
        processor = SpeakerDiarizationProcessor(multi_speaker_support=multi_speaker_support)
        processor.push_frame = AsyncMock()
        return processor

    async def test_single_speaker_latches_first_human_asr_tag(self):
        processor = await self._processor(multi_speaker_support=False)

        first = _frame("hello", _result(_word("hello", 3)))
        await processor.process_frame(first, FrameDirection.DOWNSTREAM)
        self.assertEqual(processor.primary_speaker_id, 0)
        processor.push_frame.assert_awaited_once_with(first, FrameDirection.DOWNSTREAM)

        processor.push_frame.reset_mock()
        second = _frame("background", _result(_word("background", 1)))
        await processor.process_frame(second, FrameDirection.DOWNSTREAM)
        processor.push_frame.assert_not_awaited()

    async def test_emits_growing_post_latch_snapshot_on_each_final_only(self):
        incremental_handler = AsyncMock()
        processor = SpeakerDiarizationProcessor(
            multi_speaker_support=True,
            incremental_handler=incremental_handler,
        )
        processor.push_frame = AsyncMock()

        await processor.process_frame(
            _interim_frame("hel", _result()),
            FrameDirection.DOWNSTREAM,
        )
        incremental_handler.assert_not_awaited()

        await processor.process_frame(
            _frame("hello", _result(_word("hello", 0))),
            FrameDirection.DOWNSTREAM,
        )
        # A transcription start strategy broadcasts this upstream after the
        # first final. The late signal must not clear the final or change its ID.
        await processor.process_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await processor.process_frame(
            _frame("there", _result(_word("there", 0))),
            FrameDirection.DOWNSTREAM,
        )

        first_runs, first_turn_id, _ = incremental_handler.await_args_list[0].args
        second_runs, second_turn_id, _ = incremental_handler.await_args_list[1].args
        self.assertEqual([(run.speaker_id, run.text) for run in first_runs], [(0, "hello")])
        self.assertEqual([(run.speaker_id, run.text) for run in second_runs], [(0, "hello there")])
        self.assertEqual((first_turn_id, second_turn_id), (1, 1))

    async def test_assigns_a_new_progress_key_after_finalized_fragments_are_taken(self):
        incremental_handler = AsyncMock()
        processor = SpeakerDiarizationProcessor(
            multi_speaker_support=True,
            incremental_handler=incremental_handler,
        )
        processor.push_frame = AsyncMock()

        await processor.process_frame(
            _frame("first", _result(_word("first", 0))),
            FrameDirection.DOWNSTREAM,
        )
        processor.take_turn_fragments()
        await processor.process_frame(
            _frame("second", _result(_word("second", 0))),
            FrameDirection.DOWNSTREAM,
        )

        self.assertEqual(
            [call.args[1] for call in incremental_handler.await_args_list],
            [1, 2],
        )

    async def test_incremental_snapshot_does_not_leak_non_latched_speaker(self):
        incremental_handler = AsyncMock()
        processor = SpeakerDiarizationProcessor(
            multi_speaker_support=False,
            incremental_handler=incremental_handler,
        )
        processor.push_frame = AsyncMock()

        await processor.process_frame(
            _frame("primary", _result(_word("primary", 0))),
            FrameDirection.DOWNSTREAM,
        )
        incremental_handler.reset_mock()
        await processor.process_frame(
            _frame("background", _result(_word("background", 1))),
            FrameDirection.DOWNSTREAM,
        )

        incremental_handler.assert_not_awaited()

    async def test_single_speaker_filters_mixed_result_before_aggregator(self):
        processor = await self._processor(multi_speaker_support=False)
        frame = _frame(
            "primary words background",
            _result(_word("primary", 0), _word("words", 0), _word("background", 1)),
        )

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(frame.text, " primary words")
        processor.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    async def test_single_speaker_filters_other_speaker_interim_after_latch(self):
        processor = await self._processor(multi_speaker_support=False)
        await processor.process_frame(
            _frame("primary", _result(_word("primary", 0))),
            FrameDirection.DOWNSTREAM,
        )
        processor.push_frame.reset_mock()
        interim = _interim_frame("background", _result(_word("background", 1)))
        interim.nvidia_speaker_tag = 1

        await processor.process_frame(interim, FrameDirection.DOWNSTREAM)

        processor.push_frame.assert_not_awaited()

    async def test_single_speaker_keeps_untagged_interim_after_latch(self):
        processor = await self._processor(multi_speaker_support=False)
        await processor.process_frame(
            _frame("primary", _result(_word("primary", 0))),
            FrameDirection.DOWNSTREAM,
        )
        processor.push_frame.reset_mock()
        interim = _interim_frame("and list", _result())

        await processor.process_frame(interim, FrameDirection.DOWNSTREAM)

        processor.push_frame.assert_awaited_once_with(interim, FrameDirection.DOWNSTREAM)

    async def test_multi_speaker_prefixes_each_downstream_run_once(self):
        processor = await self._processor(multi_speaker_support=True)
        frame = _frame("hello there", _result(_word("hello", 0), _word("there", 1)))

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        copies = [call.args[0] for call in processor.push_frame.await_args_list]
        self.assertEqual([copy.text for copy in copies], [" Speaker 1: hello", " Speaker 2: there"])

    async def test_one_based_asr_tags_are_numbered_by_first_appearance(self):
        processor = await self._processor(multi_speaker_support=True)
        frame = _frame("hello there", _result(_word("hello", 2), _word("there", 1)))

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        copies = [call.args[0] for call in processor.push_frame.await_args_list]
        self.assertEqual([copy.text for copy in copies], [" Speaker 1: hello", " Speaker 2: there"])
        self.assertEqual([fragment.speaker_id for fragment in processor.turn_fragments], [0, 1])

    async def test_prefixes_once_across_contiguous_finals(self):
        processor = await self._processor(multi_speaker_support=True)
        first = _frame("Okay,", _result(_word("Okay,", 0)))
        second = _frame("that's fine", _result(_word("that's", 0), _word("fine", 0)))

        await processor.process_frame(first, FrameDirection.DOWNSTREAM)
        await processor.process_frame(second, FrameDirection.DOWNSTREAM)

        texts = [call.args[0].text for call in processor.push_frame.await_args_list]
        self.assertEqual(texts, [" Speaker 1: Okay,", " that's fine"])

    async def test_multi_speaker_keeps_ui_interim_unprefixed(self):
        processor = await self._processor(multi_speaker_support=True)
        interim = _interim_frame("hello", _result(_word("hello", 1)))
        interim.nvidia_speaker_tag = 1

        await processor.process_frame(interim, FrameDirection.DOWNSTREAM)

        self.assertEqual(interim.text, "hello")
        processor.push_frame.assert_awaited_once_with(interim, FrameDirection.DOWNSTREAM)

    async def test_mapped_nvidia_run_is_not_split_again(self):
        processor = await self._processor(multi_speaker_support=True)
        frame = _frame("hello", _result(_word("hello", 1), _word("there", 2)))
        frame.nvidia_speaker_tag = 1

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(frame.text, " Speaker 2: hello")
        self.assertEqual(processor.turn_fragments[0].speaker_id, 1)

    async def test_mapped_nvidia_speaker_tag_zero_is_not_resplit(self):
        processor = await self._processor(multi_speaker_support=True)
        frame = _frame("hello", _result(_word("hello", 0), _word("there", 1)))
        frame.nvidia_speaker_tag = 0

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(frame.text, " Speaker 1: hello")
        self.assertEqual(processor.turn_fragments[0].speaker_id, 0)

    async def test_start_frame_resets_primary_speaker_for_new_connection(self):
        processor = await self._processor(multi_speaker_support=False)
        await processor.process_frame(_frame("one", _result(_word("one", 2))), FrameDirection.DOWNSTREAM)
        await processor.process_frame(_frame("two", _result(_word("two", 5))), FrameDirection.DOWNSTREAM)
        self.assertEqual(processor.primary_speaker_id, 0)

        with patch.object(FrameProcessor, "process_frame", new=AsyncMock()):
            await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

        self.assertIsNone(processor.primary_speaker_id)
        self.assertIsNone(processor.last_speaker_id)
        await processor.process_frame(_frame("three", _result(_word("three", 5))), FrameDirection.DOWNSTREAM)
        self.assertEqual(processor.primary_speaker_id, 0)

    async def test_user_started_speaking_clears_completed_turn_state(self):
        processor = await self._processor(multi_speaker_support=True)
        tagged = _frame("hello", _result(_word("hello", 0)))
        tagged.nvidia_speaker_tag = 0
        await processor.process_frame(tagged, FrameDirection.DOWNSTREAM)
        self.assertEqual(processor.last_speaker_id, 0)
        processor.take_turn_fragments()

        with patch.object(FrameProcessor, "process_frame", new=AsyncMock()):
            await processor.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        self.assertIsNone(processor.last_speaker_id)
        self.assertEqual(processor.turn_fragments, [])


class SpeakerBargeInPolicyTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _transcription_start_strategy(*, latched_speaker_only: bool):
        if latched_speaker_only:
            return TranscriptionUserTurnStartStrategy(use_interim=False)
        params = build_user_aggregator_params(welcome_enabled=False)
        return next(
            strategy
            for strategy in params.user_turn_strategies.start
            if isinstance(strategy, TranscriptionUserTurnStartStrategy)
        )

    @staticmethod
    async def _forward_to_strategy(strategy, frame, direction):  # noqa: ARG004
        await strategy.process_frame(frame)

    async def test_latch_mode_non_latched_transcript_does_not_interrupt(self):
        interrupt = AsyncMock()
        strategy = self._transcription_start_strategy(latched_speaker_only=True)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy_obj, params):  # noqa: ARG001
            if params.enable_interruptions:
                await interrupt()

        processor = SpeakerDiarizationProcessor(multi_speaker_support=False)
        processor.push_frame = AsyncMock(side_effect=partial(self._forward_to_strategy, strategy))
        await processor.process_frame(
            _frame("primary", _result(_word("primary", 0))),
            FrameDirection.DOWNSTREAM,
        )
        interrupt.reset_mock()

        await processor.process_frame(
            _frame("background", _result(_word("background", 1))),
            FrameDirection.DOWNSTREAM,
        )

        interrupt.assert_not_awaited()

    async def test_latch_mode_latched_transcript_allows_interruption_and_turn(self):
        turn_started = AsyncMock()
        strategy = self._transcription_start_strategy(latched_speaker_only=True)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy_obj, params):  # noqa: ARG001
            await turn_started(params.enable_interruptions)

        processor = SpeakerDiarizationProcessor(multi_speaker_support=False)
        processor.push_frame = AsyncMock(side_effect=partial(self._forward_to_strategy, strategy))
        await processor.process_frame(
            _frame("primary", _result(_word("primary", 0))),
            FrameDirection.DOWNSTREAM,
        )

        turn_started.assert_awaited_once_with(True)
        processor.push_frame.assert_awaited_once()

    async def test_multi_speaker_mode_any_tagged_speaker_can_interrupt(self):
        turn_started = AsyncMock()
        strategy = self._transcription_start_strategy(latched_speaker_only=False)

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy_obj, params):  # noqa: ARG001
            await turn_started(params.enable_interruptions)

        processor = SpeakerDiarizationProcessor(multi_speaker_support=True)
        processor.push_frame = AsyncMock(side_effect=partial(self._forward_to_strategy, strategy))
        await processor.process_frame(
            _frame("second speaker", _result(_word("second", 1), _word("speaker", 1))),
            FrameDirection.DOWNSTREAM,
        )

        turn_started.assert_awaited_once_with(True)


class SpeakerTurnPayloadTest(unittest.IsolatedAsyncioTestCase):
    def test_without_a_speaker_processor_the_payload_is_plain(self):
        payloads = finalized_turn_payloads(None, SimpleNamespace(content="hello", timestamp="now", user_id="user"))

        self.assertEqual(
            payloads,
            [{"type": "user-turn-finalized", "timestamp": "now", "transcript": "hello", "user_id": "user"}],
        )

    async def test_send_rtvi_payloads_queues_one_server_message_per_payload(self):
        task = SimpleNamespace(queue_frame=AsyncMock())

        await send_rtvi_payloads(task, [{"type": "a"}, {"type": "b"}])

        self.assertEqual(
            [call.args[0].data for call in task.queue_frame.await_args_list], [{"type": "a"}, {"type": "b"}]
        )

    async def test_forward_speaker_events_sends_progress_and_interim_to_the_task(self):
        task = SimpleNamespace(queue_frame=AsyncMock())
        processor = SpeakerDiarizationProcessor(multi_speaker_support=True)

        forward_speaker_events(processor, task)
        await processor.incremental_handler([SpeakerRun(1, "hello")], 3, "now")
        await processor.interim_handler("hel", "now")

        self.assertEqual(
            [(call.args[0].data["type"], call.args[0].data["transcript"]) for call in task.queue_frame.await_args_list],
            [("user-turn-progress", "hello"), ("user-turn-identifying", "hel")],
        )

    def test_builds_speaker_extras_for_finalized_turns(self):
        processor = SpeakerDiarizationProcessor(multi_speaker_support=True)
        processor.turn_fragments = [TranscriptFragment(0, "hello"), TranscriptFragment(1, "there")]

        payloads = finalized_turn_payloads(
            processor,
            SimpleNamespace(content="ignored", timestamp="now", user_id="user"),
        )

        self.assertEqual(
            [(payload["transcript"], payload["speaker_id"], payload["speaker_display_name"]) for payload in payloads],
            [("hello", 0, "Speaker 1"), ("there", 1, "Speaker 2")],
        )
        self.assertEqual(
            [(payload["turn_id"], payload["run_index"]) for payload in payloads],
            [(0, 0), (0, 1)],
        )

    def test_empty_buffer_final_does_not_reuse_the_turn_just_emitted(self):
        processor = SpeakerDiarizationProcessor(multi_speaker_support=False)
        processor.turn_sequence = 4
        processor.last_speaker_id = 0
        processor.turn_fragments = [TranscriptFragment(0, "red blue")]

        first = finalized_turn_payloads(
            processor,
            SimpleNamespace(content="red blue", timestamp="t1", user_id="user"),
        )
        processor.last_speaker_id = None
        second = finalized_turn_payloads(
            processor,
            SimpleNamespace(content="and green", timestamp="t2", user_id="user"),
        )

        self.assertEqual(
            [(payload["turn_id"], payload["run_index"], payload["transcript"]) for payload in first],
            [(4, 0, "red blue")],
        )
        self.assertEqual(second[0]["run_index"], 0)
        self.assertEqual(second[0]["transcript"], "and green")
        self.assertNotEqual(second[0]["turn_id"], first[0]["turn_id"])
        self.assertEqual(second[0]["speaker_display_name"], "Unknown")

    def test_empty_buffer_multi_speaker_final_does_not_reuse_turn_id(self):
        processor = SpeakerDiarizationProcessor(multi_speaker_support=True)
        processor.turn_sequence = 2
        processor.turn_fragments = [TranscriptFragment(1, "hello there")]

        first = finalized_turn_payloads(
            processor,
            SimpleNamespace(content="hello there", timestamp="t1", user_id="user"),
        )
        second = finalized_turn_payloads(
            processor,
            SimpleNamespace(content="hello there", timestamp="t2", user_id="user"),
        )

        self.assertEqual(first[0]["turn_id"], 2)
        self.assertEqual(first[0]["transcript"], "hello there")
        self.assertEqual([(payload["run_index"], payload["transcript"]) for payload in second], [(0, "hello there")])
        self.assertNotEqual(second[0]["turn_id"], 2)

    def test_progress_payloads_use_stable_turn_and_run_keys(self):
        payloads = progress_payloads([SpeakerRun(0, "hello"), SpeakerRun(1, "there")], turn_id=3, timestamp="now")

        self.assertEqual([payload["type"] for payload in payloads], ["user-turn-progress"] * 2)
        self.assertEqual(
            [(payload["turn_id"], payload["run_index"], payload["transcript"]) for payload in payloads],
            [(3, 0, "hello"), (3, 1, "there")],
        )


class SpeakerPrefixPolicyTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_speaker_stream_never_adds_llm_prefix(self):
        processor = SpeakerDiarizationProcessor(multi_speaker_support=False)
        processor.push_frame = AsyncMock()
        primary = _frame("hello", _result(_word("hello", 0)))
        other = _frame("background", _result(_word("background", 1)))

        await processor.process_frame(primary, FrameDirection.DOWNSTREAM)
        await processor.process_frame(other, FrameDirection.DOWNSTREAM)

        processor.push_frame.assert_awaited_once_with(primary, FrameDirection.DOWNSTREAM)
        self.assertEqual(primary.text, " hello")


class DiarizationConfigurationTest(unittest.TestCase):
    def setUp(self):
        self._api_key = patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test", "SERVICE_RECIPE": "cloud"})
        self._api_key.start()

    def tearDown(self):
        self._api_key.stop()

    def test_client_cannot_claim_diarization_support(self):
        config = filter_session_config(
            {
                "asr_id": "custom-asr",
                "asr_speaker_diarization_supported": True,
                "asr_speaker_diarization_max_speakers": 4,
            }
        )
        self.assertNotIn("asr_speaker_diarization_supported", config)
        self.assertNotIn("asr_speaker_diarization_max_speakers", config)

    def test_request_is_normalized(self):
        self.assertEqual(
            filter_session_config({"asr_speaker_diarization": True})["asr_speaker_diarization"],
            "true",
        )
        self.assertEqual(
            filter_session_config({"asr_speaker_diarization": "false"})["asr_speaker_diarization"],
            "false",
        )

    def test_parakeet_ctc_catalog_disables_diarization(self):
        catalog = load_yaml_file(PROJECT_ROOT / "services.yaml")
        self.assertIsNot(catalog["server"]["asr"]["parakeet-ctc"].get("speaker_diarization_supported"), True)

    def test_services_catalog_exposes_only_verified_dynamic_cap(self):
        entries = {entry["id"]: entry for entry in build_services_api_response()["asr"]}

        nemotron = entries["cloud-nim:nemotron-asr-streaming-english"]
        self.assertTrue(nemotron["speaker_diarization_supported"])
        self.assertEqual(nemotron["speaker_diarization_max_speakers"], 8)
        self.assertIsNot(entries["cloud-nim:parakeet-rnnt"].get("speaker_diarization_supported"), True)
        self.assertNotIn("speaker_diarization_max_speakers", entries["cloud-nim:parakeet-rnnt"])

    def test_max_speakers_is_catalog_owned(self):
        config = filter_session_config(
            {
                "asr_id": "cloud-nim:nemotron-asr-streaming-english",
                "asr_speaker_diarization": True,
                "asr_speaker_diarization_max_speakers": 99,
            }
        )
        self.assertEqual(config["asr_speaker_diarization_max_speakers"], "8")

    def test_prompt_modes_are_explicit(self):
        catalog = load_yaml_file(PROJECT_ROOT / "src/examples/generic/prompts.yaml")
        self.assertFalse(catalog["generic_assistant"].get("multi_speaker_support", False))
        self.assertTrue(catalog["multi_speaker_assistant"]["multi_speaker_support"])
        self.assertFalse(catalog["generic_assistant_without_tools"].get("multi_speaker_support", False))
        self.assertFalse(catalog["flowershop"].get("multi_speaker_support", False))

    def test_multi_speaker_prompt_answers_only_when_called(self):
        entry = load_yaml_file(PROJECT_ROOT / "src/examples/generic/prompts.yaml")["multi_speaker_assistant"]
        content = entry["content"]

        self.assertEqual(entry["temperature"], 0)
        self.assertNotIn("extra_body", entry)
        self.assertIn("Up to eight people", content)
        self.assertIn("you speak only when someone calls you by name", content)
        self.assertIn('reply exactly "..."', content)
        self.assertIn("Never address anyone as Speaker 1 or any other label", content)
        self.assertIn("saying which speaker said what", content)
        self.assertIn("Answer with one JSON object and nothing else.", content)


if __name__ == "__main__":
    unittest.main()
