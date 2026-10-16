# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

from examples.generic.silent_reply import (
    ADDRESSED_REPLY_FORMAT,
    AddressedReplyMixin,
    SilentReplyFilter,
    is_silence_reply,
    reply_field_chunks,
)
from examples.shared.speaker import (
    SpeakerDiarizationProcessor,
    continues_previous_word,
    group_fragments_by_speaker,
    join_cut_word,
    reassign_first_word,
)


class _AddressedLLM(AddressedReplyMixin, NvidiaLLMService):
    pass


def _word(text: str, speaker: int | None, start_s: float, end_s: float):
    return SimpleNamespace(word=text, speaker_tag=speaker, start_time=int(start_s * 1000), end_time=int(end_s * 1000))


def _final(transcript: str, *words) -> TranscriptionFrame:
    frame = TranscriptionFrame(text=transcript, user_id="", timestamp="t")
    frame.result = SimpleNamespace(alternatives=[SimpleNamespace(transcript=transcript, words=list(words))])
    return frame


def _interim(transcript: str) -> InterimTranscriptionFrame:
    frame = InterimTranscriptionFrame(text=transcript, user_id="", timestamp="t")
    frame.result = SimpleNamespace(alternatives=[SimpleNamespace(transcript=transcript, words=[])])
    return frame


def _aggregate(frames) -> str:
    return concatenate_aggregated_text(
        [TextPartForConcatenation(f.text, includes_inter_part_spaces=f.includes_inter_frame_spaces) for f in frames]
    )


class CutWordRuleTest(unittest.TestCase):
    def test_no_leading_space_close_after_previous_final_continues_a_word(self):
        result = _final("ors?", _word("ors?", 0, 2.72, 2.72)).result
        self.assertTrue(continues_previous_word(result, previous_end_s=2.56))

    def test_leading_space_starts_a_new_word(self):
        result = _final(" hello", _word("hello", 0, 2.72, 2.72)).result
        self.assertFalse(continues_previous_word(result, previous_end_s=2.56))

    def test_first_final_of_a_stream_is_not_a_continuation(self):
        result = _final("Hello", _word("Hello", 0, 0.4, 0.4)).result
        self.assertFalse(continues_previous_word(result, previous_end_s=None))

    def test_halves_more_than_a_second_apart_are_not_joined(self):
        result = _final("ors", _word("ors", 0, 4.0, 4.0)).result
        self.assertFalse(continues_previous_word(result, previous_end_s=2.5))

    def test_restarted_stream_with_earlier_word_times_is_not_joined(self):
        result = _final("Hello", _word("Hello", 0, 0.3, 0.3)).result
        self.assertFalse(continues_previous_word(result, previous_end_s=12.0))

    def test_only_the_first_word_moves_to_the_previous_speaker(self):
        self.assertEqual(reassign_first_word([(1, "ion what?")], 0), [(0, "ion"), (1, "what?")])
        self.assertEqual(reassign_first_word([(0, "ors? Red"), (1, "blue")], 0), [(0, "ors? Red"), (1, "blue")])

    def test_cut_punctuation_is_dropped_from_the_first_half(self):
        self.assertEqual(join_cut_word("the following col.", "ors?"), "the following colors?")


class SpeakerWordJoinProcessorTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, frames, *, multi_speaker_support: bool):
        processor = SpeakerDiarizationProcessor(multi_speaker_support=multi_speaker_support)
        processor.push_frame = AsyncMock()
        for frame in frames:
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        pushed = [call.args[0] for call in processor.push_frame.await_args_list]
        return processor, pushed

    async def test_multi_speaker_joins_cut_word_in_bubbles_and_llm_text(self):
        processor, pushed = await self._run(
            [
                _final(
                    "Repeat the col",
                    _word("Repeat", 0, 1.0, 1.0),
                    _word("the", 0, 1.3, 1.3),
                    _word("col", 0, 1.6, 1.6),
                ),
                _final("ors? Red", _word("ors?", 1, 1.84, 1.84), _word("Red", 0, 2.4, 2.4)),
            ],
            multi_speaker_support=True,
        )

        runs = group_fragments_by_speaker(processor.turn_fragments)
        self.assertEqual([(run.speaker_id, run.text) for run in runs], [(0, "Repeat the colors? Red")])
        self.assertEqual(_aggregate(pushed), "Speaker 1: Repeat the colors? Red")

    async def test_punctuation_at_the_cut_is_repaired_in_the_bubble_but_already_sent_to_the_llm(self):
        # Known limit: the first half reaches the LLM before the second half shows it was cut.
        processor, pushed = await self._run(
            [
                _final(
                    "Repeat the col.",
                    _word("Repeat", 0, 1.0, 1.0),
                    _word("the", 0, 1.3, 1.3),
                    _word("col.", 0, 1.6, 1.6),
                ),
                _final("ors?", _word("ors?", 0, 1.84, 1.84)),
            ],
            multi_speaker_support=True,
        )

        runs = group_fragments_by_speaker(processor.turn_fragments)
        self.assertEqual([(run.speaker_id, run.text) for run in runs], [(0, "Repeat the colors?")])
        self.assertEqual(_aggregate(pushed), "Speaker 1: Repeat the col.ors?")

    async def test_later_words_keep_their_own_speaker(self):
        processor, pushed = await self._run(
            [
                _final("the selecti", _word("the", 0, 1.0, 1.0), _word("selecti", 0, 1.3, 1.3)),
                _final("on what?", _word("on", 1, 1.5, 1.5), _word("what?", 1, 1.8, 1.8)),
            ],
            multi_speaker_support=True,
        )

        runs = group_fragments_by_speaker(processor.turn_fragments)
        self.assertEqual([(run.speaker_id, run.text) for run in runs], [(0, "the selection"), (1, "what?")])
        self.assertEqual(_aggregate(pushed), "Speaker 1: the selection Speaker 2: what?")

    async def test_single_speaker_keeps_cut_piece_tagged_as_another_speaker(self):
        processor, pushed = await self._run(
            [
                _final("Hello col", _word("Hello", 0, 1.0, 1.0), _word("col", 0, 1.3, 1.3)),
                _final("ors", _word("ors", 2, 1.5, 1.5)),
            ],
            multi_speaker_support=False,
        )

        self.assertEqual(_aggregate(pushed), "Hello colors")
        self.assertEqual(processor.turn_fragments[-1].text, "Hello colors")

    async def test_a_closed_turn_does_not_lend_its_speaker_to_the_next_turn(self):
        processor, _ = await self._run(
            [_final("Hello col", _word("Hello", 0, 1.0, 1.0), _word("col", 0, 1.3, 1.3))],
            multi_speaker_support=True,
        )
        processor.take_turn_fragments()

        await processor.process_frame(_final("ors", _word("ors", 1, 1.5, 1.5)), FrameDirection.DOWNSTREAM)

        self.assertEqual([(f.speaker_id, f.text) for f in processor.turn_fragments], [(1, "ors")])

    async def test_new_word_after_a_final_keeps_a_space(self):
        _, pushed = await self._run(
            [
                _final("Hello", _word("Hello", 0, 1.0, 1.0)),
                _final(" there", _word("there", 0, 1.4, 1.4)),
            ],
            multi_speaker_support=False,
        )

        self.assertEqual(_aggregate(pushed), "Hello there")

    async def test_interim_text_is_published_and_cleared_by_the_final(self):
        interim_handler = AsyncMock()
        processor = SpeakerDiarizationProcessor(multi_speaker_support=True, interim_handler=interim_handler)
        processor.push_frame = AsyncMock()

        await processor.process_frame(_interim("hel"), FrameDirection.DOWNSTREAM)
        await processor.process_frame(_interim("hel"), FrameDirection.DOWNSTREAM)
        await processor.process_frame(_interim("hello"), FrameDirection.DOWNSTREAM)
        await processor.process_frame(_final("hello", _word("hello", 0, 1.0, 1.0)), FrameDirection.DOWNSTREAM)

        self.assertEqual([call.args[0] for call in interim_handler.await_args_list], ["hel", "hello", ""])


class SilentReplyFilterTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, frames):
        silent_filter = SilentReplyFilter()
        silent_filter.push_frame = AsyncMock()
        with patch.object(FrameProcessor, "process_frame", AsyncMock()):
            for frame in frames:
                await silent_filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        return [call.args[0] for call in silent_filter.push_frame.await_args_list]

    def test_detects_only_dot_replies(self):
        self.assertTrue(is_silence_reply("..."))
        self.assertTrue(is_silence_reply(" .. . "))
        self.assertFalse(is_silence_reply(""))
        self.assertFalse(is_silence_reply("... sure"))

    async def test_silence_reply_skips_tts_but_still_flows_downstream(self):
        frames = [LLMFullResponseStartFrame(), LLMTextFrame(".."), LLMTextFrame("."), LLMFullResponseEndFrame()]

        pushed = await self._run(frames)

        self.assertEqual(pushed, frames)
        self.assertTrue(all(frame.skip_tts for frame in pushed))

    async def test_spoken_reply_is_released_at_the_first_word(self):
        start, text = LLMFullResponseStartFrame(), LLMTextFrame("Sure")
        pushed = await self._run([start, text])

        self.assertEqual(pushed, [start, text])
        self.assertFalse(any(frame.skip_tts for frame in pushed))

    async def test_interruption_drops_a_held_reply(self):
        interruption = InterruptionFrame()
        pushed = await self._run([LLMFullResponseStartFrame(), LLMTextFrame("."), interruption])

        self.assertEqual(pushed, [interruption])


def _chunk(content: str | None = None, *, usage: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [] if usage else [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2} if usage else None,
        }
    )


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


class AddressedReplyTest(unittest.IsolatedAsyncioTestCase):
    async def _contents(self, parts):
        chunks = [_chunk(part) for part in parts] + [_chunk(usage=True)]
        out = [chunk async for chunk in reply_field_chunks(_stream(chunks))]
        self.assertIsNotNone(out[-1].usage)
        return "".join(chunk.choices[0].delta.content or "" for chunk in out if chunk.choices)

    async def test_streams_only_the_reply_field(self):
        parts = ['{\n  "addressed_to_assistant": tr', 'ue,\n  "reply": "Paris is the cap', 'ital, \\"yes\\"."\n}']

        self.assertEqual(await self._contents(parts), 'Paris is the capital, "yes".')

    async def test_silence_reply_streams_as_dots(self):
        self.assertEqual(await self._contents(['{"addressed_to_assistant": false, "reply": "..."}']), "...")

    async def test_a_reply_flagged_not_addressed_streams_as_dots_even_with_words(self):
        parts = ['{"addressed_to_assistant": fal', 'se, "reply": "Sure, I will', ' remind you."}']

        self.assertEqual(await self._contents(parts), "...")

    async def test_a_flag_that_arrives_after_the_reply_started_does_not_cut_it_off(self):
        parts = ['{"reply": "Sure, here', ' it is.", "addressed_to_assistant": false}']

        self.assertEqual(await self._contents(parts), "Sure, here it is.")

    async def test_a_false_flag_literal_inside_the_reply_is_spoken(self):
        parts = ['{"addressed_to_assistant": true, "reply": "The text is ', '\\"addressed_to_assistant\\": false."}']

        self.assertEqual(await self._contents(parts), 'The text is "addressed_to_assistant": false.')

    async def test_plain_text_passes_through(self):
        self.assertEqual(await self._contents(["  ", "Sure, ", "here it is."]), "  Sure, here it is.")

    async def test_spoken_turns_use_the_json_format_but_one_shot_inference_does_not(self):
        llm = _AddressedLLM(api_key="k", settings=NvidiaLLMSettings(model="m"))
        llm._client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])
        )

        self.assertEqual(llm.build_chat_completion_params({"messages": []})["response_format"], ADDRESSED_REPLY_FORMAT)
        summary = await llm.run_inference(LLMContext([{"role": "user", "content": "hi"}]), system_instruction="s")

        self.assertEqual(summary, "summary")
        self.assertNotIn("response_format", llm._client.chat.completions.create.await_args.kwargs)
        self.assertIn("response_format", llm.build_chat_completion_params({"messages": []}))

    def test_back_to_back_speaker_lines_are_sent_as_one_line(self):
        llm = _AddressedLLM(api_key="k", settings=NvidiaLLMSettings(model="m"))
        messages = [
            {"role": "user", "content": "Greet the room."},
            {"role": "user", "content": "Speaker 1: I think their accent."},
            {"role": "user", "content": "Speaker 1: what's the accent in LA?"},
            {"role": "user", "content": "Speaker 2: no idea"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "Speaker 2: Nemotron, hi"},
        ]

        sent = llm.build_chat_completion_params({"messages": messages})["messages"]

        self.assertEqual(
            [message["content"] for message in sent],
            [
                "Greet the room.",
                "Speaker 1: I think their accent. what's the accent in LA? Speaker 2: no idea",
                "...",
                "Speaker 2: Nemotron, hi",
            ],
        )
        self.assertEqual(messages[1]["content"], "Speaker 1: I think their accent.")


if __name__ == "__main__":
    unittest.main()
