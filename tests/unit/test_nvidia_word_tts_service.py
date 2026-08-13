# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest

from pipecat.processors.aggregators.llm_response_universal import TextPartForConcatenation
from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.string import concatenate_aggregated_text
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from nvidia_word_tts import NvidiaWordTTSService, NvidiaWordTTSSettings


class NvidiaWordTTSServiceConfigTests(unittest.TestCase):
    def test_defaults_to_token_aggregation_and_word_timestamp_path(self) -> None:
        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        self.assertEqual(svc._text_aggregation_mode, TextAggregationMode.TOKEN)
        self.assertIsInstance(svc._text_aggregator, SimpleTextAggregator)
        self.assertFalse(svc._push_text_frames)
        self.assertFalse(svc._push_stop_frames)
        self.assertEqual(svc._settings.synthesis_mode, "stitched")
        self.assertIsInstance(svc._settings, NvidiaWordTTSSettings)
        self.assertTrue(svc._enable_word_time_offsets)
        self.assertEqual(svc._custom_configuration.get("enable_word_time_offsets"), "true")
        req = svc._build_base_request()
        self.assertEqual(req.custom_configuration.get("enable_word_time_offsets"), "true")
        # Current public nvidia-riva-client may lack the first-class field; when
        # present it must be enabled for the riva-speech !2703 contract.
        if hasattr(req, "enable_word_time_offsets"):
            self.assertTrue(req.enable_word_time_offsets)

    def test_can_override_aggregation_to_sentence(self) -> None:
        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        self.assertEqual(svc._text_aggregation_mode, TextAggregationMode.SENTENCE)
        self.assertFalse(svc._push_text_frames)

    def test_forces_push_text_frames_false(self) -> None:
        # Callers that need classic sentence TTSTextFrame commits must use NvidiaTTSService.
        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            push_text_frames=True,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        self.assertFalse(svc._push_text_frames)

    def test_end_of_turn_flags_set_and_cleared(self) -> None:
        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        req = svc._build_base_request()
        self.assertNotIn("riva_end_stream", req.custom_configuration)
        self.assertNotIn("is_last_request", req.custom_configuration)

        svc._set_end_of_turn_flags(req, True)
        self.assertEqual(req.custom_configuration.get("riva_end_stream"), "true")
        self.assertEqual(req.custom_configuration.get("is_last_request"), "true")

        svc._set_end_of_turn_flags(req, False)
        self.assertNotIn("riva_end_stream", req.custom_configuration)
        self.assertNotIn("is_last_request", req.custom_configuration)
        # Word-time offsets stay enabled for the turn.
        self.assertEqual(req.custom_configuration.get("enable_word_time_offsets"), "true")
        if hasattr(req, "enable_word_time_offsets"):
            self.assertTrue(req.enable_word_time_offsets)

    def test_split_preserves_leading_and_trailing_spaces(self) -> None:
        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        self.assertEqual(svc._split_text_into_chunks(" hello"), [" hello"])
        self.assertEqual(svc._split_text_into_chunks("world "), ["world "])
        self.assertEqual(svc._split_text_into_chunks(" "), [" "])
        self.assertEqual(svc._split_text_into_chunks(""), [])

    def test_run_tts_queues_space_and_punct_tokens_verbatim(self) -> None:
        """RC3 accepts space/punct-only tokens; WordTTS must not drop or coalesce them."""
        from queue import Queue

        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        # Exercise the same enqueue policy used by run_tts (no parent strip).
        q: Queue = Queue()
        for chunk in [",", " ", "created", " hello", "."]:
            for piece in svc._split_text_into_chunks(chunk):
                if piece != "":
                    q.put(piece)
        self.assertEqual(
            [q.get_nowait() for _ in range(5)],
            [",", " ", "created", " hello", "."],
        )


class SpacingFlagTests(unittest.TestCase):
    """Verify includes_inter_frame_spaces for LLM tokens vs Magpie meta words."""

    def test_llm_tokens_with_leading_spaces_need_flag_true(self) -> None:
        tokens = ["The", " sky", " turned", " blue"]
        glued = concatenate_aggregated_text(
            [TextPartForConcatenation(t, includes_inter_part_spaces=True) for t in tokens]
        )
        doubled = concatenate_aggregated_text(
            [TextPartForConcatenation(t, includes_inter_part_spaces=False) for t in tokens]
        )
        self.assertEqual(glued, "The sky turned blue")
        self.assertEqual(doubled, "The  sky  turned  blue")

    def test_meta_words_without_leading_spaces_need_flag_false(self) -> None:
        words = ["The", "sky", "turned", "blue"]
        glued = concatenate_aggregated_text(
            [TextPartForConcatenation(w, includes_inter_part_spaces=True) for w in words]
        )
        spaced = concatenate_aggregated_text(
            [TextPartForConcatenation(w, includes_inter_part_spaces=False) for w in words]
        )
        self.assertEqual(glued, "Theskyturnedblue")
        self.assertEqual(spaced, "The sky turned blue")


class MagpieWordCommitSequencerTests(unittest.TestCase):
    """WordTTS commits Magpie meta words with space injection (interim)."""

    def test_uses_magpie_word_sequencer(self) -> None:
        from nvidia_word_tts import _MagpieWordCommitSequencer

        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        self.assertIsInstance(svc._aggregated_frame_sequencer, _MagpieWordCommitSequencer)

    def test_space_insert_between_magpie_tokens(self) -> None:
        from pipecat.frames.frames import AggregatedTextFrame, AggregationType, TTSTextFrame
        from pipecat.utils.context.word_completion_tracker import WordCompletionTracker

        from nvidia_word_tts import _MagpieWordCommitSequencer

        # Spoken slots = full sentence (simulates tracker matching Magpie words).
        sentence = "I am Nemotron, created by NVIDIA"
        seq = _MagpieWordCommitSequencer(name="test")
        ctx = "ctx-1"
        frame = AggregatedTextFrame(sentence, AggregationType.SENTENCE, raw_text=sentence)
        seq.register_spoken(
            frame,
            ctx,
            tracker=WordCompletionTracker(sentence, llm_text=sentence, user_facing_text=sentence),
            append_to_context=True,
        )

        magpie_words = ["I", "am", "Nemotron,", "created", "by", "NVIDIA"]
        committed: list[str] = []
        for w in magpie_words:
            for f in seq.process_word(w, pts=0, context_id=ctx):
                if isinstance(f, TTSTextFrame) and f.append_to_context:
                    self.assertFalse(f.includes_inter_frame_spaces)
                    committed.append(f.text)

        # No remainder dump on force_complete.
        for f in seq.force_complete(last_word_pts=0):
            if isinstance(f, TTSTextFrame) and f.append_to_context:
                committed.append(f.text)

        spaced = concatenate_aggregated_text(
            [TextPartForConcatenation(t, includes_inter_part_spaces=False) for t in committed]
        )
        self.assertEqual(spaced, "I am Nemotron, created by NVIDIA")

    def test_force_complete_skips_unspoken_remainder(self) -> None:
        from pipecat.frames.frames import AggregatedTextFrame, AggregationType, TTSTextFrame
        from pipecat.utils.context.word_completion_tracker import WordCompletionTracker

        from nvidia_word_tts import _MagpieWordCommitSequencer

        text = "Hello world today"
        seq = _MagpieWordCommitSequencer(name="test")
        ctx = "ctx-1"
        frame = AggregatedTextFrame(text, AggregationType.SENTENCE, raw_text=text)
        seq.register_spoken(
            frame,
            ctx,
            tracker=WordCompletionTracker(text, llm_text=text, user_facing_text=text),
            append_to_context=True,
        )
        for f in seq.process_word("Hello", pts=0, context_id=ctx):
            pass
        leftover = [
            f.text
            for f in seq.force_complete(last_word_pts=0)
            if isinstance(f, TTSTextFrame) and f.append_to_context
        ]
        self.assertEqual(leftover, [])

    def test_force_complete_fallback_commits_remainder_when_meta_missing(self) -> None:
        from pipecat.frames.frames import AggregatedTextFrame, AggregationType, TTSTextFrame
        from pipecat.utils.context.word_completion_tracker import WordCompletionTracker

        from nvidia_word_tts import _MagpieWordCommitSequencer

        text = "Hello world"
        seq = _MagpieWordCommitSequencer(name="test")
        seq.commit_unspoken_remainder = True
        ctx = "ctx-1"
        frame = AggregatedTextFrame(text, AggregationType.SENTENCE, raw_text=text)
        seq.register_spoken(
            frame,
            ctx,
            tracker=WordCompletionTracker(text, llm_text=text, user_facing_text=text),
            append_to_context=True,
        )
        for f in seq.process_word("Hello", pts=0, context_id=ctx):
            pass
        leftover = [
            f.text
            for f in seq.force_complete(last_word_pts=0)
            if isinstance(f, TTSTextFrame) and f.append_to_context
        ]
        self.assertTrue(any("world" in t for t in leftover))


if __name__ == "__main__":
    unittest.main()
