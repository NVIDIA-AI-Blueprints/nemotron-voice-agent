# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest

from nvidia_word_tts import (
    TimedWord,
    new_words_from_meta_batch,
    normalize_durations_to_seconds,
    parse_meta_word_entries,
    word_times_from_magpie_meta,
)


class MagpieMetaTimestampHelperTests(unittest.TestCase):
    def test_token_aligned_durations(self) -> None:
        word_times, next_t, total = word_times_from_magpie_meta(
            "Hello world",
            [0.2, 0.3],
        )
        self.assertEqual(total, 2)
        self.assertEqual(word_times, [("Hello", 0.0), ("world", 0.2)])
        self.assertAlmostEqual(next_t, 0.5)

    def test_skip_already_emitted_tokens(self) -> None:
        word_times, next_t, total = word_times_from_magpie_meta(
            "Hello world again",
            [0.2, 0.3, 0.4],
            skip_tokens=2,
        )
        self.assertEqual(total, 3)
        self.assertEqual(word_times, [("again", 0.5)])
        self.assertAlmostEqual(next_t, 0.9)

    def test_char_aligned_durations_merge_to_words(self) -> None:
        text = "Hi yo"
        durations = [0.05] * len(text)
        word_times, next_t, total = word_times_from_magpie_meta(text, durations)
        self.assertEqual([w for w, _ in word_times], ["Hi", "yo"])
        self.assertEqual(total, 2)
        self.assertGreater(next_t, 0.0)

    def test_normalize_frame_count_durations(self) -> None:
        seconds = normalize_durations_to_seconds([80.0, 160.0], frame_rate_hz=80.0)
        self.assertEqual(seconds, [1.0, 2.0])

    def test_empty_inputs(self) -> None:
        self.assertEqual(word_times_from_magpie_meta("", [0.1]), ([], 0.0, 0))


def _ms(word: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {"word": word, "start_time": start_ms, "end_time": end_ms}


class MagpieBatchIngestTests(unittest.TestCase):
    def test_three_incremental_sentence_batches(self) -> None:
        s1 = parse_meta_word_entries([_ms("Hello", 0, 200), _ms("there.", 200, 500)])
        s2 = parse_meta_word_entries([_ms("How", 500, 700), _ms("are", 700, 850), _ms("you?", 850, 1100)])
        s3 = parse_meta_word_entries([_ms("Fine", 1100, 1400)])

        accepted: list[TimedWord] = []
        for batch in (s1, s2, s3):
            accepted.extend(new_words_from_meta_batch(batch, accepted))

        self.assertEqual([w.word for w in accepted], ["Hello", "there.", "How", "are", "you?", "Fine"])
        self.assertAlmostEqual(accepted[0].start_s, 0.0)
        self.assertAlmostEqual(accepted[-1].start_s, 1.1)

    def test_cumulative_growing_list_emits_suffix_only(self) -> None:
        s1 = parse_meta_word_entries([_ms("Hello", 0, 200), _ms("there.", 200, 500)])
        s1s2 = parse_meta_word_entries(
            [
                _ms("Hello", 0, 200),
                _ms("there.", 200, 500),
                _ms("How", 500, 700),
                _ms("are", 700, 850),
            ]
        )
        accepted = new_words_from_meta_batch(s1, [])
        added = new_words_from_meta_batch(s1s2, accepted)
        self.assertEqual([w.word for w in added], ["How", "are"])
        self.assertAlmostEqual(added[0].start_s, 0.5)

    def test_relative_batch_times_are_offset_after_prior_sentence(self) -> None:
        s1 = parse_meta_word_entries([_ms("Hello", 0, 200), _ms("there.", 200, 500)])
        s2_relative = parse_meta_word_entries([_ms("How", 0, 200), _ms("are", 200, 350)])
        accepted = list(s1)
        added = new_words_from_meta_batch(s2_relative, accepted)
        self.assertEqual([w.word for w in added], ["How", "are"])
        self.assertAlmostEqual(added[0].start_s, 0.5)
        self.assertAlmostEqual(added[1].start_s, 0.7)
        self.assertGreater(added[0].start_s, accepted[-1].end_s - 1e-9)

    def test_repeated_first_word_after_single_accepted_word_is_incremental(self) -> None:
        accepted = parse_meta_word_entries([_ms("Hello", 0, 200)])
        repeated_relative = parse_meta_word_entries([_ms("Hello", 0, 200), _ms("again", 200, 500)])

        added = new_words_from_meta_batch(repeated_relative, accepted)

        self.assertEqual([word.word for word in added], ["Hello", "again"])
        self.assertAlmostEqual(added[0].start_s, 0.2)
        self.assertAlmostEqual(added[1].start_s, 0.4)

    def test_matching_words_with_timestamp_mismatch_are_incremental(self) -> None:
        accepted = parse_meta_word_entries([_ms("Hello", 0, 200), _ms("there", 200, 500)])
        mismatched_relative = parse_meta_word_entries(
            [_ms("Hello", 0, 250), _ms("there", 250, 500), _ms("again", 500, 800)]
        )

        added = new_words_from_meta_batch(mismatched_relative, accepted)

        self.assertEqual([word.word for word in added], ["Hello", "there", "again"])
        self.assertAlmostEqual(added[0].start_s, 0.5)


class LateMetaDropTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_meta_after_context_clear_is_dropped(self) -> None:
        from types import SimpleNamespace

        from nvidia_word_tts import NvidiaWordTTSService

        svc = NvidiaWordTTSService(
            api_key=None,
            server="localhost:50151",
            use_ssl=False,
            model_function_map={"function_id": "", "model_name": "magpie-tts-multilingual"},
        )
        emitted: list[list[tuple[str, float]]] = []

        async def _capture(word_times, context_id=None, includes_inter_frame_spaces=None, **_kwargs):
            emitted.append(list(word_times))

        svc.add_word_timestamps = _capture  # type: ignore[method-assign]
        svc.audio_context_available = lambda _ctx: False  # type: ignore[method-assign]

        response = SimpleNamespace(
            meta=SimpleNamespace(words=[SimpleNamespace(word="Hello", start_time=0, end_time=200)])
        )
        response.HasField = lambda name: name == "meta"  # type: ignore[method-assign]
        await svc._maybe_emit_meta_timestamps(response, "gone-ctx")
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
