# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest

from nvidia_word_tts import (
    normalize_durations_to_seconds,
    word_times_from_magpie_meta,
    word_times_from_meta_words,
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

    def test_meta_words_ms(self) -> None:
        words = [
            {"word": "Hello", "start_time": 0, "end_time": 200},
            {"word": "world", "start_time": 200, "end_time": 450},
        ]
        word_times, next_t, total = word_times_from_meta_words(words)
        self.assertEqual(total, 2)
        self.assertEqual(word_times, [("Hello", 0.0), ("world", 0.2)])
        self.assertAlmostEqual(next_t, 0.45)

    def test_meta_words_skip(self) -> None:
        words = [
            {"word": "a", "start_time": 0, "end_time": 50},
            {"word": "b", "start_time": 50, "end_time": 100},
        ]
        word_times, next_t, total = word_times_from_meta_words(words, skip_tokens=1)
        self.assertEqual(total, 2)
        self.assertEqual(word_times, [("b", 0.05)])
        self.assertAlmostEqual(next_t, 0.1)

    def test_empty_inputs(self) -> None:
        self.assertEqual(word_times_from_magpie_meta("", [0.1]), ([], 0.0, 0))
        self.assertEqual(word_times_from_meta_words([]), ([], 0.0, 0))


if __name__ == "__main__":
    unittest.main()
