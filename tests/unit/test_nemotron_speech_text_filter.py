# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NVIDIA TTS input filters."""

import unittest

from examples.shared.nemotron_speech_text_filter import (
    NemotronSpeechMarkdownTextFilter,
    NemotronSpeechTextFilter,
)


class NemotronSpeechTextFilterTests(unittest.IsolatedAsyncioTestCase):
    """Verify NVIDIA TTS input cleanup."""

    async def test_punctuation_only_segments_are_removed(self) -> None:
        """Drop text that Magpie rejects as punctuation only."""
        cases = (
            # Observed: a sentence split left a lone "?" ahead of the next sentence.
            (
                "?\n\nOnce I can locate your reservation, I can help.",
                "Once I can locate your reservation, I can help.",
            ),
            ("?", ""),
            ("...", ""),
            ("Thanks for waiting.\n---\nYour flight is confirmed.", "Thanks for waiting.\nYour flight is confirmed."),
            ("- Option one\n- Option two", "Option one\nOption two"),
            ("• Window seat", "Window seat"),
            ("— and that is all.", "and that is all."),
            ("？\n您的航班已确认。", "您的航班已确认。"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(await NemotronSpeechTextFilter().filter(text), expected)

    async def test_speakable_text_is_unchanged(self) -> None:
        """Keep punctuation that belongs to words, numbers, or sentences."""
        cases = (
            "Is that right? Your flight leaves at 3 PM.",
            "The fee is $100, or -5 percent with the coupon.",
            '"Hello," she said.',
            "e.g. a window seat",
            "Grüß Gott! Ihr Flug ist bestätigt.",
            "Flight HAT045 departs JFK at 10:30.",
            "1. Check in online.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(await NemotronSpeechTextFilter().filter(text), text)

    async def test_reserved_characters_are_still_removed(self) -> None:
        """Keep the existing reserved-character cleanup."""
        self.assertEqual(
            await NemotronSpeechTextFilter().filter("PNR **ABC123** <break> {AA123}"),
            "PNR ABC123 break> AA123",
        )

    async def test_markdown_filter_also_removes_punctuation_only_segments(self) -> None:
        """Apply the same Magpie-safe cleanup after Markdown stripping."""
        filtered = await NemotronSpeechMarkdownTextFilter().filter("?\n\nOnce I can locate your reservation.")

        self.assertEqual(filtered.strip(), "Once I can locate your reservation.")
