# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from types import SimpleNamespace

from examples.shared.prewarm import _parse_tts_config


def _raw_config(parameters: dict[str, str]):
    return SimpleNamespace(model_config=[SimpleNamespace(parameters=parameters)])


class ParseTtsConfigTests(unittest.TestCase):
    def test_parses_magpie_multilingual_language_scoped_subvoices(self) -> None:
        raw = _raw_config(
            {
                "language_code": "en-US,es-US",
                "voice_name": "Magpie-Multilingual",
                "subvoices": "EN-US.Aria:0,ES-US.Isabela:1",
            }
        )
        parsed = _parse_tts_config(raw, "Magpie-Multilingual")
        self.assertEqual(parsed["languages"], ["en-US", "es-US"])
        self.assertEqual(
            parsed["voices"],
            [
                {"id": "Magpie-Multilingual.EN-US.Aria", "name": "Aria", "language": "en-US"},
                {"id": "Magpie-Multilingual.ES-US.Isabela", "name": "Isabela", "language": "es-US"},
            ],
        )

    def test_parses_magpie_zeroshot_locale_shared_subvoices(self) -> None:
        raw = _raw_config(
            {
                "language_code": "en-US,es-US,fr-FR",
                "voice_name": "Magpie-ZeroShot-Multilingual",
                "subvoices": "Male:6,Female:0",
            }
        )
        parsed = _parse_tts_config(raw, "Magpie-ZeroShot-Multilingual")
        self.assertEqual(parsed["languages"], ["en-US", "es-US", "fr-FR"])
        female = [v for v in parsed["voices"] if v["name"] == "Female"]
        male = [v for v in parsed["voices"] if v["name"] == "Male"]
        self.assertEqual(len(female), 3)
        self.assertEqual(len(male), 3)
        self.assertTrue(all(v["id"] == "Magpie-ZeroShot-Multilingual.Female" for v in female))
        self.assertTrue(all(v["id"] == "Magpie-ZeroShot-Multilingual.Male" for v in male))
        self.assertEqual({v["language"] for v in female}, {"en-US", "es-US", "fr-FR"})


if __name__ == "__main__":
    unittest.main()
