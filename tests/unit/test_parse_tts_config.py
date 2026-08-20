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
        self.assertEqual(parsed["defaultLanguage"], "en-US")
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

    def test_parses_nemo_speech_plain_subvoices_and_voices_by_language(self) -> None:
        voices_by_language = (
            '{"en-US":{"voices":["magpietts.John","magpietts.Sofia"]},'
            '"es-ES":{"voices":["magpietts.John","magpietts.Sofia"]}}'
        )
        raw = SimpleNamespace(
            model_config=[
                SimpleNamespace(
                    parameters={
                        "language_code": "en-US",
                        "voice_name": "magpietts",
                        "subvoices": "John,Sofia,Aria,Jason,Leo",
                        "voices_by_language": voices_by_language,
                    }
                ),
                SimpleNamespace(
                    parameters={
                        "language_code": "es-ES",
                        "voice_name": "magpietts",
                        "subvoices": "John,Sofia,Aria,Jason,Leo",
                        "voices_by_language": voices_by_language,
                    }
                ),
            ]
        )
        parsed = _parse_tts_config(raw, "")
        self.assertEqual(parsed["languages"], ["en-US", "es-ES"])
        self.assertEqual(parsed["defaultLanguage"], "en-US")
        john = [v for v in parsed["voices"] if v["id"] == "John"]
        self.assertEqual(len(john), 2)
        self.assertEqual({v["language"] for v in john}, {"en-US", "es-ES"})
        self.assertEqual({v["name"] for v in parsed["voices"]}, {"John", "Sofia", "Aria", "Jason", "Leo"})

    def test_parses_voice_metadata_from_each_model_config_without_cross_locale_expansion(self) -> None:
        raw = SimpleNamespace(
            model_config=[
                SimpleNamespace(
                    parameters={
                        "language_code": "en-US",
                        "voice_name": "Magpie-ZeroShot-Multilingual",
                        "subvoices": "Male:6",
                    }
                ),
                SimpleNamespace(
                    parameters={
                        "language_code": "es-ES",
                        "voice_name": "Magpie-ZeroShot-Multilingual",
                        "subvoices": "Female:0",
                    }
                ),
            ]
        )

        parsed = _parse_tts_config(raw, "Magpie-ZeroShot-Multilingual")

        self.assertEqual(parsed["languages"], ["en-US", "es-ES"])
        self.assertEqual(
            parsed["voices"],
            [
                {"id": "Magpie-ZeroShot-Multilingual.Male", "name": "Male", "language": "en-US"},
                {"id": "Magpie-ZeroShot-Multilingual.Female", "name": "Female", "language": "es-ES"},
            ],
        )

    def test_uses_first_declared_language_as_default_across_model_configs(self) -> None:
        raw = SimpleNamespace(
            model_config=[
                SimpleNamespace(parameters={"voice_name": "Magpie-Multilingual"}),
                SimpleNamespace(
                    parameters={
                        "language_code": "es-ES",
                        "voice_name": "Magpie-Multilingual",
                        "subvoices": "ES-ES.Isabela:1",
                    }
                ),
            ]
        )

        parsed = _parse_tts_config(raw, "Magpie-Multilingual")

        self.assertEqual(parsed["languages"], ["es-ES"])
        self.assertEqual(parsed["defaultLanguage"], "es-ES")


if __name__ == "__main__":
    unittest.main()
