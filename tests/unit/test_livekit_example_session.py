# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from urllib.parse import parse_qs

from examples.generic_assistant_livekit.session import token_attributes

VOICES = ["Magpie-Multilingual.EN-US.Aria", "Magpie-Multilingual.EN-US.Mia"]


def attributes(query, voices=VOICES):
    return token_attributes(parse_qs(query), voices)


def test_a_listed_voice_and_a_known_use_case_are_kept():
    assert attributes("scenario=bank&voice=Magpie-Multilingual.EN-US.Mia") == {
        "scenario": "bank",
        "voice": "Magpie-Multilingual.EN-US.Mia",
    }


def test_an_unlisted_voice_is_dropped():
    assert attributes("scenario=bank&voice=Magpie-Multilingual.EN-US.Evil") == {"scenario": "bank"}
    assert attributes("voice=Aria", voices=[]) == {"scenario": "generic"}


def test_an_unknown_or_missing_use_case_falls_back_to_the_default():
    assert attributes("scenario=nope") == {"scenario": "generic"}
    assert attributes("") == {"scenario": "generic"}
    assert attributes("scenario=") == {"scenario": "generic"}


def test_repeated_parameters_use_the_first_value():
    assert attributes("scenario=clinic&scenario=bank")["scenario"] == "clinic"
