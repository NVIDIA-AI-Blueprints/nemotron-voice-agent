# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json

import pytest

from nemotron_voice_agent.livekit.examples.generic_assistant.session import (
    MAX_PROMPT_CHARS,
    read_choices,
    session_attributes,
)

ARIA, MIA = "Magpie-Multilingual.EN-US.Aria", "Magpie-Multilingual.EN-US.Mia"
STATE = {
    "services": {
        "llm": [
            {"id": "nemotron-super", "default": True, "settings": {"temperature": {"type": "float"}}},
            {"id": "nemotron-lightning", "default": False, "settings": {}},
        ],
        "asr": [{"id": "parakeet-ctc", "default": True, "settings": {}}],
        "tts": [{"id": "magpie", "default": True}, {"id": "chatterbox", "default": False}],
    },
    "voices": {
        "magpie": {"voices": [{"id": ARIA, "name": "Aria"}, {"id": MIA, "name": "Mia"}], "default": ARIA},
        "chatterbox": {"voices": [{"id": "Olivia", "name": "Olivia"}], "default": "Olivia"},
    },
}


def services(body):
    return json.loads(session_attributes(body, STATE).get("services", "{}"))


def test_a_known_use_case_and_a_listed_voice_are_kept():
    assert session_attributes({"scenario": "bank", "voice": MIA}, STATE) == {"scenario": "bank", "voice": MIA}


def test_an_unknown_or_missing_use_case_falls_back_to_the_default():
    assert session_attributes({"scenario": "nope"}, STATE) == {"scenario": "generic"}
    assert session_attributes({}, STATE) == {"scenario": "generic"}


@pytest.mark.parametrize(
    "body,field",
    [
        ({"scenario": 7}, "scenario"),
        ({"prompt": ["x"]}, "prompt"),
        ({"voice": 1}, "voice"),
        ({"services": []}, "services"),
        ({"services": {"asr": "parakeet-ctc"}}, "services.asr"),
        ({"services": {"llm": {"key": 3}}}, "services.llm.key"),
        ({"services": {"llm": {"key": "nemotron-super", "settings": [1]}}}, "services.llm.settings"),
        ({"services": {"tts": {"custom": "my-tts:50051"}}}, "services.tts.custom"),
    ],
)
def test_a_field_of_the_wrong_type_is_refused_instead_of_dropped(body, field):
    with pytest.raises(ValueError, match=f"^{field} has the wrong type$"):
        session_attributes(body, STATE)


def test_the_voice_must_belong_to_the_chosen_tts():
    assert "voice" not in session_attributes({"voice": "Olivia"}, STATE)  # the default TTS has no Olivia
    chosen = session_attributes({"voice": "Olivia", "services": {"tts": {"key": "chatterbox"}}}, STATE)
    assert chosen["voice"] == "Olivia"
    assert "voice" not in session_attributes({"voice": ARIA, "services": {"tts": {"key": "chatterbox"}}}, STATE)


def test_offered_services_and_their_settings_are_kept():
    body = {"services": {"llm": {"key": "nemotron-super", "settings": {"temperature": 0.3, "top_k": 5}}}}
    assert services(body) == {"llm": {"key": "nemotron-super", "settings": {"temperature": 0.3}}}
    assert services({"services": {"llm": {"key": "nemotron-lightning", "settings": {"temperature": 1}}}}) == {
        "llm": {"key": "nemotron-lightning"}
    }


def test_services_and_settings_that_were_not_offered_are_dropped():
    assert services({"services": {"llm": {"key": "nemotron-bolt-74b"}}}) == {}
    assert services({"services": {"nope": {"key": "x"}}}) == {}
    weird = {"temperature": {"nested": 1}}
    assert services({"services": {"llm": {"key": "nemotron-super", "settings": weird}}}) == {
        "llm": {"key": "nemotron-super"}
    }


def test_a_custom_endpoint_is_validated_and_trimmed():
    custom = {"name": "", "base_url": " http://my-llm:8000/v1 ", "model_id": "my/model", "extra_params": ""}
    assert services({"services": {"llm": {"custom": custom}}}) == {
        "llm": {"custom": {"name": "Custom LLM", "base_url": "http://my-llm:8000/v1", "model_id": "my/model"}}
    }


@pytest.mark.parametrize(
    "slot,custom",
    [
        ("llm", {"base_url": "grpc://my-llm:8000", "model_id": "m"}),
        ("llm", {"base_url": "http://my-llm:8000/v1"}),
        ("llm", {"base_url": "http://my-llm:8000/v1", "model_id": "m", "extra_params": "[1]"}),
        ("asr", {"server": "no port", "model": "m"}),
        ("asr", {"server": "http://asr:50051", "model": "m"}),
        ("asr", {"server": "asr:99999", "model": "m"}),
        ("llm", {"base_url": "http://[::1:8000/v1", "model_id": "m"}),
        ("llm", {"base_url": "javascript:alert(1)", "model_id": "m"}),
        ("tts", {"server": "tts"}),
        ("tts", {"server": "tts:0"}),
        ("tts", {"server": "user@tts:50051"}),
        ("tts", {"server": "::1:50051"}),
        ("llm", {"base_url": "https://user:secret@my-llm/v1", "model_id": "m"}),
        ("tts", {"server": 7}),
        ("tts", {"server": "tts:50051", "voice_id": "x" * 600}),
    ],
)
def test_an_unusable_custom_endpoint_is_refused_with_a_reason(slot, custom):
    with pytest.raises(ValueError, match=f"custom {slot}"):
        session_attributes({"services": {slot: {"custom": custom}}}, STATE)


def test_a_custom_tts_gets_no_listed_voice():
    body = {"voice": ARIA, "services": {"tts": {"custom": {"server": "my-tts:50051", "voice_id": "Sofia"}}}}
    attributes = session_attributes(body, STATE)
    assert "voice" not in attributes
    assert json.loads(attributes["services"])["tts"]["custom"]["voice_id"] == "Sofia"


def test_a_custom_prompt_is_trimmed():
    assert session_attributes({"prompt": "  Be brief.  "}, STATE)["prompt"] == "Be brief."
    assert len(session_attributes({"prompt": "x" * MAX_PROMPT_CHARS}, STATE)["prompt"]) == MAX_PROMPT_CHARS
    assert "prompt" not in session_attributes({"prompt": "   "}, STATE)


def test_a_too_long_custom_prompt_is_refused_instead_of_cut():
    with pytest.raises(ValueError, match=f"longer than {MAX_PROMPT_CHARS} characters"):
        session_attributes({"prompt": "x" * (MAX_PROMPT_CHARS + 1)}, STATE)


def test_read_choices_keeps_only_known_slots_and_objects():
    raw = json.dumps({"llm": {"key": "a"}, "asr": "b", "nope": {"key": "c"}})
    assert read_choices(raw) == {"llm": {"key": "a"}}
    assert read_choices(None) == {}
    assert read_choices("{not json") == {}
    assert read_choices("[1]") == {}
