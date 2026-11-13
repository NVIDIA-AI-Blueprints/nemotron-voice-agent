# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json

from examples.generic_assistant_livekit.voices import (
    choose_voice,
    parse_set_voice,
    parse_voices,
    read_voices_file,
    with_current,
    write_voices_file,
)


def test_magpie_multilingual_returns_only_the_requested_language():
    params = [
        {
            "language_code": "en-US,es-US",
            "voice_name": "Magpie-Multilingual",
            "subvoices": "EN-US.Aria:0,EN-US.Mia:1,ES-US.Isabela:2",
        }
    ]
    assert parse_voices(params, "en-US") == [
        {"id": "Magpie-Multilingual.EN-US.Aria", "name": "Aria"},
        {"id": "Magpie-Multilingual.EN-US.Mia", "name": "Mia"},
    ]
    assert parse_voices(params, "es-US") == [{"id": "Magpie-Multilingual.ES-US.Isabela", "name": "Isabela"}]


def test_language_match_ignores_case():
    params = [{"language_code": "en-US", "voice_name": "Magpie-Multilingual", "subvoices": "EN-US.Aria:0"}]
    assert [v["name"] for v in parse_voices(params, "EN-us")] == ["Aria"]


def test_zeroshot_voices_are_shared_across_the_declared_languages():
    params = [
        {"language_code": "en-US,es-US", "voice_name": "Magpie-ZeroShot-Multilingual", "subvoices": "Male:6,Female:0"}
    ]
    expected = [
        {"id": "Magpie-ZeroShot-Multilingual.Male", "name": "Male"},
        {"id": "Magpie-ZeroShot-Multilingual.Female", "name": "Female"},
    ]
    assert parse_voices(params, "en-US") == expected
    assert parse_voices(params, "es-US") == expected
    assert parse_voices(params, "fr-FR") == []


def test_nemo_speech_plain_names_use_short_ids():
    params = [{"language_code": "en-US", "voice_name": "magpietts", "subvoices": "John,Sofia,Aria,Jason,Leo"}]
    assert parse_voices(params, "en-US") == [
        {"id": name, "name": name} for name in ("John", "Sofia", "Aria", "Jason", "Leo")
    ]


def test_nemo_speech_voices_by_language_map_and_the_voices_key():
    by_language = '{"en-US":{"voices":["magpietts.John","magpietts.Sofia"]},"es-ES":{"voices":["magpietts.John"]}}'
    params = [{"language_code": "en-US", "voice_name": "magpietts", "voices_by_language": by_language}]
    assert [v["id"] for v in parse_voices(params, "en-US")] == ["John", "Sofia"]
    assert [v["id"] for v in parse_voices([{"voices": "John,Sofia"}], "en-US")] == ["John", "Sofia"]


def test_duplicates_across_model_configs_are_removed():
    config = {"language_code": "en-US", "voice_name": "magpietts", "subvoices": "John,Sofia"}
    assert [v["id"] for v in parse_voices([config, dict(config)], "en-US")] == ["John", "Sofia"]


def test_bad_json_and_empty_params_are_ignored():
    assert parse_voices([{"voices_by_language": "{not json"}, {}], "en-US") == []
    assert parse_voices([], "en-US") == []


def test_with_current_adds_the_configured_voice_when_the_server_did_not_list_it():
    listed = [{"id": "Magpie-Multilingual.EN-US.Mia", "name": "Mia"}]
    result = with_current(listed, "Magpie-Multilingual.EN-US.Aria")
    assert result[0] == {"id": "Magpie-Multilingual.EN-US.Aria", "name": "Aria"}
    assert result[1:] == listed
    assert with_current(listed, "Magpie-Multilingual.EN-US.Mia") == listed


def test_set_voice_accepts_only_listed_voices():
    allowed = ["Magpie-Multilingual.EN-US.Aria", "Magpie-Multilingual.EN-US.Mia"]
    ok = json.dumps({"type": "set_voice", "voice": "Magpie-Multilingual.EN-US.Mia"}).encode()
    assert parse_set_voice(ok, allowed) == "Magpie-Multilingual.EN-US.Mia"
    unlisted = json.dumps({"type": "set_voice", "voice": "Magpie-Multilingual.EN-US.Evil"}).encode()
    assert parse_set_voice(unlisted, allowed) is None


def test_set_voice_ignores_other_and_malformed_messages():
    allowed = ["Aria"]
    assert parse_set_voice(b'{"type": "other", "voice": "Aria"}', allowed) is None
    assert parse_set_voice(b'{"voice": "Aria"}', allowed) is None
    assert parse_set_voice(b'{"type": "set_voice", "voice": 7}', allowed) is None
    assert parse_set_voice(b'{"type": "set_voice", "voice": ["Aria"]}', allowed) is None
    assert parse_set_voice(b"[1, 2]", allowed) is None
    assert parse_set_voice(b"not json", allowed) is None
    assert parse_set_voice(b"\xff\xfe", allowed) is None
    assert parse_set_voice(None, allowed) is None


def test_choose_voice_uses_only_listed_voices():
    available = ["Magpie-Multilingual.EN-US.Aria", "Magpie-Multilingual.EN-US.Mia"]
    assert choose_voice("Magpie-Multilingual.EN-US.Mia", available, "default") == "Magpie-Multilingual.EN-US.Mia"
    assert choose_voice("Magpie-Multilingual.EN-US.Evil", available, "default") == "default"
    assert choose_voice(None, available, "default") == "default"
    assert choose_voice(["Aria"], available, "default") == "default"


def test_voices_file_round_trip_is_atomic_and_readable(tmp_path):
    path = tmp_path / "state" / "voices.json"
    voices = [{"id": "Aria", "name": "Aria"}, {"id": "Mia", "name": "Mia"}]
    write_voices_file(path, voices, "Aria")
    assert read_voices_file(path) == {"voices": voices, "default": "Aria"}
    assert [p.name for p in path.parent.iterdir()] == ["voices.json"]  # no temporary files left behind
    write_voices_file(path, voices[:1], "Aria")
    assert read_voices_file(path)["voices"] == voices[:1]


def test_reading_a_missing_or_broken_voices_file_returns_an_empty_list(tmp_path):
    empty = {"voices": [], "default": ""}
    assert read_voices_file(tmp_path / "missing.json") == empty
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_voices_file(bad) == empty
    bad.write_text(json.dumps({"voices": "nope"}))
    assert read_voices_file(bad) == empty
    bad.write_text(json.dumps({"voices": [{"id": "x"}, {"id": "Aria", "name": "Aria"}, 7], "default": "Aria"}))
    assert read_voices_file(bad)["voices"] == [{"id": "Aria", "name": "Aria"}]
