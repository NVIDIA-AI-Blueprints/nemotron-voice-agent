# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json

from nemotron_voice_agent.livekit.examples.generic_assistant import service_state as ss


def statuses(**by_id):
    return lambda slot, entry: by_id.get(entry["id"])


def test_offline_entries_are_hidden_and_the_first_running_one_is_the_default():
    options = {"llm": [{"id": "a", "default": True}, {"id": "b", "default": False}, {"id": "c", "default": False}]}
    result = ss.with_status(options, statuses(a="offline", b="starting"))
    assert result == {
        "llm": [
            {"id": "b", "default": True, "status": "starting"},
            {"id": "c", "default": False},
        ]
    }


def test_a_slot_that_is_all_offline_keeps_its_entries():
    options = {"tts": [{"id": "a", "default": True}, {"id": "b", "default": False}]}
    result = ss.with_status(options, statuses(a="offline", b="offline"))
    assert [(e["id"], e["default"], e["status"]) for e in result["tts"]] == [
        ("a", True, "offline"),
        ("b", False, "offline"),
    ]


def test_the_state_round_trips_and_a_broken_file_reads_as_empty(tmp_path):
    path = tmp_path / "state" / ss.SERVICES_FILE
    services = {"llm": [{"id": "a", "default": True}]}
    voices = {"magpie": {"voices": [{"id": "Aria", "name": "Aria"}], "default": "Aria"}}
    ss.write_service_state(path, services, voices)
    assert ss.read_service_state(path) == {"services": services, "voices": voices}
    assert [p.name for p in path.parent.iterdir()] == [ss.SERVICES_FILE]  # no temporary files left behind
    assert ss.read_service_state(tmp_path / "missing.json") == ss.EMPTY
    path.write_text("{not json")
    assert ss.read_service_state(path) == ss.EMPTY
    path.write_text(json.dumps({"services": {"llm": [{"id": "a"}, {"name": "no id"}, 7], "asr": "x"}}))
    assert ss.read_service_state(path) == {"services": {"llm": [{"id": "a"}]}, "voices": {}}


def test_find_entry_and_voices_of_return_only_what_was_published():
    state = {
        "services": {"tts": [{"id": "magpie"}]},
        "voices": {"magpie": {"voices": [{"id": "Aria", "name": "Aria"}, {"id": "x"}, 7], "default": "Aria"}},
    }
    assert ss.find_entry(state, "tts", "magpie") == {"id": "magpie"}
    assert ss.find_entry(state, "tts", "other") is None
    assert ss.find_entry(state, "llm", "magpie") is None
    assert ss.voices_of(state, "magpie") == {"voices": [{"id": "Aria", "name": "Aria"}], "default": "Aria"}
    assert ss.voices_of(state, "other") == {"voices": [], "default": ""}


def test_only_self_hosted_entries_are_probed(monkeypatch):
    calls = []
    monkeypatch.setattr(ss, "probe_http", lambda url, ready, timeout: calls.append(("http", url)) or "ready")
    monkeypatch.setattr(ss, "probe_tcp", lambda server, timeout: calls.append(("tcp", server)) or "offline")
    assert ss.probe_entry("llm", {"source": "cloud-nim", "base_url": "https://integrate.api.nvidia.com/v1"}) is None
    llm = {"source": "self-hosted", "base_url": "http://nvidia-llm:8000/v1", "health_path": "/v1/health/ready"}
    assert ss.probe_entry("llm", llm) == "ready"
    assert ss.probe_entry("asr", {"source": "self-hosted", "server": "asr:50052"}) == "offline"
    assert calls == [("http", "http://nvidia-llm:8000/v1/health/ready"), ("tcp", "asr:50052")]
