# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import importlib

import pytest

from nemotron_voice_agent.core.session_audio import SessionAudio

pytest.importorskip("livekit.plugins.nvidia")  # the plugins belong to the example's own environment

SOFIA = [{"id": "Sofia", "name": "Sofia"}]


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")  # the module reads the LiveKit credentials when it loads
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("SERVICE_RECIPE", "server")
    for slot in ("LLM", "ASR", "TTS"):
        monkeypatch.delenv(f"{slot}_SERVICE", raising=False)
    return importlib.import_module("nemotron_voice_agent.livekit.examples.generic_assistant.agent")


def test_voices_are_asked_only_for_tts_entries_that_run(agent):
    asked = []

    def discover(cfg):
        asked.append(cfg.keys["tts"])
        return SOFIA

    def probe(slot, entry):
        return "offline" if entry["id"] == "chatterbox-multilingual-tts" else "ready"

    services, voices = agent.service_snapshot("server", {}, probe=probe, discover=discover)
    assert "chatterbox-multilingual-tts" not in [entry["id"] for entry in services["tts"]]
    assert asked == ["magpie-multilingual-tts", "magpie-zeroshot-tts"]
    assert voices["magpie-multilingual-tts"]["default"] == "Sofia"


def test_voices_found_before_are_kept_and_not_asked_again(agent):
    known = {"magpie-multilingual-tts": {"voices": SOFIA, "default": "Sofia"}}
    asked = []
    _, voices = agent.service_snapshot(
        "server", known, probe=lambda slot, entry: None, discover=lambda cfg: asked.append(cfg.keys["tts"]) or []
    )
    assert "magpie-multilingual-tts" not in asked
    assert voices == known  # a server that lists no voices adds nothing


def test_a_bad_session_choice_falls_back_to_the_defaults_with_a_reason(agent):
    cfg, error = agent.session_config({"services": '{"llm": {"custom": {"base_url": "http://x:1/v1"}}}'})
    assert cfg.keys["llm"] == agent.load_config().keys["llm"]
    assert "needs model_id" in error
    assert agent.session_config({})[1] == ""


def test_a_fallback_reason_reads_as_one_sentence_each(agent):
    _, error = agent.session_config({"services": '{"llm": {"key": "does-not-exist"}}'})
    assert ".." not in error
    assert error.endswith(". The deployment defaults are used instead.")


def test_cloud_slots_without_an_api_key_are_reported(agent):
    cloud = agent.load_config("cloud")
    assert agent.missing_api_key(cloud, env={"NVIDIA_API_KEY": "  "}) == ["llm", "asr", "tts"]
    assert agent.missing_api_key(cloud, env={"NVIDIA_API_KEY": "nvapi-test"}) == []
    assert agent.missing_api_key(agent.load_config("server"), env={}) == []


def test_the_session_log_line_names_the_choices_but_not_the_prompt_text(agent):
    cfg = agent.load_config("cloud", choices={"llm": {"key": "nemotron-super", "settings": {"temperature": 0.2}}})
    line = agent.describe_session("voice-1", cfg, "Custom prompt", "Answer in French. Secret plan.", "Mia")
    assert line.startswith("session voice-1: custom prompt (30 characters), ")
    assert "LLM=nvidia/nemotron-3.5-super" in line and "voice=Mia" in line and "'temperature': 0.2" in line
    assert "Secret plan" not in line


def test_only_nvidia_cloud_slots_get_the_api_key(agent, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    custom = {"server": "my-tts:50051"}
    cfg = agent.load_config("cloud", choices={"tts": {"custom": custom}})
    assert agent.api_key(cfg, "llm") == "nvapi-test"
    assert agent.api_key(cfg, "tts") == ""


@pytest.mark.parametrize("custom", [{"server": "my-tts:50051"}, {"server": "my-tts:443", "function_id": "f"}])
def test_a_custom_speech_endpoint_never_gets_the_environment_key(agent, monkeypatch, custom):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    cfg = agent.load_config("cloud", choices={"tts": {"custom": custom}, "asr": {"custom": custom}})
    assert not agent.build_tts(cfg).nvidia_api_key
    assert not agent.build_stt(cfg).nvidia_api_key


def test_nvidia_cloud_speech_gets_the_key(agent, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    cfg = agent.load_config("cloud")
    assert agent.build_tts(cfg).nvidia_api_key == "nvapi-test"
    assert agent.build_stt(cfg).nvidia_api_key == "nvapi-test"


def test_recording_is_off_unless_the_audio_dump_flag_is_true(agent, tmp_path):
    assert agent.audio_dump_dir({}) is None
    assert agent.audio_dump_dir({"ENABLE_AUDIO_DUMP": "false"}) is None
    assert agent.audio_dump_dir({"ENABLE_AUDIO_DUMP": "TRUE"}) == agent.REPO_ROOT / "audio_dumps"
    assert agent.audio_dump_dir({"ENABLE_AUDIO_DUMP": "true", "AUDIO_DUMP_PATH": str(tmp_path)}) == tmp_path


def test_the_session_recording_is_copied_out_of_the_job_folder(agent, tmp_path):
    source = tmp_path / "job" / "audio.ogg"
    source.parent.mkdir()
    source.write_bytes(b"OggS")
    target = agent.save_recording(source, tmp_path / "dumps", "session_AJ_1")
    assert target == tmp_path / "dumps" / "session_AJ_1.ogg"
    assert target.read_bytes() == b"OggS"
    assert agent.save_recording(tmp_path / "missing.ogg", tmp_path / "dumps", "session_AJ_2") is None


def test_the_recorded_agent_records_what_stt_received_and_tts_produced(agent, monkeypatch):
    def frame(sample):
        return agent.rtc.AudioFrame(data=sample * 160, sample_rate=16000, num_channels=1, samples_per_channel=160)

    async def stt_node(node_agent, audio, model_settings):
        async for heard in audio:
            yield len(heard.data)

    async def tts_node(node_agent, text, model_settings):
        async for _ in text:
            yield frame(b"\x02\x02")

    async def items(*values):
        for value in values:
            yield value

    async def run(node):
        return [item async for item in node]

    monkeypatch.setattr(agent.Agent.default, "stt_node", stt_node)
    monkeypatch.setattr(agent.Agent.default, "tts_node", tts_node)
    heard, spoken = SessionAudio(), SessionAudio()
    heard.start()
    spoken.start()
    recorded = agent.RecordedAgent(heard=heard, spoken=spoken, instructions="Help.")
    assert asyncio.run(run(recorded.stt_node(items(frame(b"\x01\x01"), frame(b"\x01\x01")), None))) == [160, 160]
    assert len(asyncio.run(run(recorded.tts_node(items("Hi."), None)))) == 1
    assert bytes(heard._pcm).strip(b"\x00") == b"\x01\x01" * 320
    assert bytes(spoken._pcm).strip(b"\x00") == b"\x02\x02" * 160


def test_each_recorded_stream_is_saved_and_an_empty_one_is_skipped(agent, tmp_path):
    heard, spoken = SessionAudio(), SessionAudio()
    heard.start()
    spoken.start()
    heard.add(b"\x01\x01" * 160, 16000, 1)
    saved = agent.save_streams({"asr": heard.finish(), "tts": spoken.finish()}, tmp_path / "dumps", "AJ_1")
    assert saved == [tmp_path / "dumps" / "asr_AJ_1_session.wav"]
    assert [path.name for path in (tmp_path / "dumps").iterdir()] == ["asr_AJ_1_session.wav"]
