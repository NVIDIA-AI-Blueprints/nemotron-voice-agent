# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import io
import time
import wave
from dataclasses import replace

import pytest
from PIL import Image
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from monitoring.config import MonitoringConfig
from monitoring.media import extract_inline_media
from monitoring.recorder import SessionRecorder, flatten_metrics, git_revision
from monitoring.store import LocalArtifactStore, SessionStore


class FakeProcessor:
    def __init__(self, name: str):
        self.name = name


class FakeEmitter:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(fn):
            self.handlers[name] = fn
            return fn

        return decorator

    async def emit(self, name, *args):
        await self.handlers[name](self, *args)


def _png_data_url(color=(255, 0, 0)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _settings(tmp_path) -> MonitoringConfig:
    return MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        record_audio_turns=True,
        record_audio_stereo=True,
        record_video="off",
        video_fps=1.0,
    )


def _push(observer, frame, source, destination):
    return observer.on_push_frame(
        FramePushed(
            source=source, destination=destination, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0
        )
    )


def test_flatten_metrics_names():
    assert flatten_metrics(TTFBMetricsData(processor="llm", value=0.25)) == [("ttfb", 0.25)]
    usage = LLMUsageMetricsData(
        processor="llm", value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    names = dict(flatten_metrics(usage))
    assert names["llm_usage.prompt_tokens"] == 10
    assert names["llm_usage.completion_tokens"] == 5


def test_extract_inline_media_replaces_data_urls():
    url = _png_data_url()
    stripped, found = extract_inline_media(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}, {"result": url}]
    )
    assert len(found) == 2 and found[0].sha256 == found[1].sha256
    assert stripped[0]["content"][0]["image_url"]["url"]["type"] == "media_ref"
    assert stripped[1]["result"]["mime"] == "image/png"


def test_recorder_end_to_end(tmp_path):
    settings = _settings(tmp_path)
    store = SessionStore(settings.db_url)
    store.create_schema()
    artifacts = LocalArtifactStore(settings.artifacts_dir)
    llm = FakeProcessor("NvidiaLLMService#0")
    upstream, tts = FakeProcessor("reminder"), FakeProcessor("tts")

    async def scenario():
        recorder = SessionRecorder(
            session_id="s1",
            example="multilingual-assistant",
            config={"llm": {"model": "nemotron"}},
            settings=settings,
            store=store,
            artifacts=artifacts,
            llm=llm,
            language="de-DE",
        )
        user_agg, assistant_agg, latency = FakeEmitter(), FakeEmitter(), FakeEmitter()
        recorder.attach(object(), user_agg, assistant_agg, latency)
        await recorder.start()
        observer = recorder.observers()[0]

        await user_agg.emit("on_user_turn_started", None)
        await user_agg.emit("on_user_turn_stopped", None, type("M", (), {"content": "Was siehst du?"})())

        context = LLMContext([{"role": "system", "content": "sys"}])
        context.add_message({"role": "user", "content": [{"type": "image_url", "image_url": {"url": _png_data_url()}}]})
        await _push(observer, LLMContextFrame(context=context), upstream, llm)
        await _push(observer, MetricsFrame(data=[TTFBMetricsData(processor=llm.name, value=0.4)]), llm, tts)
        await _push(observer, LLMTextFrame(text="Ein rotes "), llm, tts)
        await _push(observer, LLMTextFrame(text="Bild."), llm, tts)
        await _push(observer, LLMFullResponseEndFrame(), llm, tts)
        await _push(
            observer,
            FunctionCallResultFrame(
                function_name="take_photo",
                tool_call_id="c1",
                arguments={},
                result={"image": _png_data_url((0, 0, 255))},
            ),
            llm,
            tts,
        )
        await _push(observer, UserImageRawFrame(image=bytes(4 * 4 * 3), size=(4, 4), format="RGB"), upstream, llm)
        await latency.emit("on_latency_measured", 0.9)
        await assistant_agg.emit("on_assistant_turn_started")
        await assistant_agg.emit(
            "on_assistant_turn_stopped", type("M", (), {"content": "Ein rotes Bild.", "interrupted": False})()
        )
        recorder._queue_turn_audio("user", 1, b"\x00\x01" * 1600, 16000)
        stereo = recorder._stereo
        recorder.writer.put_task(lambda a: stereo.append(b"\x00\x00" * 3200, 16000, 2) or [])
        await recorder.close()

    asyncio.run(scenario())

    session = store.get_session("s1")
    assert session["ended_at"] is not None and session["config"]["llm"]["model"] == "nemotron"
    turns = store.rows("turns", "s1")
    assert [t["idx"] for t in turns] == [0, 1]
    assert turns[1]["user_text"] == "Was siehst du?" and turns[1]["bot_text"] == "Ein rotes Bild."

    (call,) = store.rows("llm_calls", "s1")
    assert call["output_text"] == "Ein rotes Bild." and call["ttfb"] == 0.4
    assert call["n_images"] == 1 and call["image_pixels"] == 48
    assert call["messages"][1]["content"][0]["image_url"]["url"]["type"] == "media_ref"

    media = {(m["modality"], m["source"]) for m in store.rows("media", "s1")}
    assert ("image", "context") in media
    assert ("image", "tool:take_photo") in media
    assert ("image", "user_image") in media
    assert ("audio_user", "pipeline") in media
    assert ("audio_conversation", "pipeline") in media
    for row in store.rows("media", "s1"):
        assert (settings.artifacts_dir / row["artifact_key"]).exists()

    conversation = next(m for m in store.rows("media", "s1") if m["modality"] == "audio_conversation")
    with wave.open(str(settings.artifacts_dir / conversation["artifact_key"])) as wf:
        assert wf.getnchannels() == 2 and wf.getnframes() == 1600

    metric_names = {m["name"] for m in store.rows("metrics", "s1")}
    assert {"ttfb", "user_bot_latency"} <= metric_names
    result_event = next(e for e in store.rows("events", "s1") if e["kind"] == "function_result")
    assert result_event["data"]["result"]["image"]["type"] == "media_ref"


def test_disabled_recorder_returns_none(tmp_path):
    settings = replace(_settings(tmp_path), enabled=False)
    assert SessionRecorder.create(session_id="x", example="e", config={}, settings=settings) is None


def _message(content: str, interrupted: bool = False):
    return type("M", (), {"content": content, "interrupted": interrupted})()


def test_recorder_speech_times_barge_in_and_turn_metrics(tmp_path):
    settings = _settings(tmp_path)
    store = SessionStore(settings.db_url)
    store.create_schema()
    llm, upstream, tts = FakeProcessor("NvidiaLLMService#0"), FakeProcessor("reminder"), FakeProcessor("tts")
    stamps = {}

    async def scenario():
        recorder = SessionRecorder(
            session_id="s1",
            example="multilingual-assistant",
            config={"client_tools_perceivable": ["move_head"]},
            settings=settings,
            store=store,
            artifacts=LocalArtifactStore(settings.artifacts_dir),
            llm=llm,
        )
        user_agg, assistant_agg = FakeEmitter(), FakeEmitter()
        recorder.attach(object(), user_agg, assistant_agg)
        await recorder.start()
        observer = recorder.observers()[0]

        async def push(frame, source=upstream, destination=llm):
            await _push(observer, frame, source, destination)

        async def llm_answer(text):
            await push(LLMContextFrame(context=LLMContext([{"role": "user", "content": "hi"}])))
            await assistant_agg.emit("on_assistant_turn_started")
            await push(MetricsFrame(data=[TTFBMetricsData(processor=llm.name, value=0.2)]), llm, tts)
            await push(LLMTextFrame(text=text), llm, tts)

        # Start-up artifacts: zero-valued ttfb/processing samples are not measurements.
        await push(
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor=llm.name, value=0.0),
                    ProcessingMetricsData(processor=llm.name, value=0.0),
                ]
            ),
            llm,
            tts,
        )

        # Turn 1: a tool call, then the answer is spoken.
        await push(UserStartedSpeakingFrame())
        await user_agg.emit("on_user_turn_started", None)
        stamps["speech_end_1"] = time.time() - 0.2
        await push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=stamps["speech_end_1"] + 0.2))
        await user_agg.emit("on_user_turn_stopped", None, _message("tourne la tête"))
        await push(
            FunctionCallInProgressFrame(function_name="move_head", tool_call_id="c1", arguments={"direction": "left"}),
            llm,
            tts,
        )
        await push(FunctionCallResultFrame(function_name="move_head", tool_call_id="c1", arguments={}, result={}))
        await llm_answer("answer one")
        await push(LLMFullResponseEndFrame(), llm, tts)
        await push(BotStartedSpeakingFrame())
        stamps["audio_1"] = time.time()

        # Turn 2: the user barges in. The next turn starts before the interrupted
        # assistant turn reports its text. (Turns are attributed by time: keep them apart.)
        await asyncio.sleep(0.2)
        await push(UserStartedSpeakingFrame())
        await push(InterruptionFrame())
        await user_agg.emit("on_user_turn_started", None)
        await assistant_agg.emit("on_assistant_turn_stopped", _message("answer one", interrupted=True))
        await push(BotStoppedSpeakingFrame())
        await push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()))
        await user_agg.emit("on_user_turn_stopped", None, _message("stop"))
        # The LLM is cancelled before its first token: the TTFB pipecat emits while
        # stopping its metrics is the time until the interruption.
        await push(LLMContextFrame(context=LLMContext([{"role": "user", "content": "stop"}])))
        await push(InterruptionFrame())
        await push(MetricsFrame(data=[TTFBMetricsData(processor=llm.name, value=0.5)]), llm, tts)
        await llm_answer("answer two")
        await push(LLMFullResponseEndFrame(), llm, tts)
        await push(BotStartedSpeakingFrame())
        await assistant_agg.emit("on_assistant_turn_stopped", _message("answer two"))
        await recorder.close()

    asyncio.run(scenario())

    _, first, second = store.rows("turns", "s1")
    assert first["bot_text"] == "answer one" and first["interrupted"] is True
    assert second["bot_text"] == "answer two" and second["interrupted"] is False
    assert first["user_speech_stopped_at"] == pytest.approx(stamps["speech_end_1"])
    assert first["bot_speech_started_at"] == pytest.approx(stamps["audio_1"], abs=0.05)
    assert first["user_stopped_at"] > first["user_speech_stopped_at"]  # existing column keeps its meaning
    assert (first["barge_in"], second["barge_in"]) == (False, True)

    samples = [m for m in store.rows("metrics", "s1") if m["name"] in ("ttfb", "processing")]
    assert samples and all(m["value"] > 0 for m in samples)
    calls = store.rows("llm_calls", "s1")
    assert [(c["interrupted"], c["ttfb"], c["output_text"]) for c in calls] == [
        (False, 0.2, "answer one"),
        (True, None, ""),
        (False, 0.2, "answer two"),
    ]

    kinds = [e["kind"] for e in store.rows("events", "s1")]
    assert "function_call_in_progress" in kinds
    (tool_call,) = store.rows("tool_calls", "s1")
    assert (tool_call["name"], tool_call["outcome"], tool_call["perceivable"]) == ("move_head", "ok", True)
    assert tool_call["turn_idx"] == 1 and tool_call["trigger"] == "llm" and tool_call["target"] == "client"

    one, two = store.rows("turn_metrics", "s1")
    assert (one["kind"], one["response_via"], one["barge_in"]) == ("tool", "tool", False)
    assert one["voice_latency"] == pytest.approx(stamps["audio_1"] - stamps["speech_end_1"], abs=0.05)
    assert one["response_latency"] <= one["voice_latency"]
    assert (two["kind"], two["barge_in"], two["n_llm_calls"]) == ("plain", True, 2)
    assert two["voice_latency"] is not None and store.system_samples_between(0, time.time()) == []


def test_git_revision_handles_packed_refs_worktrees_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert git_revision(repo) is None  # ref not found anywhere
    (repo / ".git" / "packed-refs").write_text(
        "# pack-refs with: peeled\nabc123 refs/heads/main\ndef456 refs/heads/dev\n"
    )
    assert git_revision(repo) == "abc123"
    (repo / ".git" / "refs" / "heads" / "main").write_text("fff999\n")
    assert git_revision(repo) == "fff999"  # a loose ref is newer than the packed one

    worktree_git = repo / ".git" / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/dev\n")
    (worktree_git / "commondir").write_text("../..\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n")
    assert git_revision(worktree) == "def456"

    (repo / ".git" / "HEAD").write_text("0123abc\n")
    assert git_revision(repo) == "0123abc"  # detached HEAD
    assert git_revision(tmp_path / "nowhere") is None  # no .git, as in the app container
    monkeypatch.setenv("GIT_SHA", "from-env")
    assert git_revision(tmp_path / "nowhere") == "from-env"
