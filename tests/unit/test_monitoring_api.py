# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring.api import create_review_router
from monitoring.config import MonitoringConfig
from monitoring.jobs.runner import _DEFAULTS
from monitoring.media import pcm16_to_wav
from monitoring.store import LocalArtifactStore, SessionStore

DREAMER = {**_DEFAULTS, "reasr": {"reference": {"name": "voxtral", "protocol": "openai"}, "candidates": []}}


@pytest.fixture
def env(tmp_path):
    settings = MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        record_audio_turns=True,
        record_audio_stereo=True,
        record_video="off",
        video_fps=1.0,
    )
    store = SessionStore(settings.db_url)
    store.create_schema()
    artifacts = LocalArtifactStore(settings.artifacts_dir)
    now = time.time()
    store.create_session(
        {
            "id": "s1",
            "example": "multilingual-assistant",
            "started_at": now - 100,
            "last_seen_at": now - 50,
            "ended_at": now - 50,
            "config": {"language": "fr-FR", "asr": {"model": "nemo"}},
            "artifact_prefix": "sessions/s1/",
        }
    )
    turns = {1: ("allume la lumière", "allume la lumière"), 2: ("il fait beau", "quel temps fait-il")}
    rows = []
    for idx, (live, _voxtral) in turns.items():
        key = f"sessions/s1/turns/{idx:03d}_user_00.wav"
        artifacts.put(key, pcm16_to_wav(bytes(range(256)) * 125, 16000))
        rows += [
            ("turns", {"session_id": "s1", "idx": idx, "user_text": live, "user_started_at": now - 90 + idx}),
            (
                "media",
                {
                    "session_id": "s1",
                    "turn_idx": idx,
                    "ts": now,
                    "modality": "audio_user",
                    "sha256": key,
                    "artifact_key": key,
                    "duration_secs": 1.0,
                },
            ),
        ]
    store.write_batch(rows)
    store.add_annotations(
        [
            {
                "session_id": "s1",
                "target_type": "turn",
                "target_id": f"s1:{idx}",
                "source": "voxtral",
                "kind": "transcript",
                "value": {"text": voxtral},
            }
            for idx, (_, voxtral) in turns.items()
        ]
    )
    app = FastAPI()
    app.include_router(create_review_router(settings, store, artifacts, DREAMER))
    return TestClient(app), store


def test_sessions_and_detail(env):
    client, _ = env
    listing = client.get("/api/review/sessions").json()
    assert listing["total"] == 1
    (summary,) = listing["sessions"]
    assert summary["user_turns"] == 2 and summary["audio_turns"] == 2 and summary["reviewed_turns"] == 0
    assert summary["language"] == "fr-FR"

    detail = client.get("/api/review/sessions/s1").json()
    turn = next(t for t in detail["turns"] if t["idx"] == 2)
    assert turn["transcripts"] == {"live:nemo": "il fait beau", "voxtral": "quel temps fait-il"}
    assert turn["audio"]["user"][0]["key"].endswith("002_user_00.wav")
    assert client.get("/api/review/sessions/nope").status_code == 404


def test_artifact_supports_range(env):
    client, _ = env
    url = "/api/review/artifacts/sessions/s1/turns/001_user_00.wav"
    full = client.get(url)
    assert full.status_code == 200 and full.content[:4] == b"RIFF"
    partial = client.get(url, headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100
    assert client.get("/api/review/artifacts/../../etc/passwd").status_code == 404


def test_asr_queue_orders_by_disagreement_and_reference_rescoring(env):
    client, store = env
    queue = client.get("/api/review/asr/queue", params={"sort": "disagreement"}).json()
    assert queue["open"] == 2 and queue["agreeing"] == 1
    assert [i["turn_idx"] for i in queue["items"]] == [2, 1]  # disagreement first
    assert queue["items"][0]["disagreement"] > 0 and queue["items"][1]["agree"]
    turn1_first = client.get("/api/review/asr/queue", params={"sort": "recent"}).json()
    assert [i["turn_idx"] for i in turn1_first["items"]] == [2, 1]  # turn 2 started later
    page = client.get("/api/review/asr/queue", params={"limit": 1, "offset": 1}).json()
    assert page["matching"] == 2 and [i["turn_idx"] for i in page["items"]] == [1]

    saved = client.post(
        "/api/review/asr/references",
        json={"annotator": "fred", "items": [{"session_id": "s1", "turn_idx": 2, "text": "il fait beau"}]},
    ).json()
    assert saved["saved"] == 1
    live = saved["wer_summary"]["s1"]["live:nemo"]
    # turn 2 now uses the human reference (live correct), turn 1 still the voxtral one (agree).
    assert live["wer"] == 0.0 and live["turns"] == 2
    assert sorted(live["reference_sources"]) == ["human:fred", "voxtral"]

    queue = client.get("/api/review/asr/queue").json()
    assert queue["open"] == 1 and queue["total"] == 2 and [i["turn_idx"] for i in queue["items"]] == [1]
    reviewed = client.get("/api/review/asr/queue", params={"status": "reviewed"}).json()
    assert reviewed["items"][0]["human_reference"]["text"] == "il fait beau"
    # Rescoring replaces derived rows rather than appending.
    client.post(
        "/api/review/asr/references",
        json={"annotator": "fred", "items": [{"session_id": "s1", "turn_idx": 1, "text": "allume la lumière"}]},
    )
    assert len(store.annotations_for("s1", kind="wer_summary")) == 1  # one live summary, not one per save
    assert client.get("/api/review/activities").json()["activities"][0]["open"] == 0


def test_validation_and_dreamer(env):
    client, store = env
    bad = client.post(
        "/api/review/asr/references",
        json={"annotator": "<script>", "items": [{"session_id": "s1", "turn_idx": 1, "text": "x"}]},
    )
    assert bad.status_code == 422
    assert client.post("/api/review/jobs", json={"kind": "nope", "session_ids": ["s1"]}).status_code == 422
    assert client.post("/api/review/jobs", json={"kind": "reasr", "session_ids": ["s1"]}).json() == {"queued": 1}
    status = client.get("/api/review/dreamer").json()
    assert status["live_sessions"] == 0 and status["pending"] == 1


def test_dreamer_controls_and_job_queue(env):
    client, store = env
    assert client.get("/api/review/dreamer").json()["paused"] is False
    assert client.post("/api/review/dreamer/pause", json={"paused": True}).json() == {"paused": True}
    status = client.get("/api/review/dreamer").json()
    assert status["paused"] is True and status["worker"] is None
    assert status["services"] == []  # DREAMER config in this test declares no on-demand service

    store.set_kv("dreamer.status", {"state": "paused", "heartbeat_at": time.time(), "poll_secs": 10})
    worker = client.get("/api/review/dreamer").json()["worker"]
    assert worker["alive"] is True and worker["state"] == "paused"
    store.set_kv("dreamer.status", {"state": "idle", "heartbeat_at": time.time() - 3600, "poll_secs": 10})
    assert client.get("/api/review/dreamer").json()["worker"]["alive"] is False

    assert client.post("/api/review/jobs/enqueue-all", json={"kind": "reasr"}).json() == {"queued": 1}
    assert client.post("/api/review/jobs/enqueue-all", json={"kind": "reasr"}).json() == {"queued": 0}
    page = client.get("/api/review/jobs", params={"status": "pending"}).json()
    assert page["total"] == 1
    job_id = page["jobs"][0]["id"]
    assert client.post(f"/api/review/jobs/{job_id}/cancel").json() == {"cancelled": job_id}
    assert client.post(f"/api/review/jobs/{job_id}/cancel").status_code == 409
    assert client.get("/api/review/jobs", params={"status": "cancelled"}).json()["total"] == 1
    # Retry = requeue from scratch.
    client.post("/api/review/jobs", json={"kind": "reasr", "session_ids": ["s1"]})
    assert client.get("/api/review/jobs", params={"status": "pending"}).json()["total"] == 1


def test_metrics_endpoint(env):
    client, store = env
    store.write_batch(
        [
            ("metrics", {"session_id": "s1", "turn_idx": 1, "ts": 1.0, "name": "user_bot_latency", "value": 0.8}),
            ("metrics", {"session_id": "s1", "turn_idx": 2, "ts": 2.0, "name": "user_bot_latency", "value": 1.2}),
            (
                "metrics",
                {"session_id": "s1", "ts": 1.0, "processor": "NvidiaLLMService#0", "name": "ttfb", "value": 0.3},
            ),
            (
                "metrics",
                {"session_id": "s1", "ts": 1.0, "processor": "NvidiaSTTService#0", "name": "ttfb", "value": 0.1},
            ),
        ]
    )
    data = client.get("/api/review/metrics", params={"group_by": "language,asr.model"}).json()
    assert data["totals"]["sessions"] == 1 and data["totals"]["turns"] == 2
    (variant,) = data["variants"]
    assert variant["key"] == {"language": "fr-FR", "asr.model": "nemo"} and variant["label"] == "fr-FR · nemo"
    assert variant["latency"]["n"] == 2 and variant["latency"]["p50"] == pytest.approx(1.0)
    assert set(variant["ttfb"]) == {"ASR", "LLM"}
    assert data["sessions"][0]["variant"] == 0 and data["sessions"][0]["latency_p50"] == pytest.approx(1.0)
    assert client.get("/api/review/metrics", params={"group_by": "bad field!"}).status_code == 422


def _turn_metric(session_id, idx, kind, response, *, voice=None, **extra):
    return (
        "turn_metrics",
        {
            "session_id": session_id,
            "idx": idx,
            "kind": kind,
            "response_latency": response,
            "voice_latency": response if voice is None else voice,
            "total_secs": response,
            "asr_secs": 0.5,
            "llm_first_secs": response - 0.6,
            "unexplained_secs": 0.1,
            "segments": [{"stage": "asr", "start": 1.0, "end": 1.5}],
            "n_llm_calls": 1,
            "computed_at": 1.0,
            **extra,
        },
    )


def _session(store, session_id, started_at, git_sha):
    store.create_session(
        {
            "id": session_id,
            "example": "multilingual-assistant",
            "started_at": started_at,
            "last_seen_at": started_at + 60,
            "ended_at": started_at + 60,
            "config": {"language": "fr-FR", "asr": {"model": "nemo"}, "git_sha": git_sha},
            "artifact_prefix": f"sessions/{session_id}/",
        }
    )


def test_metrics_by_kind_trend_and_tools(env):
    client, store = env
    now = time.time()
    _session(store, "old", now - 3 * 86400, "aaaaaaaa1111")
    _session(store, "new", now - 3600, "bbbbbbbb2222")
    rows = [_turn_metric("old", idx, "plain", 1.0 + idx / 100) for idx in range(1, 7)]
    rows += [_turn_metric("new", idx, "plain", 2.0 + idx / 100, barge_in=idx == 1) for idx in range(1, 7)]
    rows += [_turn_metric("new", 7, "tool", 1.5, voice=3.0, tool_secs=0.4, gpu_load_mean=40.0, gpu_load_peak=90.0)]
    call = {
        "session_id": "new",
        "turn_idx": 7,
        "trigger": "llm",
        "target": "client",
        "sent_at": now,
        "perceivable": True,
    }
    rows += [
        ("tool_calls", {**call, "call_id": "c1", "name": "move_head", "duration_secs": 0.1, "outcome": "ok"}),
        ("tool_calls", {**call, "call_id": "c2", "name": "move_head", "duration_secs": 5.0, "outcome": "timeout"}),
        ("tool_calls", {**call, "call_id": "c3", "name": "move_head", "duration_secs": 0.3, "outcome": "error"}),
        ("tool_calls", {**call, "call_id": "c4", "name": "move_head", "duration_secs": None, "outcome": "cancelled"}),
        # Zero-valued samples recorded before the recorder dropped them are ignored.
        ("metrics", {"session_id": "new", "ts": 1.0, "processor": "NvidiaLLMService#2", "name": "ttfb", "value": 0.0}),
        ("metrics", {"session_id": "new", "ts": 1.0, "processor": "NvidiaLLMService#2", "name": "ttfb", "value": 0.4}),
    ]
    store.write_batch(rows)

    data = client.get("/api/review/metrics", params={"group_by": "git_sha"}).json()
    assert data["kinds"] == ["plain", "tool"] and "unexplained" in data["stages"]
    totals = data["totals"]["by_kind"]
    assert totals["all"]["turns"] == 13 and totals["plain"]["turns"] == 12 and totals["tool"]["turns"] == 1
    assert totals["tool"]["response"]["p50"] == pytest.approx(1.5) and totals["tool"]["voice"]["p50"] == pytest.approx(
        3.0
    )
    assert totals["tool"]["stages"]["tool"] == {"n": 1, "p10": 0.4, "p50": 0.4, "p90": 0.4}
    assert totals["tool"]["gpu_load_mean"] == 40.0 and totals["tool"]["gpu_load_peak"] == 90.0
    assert totals["plain"]["barge_in_rate"] == pytest.approx(1 / 6)  # only turns with a known flag count

    by_label = {variant["label"]: variant for variant in data["variants"]}
    assert by_label["bbbbbbbb2222"]["by_kind"]["plain"]["response"]["n"] == 6
    assert by_label["bbbbbbbb2222"]["ttfb"]["LLM"] == {"n": 1, "p10": 0.4, "p50": 0.4, "p90": 0.4}

    revisions = [bucket for bucket in data["trend"]["by_revision"] if bucket["key"]]
    assert [bucket["label"] for bucket in revisions] == ["aaaaaaaa", "bbbbbbbb"]  # in order of first appearance
    assert revisions[0]["by_kind"]["plain"]["regression"] is None
    assert revisions[1]["by_kind"]["plain"]["regression"]["previous"] == "aaaaaaaa"
    assert revisions[1]["by_kind"]["tool"]["regression"] is None  # too few turns to judge
    assert revisions[1]["by_kind"]["plain"]["stages"]["asr"]["p50"] == 0.5
    days = data["trend"]["by_day"]
    assert len(days) >= 2 and sum(bucket["by_kind"]["all"]["turns"] for bucket in days) == 13

    (tool,) = data["tools"]
    assert (tool["name"], tool["trigger"], tool["target"], tool["perceivable"]) == ("move_head", "llm", "client", True)
    assert tool["calls"] == 4 and tool["duration"]["n"] == 3
    assert (tool["failure_rate"], tool["timeout_rate"], tool["error_rate"], tool["cancelled_rate"]) == (
        0.75,
        0.25,
        0.25,
        0.25,
    )
    assert {s["id"]: s["response_p50"] for s in data["sessions"]}["old"] == pytest.approx(1.035)


def test_session_detail_has_turn_metrics_tool_calls_and_system_samples(env):
    client, store = env
    session = store.get_session("s1")
    store.write_batch(
        [
            _turn_metric("s1", 1, "tool", 1.5),
            (
                "tool_calls",
                {
                    "session_id": "s1",
                    "call_id": "c1",
                    "turn_idx": 1,
                    "name": "move_head",
                    "trigger": "llm",
                    "target": "client",
                    "sent_at": session["started_at"] + 12,
                    "duration_secs": 0.1,
                    "outcome": "ok",
                    "perceivable": False,
                },
            ),
        ]
    )
    for offset, load in ((-30, 1.0), (10, 55.0), (20, 65.0), (500, 2.0)):
        store.add_system_sample({"ts": session["started_at"] + offset, "live_sessions": 1, "gpu_load": load})
    detail = client.get("/api/review/sessions/s1").json()
    first, second = (next(t for t in detail["turns"] if t["idx"] == idx) for idx in (1, 2))
    assert first["metrics"]["kind"] == "tool" and first["metrics"]["segments"][0]["stage"] == "asr"
    assert [call["name"] for call in first["tool_calls"]] == ["move_head"]
    assert second["metrics"] is None and second["tool_calls"] == []
    assert [sample["gpu_load"] for sample in detail["system_samples"]] == [55.0, 65.0]  # the session window only


def test_report_cli_splits_latency_by_turn_kind(env):
    from monitoring.report import collect, summarize

    _, store = env
    store.write_batch(
        [
            ("metrics", {"session_id": "s1", "turn_idx": 1, "ts": 1.0, "name": "user_bot_latency", "value": 0.8}),
            ("metrics", {"session_id": "s1", "turn_idx": 2, "ts": 2.0, "name": "user_bot_latency", "value": 2.8}),
            _turn_metric("s1", 1, "plain", 0.8),
            _turn_metric("s1", 2, "tool", 2.8),
        ]
    )
    (row,) = summarize(collect(store, since=None, session_ids=None, group_by=["language"]), ["language"])
    assert row["user_bot_p50"] == "1.800s"
    assert (row["turns[plain]"], row["user_bot_p50[plain]"]) == ("1", "0.800s")
    assert (row["turns[tool]"], row["user_bot_p50[tool]"], row["user_bot_p90[tool]"]) == ("1", "2.800s", "2.800s")
    assert not any("[vision]" in key or "response" in key for key in row)
