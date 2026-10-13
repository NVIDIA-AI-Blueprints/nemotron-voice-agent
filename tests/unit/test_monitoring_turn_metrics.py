# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, inspect, select, text

from examples.multilingual.tool_handlers import ClientToolResultBridge, build_client_tool_handler
from monitoring import schema
from monitoring.store import SessionStore
from monitoring.turn_metrics import (
    REPAIR_ANNOTATION,
    STAGES,
    backfill,
    call_outcome,
    derive,
    plan_attribution_repair,
    recompute_session,
    repair_attribution,
)

T0 = 1_000.0


class Trace:
    """Builds the rows ``derive`` reads, with times relative to the session start."""

    def __init__(self, **config):
        self.session = {"id": "s1", "started_at": T0, "ended_at": T0 + 100, "config": config}
        self.turns, self.events, self.metrics, self.llm_calls, self.samples = [], [], [], [], []

    def turn(self, idx, started, released=None, *, speech_end=None, audio=None):
        self.turns.append(
            {
                "session_id": "s1",
                "idx": idx,
                "user_started_at": T0 + started,
                "user_stopped_at": None if released is None else T0 + released,
                "user_speech_stopped_at": None if speech_end is None else T0 + speech_end,
                "bot_speech_started_at": None if audio is None else T0 + audio,
            }
        )
        return self

    def event(self, ts, kind, **data):
        self.events.append({"id": len(self.events) + 1, "ts": T0 + ts, "kind": kind, "data": data or None})
        self.events.sort(key=lambda e: e["ts"])
        return self

    def metric(self, ts, name, value, processor=None):
        self.metrics.append({"ts": T0 + ts, "name": name, "value": value, "processor": processor})
        self.metrics.sort(key=lambda m: m["ts"])
        return self

    def llm(self, started, ended, *, ttfb=None, tools=(), images=0, tokens=(100, 10)):
        self.llm_calls.append(
            {
                "id": len(self.llm_calls) + 1,
                "started_at": T0 + started,
                "ended_at": T0 + ended,
                "ttfb": ttfb,
                "function_calls": [{"name": name, "tool_call_id": call_id} for name, call_id in tools],
                "n_images": images,
                "prompt_tokens": tokens[0] if tokens else None,
                "completion_tokens": tokens[1] if tokens else None,
                "interrupted": False,
            }
        )
        return self

    def tool(self, call_id, name, sent, done=None, result=None):
        self.event(sent, "function_call_in_progress", name=name, tool_call_id=call_id)
        if done is not None:
            self.event(done, "function_result", name=name, tool_call_id=call_id, result=result or {"status": "ok"})
        return self

    def speech(self, *, asr_final, text_aggregation, tts_ttfb):
        """Add the ASR final, the first text aggregation and the TTS TTFB as ``(end, duration)`` pairs."""
        self.event(asr_final, "asr_final", text="hi")
        self.metric(text_aggregation[0], "text_aggregation", text_aggregation[1], "NvidiaTTSService#3")
        self.metric(tts_ttfb[0], "ttfb", tts_ttfb[1], "NvidiaTTSService#3")
        return self

    def build(self):
        return {
            "session": self.session,
            "turns": self.turns,
            "events": self.events,
            "metrics": self.metrics,
            "llm_calls": self.llm_calls,
            "samples": self.samples,
        }


def _plain() -> Trace:
    # Speech ends at 10.0; turn released 10.6; LLM 10.6 -> first token 11.0; sentence 11.2; audio 11.5.
    return (
        Trace()
        .turn(1, 8.0, 10.6, speech_end=10.0, audio=11.5)
        .speech(asr_final=10.5, text_aggregation=(11.2, 0.2), tts_ttfb=(11.45, 0.25))
        .llm(10.6, 12.0, ttfb=0.4)
    )


def _stage_sum(row) -> float:
    return sum(row[f"{stage}_secs"] or 0.0 for stage in STAGES) + row["unexplained_secs"]


def test_plain_turn_stages_add_up_to_the_total():
    (row,), calls, updates = derive(_plain().build())
    assert row["kind"] == "plain" and calls == []
    assert row["voice_latency"] == pytest.approx(1.5) and row["response_latency"] == pytest.approx(1.5)
    assert row["response_via"] == "audio" and row["total_secs"] == pytest.approx(1.5)
    assert row["asr_secs"] == pytest.approx(0.5) and row["turn_detection_secs"] == pytest.approx(0.1)
    assert row["llm_first_secs"] == pytest.approx(0.4)
    assert row["text_aggregation_secs"] == pytest.approx(0.2) and row["tts_secs"] == pytest.approx(0.25)
    assert row["tool_secs"] is None and row["llm_later_secs"] is None and row["intent_match_secs"] is None
    assert row["unexplained_secs"] == pytest.approx(0.05)
    assert _stage_sum(row) == pytest.approx(row["total_secs"])
    assert [s["stage"] for s in row["segments"]] == ["asr", "turn_detection", "llm_first", "text_aggregation", "tts"]
    assert row["n_llm_calls"] == 1 and row["prompt_tokens"] == 100 and row["completion_tokens"] == 10
    assert updates == {1: {}}  # nothing to patch: the recorder already stored the speech times


def _tool_turn(**config) -> Trace:
    # LLM 1: 10.6 -> tool sent 11.4; result 11.9; LLM 2: 11.9 -> first token 12.1; audio 12.6.
    return (
        Trace(**config)
        .turn(1, 8.0, 10.6, speech_end=10.0, audio=12.6)
        .speech(asr_final=10.5, text_aggregation=(12.3, 0.2), tts_ttfb=(12.55, 0.25))
        .llm(10.6, 11.4, ttfb=0.3, tools=[("move_head", "c1")])
        .tool("c1", "move_head", 11.4, 11.9)
        .llm(11.9, 13.0, ttfb=0.2, tokens=(120, 20))
    )


def test_tool_turn_has_tool_and_later_llm_stages():
    (row,), (call,), _ = derive(_tool_turn().build())
    assert row["kind"] == "tool" and row["n_llm_calls"] == 2 and row["n_tool_calls"] == 1
    assert row["llm_first_secs"] == pytest.approx(0.8)  # the whole first call, up to the tool call
    assert row["tool_secs"] == pytest.approx(0.5) and row["llm_later_secs"] == pytest.approx(0.2)
    assert row["prompt_tokens"] == 220 and row["completion_tokens"] == 30
    assert row["voice_latency"] == pytest.approx(2.6) and row["response_latency"] == pytest.approx(2.6)
    assert _stage_sum(row) == pytest.approx(row["total_secs"])
    assert call == {
        "session_id": "s1",
        "call_id": "c1",
        "turn_idx": 1,
        "name": "move_head",
        "trigger": "llm",
        "target": "client",
        "sent_at": T0 + 11.4,
        "duration_secs": pytest.approx(0.5),
        "outcome": "ok",
        "perceivable": False,
    }


def test_perceivable_tool_call_is_the_first_response():
    (row,), (call,), _ = derive(_tool_turn(client_tools_perceivable=["move_head"]).build())
    assert call["perceivable"] is True
    assert row["response_latency"] == pytest.approx(1.4) and row["response_via"] == "tool"
    assert row["voice_latency"] == pytest.approx(2.6)  # still measured, and still what the stages split


def test_perceivable_tool_call_without_any_bot_audio():
    trace = (
        Trace(client_tools_perceivable=["dance"])
        .turn(1, 8.0, 10.6, speech_end=10.0)
        .event(10.5, "asr_final", text="danse")
        .llm(10.6, 11.4, ttfb=0.3, tools=[("dance", "c1")])
        .tool("c1", "dance", 11.4, 11.5)
    )
    (row,), _, _ = derive(trace.build())
    assert row["voice_latency"] is None and row["response_latency"] == pytest.approx(1.4)
    assert row["llm_first_secs"] == pytest.approx(0.8) and row["total_secs"] == pytest.approx(1.4)


def test_spoken_call_that_also_calls_a_tool_uses_its_first_token():
    # "With pleasure!" is spoken from the same call that then requests the tool.
    trace = (
        Trace()
        .turn(1, 8.0, 10.6, speech_end=10.0, audio=11.5)
        .speech(asr_final=10.5, text_aggregation=(11.2, 0.2), tts_ttfb=(11.45, 0.25))
        .llm(10.6, 11.6, ttfb=0.4, tools=[("dance", "c1")])
        .tool("c1", "dance", 11.6, 11.7)
        .llm(11.7, 12.5, ttfb=0.2)
    )
    (row,), _, _ = derive(trace.build())
    assert row["kind"] == "tool" and row["llm_first_secs"] == pytest.approx(0.4)
    assert row["tool_secs"] is None and row["tts_secs"] == pytest.approx(0.25)


def test_turn_without_answer_has_no_latency():
    (row,), _, _ = derive(Trace().turn(1, 8.0, 10.6, speech_end=10.0).llm(10.6, 10.9).build())
    assert row["voice_latency"] is None and row["response_latency"] is None and row["total_secs"] is None
    assert row["segments"] == [] and row["kind"] == "plain"


def test_older_recordings_fall_back_to_events_and_patch_the_turn():
    trace = (
        Trace()
        .turn(1, 8.0, 10.6)
        .event(10.5, "asr_final", text="hi")
        .llm(10.6, 12.0, ttfb=0.4)
        .event(11.5, "bot_started_speaking")
        .event(11.5, "latency_breakdown", user_turn_start_time=T0 + 10.0, user_turn_secs=0.6)
    )
    (row,), _, updates = derive(trace.build())
    assert row["voice_latency"] == pytest.approx(1.5)
    assert updates[1]["user_speech_stopped_at"] == T0 + 10.0 and updates[1]["bot_speech_started_at"] == T0 + 11.5


def test_barge_in_is_user_speech_starting_while_the_bot_speaks():
    trace = (
        Trace()
        .turn(1, 5.0)
        .turn(2, 20.0)
        .turn(3, 40.0)
        .event(4.99, "user_started_speaking")
        .event(8.0, "bot_started_speaking")
        .event(19.99, "user_started_speaking")  # the bot is still speaking: barge-in on turn 2
        .event(19.99, "interruption")
        .event(20.0, "bot_stopped_speaking")
        .event(25.0, "bot_started_speaking")
        .event(30.0, "bot_stopped_speaking")
        .event(39.99, "user_started_speaking")  # the bot finished 10 s ago
        .event(39.99, "interruption")  # every user turn interrupts: an interruption is not a barge-in
    )
    rows, _, updates = derive(trace.build())
    assert [row["barge_in"] for row in rows] == [False, True, False]
    assert [updates[idx]["barge_in"] for idx in (1, 2, 3)] == [False, True, False]


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        ({"status": "ok"}, "ok"),
        ("plain text", "ok"),
        ({"error": "robot unreachable"}, "error"),
        ({"status": "error"}, "error"),
        ({"error": "Timed out waiting for the client to answer this request."}, "timeout"),
        (None, "timeout"),
    ],
)
def test_call_outcome(result, outcome):
    assert call_outcome(result) == outcome


def test_client_tool_timeout_handler_result_is_classified_as_timeout():
    results = []

    async def scenario():
        async def result_callback(result):
            results.append(result)

        params = SimpleNamespace(tool_call_id="c1", function_name="nap", result_callback=result_callback)
        await build_client_tool_handler(0.01, ClientToolResultBridge())(params)

    asyncio.run(scenario())
    assert [call_outcome(result) for result in results] == ["timeout"]


def test_tool_call_rows_and_outcomes():
    trace = (
        Trace()
        .turn(1, 5.0)
        .turn(2, 30.0)
        .tool("ok", "move_head", 6.0, 6.1)
        .tool("bad", "dance", 7.0, 7.3, {"error": "no motor"})
        .tool("slow", "go_to_sleep", 8.0, 13.0, {"error": "Timed out waiting for the client to answer this request."})
        .event(13.6, "function_result", name="go_to_sleep", tool_call_id="slow", result={"status": "sleeping"})
        .tool("cut", "search", 14.0)
        .event(14.5, "function_call_cancelled", name="search", tool_call_id="cut")
        .tool("lost", "search", 31.0)
        # Recordings made before ``function_call_in_progress`` only have the batch event.
        .event(32.0, "function_calls", calls=[{"name": "old", "tool_call_id": "legacy"}])
        .event(32.4, "function_result", name="old", tool_call_id="legacy", result={})
    )
    _, calls, _ = derive(trace.build())
    by_id = {call["call_id"]: call for call in calls}
    assert {k: v["outcome"] for k, v in by_id.items()} == {
        "ok": "ok",
        "bad": "error",
        "slow": "timeout",  # the first result wins; the late client answer is ignored
        "cut": "cancelled",
        "lost": "cancelled",  # never answered
        "legacy": "ok",
    }
    assert by_id["slow"]["duration_secs"] == pytest.approx(5.0)
    assert by_id["cut"]["duration_secs"] == pytest.approx(0.5) and by_id["lost"]["duration_secs"] is None
    assert by_id["legacy"]["duration_secs"] == pytest.approx(0.4)
    assert {call["turn_idx"] for call in calls if call["call_id"] in ("lost", "legacy")} == {2}
    assert all(call["trigger"] == "llm" and call["target"] == "client" for call in calls)


def _intent_events(trace: Trace, at: float, *, target: str, **data) -> Trace:
    """What the intent engine reports through the recorder hooks (synthetic)."""
    trace.metric(at, "intent_match_secs", 0.01, "IntentEngine")
    if target:
        trace.metric(at + 0.3, "intent_target_secs", 0.3, "IntentEngine")
        return trace.event(at + 0.3, "intent_engine", target=target, **data)
    return trace.event(at, "intent_engine", **data)


def test_intent_handled_turn_with_home_assistant():
    trace = (
        Trace(intent_engine={"enabled": True, "timeout_secs": 3.0})
        .turn(1, 8.0, 10.6, speech_end=10.0, audio=11.3)
        .speech(asr_final=10.5, text_aggregation=(11.0, 0.05), tts_ttfb=(11.25, 0.25))
    )
    _intent_events(
        trace, 10.61, target="home_assistant", handled_by="intent", intent="HassTurnOn", response_type="action_done"
    )
    (row,), (call,), _ = derive(trace.build())
    assert row["kind"] == "intent" and row["n_llm_calls"] == 0
    assert row["intent_match_secs"] == pytest.approx(0.01) and row["tool_secs"] == pytest.approx(0.3)
    assert row["llm_first_secs"] is None and _stage_sum(row) == pytest.approx(row["total_secs"])
    assert (call["name"], call["trigger"], call["target"]) == ("HassTurnOn", "intent", "home_assistant")
    assert call["outcome"] == "ok" and call["duration_secs"] == pytest.approx(0.3)
    assert call["sent_at"] == pytest.approx(T0 + 10.61) and call["perceivable"] is False
    assert row["response_latency"] == row["voice_latency"]  # Home Assistant actions are never perceivable


def test_intent_client_tool_call_uses_the_function_frames():
    trace = Trace(client_tools_perceivable=["move_head"]).turn(1, 8.0, 10.6, speech_end=10.0, audio=11.3)
    trace.metric(10.61, "intent_match_secs", 0.01, "IntentEngine")
    trace.tool("intent-abc123", "move_head", 10.62, 10.9)
    trace.metric(10.91, "intent_target_secs", 0.3, "IntentEngine")
    trace.event(10.91, "intent_engine", handled_by="intent", intent="LookLeft", target="client_tool", response_type="x")
    (row,), (call,), _ = derive(trace.build())
    assert row["kind"] == "intent" and (call["trigger"], call["target"]) == ("intent", "client")
    assert row["response_via"] == "tool" and row["response_latency"] == pytest.approx(0.62)
    assert row["tool_secs"] == pytest.approx(0.3)  # the target time, not counted twice with the call


def test_intent_engine_fallbacks_stay_llm_turns():
    missed = _plain()
    _intent_events(missed, 10.61, target="", handled_by="llm", reason="no_match")
    (row,), calls, _ = derive(missed.build())
    assert row["kind"] == "plain" and calls == [] and row["intent_match_secs"] == pytest.approx(0.01)

    failed = Trace(intent_engine={"enabled": True, "timeout_secs": 0.3}).turn(1, 8.0, 10.6, speech_end=10.0, audio=12.0)
    _intent_events(
        failed, 10.61, target="home_assistant", handled_by="llm", intent="HassTurnOn", reason="target_unavailable"
    )
    failed.llm(10.92, 12.5, ttfb=0.4)
    (row,), (call,), _ = derive(failed.build())
    assert row["kind"] == "tool" and call["outcome"] == "timeout" and call["trigger"] == "intent"

    dry = _plain()
    _intent_events(dry, 10.6, target="home_assistant", handled_by="llm", intent="HassTurnOn", reason="dry_run")
    assert derive(dry.build())[1] == []  # nothing was sent


def test_turn_kind_precedence():
    vision = _tool_turn()
    vision.llm_calls[0]["n_images"] = 1
    assert derive(vision.build())[0][0]["kind"] == "vision"  # image input wins over tool calls
    assert derive(vision.build())[0][0]["n_images"] == 1
    vision.event(10.61, "intent_engine", handled_by="intent", intent="X", target="client_tool")
    assert derive(vision.build())[0][0]["kind"] == "intent"


def test_gpu_load_is_averaged_over_the_turn():
    trace = _plain().turn(2, 20.0).event(15.0, "bot_stopped_speaking")
    trace.samples = [
        {"ts": T0 + 7.0, "gpu_load": 99.0},  # before the turn
        {"ts": T0 + 9.0, "gpu_load": 20.0},
        {"ts": T0 + 11.0, "gpu_load": 80.0},
        {"ts": T0 + 12.0, "gpu_load": None},
        {"ts": T0 + 18.0, "gpu_load": 1.0},  # after the bot stopped
    ]
    row = derive(trace.build())[0][0]
    assert row["gpu_load_mean"] == pytest.approx(50.0) and row["gpu_load_peak"] == 80.0
    assert derive(_plain().build())[0][0]["gpu_load_mean"] is None


def test_derive_only_selected_turns():
    trace = _tool_turn().turn(2, 30.0).tool("c2", "dance", 31.0, 31.1)
    rows, calls, updates = derive(trace.build(), only={2})
    assert [row["idx"] for row in rows] == [2] and [call["call_id"] for call in calls] == ["c2"] and set(updates) == {2}


# ------------------------------------------------------------- store round trip
def _store_with(trace: Trace, tmp_path) -> SessionStore:
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    store.create_session({**trace.session, "example": "e", "last_seen_at": T0, "artifact_prefix": "p/"})
    rows = [("turns", turn) for turn in trace.turns]
    rows += [
        ("events", {"session_id": "s1", **{k: v for k, v in event.items() if k != "id"}}) for event in trace.events
    ]
    rows += [("metrics", {"session_id": "s1", **metric}) for metric in trace.metrics]
    rows += [("llm_calls", {"session_id": "s1", **{k: v for k, v in c.items() if k != "id"}}) for c in trace.llm_calls]
    store.write_batch(rows)
    for sample in trace.samples:
        store.add_system_sample({"live_sessions": 1, **sample})
    return store


def test_recompute_session_is_idempotent_and_patches_turns(tmp_path):
    trace = _tool_turn(client_tools_perceivable=["move_head"]).turn(2, 30.0)
    trace.event(7.99, "user_started_speaking").event(29.99, "user_started_speaking")
    store = _store_with(trace, tmp_path)
    assert recompute_session(store, "s1") == 2
    first = store.rows("turn_metrics", "s1")
    assert backfill(store) == {"s1": 2}
    second = store.rows("turn_metrics", "s1")
    assert len(second) == 2 and len(store.rows("tool_calls", "s1")) == 1

    def stable(rows):
        return [{k: v for k, v in row.items() if k != "computed_at"} for row in rows]

    assert stable(first) == stable(second)
    assert first[0]["response_latency"] == pytest.approx(1.4) and first[0]["response_via"] == "tool"
    assert [turn["barge_in"] for turn in store.rows("turns", "s1")] == [False, False]
    assert recompute_session(store, "missing") == 0

    # The live path recomputes one finished turn and leaves the others alone.
    assert recompute_session(store, "s1", only_turn=1) == 1
    assert len(store.rows("turn_metrics", "s1")) == 2 and len(store.rows("tool_calls", "s1")) == 1


def test_schema_migration_from_version_3(tmp_path):
    store = _store_with(_plain(), tmp_path)
    new_columns = ("user_speech_stopped_at", "bot_speech_started_at", "barge_in")
    with store.engine.begin() as conn:  # put the database back into its version 3 shape
        for column in new_columns:
            conn.execute(text(f"ALTER TABLE turns DROP COLUMN {column}"))
        for table in ("turn_metrics", "tool_calls", "system_samples"):
            conn.execute(text(f"DROP TABLE {table}"))
        conn.execute(schema.schema_version.delete().where(schema.schema_version.c.version > 3))
        conn.execute(schema.schema_version.insert().values(version=3))

    migrated = SessionStore(str(store.engine.url))
    migrated.create_schema()
    migrated.create_schema()  # idempotent
    assert set(new_columns) <= {column["name"] for column in inspect(migrated.engine).get_columns("turns")}
    with migrated.engine.connect() as conn:
        assert conn.execute(select(func.max(schema.schema_version.c.version))).scalar() == schema.SCHEMA_VERSION == 4
    (turn,) = migrated.rows("turns", "s1")
    assert turn["barge_in"] is None and turn["user_stopped_at"] == T0 + 10.6  # existing data untouched
    # The speech times were dropped with the columns, so this turn cannot be timed.
    assert backfill(migrated) == {"s1": 1} and migrated.rows("turn_metrics", "s1")[0]["voice_latency"] is None


# ----------------------------------------------------------- attribution repair
def _old_session(tmp_path, turns, outputs, **config) -> SessionStore:
    """A session as an old recorder stored it: ``turns`` are ``(idx, bot_text, interrupted, bot_stopped)``."""
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    store.create_session(
        {"id": "s1", "example": "e", "started_at": T0, "ended_at": T0 + 100, "config": config, "artifact_prefix": "p/"}
    )
    rows = []
    for idx, bot_text, interrupted, bot_stopped in turns:
        start = T0 + 10 * idx
        rows.append(
            (
                "turns",
                {
                    "session_id": "s1",
                    "idx": idx,
                    "user_started_at": start,
                    "bot_started_at": start + 2,
                    "bot_stopped_at": None if bot_stopped is None else start + bot_stopped,
                    "bot_text": bot_text,
                    "interrupted": interrupted,
                },
            )
        )
    for turn_idx, offset, output in outputs:
        rows.append(
            (
                "llm_calls",
                {
                    "session_id": "s1",
                    "turn_idx": turn_idx,
                    "started_at": T0 + 10 * turn_idx + offset,
                    "output_text": output,
                },
            )
        )
    store.write_batch(rows)
    return store


OLD_TURNS = [
    (1, "", False, 3),  # the tool call's empty response; the spoken answer landed on turn 2
    (2, "First answer, cut by the user.", True, 0.01),  # only turn 1's text; its own went to turn 3
    (3, "Second answer.\n\nAlso cut. Third answer, played to the end.", True, 8),
    (4, "Fourth answer.", True, 5),  # last turn, session cancelled mid-response: a genuine flag
]
OLD_OUTPUTS = [
    (1, 2, ""),
    (1, 3, "First answer, cut by the user. And more that was never spoken."),
    (2, 2, "Second answer.  Also cut."),
    (3, 2, "Third answer, played to the end."),
    (4, 2, "Fourth answer."),
]


def test_repair_moves_interrupted_responses_back_to_their_turn(tmp_path):
    store = _old_session(tmp_path, OLD_TURNS, OLD_OUTPUTS)
    before = store.rows("turns", "s1")

    plan = repair_attribution(store, "s1", dry_run=True)
    assert [(m["from_turn"], m["to_turn"]) for m in plan["moves"]] == [(2, 1), (3, 2)]
    assert store.rows("turns", "s1") == before and store.annotations_for("s1", kind=REPAIR_ANNOTATION) == []

    report = repair_attribution(store, "s1")
    one, two, three, four = store.rows("turns", "s1")
    assert (one["bot_text"], one["interrupted"]) == ("First answer, cut by the user.", True)
    assert (two["bot_text"], two["interrupted"]) == ("Second answer.\n\nAlso cut.", True)
    assert (three["bot_text"], three["interrupted"]) == ("Third answer, played to the end.", False)
    assert (four["bot_text"], four["interrupted"]) == ("Fourth answer.", True)  # untouched
    # A cut response stopped when the next user turn started.
    assert one["bot_stopped_at"] == two["user_started_at"] and two["bot_stopped_at"] == three["user_started_at"]
    assert three["bot_stopped_at"] == T0 + 38 and one["user_started_at"] == T0 + 10
    assert [turn["idx"] for turn in report["turns"]] == [1, 2, 3]

    # Safe to re-run: the audit annotation marks the session as done.
    (annotation,) = store.annotations_for("s1", kind=REPAIR_ANNOTATION)
    assert annotation["value"]["moves"] == report["moves"]
    after = store.rows("turns", "s1")
    assert repair_attribution(store, "s1") is None and store.rows("turns", "s1") == after
    assert repair_attribution(store, "missing") is None


def test_repair_skips_sessions_from_a_fixed_recorder_and_unrelated_text(tmp_path):
    store = _old_session(tmp_path, OLD_TURNS, OLD_OUTPUTS, recorder_version=2)
    assert repair_attribution(store, "s1") is None  # new recordings are already right

    turns = [
        {"idx": 1, "bot_text": "Hello there, how are you?"},
        {"idx": 2, "bot_text": "Hello there, how are you? Fine."},
    ]
    calls = [
        {"turn_idx": 1, "started_at": 1.0, "output_text": "Hello there, how are you?"},
        {"turn_idx": 2, "started_at": 2.0, "output_text": "Hello there, how are you? Fine."},
    ]
    assert plan_attribution_repair(turns, calls) == []  # turn 1 already has it: the bot repeated itself
    turns = [{"idx": 1, "bot_text": ""}, {"idx": 2, "bot_text": "Something else entirely."}]
    assert plan_attribution_repair(turns, calls[:1]) == []
    turns[1]["bot_text"] = "Hello there, how are you? Unrelated tail nobody generated."
    assert plan_attribution_repair(turns, calls) == []  # the rest is not turn 2's own answer: left alone
    assert plan_attribution_repair([{"idx": 2, "bot_text": "Hello there"}], calls) == []  # no previous turn


@pytest.mark.skipif(not os.getenv("MONITORING_TEST_DB"), reason="set MONITORING_TEST_DB to a recorded database")
def test_repair_on_a_copy_of_a_recorded_database(tmp_path):
    copy = tmp_path / "voice_agent.db"
    shutil.copy(Path(os.environ["MONITORING_TEST_DB"]), copy)
    store = SessionStore(f"sqlite:///{copy}")
    store.create_schema()
    reports = [r for sid in store.all_session_ids() if (r := repair_attribution(store, sid)) is not None]
    moved = [(r["session_id"], m["to_turn"]) for r in reports for m in r["moves"]]
    with store.engine.connect() as conn:
        turns = [dict(r) for r in conn.execute(select(schema.turns)).mappings()]
        outputs = {}
        for call in conn.execute(select(schema.llm_calls).order_by(schema.llm_calls.c.started_at)).mappings():
            outputs.setdefault((call["session_id"], call["turn_idx"]), []).append(call["output_text"] or "")
    by_key = {(t["session_id"], t["idx"]): t for t in turns}
    for key in moved:
        # The moved text is what that turn's own LLM call generated, and the turn is marked as cut.
        moved_text = "".join(by_key[key]["bot_text"].split())
        assert moved_text and "".join(" ".join(outputs[key]).split()).endswith(moved_text)
        assert by_key[key]["interrupted"]
    assert not [
        t for t in turns if t["bot_stopped_at"] and t["bot_started_at"] and t["bot_stopped_at"] < t["bot_started_at"]
    ]
    assert all(repair_attribution(store, sid) is None for sid in store.all_session_ids())  # idempotent
    assert backfill(store)


@pytest.mark.skipif(not os.getenv("MONITORING_TEST_DB"), reason="set MONITORING_TEST_DB to a recorded database")
def test_backfill_on_a_copy_of_a_recorded_database(tmp_path):
    copy = tmp_path / "voice_agent.db"
    shutil.copy(Path(os.environ["MONITORING_TEST_DB"]), copy)
    store = SessionStore(f"sqlite:///{copy}")
    store.create_schema()
    first = backfill(store)
    assert backfill(store) == first and sum(first.values()) > 0
    with store.engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(select(schema.turn_metrics)).mappings()]
        latency = {
            (m.session_id, m.turn_idx): m.value
            for m in conn.execute(select(schema.metrics).where(schema.metrics.c.name == "user_bot_latency"))
        }
    timed = [row for row in rows if row["voice_latency"] is not None]
    assert len(timed) == len(latency)
    for row in timed:
        # Same anchors as pipecat's observer, and the stages never exceed the total.
        assert row["voice_latency"] == pytest.approx(latency[(row["session_id"], row["idx"])], abs=0.01)
        assert _stage_sum(row) == pytest.approx(row["total_secs"]) and row["unexplained_secs"] >= 0
        assert row["response_latency"] == row["voice_latency"]  # old sessions have no perceivable flags
