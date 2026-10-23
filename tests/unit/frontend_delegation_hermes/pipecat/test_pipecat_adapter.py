# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""C7, C12, C14: the τ² contract on the Pipecat adapter (stub speech, fake backend, real Pipecat pipeline)."""

from __future__ import annotations

import base64
import json
import unittest

from _pipecat_fakes import (
    Client,
    failing_tts_app,
    pcmu_silence,
    pcmu_speech,
    stub_app,
    stub_runtime,
    tau2_session_update,
)
from fastapi.testclient import TestClient


def _output(call_id: str, output: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


class PipecatContractTests(unittest.TestCase):
    def test_handshake_and_health(self) -> None:
        app = stub_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime?model=pine-pipecat") as ws:
                session = Client(ws)
                created = session.wait_for("session.created")
                self.assertEqual(session.events[0]["type"], "session.created")
                self.assertTrue(created["session"]["id"].startswith("sess_"))
                session.send(tau2_session_update("mock"))
                updated = session.wait_for("session.updated")
                self.assertEqual(updated["session"]["audio"]["output"]["format"]["type"], "audio/pcmu")
                self.assertEqual(session.of_type("error"), [])
            health = app.state.runtime.health()
            self.assertEqual((health["status"], health["adapter"], health["profile"]), ("ok", "pipecat", "eval"))
            self.assertFalse(health["pipecat"]["smart_turn"])

    def test_eval_settings_end_turns_on_audio_time_only(self) -> None:
        health = stub_runtime(settings="adapters/pipecat.eval.yaml").health()
        self.assertEqual(health["pipecat"]["vad"], {"stop_secs": 0.8, "audio_idle_timeout": 0.0})
        self.assertEqual(health["pipecat"]["turn"], {"user_speech_timeout": 0.0})

    def test_demo_settings_keep_pipecat_defaults(self) -> None:
        health = stub_runtime(settings="adapters/pipecat.yaml").health()
        self.assertEqual((health["pipecat"]["vad"], health["pipecat"]["turn"]), ({}, {}))

    def test_turn_tool_round_trip_and_transcript_before_audio(self) -> None:
        with TestClient(stub_app()) as client, client.websocket_connect("/v1/realtime?model=pine-pipecat") as ws:
            session = Client(ws)
            session.wait_for("session.created")
            session.send(tau2_session_update("mock"))
            session.wait_for("session.updated")
            session.audio(pcmu_silence(300) + pcmu_speech(600) + pcmu_silence(1200))
            started = session.wait_for("input_audio_buffer.speech_started")
            # Onset on the input-audio clock: speech starts at 300 ms; Pipecat's VAD confirms it after
            # start_secs plus its volume smoothing, so the reported onset is measured late, not exact.
            self.assertGreaterEqual(started["audio_start_ms"], 280)
            self.assertLessEqual(started["audio_start_ms"], 600)
            transcript = session.wait_for("conversation.item.input_audio_transcription.completed")
            self.assertEqual(transcript["transcript"], "utterance 1")
            call = session.wait_for("response.function_call_arguments.done")
            tools = [tool["name"] for tool in tau2_session_update("mock")["session"]["tools"]]
            self.assertEqual(call["name"], tools[0])
            session.send(_output(call["call_id"], json.dumps({"ok": True})))
            session.send({"type": "response.create"})
            session.audio(pcmu_silence(1500))
            session.settle(2.0)
            types = session.types()
            self.assertNotIn("error", types)
            self.assertIn("response.output_audio.delta", types)
            # Every announced response finishes exactly once.
            created = [e["response"]["id"] for e in session.of_type("response.created")]
            done = [e["response"]["id"] for e in session.of_type("response.done")]
            self.assertEqual(sorted(created), sorted(done))
            # Transcript text of an item precedes its audio.
            seen_text: set[str] = set()
            for event in session.events:
                if event["type"] == "response.output_audio_transcript.delta":
                    seen_text.add(event["item_id"])
                elif event["type"] == "response.output_audio.delta":
                    self.assertIn(event["item_id"], seen_text, "audio before its transcript")
            # About 100 ms of 8 kHz μ-law per delta (the streaming resampler may carry a few samples over).
            sizes = [len(base64.b64decode(e["delta"])) for e in session.of_type("response.output_audio.delta")]
            self.assertGreater(min(sizes), 0)
            self.assertLessEqual(max(sizes), 1200)

    def test_unknown_call_id_and_manual_commit_are_errors(self) -> None:
        with TestClient(stub_app()) as client, client.websocket_connect("/v1/realtime") as ws:
            session = Client(ws)
            session.wait_for("session.created")
            session.send(tau2_session_update("mock"))
            session.wait_for("session.updated")
            session.send(_output("call_unknown", "x"))
            error = session.wait_for("error")
            self.assertEqual(error["error"]["param"], "item.call_id")
            session.send({"type": "input_audio_buffer.commit"})
            commit_error = session.wait_for("error", 2)
            self.assertEqual(commit_error["error"]["code"], "unsupported_event")

    def test_barge_in_keeps_every_response_lifecycle_and_tool_call(self) -> None:
        with TestClient(stub_app()) as client, client.websocket_connect("/v1/realtime") as ws:
            session = Client(ws)
            session.wait_for("session.created")
            session.send(tau2_session_update("mock"))
            session.wait_for("session.updated")
            session.audio(pcmu_silence(300) + pcmu_speech(600) + pcmu_silence(1000))
            session.wait_for("response.output_audio.delta")
            session.audio(pcmu_speech(600) + pcmu_silence(1500))  # talk over the agent
            session.settle(3.0)
            created = [e["response"]["id"] for e in session.of_type("response.created")]
            done = [e["response"]["id"] for e in session.of_type("response.done")]
            self.assertEqual(sorted(created), sorted(done), "a response lost its response.done")
            calls = session.of_type("response.function_call_arguments.done")
            items = [e for e in session.of_type("response.output_item.done") if e["item"]["type"] == "function_call"]
            self.assertEqual(len(calls), len(items))
            self.assertGreaterEqual(len(session.of_type("input_audio_buffer.speech_started")), 2)

    def test_tts_failure_still_finishes_the_response(self) -> None:
        with TestClient(failing_tts_app()) as client, client.websocket_connect("/v1/realtime") as ws:
            session = Client(ws)
            session.wait_for("session.created")
            session.send(tau2_session_update("mock"))
            session.wait_for("session.updated")
            session.audio(pcmu_silence(300) + pcmu_speech(600) + pcmu_silence(1000))
            session.wait_for("response.output_audio_transcript.delta", timeout=20.0)
            session.settle(6.0)
            created = [e["response"]["id"] for e in session.of_type("response.created")]
            done = [e["response"]["id"] for e in session.of_type("response.done")]
            self.assertTrue(created)
            self.assertEqual(sorted(created), sorted(done), "a failed synthesis left a response open")
            self.assertEqual(session.of_type("response.output_audio.delta"), [])

    def test_capacity_refusal_is_server_busy(self) -> None:
        with TestClient(stub_app(max_sessions=1)) as client, client.websocket_connect("/v1/realtime") as first:
            Client(first).wait_for("session.created")
            with client.websocket_connect("/v1/realtime") as second:
                refused = json.loads(second.receive_text())
                self.assertEqual(refused["error"]["code"], "server_busy")


if __name__ == "__main__":
    unittest.main()
