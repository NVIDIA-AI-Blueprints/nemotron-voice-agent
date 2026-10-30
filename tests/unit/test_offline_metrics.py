# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify extraction distinguishes RPC measurements from source presentation."""

# ruff: noqa: D101,D102,D103,D107

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools/offline-eval/extract_metrics.py"
_SPEC = importlib.util.spec_from_file_location("offline_metrics", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
metrics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = metrics
_SPEC.loader.exec_module(metrics)


def recorded_case(directory, *, instrumented=False, playback=True):
    provider = {"llm_first_text_ns": 30_000_000, "llm_complete_ns": 90_000_000, "first_audio_ready_ns": 100_000_000}
    if instrumented:
        provider.update(
            tts_first_input_ns=60_000_000,
            tts_first_audio_latency_ns=40_000_000,
            tts_stream_elapsed_ns=140_000_000,
            llm_provider_metrics={
                "rounds": [
                    {
                        "round_index": 0,
                        "first_text_latency_ns": 5_000_000,
                        "first_response_latency_ns": 2_000_000,
                        "stream_elapsed_ns": 60_000_000,
                    }
                ]
            },
        )
    saved_metrics = {
        "recognition": {"asr_observed_wall_ns": 300_000_000},
        "responses": [
            {
                "response_id": "r1",
                "turn_ids": ["t1", "t2"],
                "request_sample": 4800,
                "interrupted": not playback,
                "compute_ns": 200_000_000,
                "generated_samples": 4000,
                "committed_samples": 4000 if playback else 0,
                "provider": provider,
            }
        ],
    }
    if instrumented:
        saved_metrics["recognition"].update(asr_first_response_ns=10_000_000, asr_first_transcript_ns=20_000_000)
        saved_metrics["server"] = {
            "runtime_run_wall_ns": 500_000_000,
            "source_sealed_to_precomplete_wall_ns": 520_000_000,
        }
    result = {
        "source_duration_s": 2,
        "output_duration_s": 2,
        "upload_wall_s": 0.001,
        "total_wall_s": 0.6,
        "server_completion": {"metrics": saved_metrics},
    }
    (directory / "result.json").write_text(json.dumps(result))
    events = [
        {"type": "turn_endpoint", "turn_id": "t1", "sample": 2000, "speech_end_sample": 1000, "reason": "prediction"},
        {
            "type": "turn_endpoint",
            "turn_id": "t2",
            "sample": 4000,
            "speech_end_sample": 3200,
            "reason": "source_eof",
            "compute_ns": 2_000_000,
        },
    ]
    if playback:
        events.append({"type": "bot_audio", "response_id": "r1", "sample": 6400, "end_sample": 10400})
    (directory / "server_events.jsonl").write_text("\n".join(json.dumps(event) for event in events))


class MetricsTests(unittest.TestCase):
    def test_shared_policy_commit_separates_provider_audio_from_output_buffer_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded_case(directory)
            result = json.loads((directory / "result.json").read_text())
            record = result["server_completion"]["metrics"]["responses"][0]
            record.update(
                turn_ids=["turn-1"],
                endpoint_sample=4000,
                latest_vad_speech_end_sample=3200,
                first_packed_audio_ready_ns=150_000_000,
            )
            (directory / "result.json").write_text(json.dumps(result))
            (directory / "server_events.jsonl").write_text(
                json.dumps({"type": "bot_audio", "response_id": "r1", "sample": 7200, "end_sample": 11200})
            )
            row = metrics.extract_case(directory, "shared")["responses"][0]
            self.assertEqual(row["response_first_audio_ms"], 100)
            self.assertEqual(row["output_buffer_wait_ms"], 50)
            self.assertEqual(row["rendered_turn_gap_ms"], 250)
            self.assertEqual(row["endpoint_reason"], "shared_Pipecat_turn_commit")
            record["latest_vad_speech_end_sample"] = None
            (directory / "result.json").write_text(json.dumps(result))
            quiet = metrics.extract_case(directory, "quiet")["responses"][0]
            self.assertIsNone(quiet["rendered_turn_gap_ms"])
            self.assertIsNone(quiet["endpoint_wait_ms"])

    def test_old_recordings_keep_unobserved_timers_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded_case(directory)
            case = metrics.extract_case(directory, "original")
        self.assertIsNone(case["session"]["asr_first_response_ms"])
        self.assertIsNone(case["session"]["server_batch_e2e_ms"])
        self.assertIsNone(case["responses"][0]["tts_first_audio_latency_ms"])
        self.assertEqual(case["session"]["asr_total_wall_ms"], 300)

    def test_coalesced_turn_gap_uses_last_owned_turn_and_exact_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded_case(directory)
            row = metrics.extract_case(directory, "original")["responses"][0]
        self.assertEqual(row["coalesced_turn_count"], 2)
        self.assertEqual(row["endpoint_reason"], "source_eof")
        self.assertEqual(row["endpoint_wait_ms"], 50)
        self.assertEqual(row["post_endpoint_ready_wait_ms"], 50)
        self.assertEqual(row["rendered_turn_gap_ms"], 200)

    def test_cancelled_before_playback_has_no_audible_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded_case(directory, playback=False)
            case = metrics.extract_case(directory, "original")
        self.assertEqual(case["session"]["response_count"], 1)
        self.assertEqual(case["session"]["committed_response_count"], 0)
        self.assertIsNone(case["session"]["mean_rendered_turn_gap_ms"])
        self.assertIsNone(case["responses"][0]["rendered_turn_gap_ms"])
        self.assertEqual(case["responses"][0]["response_first_audio_ms"], 100)

    def test_measured_rpc_timers_are_not_inferred_from_response_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorded_case(directory, instrumented=True)
            case = metrics.extract_case(directory, "instrumented")
        row = case["responses"][0]
        self.assertEqual(row["llm_rpc_ttft_ms"], 5)
        self.assertEqual(row["llm_first_text_ms"], 30)
        self.assertEqual(row["tts_first_audio_latency_ms"], 40)
        self.assertNotEqual(
            row["tts_first_audio_latency_ms"], row["response_first_audio_ms"] - row["llm_first_text_ms"]
        )
        self.assertEqual(case["session"]["server_batch_e2e_ms"], 520)
        self.assertEqual(case["session"]["asr_first_transcript_ms"], 20)


if __name__ == "__main__":
    unittest.main()
