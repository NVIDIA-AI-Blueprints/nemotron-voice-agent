# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject false pairings and retain observed timing scopes in saved reports."""

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "offline-eval" / "report_comparison.py"
_SPEC = importlib.util.spec_from_file_location("nva_comparison_report_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
reporter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reporter
_SPEC.loader.exec_module(reporter)


def save(path, document):
    """Persist a minimal, explicit public fixture document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document) + "\n")


def wav(path, samples):
    """Write known canonical samples without invoking a model."""
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def event(kind, elapsed_ms, **fields):
    """Represent both original monotonic nanoseconds and their exact ms conversion."""
    return {"event": kind, "elapsed_ns": round(elapsed_ms * 1_000_000), "elapsed_ms": elapsed_ms, **fields}


def trace_document(*, outcome="returned", session="sess-exact"):
    """Supply genuinely different frame and observer clocks to prevent accidental mixing."""
    return {
        "schema": "nva.evaluation.trace.v1",
        "session_id": "trace-uuid",
        "configuration": {"realtime_session_id": session},
        "dropped_events": 0,
        "events": [
            event("asr_worker_started", 0),
            event("asr_request_supplied", 1, audio_bytes=1024),
            event("asr_provider_result", 10, results=[]),
            event("asr_provider_result", 20, results=[{"is_final": False, "transcript": "hello"}]),
            event("asr_provider_result", 30, results=[{"is_final": True, "transcript": "hello world"}]),
            event("LLMFullResponseStartFrame", 40, llm_request_index=1, pipeline_timestamp_ns=100_000_000),
            event("llm_text", 45, llm_request_index=1, text="hello", pipeline_timestamp_ns=107_000_000),
            event(
                "tts_text_submitted",
                47,
                context_id="tts-one",
                llm_request_index=1,
                text="hello",
                synthesis_mode="stitched",
            ),
            event("tts_worker_started", 49, context_id="tts-one", llm_request_index=1),
            event(
                "tts_pcm_ready",
                55,
                context_id="tts-one",
                llm_request_index=1,
                audio_bytes=32,
                sample_rate=16000,
                num_channels=1,
            ),
            event("LLMFullResponseEndFrame", 60, llm_request_index=1, pipeline_timestamp_ns=120_000_000),
            event(
                "tts_worker_finished",
                70,
                context_id="tts-one",
                llm_request_index=1,
                outcome=outcome,
                stop_requested=True,
            ),
            event("asr_worker_finished", 80, iterator_closed=True),
        ],
    }


class TraceScopeTests(unittest.TestCase):
    """Keep service callbacks, request frames, and successful completion claims distinct."""

    def test_units_baselines_and_frame_clock_are_distinct(self):
        """ASR callback is not first text, and observer delay is not LLM TTFT."""
        tables = reporter.trace_tables(trace_document(), "case")
        asr = tables["asr_workers"][0]
        self.assertEqual(asr["first_callback_from_worker_start_ms"], 10)
        self.assertEqual(asr["first_nonempty_transcript_from_worker_start_ms"], 20)
        self.assertEqual(asr["first_final_transcript_from_first_audio_supply_ms"], 29)
        self.assertEqual(asr["observed_worker_lifetime_ms"], 80)
        llm = tables["llm_requests"][0]
        self.assertEqual(llm["llm_frame_to_first_nonempty_text_ms"], 7)
        self.assertEqual(llm["llm_frame_to_end_ms"], 20)
        self.assertIsNone(llm["llm_frame_to_first_generated_pcm_ms"])
        self.assertFalse(llm["wire_association_proven"])
        self.assertIsNone(llm["wire_response_id"])
        tts = tables["tts_contexts"][0]
        self.assertEqual(tts["submission_to_first_pcm_ms"], 8)
        self.assertEqual(tts["submission_to_worker_return_ms"], 23)
        self.assertEqual(tts["generated_pcm_duration_ms"], 1)
        self.assertEqual(tts["completion_scope"], "provider_success_unknown")
        self.assertFalse(tts["rpc_success_confirmed"])

    def test_cancelled_worker_is_not_reported_as_successful_completion(self):
        """Preserve observed cancellation lifetime but leave return/EOF latency absent."""
        tts = reporter.trace_tables(trace_document(outcome="cancelled"), "case")["tts_contexts"][0]
        self.assertEqual(tts["worker_outcome"], "cancelled")
        self.assertEqual(tts["worker_observed_lifetime_ms"], 21)
        self.assertIsNone(tts["worker_returned_lifetime_ms"])
        self.assertIsNone(tts["submission_to_worker_return_ms"])
        self.assertFalse(tts["rpc_success_confirmed"])

    def test_only_explicit_unambiguous_response_id_correlation_is_accepted(self):
        """Equal counts alone remain unjoined; direct one-to-many correlation also fails."""
        trace = trace_document()
        trace["events"].insert(6, event("realtime_response_created", 41, llm_request_index=1, response_id="r1"))
        self.assertEqual(reporter.trace_tables(trace, "case")["llm_requests"][0]["wire_response_id"], "r1")
        trace["events"].insert(7, event("realtime_response_created", 42, llm_request_index=1, response_id="r2"))
        self.assertFalse(reporter.trace_tables(trace, "case")["llm_requests"][0]["wire_association_proven"])
        trace["events"].pop(7)
        trace["dropped_events"] = 1
        self.assertIsNone(reporter.trace_tables(trace, "case")["llm_requests"][0]["wire_response_id"])
        trace["dropped_events"] = 0
        trace["observation_errors"] = 1
        self.assertIsNone(reporter.trace_tables(trace, "case")["llm_requests"][0]["wire_response_id"])

    def test_missing_events_remain_null_and_clock_inconsistency_fails(self):
        """Do not manufacture a first transcript or translate unrelated clock origins."""
        trace = trace_document()
        trace["events"] = [item for item in trace["events"] if item["event"] not in {"asr_provider_result", "llm_text"}]
        tables = reporter.trace_tables(trace, "case")
        self.assertIsNone(tables["asr_workers"][0]["first_nonempty_transcript_from_worker_start_ms"])
        self.assertIsNone(tables["llm_requests"][0]["llm_frame_to_first_nonempty_text_ms"])
        trace["events"][0]["elapsed_ms"] = 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            reporter.trace_tables(trace, "case")

    def test_overlapping_asr_workers_are_not_assigned_by_ordinal(self):
        """Lack of worker identifiers must remain visible when two workers overlap."""
        trace = trace_document()
        trace["events"].insert(2, event("asr_worker_started", 2))
        rows = reporter.trace_tables(trace, "case")["asr_workers"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["association_ambiguous"] for row in rows))
        self.assertTrue(all(row["first_callback_from_worker_start_ms"] is None for row in rows))


class PairedReportTests(unittest.TestCase):
    """Read saved evidence strictly and export matching audio inventory and valid turns."""

    def setUp(self):
        """Build small complete online/offline runs without network or generated audio."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.offline, self.online, self.traces = self.root / "offline", self.root / "online", self.root / "traces"
        self.off_case, self.on_case = self.offline / "case", self.online / "case"
        self.off_case.mkdir(parents=True)
        self.on_case.mkdir(parents=True)
        self.traces.mkdir()
        source = [1, 2, 0, -3] * 1000
        source_hash = hashlib.sha256(struct.pack("<" + "h" * len(source), *source)).hexdigest()
        for directory in (self.off_case, self.on_case):
            wav(directory / "input.wav", source)
            wav(directory / "full_output.wav", [0] * 1080 + [9] * 5 + [0] * 2915)
        metrics = {
            "recognition": {"asr_observed_wall_ns": 200_000_000, "asr_first_response_ns": 10_000_000},
            "server": {"source_sealed_to_precomplete_wall_ns": 400_000_000},
            "responses": [
                {
                    "response_id": "response-1",
                    "turn_ids": ["turn-1"],
                    "interrupted": True,
                    "request_sample": 1000,
                    "compute_ns": 10_000_000,
                    "generated_samples": 10,
                    "committed_samples": 5,
                    "provider": {"first_audio_ready_ns": 5_000_000, "tts_first_audio_latency_ns": 3_000_000},
                }
            ],
        }
        save(
            self.off_case / "result.json",
            {
                "source_duration_s": 0.25,
                "output_duration_s": 0.25,
                "upload_wall_s": 0.01,
                "total_wall_s": 0.5,
                "source_sha256": source_hash,
                "server_completion": {"status": "ok", "metrics": metrics},
            },
        )
        save(
            self.off_case / "server_manifest.json",
            {"metrics": metrics, "messages": [{"role": "user", "content": "hello"}]},
        )
        events = [
            {"type": "turn_endpoint", "turn_id": "turn-1", "sample": 800, "speech_end_sample": 400},
            {"type": "bot_audio", "response_id": "response-1", "sample": 1080, "end_sample": 1085},
        ]
        (self.off_case / "server_events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
        save(self.on_case / "accepted_session.json", {"id": "sess-exact"})
        save(
            self.on_case / "result.json",
            {
                "mode": "online_1x_normal_realtime_pipeline",
                "completion_reason": "post_input_idle",
                "client_setup_ms": 20,
                "client_total_wall_ms": 5270,
                "client_input_send_ms": 250,
                "input_pcm_sha256": source_hash,
                "responses": [
                    {
                        "response_id": "r1",
                        "committed_audio_samples": 5,
                        "last_audio_receive_ms": 200,
                        "server_e2e_ms": [30],
                        "llm_ttft_ms": [7],
                        "tts_ttfb_ms": [8],
                    },
                    {"response_id": "cancelled-before-pcm", "committed_audio_samples": 0, "llm_ttft_ms": [999]},
                ],
            },
        )
        save(self.on_case / "send_trace.json", [{"start_sample": 0, "send_start_ms": 0.5}])
        save(self.on_case / "transcripts.json", [{"transcript": "hello"}])
        save(self.traces / "trace.json", trace_document())

    def build(self):
        """Extract this pair into a new directory."""
        return reporter.build_report(self.offline, self.online, self.traces, self.root / "report")

    def test_report_keeps_valid_responses_units_inventory_and_transcripts(self):
        """Do not let an unplayed cancelled request change valid-response means."""
        report = self.build()
        case = report["cases"][0]
        self.assertTrue(case["paired_input_pcm_equal"])
        self.assertEqual(case["input_duration_ms"], 250)
        self.assertEqual(case["offline_server_batch_e2e_ms"], 400)
        self.assertEqual(case["offline_asr_stream_wall_ms"], 200)
        self.assertEqual(case["online_mean_valid_wire_server_e2e_ms"], 30)
        self.assertEqual(case["online_mean_valid_wire_llm_first_choices_chunk_ms"], 7)
        self.assertEqual(case["online_valid_response_count"], 1)
        self.assertEqual(case["online_client_idle_close_overhead_ms"], 5000)
        self.assertEqual(case["online_max_send_deadline_lateness_ms"], 0.5)
        self.assertEqual(len(report["online_valid_responses"]), 1)
        self.assertEqual(len(report["wav_inventory"]), 4)
        self.assertEqual(report["transcripts"][0]["messages"][0]["content"], "hello")
        self.assertTrue((self.root / "report/cases_ms.csv").is_file())
        self.assertTrue((self.root / "report/comparison.json").is_file())

    def test_same_filename_with_changed_pcm_fails_before_writing_report(self):
        """File naming and declared metadata cannot establish source equality."""
        wav(self.on_case / "input.wav", [1] * 4000)
        with self.assertRaisesRegex(ValueError, "source PCM differs"):
            self.build()
        self.assertFalse((self.root / "report").exists())

    def test_observation_errors_are_visible_in_case_quality_and_warnings(self):
        """Caught passive trace failures must not masquerade as complete timing evidence."""
        trace = trace_document()
        trace["observation_errors"] = 2
        save(self.traces / "trace.json", trace)
        report = self.build()
        self.assertEqual(report["cases"][0]["server_trace_observation_errors"], 2)
        self.assertTrue(any("observation errors" in warning for warning in report["warnings"]))

    def test_wrong_or_duplicate_server_session_id_cannot_be_paired(self):
        """Do not silently pair the only available trace when its session differs."""
        save(self.traces / "trace.json", trace_document(session="other-session"))
        with self.assertRaisesRegex(ValueError, "exactly one server trace"):
            self.build()
        save(self.traces / "trace.json", trace_document())
        save(self.traces / "duplicate.json", trace_document())
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.build()
        self.assertFalse((self.root / "report").exists())

    def test_missing_output_or_missing_results_fail_without_fabricating_audio(self):
        """A report requires genuine outputs and complete result evidence for both modes."""
        (self.on_case / "full_output.wav").unlink()
        with self.assertRaisesRegex(ValueError, "Required recording is missing"):
            self.build()
        (self.on_case / "result.json").unlink()
        with self.assertRaisesRegex(ValueError, "No completed online recordings"):
            self.build()
        self.assertFalse((self.root / "report").exists())


if __name__ == "__main__":
    unittest.main()
