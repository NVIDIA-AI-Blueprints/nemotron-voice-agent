# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify paired replay packet preservation, receive-clock placement, and units."""

import asyncio
import base64
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "offline-eval" / "compare_live.py"
_SPEC = importlib.util.spec_from_file_location("nva_live_comparison_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
client = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = client
_SPEC.loader.exec_module(client)


def pcm(*samples):
    """Encode exact little-endian known samples."""
    return struct.pack("<" + "h" * len(samples), *samples)


def audio_event(response_id, data, kind="response.output_audio.delta"):
    """Construct canonical or compatibility alias output events."""
    return {"type": kind, "response_id": response_id, "delta": base64.b64encode(data).decode()}


class PlaybackTests(unittest.TestCase):
    """Do not manufacture low latency by backdating audio or consuming aliases."""

    def test_queue_anchors_receipt_and_preserves_real_underrun(self):
        """Queue early arrivals contiguously, but retain gaps after a late arrival."""
        capture = client.PlaybackCapture()
        capture.consume({"type": "response.created", "response": {"id": "r1"}}, 0)
        capture.consume(audio_event("r1", pcm(1) * 160), 100)
        capture.consume(audio_event("r1", pcm(2) * 160), 105)
        capture.consume(audio_event("r1", pcm(3) * 160), 125)
        capture.consume(audio_event("r1", pcm(99) * 160, "response.audio.delta"), 125.1)
        self.assertEqual([span.start_sample for span in capture.spans], [1600, 1760, 2000])
        rendered = capture.render(16_000)
        self.assertEqual(len(rendered), 32_000)
        self.assertEqual(rendered[1600 * 2 : 1920 * 2], pcm(1) * 160 + pcm(2) * 160)
        self.assertEqual(rendered[1920 * 2 : 2000 * 2], bytes(160))
        self.assertEqual(rendered[2000 * 2 : 2160 * 2], pcm(3) * 160)

    def test_interruption_clips_already_queued_tail_and_drops_late_cancelled_pcm(self):
        """Retain heard PCM and clear only the unplayed part of an interrupted response."""
        capture = client.PlaybackCapture()
        capture.consume({"type": "response.created", "response": {"id": "r1"}}, 0)
        capture.consume(audio_event("r1", pcm(7) * 1600), 100)
        capture.consume({"type": "input_audio_buffer.speech_started"}, 125)
        capture.consume(audio_event("r1", pcm(9) * 1600), 130)
        self.assertEqual(len(capture.spans), 1)
        self.assertEqual(capture.spans[0].end_sample, 2000)
        self.assertEqual(capture.spans[0].pcm, pcm(7) * 400)
        capture.consume({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 200}, 205)
        capture.consume({"type": "response.created", "response": {"id": "r2"}}, 220)
        capture.consume(audio_event("r2", pcm(8) * 160), 250)
        rows = capture.response_rows()
        self.assertEqual(rows[1]["gateway_stop_to_playback_ms"], 50)
        self.assertEqual(rows[1]["client_stop_to_first_audio_ms"], 45)
        self.assertEqual(rows[1]["committed_audio_samples"], 160)
        self.assertEqual(len(capture.render(4000)), 4160 * 2)

    def test_unplayed_response_has_no_playback_latency_and_pcm_errors_fail(self):
        """Keep missing measurements absent rather than inventing zero latency."""
        capture = client.PlaybackCapture()
        capture.consume({"type": "response.created", "response": {"id": "r1"}}, 0)
        capture.consume({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}}, 10)
        row = capture.response_rows()[0]
        self.assertIsNone(row["first_playback_ms"])
        self.assertIsNone(row["gateway_stop_to_playback_ms"])
        self.assertEqual(row["committed_audio_samples"], 0)
        with self.assertRaisesRegex(ValueError, "preceding response.created"):
            capture.consume(audio_event("unknown", pcm(1)), 20)
        capture.consume({"type": "response.created", "response": {"id": "r2"}}, 25)
        with self.assertRaisesRegex(ValueError, "partial PCM16"):
            capture.consume(audio_event("r2", b"x"), 30)

    def test_late_cancelled_old_response_does_not_clip_new_response(self):
        """The server permits a newer response during the old response finish sequence."""
        capture = client.PlaybackCapture()
        capture.consume({"type": "response.created", "response": {"id": "r1"}}, 0)
        capture.consume({"type": "response.created", "response": {"id": "r2"}}, 90)
        capture.consume(audio_event("r2", pcm(9) * 1600), 100)
        capture.consume({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}}, 125)
        self.assertEqual(capture.spans[0].end_sample, 3200)
        self.assertEqual(capture.spans[0].pcm, pcm(9) * 1600)
        self.assertEqual(capture.active_response_id, "r2")


class MetricTests(unittest.TestCase):
    """Require documented units and preserve missing server observations."""

    def test_native_arrays_and_compact_aggregates_convert_seconds_to_ms(self):
        """Convert durations only; counts and throughput are not multiplied."""
        response = {
            "metadata": {
                "nvidia_metrics": json.dumps(
                    {
                        "server_e2e": [0.75, 1.25],
                        "asr_ttfb": [0.08, None, True],
                        "llm_ttft": {"average": 0.03, "count": 2},
                        "tts_ttfb": [],
                        "llm_tokens_per_sec": [90],
                    }
                )
            }
        }
        metrics = client.server_metrics_ms(response)
        self.assertEqual(metrics["server_e2e_ms"], [750, 1250])
        self.assertEqual(metrics["asr_ttfb_ms"], [80])
        self.assertEqual(metrics["llm_ttft_ms"], {"average_ms": 30, "count": 2})
        self.assertIsNone(metrics["tts_ttfb_ms"])
        self.assertIsNone(metrics["llm_processing_time_ms"])
        self.assertNotIn("llm_tokens_per_sec_ms", metrics)

    def test_bad_and_absent_metadata_remain_missing(self):
        """Reject invalid timer data without replacing it with zeros."""
        for metadata in (
            {},
            {"nvidia_metrics": "bad"},
            {"nvidia_metrics": "[]"},
            {"nvidia_metrics": '{"asr_ttfb":[NaN,"x",true]}'},
        ):
            with self.subTest(metadata=metadata):
                self.assertTrue(
                    all(value is None for value in client.server_metrics_ms({"metadata": metadata}).values())
                )

    def test_session_pcm_rates_are_exact_and_other_config_is_retained(self):
        """Preserve provider selectors while rejecting an accidental 24 kHz replay."""
        config = {"prompt": {"id": "test"}, "temperature": 0}
        actual = client.session_patch(config)
        self.assertNotIn("audio", config)
        self.assertEqual(actual["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 16000})
        self.assertEqual(actual["prompt"], {"id": "test"})
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            client.session_patch({"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}})
        with self.assertRaisesRegex(ValueError, "environment") as failure:
            client.session_patch({"nvidia": {"NVIDIA_API_KEY": "private-test-value"}})
        self.assertNotIn("private-test-value", str(failure.exception))


class WireReplayTests(unittest.IsolatedAsyncioTestCase):
    """Exercise sender/receiver together without providers or a network socket."""

    async def test_short_last_packet_is_unchanged_and_alias_does_not_duplicate_recording(self):
        """Use exact source bytes and save server metrics, clocks, and genuine PCM."""

        class Socket:
            def __init__(self):
                self.incoming = asyncio.Queue()
                self.uploads = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def send(self, raw):
                event = json.loads(raw)
                if event["type"] == "session.update":
                    await self.incoming.put({"type": "session.updated", "session": event["session"]})
                else:
                    self.uploads.append(base64.b64decode(event["audio"]))
                    if len(self.uploads) == 3:
                        events = [
                            {"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 0},
                            {"type": "response.created", "response": {"id": "r1"}},
                            audio_event("r1", pcm(7, 8)),
                            audio_event("r1", pcm(7, 8), "response.audio.delta"),
                            {
                                "type": "response.done",
                                "response": {
                                    "id": "r1",
                                    "status": "completed",
                                    "metadata": {"nvidia_metrics": '{"llm_ttft":[0.04]}'},
                                },
                            },
                        ]
                        for value in events:
                            await self.incoming.put(value)

            async def recv(self):
                return json.dumps(await self.incoming.get())

        socket = Socket()
        with tempfile.TemporaryDirectory() as root, patch.object(client.websockets, "connect", return_value=socket):
            output = Path(root) / "live"
            original = pcm(1, 2, 0, 0, -4)
            result = await client.run_live(
                "ws://test/v1/realtime",
                original,
                output,
                {},
                chunk_samples=2,
                post_input_idle_ms=1,
                max_post_input_ms=500,
            )
            self.assertEqual(socket.uploads, [pcm(1, 2), pcm(0, 0), pcm(-4)])
            self.assertEqual(client.runner.client.read_pcm(output / "input.wav"), original)
            self.assertEqual(result["responses"][0]["committed_audio_samples"], 2)
            self.assertEqual(result["responses"][0]["llm_ttft_ms"], [40])
            self.assertEqual(result["completion_reason"], "post_input_idle")
            trace = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(value["event"].get("pcm_bytes", 0) for value in trace), 8)
            self.assertNotIn("delta", trace[2]["event"])
            self.assertTrue((output / "comparison.wav").is_file())

    def test_flat_config_maps_selectors_and_never_forwards_service_endpoints_or_keys(self):
        """Live routing is catalog-owned, while prompt/voice/settings stay explicit."""
        session = client.flat_to_session(
            {
                "prompt_key": "generic_assistant_without_tools",
                "base_url": "http://private",
                "asr_server": "private:1",
                "tts_server": "private:2",
                "api_key": "secret",
                "llm_id": "chosen",
                "model_id": "model",
                "temperature": 0,
                "max_tokens": 256,
                "tts_voice_id": "Aria",
                "extra_params": {"seed": 1},
            }
        )
        self.assertEqual(session["nvidia"], {"llm_id": "chosen", "model_id": "model", "extra_params": {"seed": 1}})
        self.assertEqual(session["max_output_tokens"], 256)
        self.assertEqual(session["temperature"], 0)
        self.assertEqual(session["audio"]["output"]["voice"], "Aria")
        self.assertNotIn("secret", json.dumps(session))
        self.assertNotIn("private", json.dumps(session))
        with self.assertRaisesRegex(ValueError, "system_prompt override"):
            client.flat_to_session({"prompt_key": "catalog", "system_prompt": "control role text"})

    def test_prepared_pair_preserves_source_then_adds_identical_explicit_silence(self):
        """The reusable manifest must reference exactly the bytes both modes send."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "original.wav"
            client.runner.client.write_pcm(source, pcm(1, 0, -32768, 32767))
            cases = client.runner.load_cases([source])
            result = client.prepare_paired_sources(cases, directory / "prepared", trailing_silence_ms=2)
            prepared = client.runner.load_cases(manifest=directory / "prepared/cases.json")
            self.assertEqual(
                client.runner.client.read_pcm(prepared[0].input_path), pcm(1, 0, -32768, 32767) + bytes(64)
            )
            provenance = result["cases"][0]["provenance"]["paired_source"]
            self.assertEqual(provenance["original_samples"], 4)
            self.assertEqual(provenance["prepared_samples"], 36)
            self.assertEqual(provenance["trailing_silence_ms"], 2)
            self.assertNotEqual(provenance["original_pcm_sha256"], provenance["prepared_pcm_sha256"])
            self.assertEqual(client.runner.client.read_pcm(source), pcm(1, 0, -32768, 32767))

    def test_mode_comparison_retains_both_offsets_and_silent_tail(self):
        """Pair modes with zero padding only; do not align them by their first speech."""
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "modes.wav"
            client.write_mode_comparison(pcm(0, 9, 10), pcm(0, 0, 7, 8), target)
            with wave.open(str(target), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getframerate()), (2, 16000))
                self.assertEqual(audio.readframes(4), pcm(0, 0, 9, 0, 10, 7, 0, 8))

    async def test_insecure_credentials_and_invalid_bounds_prevent_connect(self):
        """Validate input before making an authenticated request or creating audio."""
        with tempfile.TemporaryDirectory() as root, patch.object(client.websockets, "connect") as connect:
            with self.assertRaisesRegex(ValueError, "unencrypted"):
                await client.run_live("ws://test", pcm(1), Path(root) / "a", {}, api_key="test-key")
            with self.assertRaisesRegex(ValueError, "Timeouts"):
                await client.run_live("ws://test", pcm(1), Path(root) / "b", {}, post_input_idle_ms=-1)
            connect.assert_not_called()
            self.assertFalse((Path(root) / "a").exists())

    async def test_semantic_turn_waiting_for_first_response_cannot_finish_as_idle_success(self):
        """A delayed first token must reach the absolute failure bound, not create silent success."""

        class Socket:
            def __init__(self):
                self.incoming = asyncio.Queue()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def send(self, raw):
                message = json.loads(raw)
                if message["type"] == "session.update":
                    await self.incoming.put({"type": "session.updated", "session": message["session"]})
                else:
                    await self.incoming.put({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 0})

            async def recv(self):
                return json.dumps(await self.incoming.get())

        with tempfile.TemporaryDirectory() as root, patch.object(client.websockets, "connect", return_value=Socket()):
            output = Path(root) / "pending"
            with self.assertRaisesRegex(TimeoutError, "absolute post-input bound"):
                await client.run_live(
                    "ws://test",
                    pcm(1),
                    output,
                    {},
                    post_input_idle_ms=1,
                    max_post_input_ms=250,
                )
            self.assertFalse((output / "full_output.wav").exists())
            self.assertFalse((output / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
