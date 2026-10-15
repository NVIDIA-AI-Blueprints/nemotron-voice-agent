# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Verify case preservation, listening-channel alignment, and truthful failures."""

import asyncio
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "offline-eval" / "run_fdb_samples.py"
_SPEC = importlib.util.spec_from_file_location("nva_offline_fdb_runner_test", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)
client = runner.client


def pcm(values):
    """Encode known signed samples without changing their amplitude."""
    return struct.pack("<" + "h" * len(values), *values)


class ComparisonTests(unittest.TestCase):
    """Make sample-level stereo listening alignment independently inspectable."""

    def test_channels_are_exact_and_source_tail_is_silent(self):
        """Preserve both channel byte streams, offsets, and extended bot tail."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source, response, comparison = [
                directory / name for name in ("input.wav", "full_output.wav", "comparison.wav")
            ]
            source_pcm = pcm([-32768, -7, 0, 5, 32767])
            response_pcm = pcm([0, 0, 91, 92, 93, -100, 101, 102])
            client.write_pcm(source, source_pcm)
            client.write_pcm(response, response_pcm)
            metadata = runner.write_comparison(source, response, comparison)
            with wave.open(str(comparison), "rb") as output:
                self.assertEqual((output.getnchannels(), output.getframerate(), output.getnframes()), (2, 16000, 8))
                samples = output.readframes(8)
            left = bytes(value for index in range(0, len(samples), 4) for value in samples[index : index + 2])
            right = bytes(value for index in range(2, len(samples), 4) for value in samples[index : index + 2])
            self.assertEqual(left, source_pcm + bytes(6))
            self.assertEqual(right, response_pcm)
            self.assertEqual(client.read_pcm(source), source_pcm)
            self.assertEqual(client.read_pcm(response), response_pcm)
            self.assertEqual(metadata["user_tail_padding_samples"], 3)
            self.assertEqual(metadata["alignment_adjustment_samples"], 0)

    def test_short_response_is_not_disguised_by_padding(self):
        """Reject an incomplete server horizon instead of silently repairing it."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source, response, comparison = [
                directory / name for name in ("input.wav", "full_output.wav", "comparison.wav")
            ]
            client.write_pcm(source, pcm([1, 2, 3]))
            client.write_pcm(response, pcm([0, 9]))
            with self.assertRaisesRegex(ValueError, "entire source horizon"):
                runner.write_comparison(source, response, comparison)
            self.assertFalse(comparison.exists())
            self.assertFalse(comparison.with_name(comparison.name + ".partial").exists())

    def test_comparison_preserves_samples_across_io_block_boundary(self):
        """Keep offsets exact beyond the renderer's bounded I/O block size."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source, response, comparison = [
                directory / name for name in ("input.wav", "full_output.wav", "comparison.wav")
            ]
            source_pcm = pcm([3]) * 65537
            bot_pcm = bytes(65535 * 2) + pcm([42, 43, 44, 45])
            client.write_pcm(source, source_pcm)
            client.write_pcm(response, bot_pcm)
            runner.write_comparison(source, response, comparison)
            with wave.open(str(comparison), "rb") as output:
                output.setpos(65535)
                self.assertEqual(output.readframes(4), pcm([3, 42, 3, 43, 0, 44, 0, 45]))


class BatchTests(unittest.IsolatedAsyncioTestCase):
    """Prevent failed/pending trials from masquerading as generated TTS audio."""

    async def test_prepare_preserves_independent_cases_and_never_calls_server(self):
        """Retain exact case PCM and provenance without inventing output files."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            first, second = directory / "one.wav", directory / "two.wav"
            client.write_pcm(first, pcm([1, 2, 3]))
            client.write_pcm(second, pcm([-5, 8]))
            manifest = directory / "cases.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"id": "fdb-1", "input": "one.wav", "provenance": {"dataset": "fixture"}},
                            {"id": "fdb-2", "input": "two.wav"},
                        ]
                    }
                )
            )
            cases = runner.load_cases(manifest=manifest)
            self.assertEqual(cases[0].input_path, first.resolve())
            with patch.object(client, "run_client", AsyncMock()) as request:
                summary = await runner.run_batch(cases, directory / "prepared", url=None, prepare_only=True)
            request.assert_not_awaited()
            self.assertEqual([case["status"] for case in summary["cases"]], ["prepared_only", "prepared_only"])
            self.assertEqual(client.read_pcm(directory / "prepared/fdb-1/input.wav"), pcm([1, 2, 3]))
            self.assertEqual(client.read_pcm(directory / "prepared/fdb-2/input.wav"), pcm([-5, 8]))
            self.assertEqual(summary["cases"][0]["provenance"], {"dataset": "fixture"})
            self.assertFalse(list((directory / "prepared").rglob("*output.wav")))
            self.assertFalse(list((directory / "prepared").rglob("comparison.wav")))

    async def test_missing_backend_records_failure_without_false_tts(self):
        """Keep connection errors public and sanitized, with only the true input."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "case.wav"
            client.write_pcm(source, pcm([1, 0, 2]))
            with patch.object(
                client, "run_client", AsyncMock(side_effect=ConnectionError("secret URL must not leak"))
            ) as request:
                summary = await runner.run_batch(
                    runner.load_cases([source]), directory / "failed", url="ws://unavailable/api/ws/offline"
                )
            self.assertEqual(summary["failed_cases"], 1)
            self.assertEqual(summary["cases"][0]["status"], "failed")
            self.assertEqual(request.await_args.kwargs["upload_speed"], 0)
            self.assertEqual(request.await_args.kwargs["chunk_samples"], 65536)
            self.assertNotIn("secret URL", (directory / "failed/summary.json").read_text())
            self.assertFalse((directory / "failed/case/full_output.wav").exists())
            self.assertFalse((directory / "failed/case/comparison.wav").exists())

    async def test_invalid_source_selection_prevents_all_provider_requests(self):
        """Reject oversized sources without clipping them or issuing partial runs."""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "case.wav"
            original = pcm([1]) * 16001
            client.write_pcm(source, original)
            with (
                patch.object(client, "run_client", AsyncMock()) as request,
                self.assertRaisesRegex(ValueError, "never truncated"),
            ):
                await runner.run_batch(
                    runner.load_cases([source]), directory / "too-long", url="ws://endpoint", max_input_seconds=1
                )
            request.assert_not_awaited()
            self.assertEqual(client.read_pcm(source), original)
            self.assertFalse((directory / "too-long").exists())


class PacketNegotiationTests(unittest.IsolatedAsyncioTestCase):
    """The bulk client must obey an endpoint's smaller negotiated packet limit."""

    async def test_bulk_packet_limit_is_negotiated_and_exact_pcm_is_preserved(self):
        """Cap 65536 requested samples to three without dropping the short tail."""

        class Socket:
            """Minimal connection accepting only upload/control traffic."""

            def __init__(self):
                self.uploads = []
                self.completed = asyncio.Event()

            async def send(self, value):
                if isinstance(value, bytes):
                    self.uploads.append(client.decode_packet(value, b"NVAI"))
                elif json.loads(value)["type"] == "offline.input.complete":
                    self.completed.set()

            async def recv(self):
                return json.dumps(
                    {
                        "type": "offline.session.accepted",
                        "protocol": client.PROTOCOL,
                        "capabilities": list(client.REQUIRED),
                        "output_sample_rate": 16000,
                        "limits": {"max_packet_samples": 3},
                    }
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                await self.completed.wait()
                return json.dumps(
                    {
                        "type": "offline.session.complete",
                        "status": "ok",
                        "input_samples": 8,
                        "rendered_samples": 8,
                        "output_packets": 0,
                    }
                )

        socket = Socket()

        @asynccontextmanager
        async def connect(*args, **kwargs):
            yield socket

        with tempfile.TemporaryDirectory() as root, patch("websockets.asyncio.client.connect", connect):
            source = pcm([1, 2, 3, 4, 5, 6, 7, 8])
            result = await client.run_client("ws://test", source, Path(root), chunk_samples=65536)
        self.assertEqual(
            [(sequence, offset, len(audio) // 2) for sequence, offset, audio in socket.uploads],
            [(0, 0, 3), (1, 3, 3), (2, 6, 2)],
        )
        self.assertEqual(b"".join(audio for _, _, audio in socket.uploads), source)
        self.assertEqual(result["input_packets"], 3)
        self.assertEqual(result["chunk_samples"], 3)
        self.assertEqual(result["requested_chunk_samples"], 65536)


if __name__ == "__main__":
    unittest.main()
