# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from examples.generic.offline.models import AudioSpan, SessionResult
from examples.generic.offline.protocol import encode_packet


class RouteTests(unittest.TestCase):
    def test_actual_nva_app_offline_route_preserves_normal_route_and_records(self):
        class Runtime:
            configuration = {"model": "scripted-route-test"}

            async def run(self, source):
                extent = source.sample_count + 12
                return SessionResult(
                    (AudioSpan(source.sample_count + 10, b"\x01\x00" * 2, "one"),),
                    extent,
                    ({"event": "test", "sample": source.sample_count},),
                    (),
                )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"OFFLINE_AUDIO_DUMP_PATH": directory}),
            patch("examples.generic.offline.factory.build_runtime", return_value=Runtime()) as factory,
        ):
            app = server.create_app()
            self.assertTrue(any(getattr(route, "path", None) == "/api/ws" for route in app.routes))
            with TestClient(app) as client, client.websocket_connect("/api/ws/offline") as socket:
                socket.send_json(
                    {
                        "type": "offline.session.start",
                        "protocol": "nva.offline.v1",
                        "sample_rate": 16000,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                        "input_samples": 16000,
                        "config": {"prompt_content": "custom evaluation"},
                    }
                )
                self.assertEqual(socket.receive_json()["type"], "offline.session.accepted")
                factory.assert_not_called()
                pcm = bytes(32000)
                socket.send_bytes(encode_packet(b"NVAI", 0, 0, pcm))
                socket.send_json(
                    {
                        "type": "offline.input.complete",
                        "input_samples": 16000,
                        "input_packets": 1,
                        "sha256": hashlib.sha256(pcm).hexdigest(),
                    }
                )
                self.assertTrue(socket.receive_bytes().startswith(b"NVAO"))
                complete = None
                while complete is None:
                    event = socket.receive_json()
                    if event["type"] == "offline.session.complete":
                        complete = event
                self.assertEqual(complete["rendered_samples"], 16012)
            self.assertEqual(factory.call_args.args[0]["prompt_content"], "custom evaluation")
            records = list(Path(directory).glob("*/manifest.json"))
            self.assertEqual(len(records), 1)
            self.assertIn('"configuration_fingerprint"', records[0].read_text())
