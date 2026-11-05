# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import http.client
import os
import socket
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import examples_registry
import server
from utils import set_service_context


def _entry(entry_id: str, source: str = "self-hosted", **fields) -> dict:
    return {"id": entry_id, "source": source, "selected": False, **fields}


class ServiceStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        server._seen_service_targets.clear()
        server._failed_service_targets.clear()

    def test_unresolvable_host_is_offline_and_refused_port_is_starting(self) -> None:
        with patch("socket.create_connection", side_effect=socket.gaierror):
            self.assertEqual(server._probe_tcp("nemo-speech:50051", 1), "offline")
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            self.assertEqual(server._probe_tcp("nemo-speech:50051", 1), "starting")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(socket.gaierror())):
            self.assertEqual(server._probe_http("http://llm:8000/health", False, 1), "offline")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("u", 503, "", None, None)):
            self.assertEqual(server._probe_http("http://llm:8000/health", False, 1), "starting")

    def test_refused_local_port_is_offline(self) -> None:
        refused = urllib.error.URLError(ConnectionRefusedError())
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            self.assertEqual(server._probe_tcp("localhost:50051", 1), "offline")
        with patch("urllib.request.urlopen", side_effect=refused):
            self.assertEqual(server._probe_http("http://localhost:18000/health", False, 1), "offline")
            self.assertEqual(server._probe_http("http://[::1]:18000/health", False, 1), "offline")
            self.assertEqual(server._probe_http("http://llm:8000/health", False, 1), "starting")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(ConnectionResetError())):
            self.assertEqual(server._probe_http("http://localhost:18000/health", False, 1), "starting")

    def test_truncated_health_response_is_starting(self) -> None:
        for error in (http.client.IncompleteRead(b"{"), http.client.BadStatusLine("")):
            with patch("urllib.request.urlopen", side_effect=error):
                self.assertEqual(server._probe_http("http://llm:8000/health", True, 1), "starting")

    def test_service_that_disappears_after_being_seen_is_error_until_ready(self) -> None:
        self.assertEqual(server._remembered_status("llm", "offline"), "offline")
        self.assertEqual(server._remembered_status("llm", "starting"), "starting")
        self.assertEqual(server._remembered_status("llm", "offline"), "error")
        self.assertEqual(server._remembered_status("llm", "starting"), "error")
        self.assertEqual(server._remembered_status("llm", "ready"), "ready")
        self.assertEqual(server._remembered_status("llm", "starting"), "starting")

    def test_catalog_hides_offline_entries_and_selects_first_visible(self) -> None:
        catalog = {
            "llm": [
                _entry("self-hosted:lightning", base_url="http://nvidia-llm-vllm:8000/v1"),
                _entry("self-hosted:super", base_url="http://nemotron-35-super-vl-vllm:8000/v1"),
                _entry("cloud-nim:lightning", "cloud-nim", base_url="https://integrate.api.nvidia.com/v1"),
            ],
            "asr": [_entry("self-hosted:asr", server="nemo-speech:50051")],
        }
        statuses = {"http://nemotron-35-super-vl-vllm:8000/health": "starting"}
        set_service_context(examples_registry.EXAMPLES["generic-assistant"])
        with (
            patch.dict(os.environ, {"APP_RUNTIME": "container", "SERVICE_RECIPE": "single-gpu", "NVIDIA_API_KEY": ""}),
            patch.object(server, "_probe_status", side_effect=lambda target, _: statuses.get(target, "offline")),
        ):
            result = asyncio.run(server._with_service_status(catalog))

        self.assertEqual([entry["id"] for entry in result["llm"]], ["self-hosted:super", "cloud-nim:lightning"])
        self.assertEqual([entry.get("status") for entry in result["llm"]], ["starting", None])
        self.assertEqual([entry["selected"] for entry in result["llm"]], [True, False])
        self.assertEqual([entry["status"] for entry in result["asr"]], ["offline"])

    def test_speech_entry_with_health_path_uses_http_readiness(self) -> None:
        asr_server = "http://speech.invalid:9000/v1"
        entry = _entry("self-hosted:transcribe", server=asr_server, health_path="/v1/health/ready")
        catalog = [("self-hosted", {"asr": {"transcribe": entry}})]
        for body, status in (
            (b'{"status":"ready"}', "ready"),
            (b'{"status":"starting"}', "starting"),
            (404, "starting"),
        ):
            server._seen_service_targets.clear()
            response = MagicMock()
            response.__enter__.return_value.read.return_value = body
            error = urllib.error.HTTPError(asr_server, body, "Not Found", None, None) if isinstance(body, int) else None
            with (
                self.subTest(status=status),
                patch("utils.service_catalog_sources", return_value=catalog),
                patch("urllib.request.urlopen", side_effect=error, return_value=response) as http,
                patch("socket.create_connection") as connect,
            ):
                result = asyncio.run(server._with_service_status({"asr": [entry]}))
                self.assertEqual(result["asr"][0]["status"], status)
                check = server._ensure_asr_ready_for_connection({"asr_server": asr_server}, {"services": {"asr": []}})
                if status == "ready":
                    asyncio.run(check)
                else:
                    with self.assertRaisesRegex(RuntimeError, "ASR service is still starting"):
                        asyncio.run(check)
                self.assertEqual(http.call_args.args[0], "http://speech.invalid:9000/v1/health/ready")
                connect.assert_not_called()

    def test_connect_check_reports_missing_service_as_not_running(self) -> None:
        with (
            patch("socket.create_connection", side_effect=socket.gaierror),
            self.assertRaisesRegex(RuntimeError, "not running at nemo-speech:50051"),
        ):
            server._check_service_port("ASR", "nemo-speech:50051")

    def test_speech_health_path_without_url_scheme_uses_port_fallback(self) -> None:
        for address, expected in (
            ("nemotron-asr-streaming-english:50052", "http://nemotron-asr-streaming-english:9001/v1/health/ready"),
            ("127.0.0.1:50152", "http://127.0.0.1:9001/v1/health/ready"),
            ("speech.invalid:50052", ""),
        ):
            entry = _entry("self-hosted:streaming", server=address, health_path="/custom/ready")
            with (
                self.subTest(address=address),
                patch("utils.service_catalog_sources", return_value=[("self-hosted", {"asr": {"streaming": entry}})]),
            ):
                self.assertEqual(server._local_speech_ready_url("ASR", address), expected)


if __name__ == "__main__":
    unittest.main()
