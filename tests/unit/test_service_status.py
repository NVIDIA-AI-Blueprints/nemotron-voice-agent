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
        self.http_asr_server = "http://speech.invalid:9000/custom/v1/"
        self.http_asr_health = "http://speech.invalid:9000/v1/health/ready"
        self.http_asr_entry = _entry("self-hosted:transcribe", server=self.http_asr_server, asr_type="transcribe")
        self.http_asr_catalog = {"asr": [self.http_asr_entry]}
        self.http_asr_example = {"services": {"asr": ["asr"]}}

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

    def test_http_asr_missing_health_route_checks_tcp_for_catalog_and_connect(self) -> None:
        missing = urllib.error.HTTPError(self.http_asr_health, 404, "Not Found", None, None)
        for error, status in ((None, "ready"), (socket.gaierror(), "offline")):
            server._seen_service_targets.clear()
            with (
                self.subTest(status=status),
                patch("urllib.request.urlopen", side_effect=missing),
                patch("socket.create_connection", side_effect=error) as connect,
                patch.object(server, "load_service_entry_by_id", return_value=self.http_asr_entry),
            ):
                result = asyncio.run(server._with_service_status(self.http_asr_catalog))
                self.assertEqual(result["asr"][0]["status"], status)
                connect.assert_called_once_with(("speech.invalid", 9000), timeout=server._STATUS_PROBE_TIMEOUT_SECS)
                connect.reset_mock()
                check = server._ensure_asr_ready_for_connection(
                    {"asr_id": "self-hosted:transcribe", "asr_server": self.http_asr_server}, self.http_asr_example
                )
                if error:
                    with self.assertRaisesRegex(RuntimeError, "ASR service is not running"):
                        asyncio.run(check)
                else:
                    asyncio.run(check)
                connect.assert_called_once_with(("speech.invalid", 9000), timeout=server._CONNECT_HEALTH_TIMEOUT_SECS)

    def test_http_asr_other_health_responses_do_not_fall_back(self) -> None:
        for body in (401, 403, 429, 500, 503, b'{"ready":true}', b'{"ready":false}', b"invalid"):
            status = "ready" if body == b'{"ready":true}' else "starting"
            response = MagicMock()
            response.__enter__.return_value.read.return_value = body
            error = (
                urllib.error.HTTPError(self.http_asr_health, body, "Unavailable", None, None)
                if isinstance(body, int)
                else None
            )
            with (
                self.subTest(body=body),
                patch("urllib.request.urlopen", side_effect=error, return_value=response),
                patch("socket.create_connection") as connect,
                patch.object(server, "load_service_entry", return_value=self.http_asr_entry),
            ):
                result = asyncio.run(server._with_service_status(self.http_asr_catalog))
                self.assertEqual(result["asr"][0]["status"], status)
                check = server._ensure_asr_ready_for_connection(
                    {"asr_server": self.http_asr_server}, self.http_asr_example
                )
                if status == "starting":
                    with self.assertRaisesRegex(RuntimeError, "ASR service is still starting"):
                        asyncio.run(check)
                else:
                    asyncio.run(check)
                connect.assert_not_called()

    def test_missing_health_route_does_not_fall_back_for_llm_tts_or_grpc_asr(self) -> None:
        missing = urllib.error.HTTPError(self.http_asr_health, 404, "Not Found", None, None)
        for label, target in (
            ("LLM", self.http_asr_server),
            ("TTS", self.http_asr_server),
            ("ASR", "nemo-speech:50051"),
            ("ASR", self.http_asr_server),
        ):
            with (
                self.subTest(label=label),
                patch("urllib.request.urlopen", side_effect=missing),
                patch("socket.create_connection") as connect,
                self.assertRaisesRegex(RuntimeError, f"{label} service is still starting"),
            ):
                server._check_http_health(label, target, self.http_asr_health, expects_ready_json=True)
            connect.assert_not_called()

    def test_non_transcribe_http_asr_uses_tcp_for_catalog_and_connect(self) -> None:
        entry = _entry("self-hosted:streaming", server=self.http_asr_server)
        for asr_id in ("self-hosted:streaming", ""):
            with (
                self.subTest(asr_id=asr_id),
                patch.object(server, "load_service_entry_by_id", return_value=entry),
                patch.object(server, "load_service_entry", return_value=entry),
                patch("urllib.request.urlopen") as http,
                patch("socket.create_connection") as connect,
            ):
                result = asyncio.run(server._with_service_status({"asr": [entry]}))
                self.assertEqual(result["asr"][0]["status"], "ready")
                asyncio.run(
                    server._ensure_asr_ready_for_connection(
                        {"asr_id": asr_id, "asr_server": self.http_asr_server}, self.http_asr_example
                    )
                )
                http.assert_not_called()
                self.assertEqual(connect.call_count, 2)

    def test_connect_check_reports_missing_service_as_not_running(self) -> None:
        with (
            patch("socket.create_connection", side_effect=socket.gaierror),
            self.assertRaisesRegex(RuntimeError, "not running at nemo-speech:50051"),
        ):
            server._check_service_port("ASR", "nemo-speech:50051")


if __name__ == "__main__":
    unittest.main()
