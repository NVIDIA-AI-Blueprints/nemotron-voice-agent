# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import http.client
import os
import socket
import unittest
import urllib.error
from unittest.mock import patch

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

    def test_connect_check_reports_missing_service_as_not_running(self) -> None:
        with (
            patch("socket.create_connection", side_effect=socket.gaierror),
            self.assertRaisesRegex(RuntimeError, "not running at nemo-speech:50051"),
        ):
            server._check_service_port("ASR", "nemo-speech:50051")


if __name__ == "__main__":
    unittest.main()
