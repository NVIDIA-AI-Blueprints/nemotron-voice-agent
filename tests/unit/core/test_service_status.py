# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import http.client
import socket
import unittest
import urllib.error
from unittest.mock import patch

from nemotron_voice_agent.core.service_status import parse_endpoint, probe_http, probe_tcp


class ParseEndpointTest(unittest.TestCase):
    def test_parses_addresses_and_returns_none_for_unusable_ones(self) -> None:
        cases = {
            "tts:50051": ("tts", 50051),
            "[::1]:50051": ("::1", 50051),
            "https://integrate.api.nvidia.com/v1": ("integrate.api.nvidia.com", 443),
            "http://llm:8000/v1": ("llm", 8000),
            "tts:99999": None,
            "[::1:50051": None,
            "": None,
        }
        for server, expected in cases.items():
            with self.subTest(server=server):
                self.assertEqual(parse_endpoint(server), expected)


class ProbeTest(unittest.TestCase):
    def test_unresolvable_host_is_offline_and_refused_port_is_starting(self) -> None:
        with patch("socket.create_connection", side_effect=socket.gaierror):
            self.assertEqual(probe_tcp("nemo-speech:50051", 1), "offline")
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            self.assertEqual(probe_tcp("nemo-speech:50051", 1), "starting")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(socket.gaierror())):
            self.assertEqual(probe_http("http://llm:8000/health", False, 1), "offline")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("u", 503, "", None, None)):
            self.assertEqual(probe_http("http://llm:8000/health", False, 1), "starting")

    def test_refused_local_port_is_offline(self) -> None:
        refused = urllib.error.URLError(ConnectionRefusedError())
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            self.assertEqual(probe_tcp("localhost:50051", 1), "offline")
        with patch("urllib.request.urlopen", side_effect=refused):
            self.assertEqual(probe_http("http://localhost:18000/health", False, 1), "offline")
            self.assertEqual(probe_http("http://[::1]:18000/health", False, 1), "offline")
            self.assertEqual(probe_http("http://llm:8000/health", False, 1), "starting")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(ConnectionResetError())):
            self.assertEqual(probe_http("http://localhost:18000/health", False, 1), "starting")

    def test_truncated_health_response_is_starting(self) -> None:
        for error in (http.client.IncompleteRead(b"{"), http.client.BadStatusLine("")):
            with patch("urllib.request.urlopen", side_effect=error):
                self.assertEqual(probe_http("http://llm:8000/health", True, 1), "starting")
