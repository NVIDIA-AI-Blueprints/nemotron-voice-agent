# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Speech endpoints: channel auth from the catalog, handed to the shared Pipecat services.

v1's own Riva stream adapters were deleted with v1's host; the Pipecat adapter gives the same
resolved endpoint (``core/config/catalog.py``) to ``NvidiaForceEouSTTService`` and Pipecat's
``NvidiaTTSService`` (``adapters/pipecat/services.py``, plan §3.2), which open the channels.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from _fbv_core_fakes import base_config

from examples.frontend_backend_verdict.adapters.pipecat.services import build_stt, build_tts
from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_verdict.core.config.catalog import build_endpoint


def _nvcf(kind: str):  # noqa: ANN202
    return build_endpoint(
        kind,
        {"server": "grpc.nvcf.nvidia.com:443", "function_id": "fn-123", "model": "m", "voice_id": "v"},
        source="test",
        environ={"NVIDIA_API_KEY": "nvapi-secret"},
    )


class AuthTests(unittest.TestCase):
    def test_nvcf_endpoint_uses_ssl_function_id_and_bearer(self) -> None:
        endpoint = _nvcf("asr")
        self.assertTrue(endpoint.use_ssl)
        self.assertEqual(endpoint.metadata(), [["function-id", "fn-123"], ["authorization", "Bearer nvapi-secret"]])
        self.assertNotIn("nvapi-secret", endpoint.describe())
        self.assertNotIn("nvapi-secret", repr(endpoint))

    def test_local_endpoint_is_insecure_without_function_id(self) -> None:
        endpoint = build_endpoint("tts", {"server": "nemo-speech:50051"}, source="test", environ={})
        self.assertFalse(endpoint.use_ssl)
        self.assertEqual(endpoint.metadata(), [])


class PipecatServiceTests(unittest.TestCase):
    """The endpoint reaches the shared services unchanged (no channel is opened here)."""

    def setUp(self) -> None:
        config = base_config()
        self.config = replace(
            config,
            asr=replace(config.asr, endpoint=_nvcf("asr")),
            tts=replace(config.tts, endpoint=_nvcf("tts")),
        )
        self.settings = load_pipecat_settings()

    def test_stt_gets_the_nvcf_endpoint(self) -> None:
        with mock.patch("examples.shared.nvidia_force_eou_stt.NvidiaForceEouSTTService") as service:
            build_stt(self.config, self.settings)
        kwargs = service.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "nvapi-secret")
        self.assertEqual(kwargs["server"], "grpc.nvcf.nvidia.com:443")
        self.assertTrue(kwargs["use_ssl"])
        self.assertEqual(kwargs["model_function_map"], {"function_id": "fn-123", "model_name": "m"})
        self.assertEqual(kwargs["ttfs_p99_latency"], self.settings.turn.stt_ttfs_p99)

    def test_tts_gets_the_nvcf_endpoint_and_the_profile_rate(self) -> None:
        with mock.patch("pipecat.services.nvidia.tts.NvidiaTTSService") as service:
            build_tts(self.config, self.settings)
        kwargs = service.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "nvapi-secret")
        self.assertTrue(kwargs["use_ssl"])
        self.assertEqual(kwargs["model_function_map"], {"function_id": "fn-123", "model_name": "m"})
        self.assertEqual(kwargs["sample_rate"], self.config.tts.sample_rate)

    def test_local_endpoint_sends_no_key(self) -> None:
        local = build_endpoint("asr", {"server": "nemo-speech:50051"}, source="test", environ={})
        config = replace(self.config, asr=replace(self.config.asr, endpoint=local))
        with mock.patch("examples.shared.nvidia_force_eou_stt.NvidiaForceEouSTTService") as service:
            build_stt(config, self.settings)
        kwargs = service.call_args.kwargs
        self.assertIsNone(kwargs["api_key"])
        self.assertFalse(kwargs["use_ssl"])
        self.assertNotIn("model_function_map", kwargs)


if __name__ == "__main__":
    unittest.main()
