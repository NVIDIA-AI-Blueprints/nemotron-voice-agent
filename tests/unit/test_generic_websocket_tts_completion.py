# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify generic TTS construction selects the shared policy by actual transport."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipecat.services.nvidia.tts import NvidiaTTSSynthesisMode
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

from examples.generic import pipeline
from examples.shared.evaluation_trace import EvaluationTrace
from examples.shared.nvidia_tts_explicit_eof import EXPLICIT_EOF_COMPLETION_POLICY


class _ReachedPipelineAssembly(RuntimeError):
    """Stop after the actual provider constructors and policy observations."""


class GenericWebsocketTTSCompletionTests(unittest.IsolatedAsyncioTestCase):
    """Keep traced and ordinary WebSocket services aligned without model requests."""

    async def construct_tts(self, *, websocket=True, mode="stitched", traced=False, body=None):
        """Build through TTS with real settings and inert network clients."""
        transport = Mock(spec=FastAPIWebsocketTransport) if websocket else object()
        trace = Mock(spec=EvaluationTrace) if traced else None
        catalogs = {
            "llm": {"model_id": "test-model", "base_url": "http://llm/v1"},
            "asr": {"server": "asr:50051"},
            "tts": {"server": "tts:50051", "voice_id": "test-voice", "synthesis_mode": mode},
        }
        with ExitStack() as stack:
            for name, value in (
                ("create_transport", transport),
                ("create_evaluation_trace", trace),
                ("resolve_prompt", ("test", "system prompt")),
                ("nvidia_api_key", "test-speech-key"),
                ("nvidia_llm_api_key", "test-llm-key"),
                ("resolve_tools_available", []),
                ("build_tools_schema", (None, [])),
                ("load_ipa_dictionary", None),
            ):
                stack.enter_context(patch.object(pipeline, name, return_value=value))
            stack.enter_context(patch.object(pipeline, "load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch.object(pipeline.examples_registry, "welcome_message_enabled", return_value=False))
            native_stt = stack.enter_context(patch.object(pipeline, "NvidiaSTTService"))
            observed_stt = stack.enter_context(patch.object(pipeline, "EvaluationNvidiaSTTService"))
            stack.enter_context(patch.object(pipeline, "NvidiaLLMService"))
            native = stack.enter_context(patch.object(pipeline, "NvidiaTTSService"))
            observed = stack.enter_context(patch.object(pipeline, "EvaluationNvidiaTTSService"))
            stack.enter_context(patch.object(pipeline, "build_context_messages", side_effect=_ReachedPipelineAssembly))
            with self.assertRaises(_ReachedPipelineAssembly):
                await pipeline.bot(SimpleNamespace(body=body or {}))
        selected, other = (observed, native) if traced else (native, observed)
        self.constructed_stt = (observed_stt if traced else native_stt).call_args.kwargs
        selected.assert_called_once()
        other.assert_not_called()
        return selected.call_args.kwargs, trace

    async def test_traced_and_ordinary_asr_receive_the_same_deployment_history(self):
        """The configured endpointing window reaches both real construction paths."""
        for traced in (False, True):
            with self.subTest(traced=traced), patch.dict(os.environ, {"ASR_STOP_HISTORY_MS": "400"}):
                await self.construct_tts(traced=traced)
                self.assertEqual(self.constructed_stt["stop_history"], 400)

    async def test_ordinary_and_traced_websocket_use_identical_eof_options(self):
        """Tracing never changes the selected native completion policy or settings."""
        ordinary, _ = await self.construct_tts()
        observed, trace = await self.construct_tts(traced=True)
        self.assertIsNone(ordinary["stop_frame_timeout_s"])
        self.assertIsNone(observed["stop_frame_timeout_s"])
        self.assertEqual(ordinary["settings"], observed["settings"])
        self.assertEqual(ordinary["settings"].synthesis_mode, "stitched")
        self.assertEqual(ordinary["settings"].voice, "test-voice")
        self.assertIs(observed["evaluation_trace"], trace)
        trace.record.assert_called_once_with("tts_context_completion_policy", policy=EXPLICIT_EOF_COMPLETION_POLICY)

    async def test_websocket_synthesis_enum_uses_explicit_eof(self):
        """Catalog enum and string representations select the same policy."""
        kwargs, _ = await self.construct_tts(mode=NvidiaTTSSynthesisMode.STITCHED)
        self.assertIsNone(kwargs["stop_frame_timeout_s"])

    async def test_webrtc_and_eval_transport_keep_native_policy(self):
        """Non-WebSocket transports retain their existing context idle behavior."""
        for traced in (False, True):
            with self.subTest(traced=traced):
                kwargs, trace = await self.construct_tts(websocket=False, traced=traced)
                self.assertNotIn("stop_frame_timeout_s", kwargs)
                if trace:
                    trace.record.assert_called_once_with("tts_context_completion_policy", policy="native_idle")

    async def test_nonstitched_and_unspecified_modes_keep_native_policy(self):
        """The compatibility contract is never applied to per-sentence synthesis."""
        for mode in ("per_sentence", ""):
            with self.subTest(mode=mode):
                kwargs, trace = await self.construct_tts(mode=mode, traced=True)
                self.assertNotIn("stop_frame_timeout_s", kwargs)
                trace.record.assert_called_once_with("tts_context_completion_policy", policy="native_idle")

    async def test_effective_session_mode_controls_policy_without_changing_catalog(self):
        """A session's normal synthesis override selects its matching completion policy."""
        kwargs, _ = await self.construct_tts(mode="per_sentence", body={"tts_synthesis_mode": "stitched"})
        self.assertIsNone(kwargs["stop_frame_timeout_s"])
        self.assertEqual(kwargs["settings"].synthesis_mode, "stitched")
        kwargs, _ = await self.construct_tts(mode="stitched", body={"tts_synthesis_mode": "per_sentence"})
        self.assertNotIn("stop_frame_timeout_s", kwargs)
        self.assertEqual(kwargs["settings"].synthesis_mode, "per_sentence")

    async def test_version_guard_rejects_unvalidated_websocket_contract(self):
        """A dependency drift fails before TTS starts rather than silently changing behavior."""
        with (
            patch("examples.shared.nvidia_tts_explicit_eof.version", return_value="1.13.0"),
            self.assertRaisesRegex(RuntimeError, "requires exactly Pipecat 1.12.0"),
        ):
            await self.construct_tts()
