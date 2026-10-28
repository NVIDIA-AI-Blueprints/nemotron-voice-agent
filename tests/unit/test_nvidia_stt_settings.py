# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared endpointing settings preserve native defaults and reject bad overrides."""

import inspect
import os
import unittest
from unittest.mock import patch

from pipecat.services.nvidia.stt import NvidiaSTTService

from examples.shared.nvidia_stt_settings import resolve_stt_stop_history_ms


class NvidiaSTTSettingsTests(unittest.TestCase):
    """Validate the server-owned input before either ASR adapter starts."""

    def test_absence_preserves_the_native_constructor_default(self):
        """Deployments that do not opt in keep current Pipecat endpointing."""
        with patch.dict(os.environ, {}, clear=True):
            actual = resolve_stt_stop_history_ms()
        self.assertEqual(actual, inspect.signature(NvidiaSTTService).parameters["stop_history"].default)
        self.assertEqual(actual, 320)

    def test_valid_overrides_include_the_baseline_and_boundaries(self):
        """Explicit millisecond windows are not rounded or clamped."""
        for value in (1, 320, 400, 10000):
            with self.subTest(value=value), patch.dict(os.environ, {"ASR_STOP_HISTORY_MS": str(value)}):
                self.assertEqual(resolve_stt_stop_history_ms(), value)

    def test_invalid_overrides_fail_without_silent_fallback(self):
        """Bad deployment input must not masquerade as a matched configuration."""
        for value in ("", "0", "-1", "1.5", "true", " 400 ", "10001"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"ASR_STOP_HISTORY_MS": value}),
                self.assertRaisesRegex(ValueError, "ASR_STOP_HISTORY_MS"),
            ):
                resolve_stt_stop_history_ms()
