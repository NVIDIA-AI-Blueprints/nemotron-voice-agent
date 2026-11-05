# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The frontend adapter's contracts with its neighbors: the context meter it serves."""

import unittest
from types import SimpleNamespace

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter


class ContextMeterTests(unittest.TestCase):
    def test_the_context_window_is_the_talkers_once_the_voice_is_built(self):
        self.assertIsNone(CascadeAdapter.context_window_tokens(SimpleNamespace(voice=None)))
        memory = SimpleNamespace(context_tokens=16_000)
        started = SimpleNamespace(voice=object(), live_config=SimpleNamespace(memory=memory))
        self.assertEqual(CascadeAdapter.context_window_tokens(started), 16_000)
