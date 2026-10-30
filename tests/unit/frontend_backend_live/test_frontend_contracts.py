# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The frontend adapter's contracts with its neighbors: the endings the cascade reads, the context meter it serves."""

import unittest
from types import SimpleNamespace

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.cascade.ports import Ending
from tests.unit.contracts.endings import ENDINGS


class CascadePortTests(unittest.TestCase):
    def test_the_endings_are_the_fixtures(self):
        self.assertEqual([ending.value for ending in Ending], list(ENDINGS))


class ContextMeterTests(unittest.TestCase):
    def test_the_context_window_is_the_talkers_memory_budget_once_the_cascade_is_built(self):
        self.assertIsNone(CascadeAdapter.context_window_tokens(SimpleNamespace(cascade=None)))
        started = SimpleNamespace(cascade=SimpleNamespace(memory=SimpleNamespace(context_tokens=16_000)))
        self.assertEqual(CascadeAdapter.context_window_tokens(started), 16_000)
