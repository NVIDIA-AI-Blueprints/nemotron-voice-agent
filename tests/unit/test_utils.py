# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import unittest
from unittest.mock import patch

from utils import parse_env_float


class ParseEnvFloatTests(unittest.TestCase):
    def test_accepts_finite_value(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT_SETTING": "1.25"}):
            self.assertEqual(parse_env_float("TEST_FLOAT_SETTING", 2.0), 1.25)

    def test_rejects_invalid_and_non_finite_values(self) -> None:
        for raw in ("invalid", "NaN", "Infinity", "-Infinity"):
            with self.subTest(raw=raw), patch.dict(os.environ, {"TEST_FLOAT_SETTING": raw}):
                self.assertEqual(parse_env_float("TEST_FLOAT_SETTING", 2.0), 2.0)

    def test_clamps_value_below_minimum(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT_SETTING": "0.25"}):
            self.assertEqual(parse_env_float("TEST_FLOAT_SETTING", 2.0, min_value=0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
