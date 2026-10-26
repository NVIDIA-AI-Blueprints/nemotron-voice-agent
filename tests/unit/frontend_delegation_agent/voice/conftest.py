# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The voice-layer tests also use the core tests' helpers (the test head and the config variants)."""

import sys
from pathlib import Path

_CORE_TESTS = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_TESTS) not in sys.path:
    sys.path.insert(0, str(_CORE_TESTS))
