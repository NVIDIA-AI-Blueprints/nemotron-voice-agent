# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests run with each config file's default prompt profile, whatever the shell exports."""

import pytest


@pytest.fixture(autouse=True)
def _default_prompt_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FDH_PROMPT_PROFILE", raising=False)
