# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import pytest


@pytest.fixture(autouse=True)
def custom_endpoints_on(monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_CUSTOM_ENDPOINTS", "1")
