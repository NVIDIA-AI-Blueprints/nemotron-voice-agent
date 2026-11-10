# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend adapters a delegation core can be built with, by the name ``backend.adapter`` gives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module

from examples.frontend_backend_live.delegation_core.backend.port import AdapterSource

# Loads one adapter's source class on first use, so a harness's packages load only for a harness.
AdapterLoader = Callable[[], type[AdapterSource]]

ADAPTERS: Mapping[str, AdapterLoader] = {
    "llm": lambda: import_module("examples.frontend_backend_live.delegation_core.backend.llm").LlmSource,
    "llm-agent": lambda: import_module("examples.frontend_backend_live.delegation_core.backend.agent").AgentSource,
    "harness": lambda: import_module("examples.frontend_backend_live.delegation_core.backend.harness").HarnessSource,
}
