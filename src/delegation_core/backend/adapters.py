# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend adapters a delegation core can be built with, by the name ``backend.adapter`` gives.

``llm`` is the backend LLM, directly connected (the default); ``llm-agent`` is a model conductor with LLM workers;
``harness`` is an external agent runtime over its protocol, such as Hermes over ACP. Each entry imports its
``AdapterSource`` only when a core is built with it, so a harness's protocol packages load only for a harness. Tests
and the emulator pass their own table to ``build_core(adapters=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module

from delegation_core.backend.port import AdapterSource

# Loads one adapter's source class, on first use.
AdapterLoader = Callable[[], type[AdapterSource]]

ADAPTERS: Mapping[str, AdapterLoader] = {
    "llm": lambda: import_module("delegation_core.backend.llm").LlmSource,
    "llm-agent": lambda: import_module("delegation_core.backend.agent").AgentSource,
    "harness": lambda: import_module("delegation_core.backend.harness").HarnessSource,
}
