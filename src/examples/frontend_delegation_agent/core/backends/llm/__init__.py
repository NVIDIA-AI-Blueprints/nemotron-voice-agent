# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LLM backend: a lightweight OpenAI-SDK tool loop that runs in the voice process (llm-backend-plan.md).

It reproduces what Hermes sends to the model and the Hermes behaviours seen in real sessions,
and nothing else (section 6.6 of the plan lists what it leaves out). This module and
``settings.py`` are stdlib only; ``host.py`` imports the OpenAI SDK.
"""

from __future__ import annotations

from examples.frontend_delegation_agent.core.backends.base import BackendSpec, Placement

_PACKAGE = "examples.frontend_delegation_agent.core.backends.llm"

#: Hermes' text the LLM backend renders itself, from ``prompts/<profile>/backend/llm.yaml``.
HARNESS_KEYS: tuple[str, ...] = (
    "harness_system",
    "harness_steer_wrap",
    "harness_redirect_user",
    "harness_placeholder",
    "harness_interrupted",
    "harness_unknown_tool",
    "harness_invalid_tool_exit",
    "harness_invalid_args",
    "harness_tool_loop_warning",
    "harness_loop_exact_failure",
    "harness_loop_same_tool_failure",
    "harness_iteration_limit",
    "harness_continue_now",
)

SPEC = BackendSpec(
    kind="llm",
    host=f"{_PACKAGE}.host:LLMAgentHost",
    placements=frozenset({Placement.IN_PROCESS}),
    settings=f"{_PACKAGE}.settings:LLMSettings",
    required_prompt_keys=HARNESS_KEYS,
    worker_prompts=f"{_PACKAGE}.settings:worker_prompts",
)
