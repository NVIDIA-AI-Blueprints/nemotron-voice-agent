# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent (``backend.adapter: llm-agent``): a model conductor with LLM workers.

The conductor places each request through the control tools (start a Work, split it into tasks, steer, cancel, or
answer a status question) and the IM carries each decision out. The workers are backend LLM runs (``backend/llm``), up
to ``max_running`` tasks at once.
"""

from .adapter import AgentAdapter
from .conductor import LlmConductor
from .settings import AgentSettings, ConductorSettings
from .source import AgentSource

__all__ = ["AgentAdapter", "AgentSettings", "AgentSource", "ConductorSettings", "LlmConductor"]
