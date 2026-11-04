# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM, directly connected to a model provider (``backend.adapter: llm``, the default).

``LlmSource`` is built once per delegation core, ``LlmAdapter`` once per IM session, and ``LlmRun`` runs one task's
rounds; the backend LLM agent (``llm-agent``) runs its workers as ``LlmRun`` too.
"""

from .adapter import LlmAdapter
from .run import LlmRun
from .settings import LlmSettings
from .source import LlmSource

__all__ = ["LlmAdapter", "LlmRun", "LlmSettings", "LlmSource"]
