# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend model providers that ``backend.llm.provider`` names, and how to add one.

To add a provider:

1. Subclass :class:`Backend` (``base.py``) and give it a ``name``.
2. Implement ``_events(payload)``, an async generator of one round's Responses events: ``response.created`` once the
   model answers, each item's ``response.output_item.added`` and ``response.output_item.done``, then one terminal.
   Yield ``None`` while the model works with nothing to pass on; :meth:`Backend.stream` adds retries and time limits.
3. If its client raises other errors than the OpenAI client does, set ``transient_errors``; list the request settings
   it cannot honor in ``unsupported(settings)``.
4. Add the class to :data:`BACKENDS`, and bind ``ProviderContract`` (``tests/unit/contracts/backend.py``) to it.
"""

from collections.abc import Mapping

from .base import Backend, BrokenStreamError, ProviderLimits, make_client
from .chat_completions import ChatCompletionsBackend
from .openai_responses import OpenAIResponsesBackend

# Each provider by its name in backend.llm.provider.
BACKENDS: Mapping[str, type[Backend]] = {
    provider.name: provider for provider in (ChatCompletionsBackend, OpenAIResponsesBackend)
}

__all__ = [
    "BACKENDS",
    "Backend",
    "BrokenStreamError",
    "ChatCompletionsBackend",
    "OpenAIResponsesBackend",
    "ProviderLimits",
    "make_client",
]
