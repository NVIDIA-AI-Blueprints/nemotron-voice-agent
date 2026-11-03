# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's frontend model: the base class, the call spec, and the providers by name.

``FRONTENDS`` maps each provider name ``config.yaml`` may use (``frontend.cascade.provider``) to its class; the agent's
build method picks from it. To add a provider, subclass ``Frontend`` with its ``name``, add it to ``FRONTENDS``, and
name it in ``config.yaml``. The backend role (``Backend``, ``BACKENDS``) is in the delegation core
(``delegation_core.backend.providers``).
"""

from .base import CallSpec, Frontend
from .chat_completions import ChatCompletionsFrontend
from .openai_responses import OpenAIResponsesFrontend

__all__ = ["FRONTENDS", "CallSpec", "Frontend"]

FRONTENDS: dict[str, type[Frontend]] = {
    provider.name: provider for provider in (ChatCompletionsFrontend, OpenAIResponsesFrontend)
}
