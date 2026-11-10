# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend model providers by the API each model's catalog entry names: :class:`Backend` subclasses in ``BACKENDS``."""

from collections.abc import Mapping

from examples.frontend_backend_live.delegation_core.tables import pick
from examples.shared.delegation_contract import ModelEndpoint

from .base import Backend, BrokenStreamError, ProviderLimits, make_client
from .chat_completions import ChatCompletionsBackend
from .openai_responses import OpenAIResponsesBackend
from .retry import back_off

BACKENDS: Mapping[str, type[Backend]] = {
    provider.name: provider for provider in (ChatCompletionsBackend, OpenAIResponsesBackend)
}


def provider_class(endpoint: ModelEndpoint) -> type[Backend]:
    """Return the provider of the API ``endpoint``'s server speaks: the ``BACKENDS`` entry its ``api`` names."""
    return pick(BACKENDS, endpoint.api, "backend API")


def provider_for(endpoint: ModelEndpoint, limits: ProviderLimits) -> Backend:
    """Return a provider of ``endpoint`` on the API its server speaks, opening and timing its rounds by ``limits``."""
    return provider_class(endpoint)(endpoint, limits)


__all__ = [
    "BACKENDS",
    "Backend",
    "BrokenStreamError",
    "ChatCompletionsBackend",
    "OpenAIResponsesBackend",
    "ProviderLimits",
    "back_off",
    "make_client",
    "provider_class",
    "provider_for",
]
