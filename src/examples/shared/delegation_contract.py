# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation contract constants for Responses requests and events, and a resolved model endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

TERMINALS: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})
# The request options besides the model, the instructions, the tools, the tool choice and the output limit.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# Metadata keys: the delegation a request starts or continues, and the mark of a typed run, valued "true".
DELEGATION_ID = "live_delegation_id"
TYPED_RUN = "live_typed_run"
PROGRESS_PHASE = "commentary"
# The API a model endpoint speaks when its services.yaml entry names none: chat completions.
CHAT_COMPLETIONS: Final = "chat-completions"


def is_progress(item: dict) -> bool:
    """Return whether a message item is progress (phase ``commentary``) rather than a final answer."""
    return item.get("phase") == PROGRESS_PHASE


@dataclass(frozen=True)
class ModelEndpoint:
    """A model endpoint from a ``services.yaml`` entry and the session body; ``extra_params`` reach the client as is."""

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)
    # The API its server speaks: the entry's ``api``, a provider name of the delegation core's ``BACKENDS``.
    api: str = CHAT_COMPLETIONS
