# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation contract of the Live gateway, the delegation core and the frontend adapter, and the model endpoint.

The gateway sends each delegation to the core as a Responses request with these options and metadata keys, and each of
the three knows a response's end by its terminal event. The example's catalog resolves a :class:`ModelEndpoint` for the
core's backend models and for the frontend's talker. It imports only the standard library, so the delegation core
imports it without loading Pipecat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# The terminal events of a response (and of a backend attempt).
TERMINALS: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})
# The options a request may carry beside the model, the instructions, the tools, the tool choice and the output limit.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# The metadata keys the core reads: the delegation a request starts or continues, and the mark of a typed run (the
# client's typed messages, or nothing, run as a delegation of its own), whose value is "true".
DELEGATION_ID = "live_delegation_id"
TYPED_RUN = "live_typed_run"
# The phase that marks a message item as progress; a message with any other phase, or none, is a final answer.
PROGRESS_PHASE = "commentary"


def is_progress(item: dict) -> bool:
    """Whether a message item is progress: phase ``commentary``. A message with no phase is a final answer."""
    return item.get("phase") == PROGRESS_PHASE


@dataclass(frozen=True)
class ModelEndpoint:
    """A model endpoint, resolved from a ``services.yaml`` entry and the session body.

    ``extra_params`` are passed to the client verbatim (``temperature``, ``max_tokens``, ``extra_body``,
    ``stream_options`` ...), so settings written there by ``settings.yaml`` take effect unchanged.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)
