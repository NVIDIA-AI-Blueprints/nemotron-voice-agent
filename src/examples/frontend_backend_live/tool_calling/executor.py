# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface between the thinker and the domain's tools."""

from typing import Protocol


class ToolExecutor(Protocol):
    """Runs the thinker's function calls for one session.

    Implementations must be idempotent per ``call_id``: a repeated call must not add a second item or place
    a second order. The thinker never retries a round after a function call was emitted, but a provider
    may deliver the same call twice.
    """

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one tool and return its result as a JSON string.

        An executor that runs the tools of a whole round together, such as ``ClientToolGateway``, may instead provide
        ``execute_batch(calls)``; the worker prefers it.
        """
