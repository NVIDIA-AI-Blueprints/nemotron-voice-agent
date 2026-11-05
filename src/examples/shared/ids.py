# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ids and size estimates of the Live gateway, the delegation core and the frontend adapter.

It imports only the standard library, so the delegation core imports it without loading Pipecat.
"""

from uuid import uuid4


def uid(prefix: str) -> str:
    """Return a unique id such as ``event_3f9a...``."""
    return f"{prefix}_{uuid4().hex}"


def approx_tokens(text: str) -> int:
    """Return a cheap token estimate: about four characters per token."""
    return (len(text) + 3) // 4
