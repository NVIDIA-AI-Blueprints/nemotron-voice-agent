# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core's ids and size estimates, its own copies of the live protocol's."""

from uuid import uuid4


def uid(prefix: str) -> str:
    """Return a unique id such as ``event_3f9a...``."""
    return f"{prefix}_{uuid4().hex}"


def approx_tokens(text: str) -> int:
    """Return a cheap token estimate (about four characters per token) for the protocol's size limits."""
    return (len(text) + 3) // 4
