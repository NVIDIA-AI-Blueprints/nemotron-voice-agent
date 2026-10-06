# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime-shaped opaque ids.

Same shapes as ``realtime.conversation`` (``item_<12 hex>``, ``resp_<12 hex>``).
They are generated here rather than imported because the ``realtime`` package
eagerly imports its Pipecat gateway, and ``wire/`` must stay Pipecat-free.
"""

from __future__ import annotations

import uuid


def new_event_id() -> str:
    """Return a server event id."""
    return f"event_{uuid.uuid4().hex[:20]}"


def new_session_id() -> str:
    """Return a Realtime session id."""
    return f"sess_{uuid.uuid4().hex[:20]}"


def new_item_id() -> str:
    """Return a conversation item id."""
    return f"item_{uuid.uuid4().hex[:12]}"


def new_response_id() -> str:
    """Return a response id."""
    return f"resp_{uuid.uuid4().hex[:12]}"
