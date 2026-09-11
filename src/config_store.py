# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central in-memory store for runtime configuration discovered from services.

Any module can write to or read from this store. Data is populated during
server startup (e.g. by prewarm) and consumed by API endpoints or pipeline code.
"""

_store: dict[str, object] = {}


def set(key: str, value: object) -> None:
    """Store a value by key."""
    _store[key] = value


def get(key: str, default: object = None) -> object:
    """Retrieve a value by key, returning default if absent."""
    return _store.get(key, default)
