# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every contract test class shares: its base and an event collector.

A contract is a mixin of test methods; each implementation's test subclasses it with a ``unittest`` case and the factory
methods the contract asks for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator


class ContractCase:
    """A contract's base: a mixin of test methods that each implementation's test case subclasses."""


async def collect(events: AsyncIterator[dict], limit: int = 1000) -> list[dict]:
    """Every event of an async iterator, in order (at most ``limit``, so a broken adapter cannot loop forever)."""
    out = []
    async for event in events:
        out.append(event)
        if len(out) >= limit:
            break
    return out
