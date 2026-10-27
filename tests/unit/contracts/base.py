# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every contract test class shares: known failures and an event collector.

A contract is a mixin of test methods; each implementation's test subclasses it with a ``unittest`` case and the factory
methods the contract asks for. ``EXPECTED_FAILURES`` maps a test method's name to the change that fixes it
(``{"test_a_stop_is_confirmed": "stops are confirmed"}``): that test is marked expected-to-fail, and it fails as an
unexpected success once the change lands.
"""

from __future__ import annotations

import inspect
import unittest
from collections.abc import AsyncIterator, Mapping
from typing import ClassVar


class ContractCase:
    """A contract's base: known failures, each named by the change that fixes it."""

    EXPECTED_FAILURES: ClassVar[Mapping[str, str]] = {}

    def __init_subclass__(cls, **kwargs):
        """Mark each test that ``EXPECTED_FAILURES`` names as expected to fail, in this class only."""
        super().__init_subclass__(**kwargs)
        for name in cls.__dict__.get("EXPECTED_FAILURES", {}):
            test = getattr(cls, name, None)
            if test is None:
                raise TypeError(f"{cls.__name__}.EXPECTED_FAILURES names {name}, which is no test of the contract")
            setattr(cls, name, unittest.expectedFailure(_own_copy(test)))


def _own_copy(test):
    """A new function that runs ``test``: ``expectedFailure`` marks the function it gets, never the shared one."""
    if inspect.iscoroutinefunction(test):

        async def run(self):
            return await test(self)
    else:

        def run(self):
            return test(self)

    run.__name__, run.__qualname__, run.__doc__ = test.__name__, test.__qualname__, test.__doc__
    return run


async def collect(events: AsyncIterator[dict], limit: int = 1000) -> list[dict]:
    """Every event of an async iterator, in order (at most ``limit``, so a broken adapter cannot loop forever)."""
    out = []
    async for event in events:
        out.append(event)
        if len(out) >= limit:
            break
    return out
