# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The display sink for a host that shows nothing, such as a Live session whose client draws from the gateway relay."""

from __future__ import annotations

from collections.abc import Sequence

from examples.frontend_backend_live.cascade.ports import Ending, ReadyResult


class NullDisplay:
    """A ``DisplaySink`` that shows nothing."""

    def started(self, delegation_id: str, request: str) -> None:
        """Nothing to show."""

    def progress(self, delegation_id: str, line: str) -> None:
        """Nothing to show."""

    def shown(self, delegation_id: str, item: dict) -> None:
        """Nothing to show."""

    def answered(self, result: ReadyResult) -> None:
        """Nothing to show."""

    def spoken(self, delegation_ids: Sequence[str], words: str) -> None:
        """Nothing to show."""

    def ended(self, delegation_id: str, how: Ending) -> None:
        """Nothing to show."""
