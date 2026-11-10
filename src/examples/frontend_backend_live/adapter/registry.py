# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend adapters a deployment can choose by name: ``frontend.adapter`` in the example's settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import FrontendAdapterClass

FRONTEND_ADAPTERS: Mapping[str, FrontendAdapterClass] = {"cascade": CascadeAdapter}
