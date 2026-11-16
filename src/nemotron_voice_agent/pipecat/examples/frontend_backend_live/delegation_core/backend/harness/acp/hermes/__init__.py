# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes over ACP (``backend.harness.agent: hermes``), loaded only when a core is built with ``agent: hermes``."""

from .agent import HermesAgent

__all__ = ["HermesAgent"]
