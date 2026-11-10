# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server tool sources by their ``backend.tools`` name: ``port.ToolSource`` classes with ``from_settings(block)``."""

from collections.abc import Mapping

from examples.frontend_backend_live.delegation_core.backend.tools.cafe.tools import CafeTools

TOOL_SOURCES: Mapping[str, type] = {source.label: source for source in (CafeTools,)}

__all__ = ["TOOL_SOURCES", "CafeTools"]
