# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool sources the backend runs on the server, by the name ``backend.tools`` gives them.

A tool source implements ``ToolSource`` (``backend/port.py``) and builds itself from its ``backend.tools`` block, which
names URLs and keys by environment variable so it holds no secret. Each IM session gets a fresh set, so per-session
state is never shared. Adding a tool set is one class and one entry here.
"""

from collections.abc import Mapping

from delegation_core.backend.tools.cafe.tools import CafeTools

# Each tool source by its name in backend.tools, which is also the server label of its calls.
TOOL_SOURCES: Mapping[str, type] = {source.label: source for source in (CafeTools,)}

__all__ = ["TOOL_SOURCES", "CafeTools"]
