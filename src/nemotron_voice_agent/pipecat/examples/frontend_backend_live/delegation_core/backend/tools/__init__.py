# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server tool sources by their ``backend.tools`` name: ``port.ToolSource`` classes with ``from_settings(block)``."""

from collections.abc import Mapping

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.tools import (
    AirlineTools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.cafe.tools import (
    CafeTools,
)

TOOL_SOURCES: Mapping[str, type] = {source.label: source for source in (AirlineTools, CafeTools)}

__all__ = ["TOOL_SOURCES", "AirlineTools", "CafeTools"]
