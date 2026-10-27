# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External agent backend (``backend.adapter: harness``): an agent its operator deploys, reached over its protocol.

Three layers: this generic package (conductor, worker runs, steering, tool bridge), a protocol package such as ``acp``
(client and transports), and agent classes for agents with behavior of their own. The protocol packages (``acp``,
``mcp``, ``websockets``) load only when a core is built with the harness; this module imports none of them.
"""

from .adapter import HarnessAdapter
from .conductor import HarnessConductor
from .settings import HarnessSettings
from .source import HarnessSource
from .worker import WorkerRun

__all__ = ["HarnessAdapter", "HarnessConductor", "HarnessSettings", "HarnessSource", "WorkerRun"]
